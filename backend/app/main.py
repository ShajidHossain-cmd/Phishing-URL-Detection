"""URLGuard API: offline URL lexical classifier; no URL is fetched."""
from contextlib import asynccontextmanager
import hashlib
import logging
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.feature_engineering import FEATURES, extract_features

logger = logging.getLogger('urlguard')
BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv('MODEL_PATH', str(BASE/'models'/'random_forest_model.joblib')))
MAX_URL_CHARS = 2048
MAX_REQUEST_BYTES = 4096


def error_response(status: int, code: str, message: str, details=None):
    return JSONResponse(status_code=status, content={'error': {'code': code, 'message': message, 'details': details or {}}})


def validate_url(value: str) -> str:
    value = value.strip()
    if not value or len(value) > MAX_URL_CHARS:
        raise ValueError('URL must contain 1 to 2048 characters')
    if any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in value):
        raise ValueError('URL must not contain control characters')
    if re.search(r'\s', value):
        raise ValueError('URL must not contain whitespace')
    if re.match(r'^[a-z][a-z0-9+.-]*:', value, flags=re.I) and not value.lower().startswith(('http://', 'https://')):
        raise ValueError('Only HTTP and HTTPS URL schemes are supported')
    if value.startswith('//') or '#' in value and value.startswith('#'):
        raise ValueError('URL must include a hostname')
    candidate = value if value.lower().startswith(('http://','https://')) else 'http://' + value
    try:
        parsed = urlsplit(candidate)
        _ = parsed.port
        if not parsed.hostname or ('.' not in parsed.hostname and parsed.hostname != 'localhost' and not re.fullmatch(r'\d+\.\d+\.\d+\.\d+', parsed.hostname)):
            raise ValueError('URL must contain a valid hostname')
        if parsed.username or parsed.password:
            raise ValueError('URLs with embedded credentials are not supported')
    except ValueError as exc:
        raise ValueError(f'URL is invalid: {exc}') from exc
    return value


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=MAX_URL_CHARS, description='HTTP(S) URL, with or without a scheme', examples=['https://example.com/login'])

    @field_validator('url')
    @classmethod
    def check_url(cls, v: str) -> str:
        return validate_url(v)


class PredictResponse(BaseModel):
    url: str
    prediction: str = Field(description='benign or phishing')
    confidence: float = Field(ge=0, le=1, description='Uncalibrated probability assigned to predicted class')
    probabilities: dict[str, float] = Field(description="Per-class uncalibrated probability", examples=[{"benign": 0.83, "phishing": 0.17}])
    model: str = 'RandomForestClassifier'


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model():
    if not MODEL_PATH.exists():
        logger.warning('Model absent: %s. See README training step.', MODEL_PATH)
        return None
    expected_hash = os.getenv('MODEL_SHA256', '').strip().lower()
    if not expected_hash:
        logger.warning('MODEL_SHA256 is not configured; load only a trusted model file')
    else:
        with MODEL_PATH.open('rb') as model_file:
            actual = hashlib.file_digest(model_file, 'sha256').hexdigest()
        if actual != expected_hash:
            raise RuntimeError('Model checksum mismatch')
    model = joblib.load(MODEL_PATH)  # Only load self-trained trusted local artifacts.
    if not callable(getattr(model, 'predict_proba', None)):
        raise RuntimeError('Model must support predict_proba')
    if list(getattr(model, 'feature_names_in_', [])) != FEATURES:
        raise RuntimeError('Model training feature order differs from inference extractor')
    if list(model.classes_) != [0, 1]:
        raise RuntimeError('Model classes must be [0, 1]')
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()  # Once per application startup, never per request.
    yield
    app.state.model = None

app = FastAPI(title='URLGuard Phishing Detection API', version='1.0.0',
              description='Offline lexical URL screening. URLs are never opened or downloaded.', lifespan=lifespan)
origins = [s.strip() for s in os.getenv('FRONTEND_ORIGINS', 'http://localhost:5173').split(',') if s.strip()]
if not origins or '*' in origins:
    raise RuntimeError('FRONTEND_ORIGINS must list exact trusted frontend origins')
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


class BodyLimitMiddleware:
    """Bound raw incoming request bytes even without Content-Length."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            await self.app(scope, receive, send)
            return
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                response = error_response(413, 'REQUEST_TOO_LARGE', 'Request body exceeds 4096 bytes')
                await response(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get('more_body', False):
                break
        body = b''.join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return {'type': 'http.disconnect'}
            delivered = True
            return {'type': 'http.request', 'body': body, 'more_body': False}

        await self.app(scope, replay, send)


app.add_middleware(BodyLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    fields = [{'field': '.'.join(map(str, e['loc'])), 'message': str(e['msg'])} for e in exc.errors()]
    return error_response(422, 'VALIDATION_ERROR', 'Invalid request data', {'fields': fields})


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return error_response(exc.status_code, 'HTTP_ERROR', str(exc.detail))


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception('Unexpected error while handling request')
    return error_response(500, 'INTERNAL_ERROR', 'An unexpected server error occurred')


@app.get('/api/health', response_model=StatusResponse, summary='Check API and model readiness')
def health(request: Request):
    loaded = request.app.state.model is not None
    if not loaded:
        return error_response(503, 'MODEL_UNAVAILABLE', 'Prediction model is not installed')
    return StatusResponse(status='ok', model_loaded=True)


@app.get('/api/model/info', summary='Describe the deployed model')
def model_info(request: Request):
    if request.app.state.model is None:
        return error_response(503, 'MODEL_UNAVAILABLE', 'Prediction model is not installed')
    return {'name': 'RandomForestClassifier', 'labels': {'0': 'benign', '1': 'phishing'},
            'feature_count': len(FEATURES), 'network_requests': False}


def predict_url(url: str, model) -> PredictResponse:
    """Feature order and transformations mirror A2 RandomForest fit(X_train) exactly."""
    frame = pd.DataFrame([extract_features(url)], columns=FEATURES)
    probabilities = model.predict_proba(frame)[0]
    classes = [int(x) for x in model.classes_]
    scores = {int(c): float(p) for c, p in zip(classes, probabilities)}
    label = int(model.predict(frame)[0])
    return PredictResponse(url=url, prediction='phishing' if label == 1 else 'benign',
                           confidence=scores[label], probabilities={'benign': scores[0], 'phishing': scores[1]})


@app.post('/api/predict', response_model=PredictResponse, summary='Classify a URL without visiting it',
          responses={422: {'description': 'Invalid URL / request'}, 413: {'description': 'Body too large'},
                     503: {'description': 'Model unavailable'}, 500: {'description': 'Unexpected inference error'}},
          openapi_extra={'requestBody': {'content': {'application/json': {'example': {'url': 'https://example.com/login'}}}}})
async def predict(payload: PredictRequest, request: Request):
    model = request.app.state.model
    if model is None:
        return error_response(503, 'MODEL_UNAVAILABLE', 'Prediction model is not installed')
    return await run_in_threadpool(predict_url, payload.url, model)


@app.get('/api', summary='API index', description='Lists publicly available application endpoints.')
def api_index():
    return {'service': 'URLGuard', 'endpoints': ['/api/health', '/api/model/info', '/api/predict', '/docs']}
