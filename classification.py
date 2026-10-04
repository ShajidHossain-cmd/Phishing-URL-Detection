from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("data_feature/processed_merged_urls_v2.csv")

y = df["label"]
X = df.drop(columns=["url", "label", "source"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline model: Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# Advanced model: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\n--- Random Forest Results ---")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

# Beyond-unit model 1: HistGradientBoosting
hgb_model = HistGradientBoostingClassifier(
    random_state=42
)

hgb_model.fit(X_train, y_train)
hgb_pred = hgb_model.predict(X_test)
hgb_accuracy = accuracy_score(y_test, hgb_pred)
hgb_precision = precision_score(y_test, hgb_pred)
hgb_recall = recall_score(y_test, hgb_pred)
hgb_f1 = f1_score(y_test, hgb_pred)

print("\n--- HistGradientBoosting Results ---")
print("Accuracy:", hgb_accuracy)
print("Precision:", hgb_precision)
print("Recall:", hgb_recall)
print("F1 Score:", hgb_f1)
print("\nHistGradientBoosting Classification Report:")
print(classification_report(y_test, hgb_pred))

hgb_cm = confusion_matrix(y_test, hgb_pred)

print("\nHistGradientBoosting Confusion Matrix:")
print(hgb_cm)

# Beyond-unit model 2: Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)

print("\n--- Gaussian Naive Bayes Results ---")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("\nGaussian Naive Bayes Classification Report:")
print(classification_report(y_test, nb_pred))

nb_cm = confusion_matrix(y_test, nb_pred)

print("\nGaussian Naive Bayes Confusion Matrix:")
print(nb_cm)



# Logistic Regression error analysis
error_analysis = df.loc[X_test.index, ["url", "label", "source"]].copy()

error_analysis["predicted"] = y_pred

# False positives: actual 0, predicted 1
false_positives = error_analysis[
    (error_analysis["label"] == 0) &
    (error_analysis["predicted"] == 1)
]

# False negatives: actual 1, predicted 0
false_negatives = error_analysis[
    (error_analysis["label"] == 1) &
    (error_analysis["predicted"] == 0)
]

print("\nNumber of False Positives:", len(false_positives))
print("Number of False Negatives:", len(false_negatives))

print("\nExample False Positives:")
print(false_positives.head(10))

print("\nExample False Negatives:")
print(false_negatives.head(10))
# Random Forest error analysis
rf_error_analysis = df.loc[X_test.index, ["url", "label", "source"]].copy()

rf_error_analysis["predicted"] = rf_pred

# False positives: actual 0, predicted 1
rf_false_positives = rf_error_analysis[
    (rf_error_analysis["label"] == 0) &
    (rf_error_analysis["predicted"] == 1)
]

# False negatives: actual 1, predicted 0
rf_false_negatives = rf_error_analysis[
    (rf_error_analysis["label"] == 1) &
    (rf_error_analysis["predicted"] == 0)
]

print("\n--- Random Forest Error Analysis ---")
print("Number of False Positives:", len(rf_false_positives))
print("Number of False Negatives:", len(rf_false_negatives))

print("\nRandom Forest Example False Positives:")
print(rf_false_positives.head(10))

print("\nRandom Forest Example False Negatives:")
print(rf_false_negatives.head(10))

# Cross-source evaluation: Kaggle -> UCI

kaggle_data = df[df["source"] == "kaggle_sid321axn"]
uci_data = df[df["source"] == "uci_phiusiil"]

X_train_cross = kaggle_data.drop(columns=["url", "label", "source"])
y_train_cross = kaggle_data["label"]

X_test_cross = uci_data.drop(columns=["url", "label", "source"])
y_test_cross = uci_data["label"]


# Logistic Regression
cross_scaler = StandardScaler()

X_train_cross_scaled = cross_scaler.fit_transform(X_train_cross)
X_test_cross_scaled = cross_scaler.transform(X_test_cross)

cross_lr = LogisticRegression(max_iter=1000)
cross_lr.fit(X_train_cross_scaled, y_train_cross)

cross_lr_pred = cross_lr.predict(X_test_cross_scaled)

cross_lr_accuracy = accuracy_score(y_test_cross, cross_lr_pred)
cross_lr_precision = precision_score(y_test_cross, cross_lr_pred)
cross_lr_recall = recall_score(y_test_cross, cross_lr_pred)
cross_lr_f1 = f1_score(y_test_cross, cross_lr_pred)
cross_lr_cm = confusion_matrix(y_test_cross, cross_lr_pred)

print("\n--- Cross-Source: Train Kaggle -> Test UCI (Logistic Regression) ---")
print("Accuracy:", cross_lr_accuracy)
print("Precision:", cross_lr_precision)
print("Recall:", cross_lr_recall)
print("F1 Score:", cross_lr_f1)
print("Confusion Matrix:")
print(cross_lr_cm)


# Random Forest
cross_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

cross_rf.fit(X_train_cross, y_train_cross)

cross_rf_pred = cross_rf.predict(X_test_cross)

cross_rf_accuracy = accuracy_score(y_test_cross, cross_rf_pred)
cross_rf_precision = precision_score(y_test_cross, cross_rf_pred)
cross_rf_recall = recall_score(y_test_cross, cross_rf_pred)
cross_rf_f1 = f1_score(y_test_cross, cross_rf_pred)
cross_rf_cm = confusion_matrix(y_test_cross, cross_rf_pred)

print("\n--- Cross-Source: Train Kaggle -> Test UCI (Random Forest) ---")
print("Accuracy:", cross_rf_accuracy)
print("Precision:", cross_rf_precision)
print("Recall:", cross_rf_recall)
print("F1 Score:", cross_rf_f1)
print("Confusion Matrix:")
print(cross_rf_cm)

# Cross-source evaluation: UCI -> Kaggle
X_train_reverse = uci_data.drop(columns=["url", "label", "source"])
y_train_reverse = uci_data["label"]

X_test_reverse = kaggle_data.drop(columns=["url", "label", "source"])
y_test_reverse = kaggle_data["label"]

reverse_scaler = StandardScaler()

X_train_reverse_scaled = reverse_scaler.fit_transform(X_train_reverse)
X_test_reverse_scaled = reverse_scaler.transform(X_test_reverse)

reverse_lr = LogisticRegression(max_iter=1000)
reverse_lr.fit(X_train_reverse_scaled, y_train_reverse)

reverse_lr_pred = reverse_lr.predict(X_test_reverse_scaled)

reverse_lr_accuracy = accuracy_score(y_test_reverse, reverse_lr_pred)
reverse_lr_precision = precision_score(y_test_reverse, reverse_lr_pred)
reverse_lr_recall = recall_score(y_test_reverse, reverse_lr_pred)
reverse_lr_f1 = f1_score(y_test_reverse, reverse_lr_pred)
reverse_lr_cm = confusion_matrix(y_test_reverse, reverse_lr_pred)

print("\n--- Cross-Source: Train UCI -> Test Kaggle (Logistic Regression) ---")
print("Accuracy:", reverse_lr_accuracy)
print("Precision:", reverse_lr_precision)
print("Recall:", reverse_lr_recall)
print("F1 Score:", reverse_lr_f1)
print("Confusion Matrix:")
print(reverse_lr_cm)


reverse_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

reverse_rf.fit(X_train_reverse, y_train_reverse)

reverse_rf_pred = reverse_rf.predict(X_test_reverse)

reverse_rf_accuracy = accuracy_score(y_test_reverse, reverse_rf_pred)
reverse_rf_precision = precision_score(y_test_reverse, reverse_rf_pred)
reverse_rf_recall = recall_score(y_test_reverse, reverse_rf_pred)
reverse_rf_f1 = f1_score(y_test_reverse, reverse_rf_pred)
reverse_rf_cm = confusion_matrix(y_test_reverse, reverse_rf_pred)

print("\n--- Cross-Source: Train UCI -> Test Kaggle (Random Forest) ---")
print("Accuracy:", reverse_rf_accuracy)
print("Precision:", reverse_rf_precision)
print("Recall:", reverse_rf_recall)
print("F1 Score:", reverse_rf_f1)
print("Confusion Matrix:")
print(reverse_rf_cm)

# Source distribution analysis
print("\n--- Kaggle Label Distribution ---")
print(kaggle_data["label"].value_counts())
print("\nKaggle Label Percentages:")
print(kaggle_data["label"].value_counts(normalize=True) * 100)

print("\n--- UCI Label Distribution ---")
print(uci_data["label"].value_counts())
print("\nUCI Label Percentages:")
print(uci_data["label"].value_counts(normalize=True) * 100)
print("\n--- Mean Feature Values by Source ---")
source_feature_means = df.groupby("source")[X.columns].mean()
print(source_feature_means.T)
# Logistic Regression 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
lr_cv_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

lr_cv_scores = cross_val_score(
    lr_cv_model,
    X,
    y,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

print("\n--- Logistic Regression 5-Fold Cross-Validation ---")
print("F1 scores:", lr_cv_scores)
print("Mean F1:", lr_cv_scores.mean())
print("Standard Deviation:", lr_cv_scores.std())

# Final model comparison
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "HistGradientBoosting", "GaussianNB"],
    "Accuracy": [accuracy, rf_accuracy, hgb_accuracy, nb_accuracy],
    "Precision": [precision, rf_precision, hgb_precision, nb_precision],
    "Recall": [recall, rf_recall, hgb_recall, nb_recall],
    "F1": [f1, rf_f1, hgb_f1, nb_f1]
})
print("\n--- Final Model Comparison ---")
print(comparison)

# F1 score comparison visualisation

model_names = ["Logistic Regression", "Random Forest", "HistGradientBoosting", "GaussianNB"]

f1_scores = [f1, rf_f1, hgb_f1, nb_f1]

plt.figure(figsize=(10, 6))
plt.bar(model_names, f1_scores)

plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison of Classification Models")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("model_f1_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
# Random Forest confusion matrix visualisation

disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Benign", "Phishing"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()