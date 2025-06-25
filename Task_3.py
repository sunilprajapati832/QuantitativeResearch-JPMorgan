import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib
import warnings
warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("customer_loan_data.csv")

# Preprocessing
X = df.drop(columns=["customer_id", "default"])
y = df["default"]
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "roc_auc": auc,
        "y_prob": y_prob,
        "y_pred": y_pred
    }

# Model Comparison
print("\nModel Comparison:")
for name in results:
    print(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
          f"ROC AUC: {results[name]['roc_auc']:.4f}")

# Best Model
best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = results[best_model_name]["model"]
print(f"\nBest Model: {best_model_name}")

# Confusion Matrix
cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curves
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC: {res['roc_auc']:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()

# Prediction Function
recovery_rate = 0.10

def predict_expected_loss(features_dict):
    df_input = pd.DataFrame([features_dict])
    df_input = df_input.fillna(X.median())
    scaled_input = scaler.transform(df_input)

    pd_prob = best_model.predict_proba(scaled_input)[0][1]
    loan_amt = features_dict["loan_amt_outstanding"]
    expected_loss = pd_prob * loan_amt * (1 - recovery_rate)

    return expected_loss, pd_prob

# Test Example
sample = {
    "credit_lines_outstanding": 3,
    "loan_amt_outstanding": 30000,
    "total_debt_outstanding": 50000,
    "income": 65000,
    "years_employed": 4,
    "fico_score": 700
}

expected_loss, pd_prob = predict_expected_loss(sample)
print(f"\nPredicted PD: {pd_prob:.4f}")
print(f"Expected Loss: ₹{expected_loss:.2f}")

# Save Model & Scaler
joblib.dump(best_model, "pd_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel and scaler saved successfully as pd_model.pkl and scaler.pkl")
