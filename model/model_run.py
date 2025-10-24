import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================
#       Load Dataset
# ==========================
df = pd.read_csv("C:\\Users\\Zobayer Akib\\OneDrive\\Desktop\\Fast Api\\model\\heart.csv")


# ==========================
#    Data Preprocessing
# ==========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
#       Train Models
# ==========================
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ==========================
#       Evaluate Models
# ==========================
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

print(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# ==========================
#   Choose Best Model
# ==========================
if rf_acc > log_reg_acc:
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = log_reg
    best_name = "Logistic Regression"

print(f"\n Best Model: {best_name} (Accuracy: {max(rf_acc, log_reg_acc):.4f})")

# Optional: detailed report for the best model
print("\nClassification Report:")
if best_name == "Random Forest":
    print(classification_report(y_test, rf_model.predict(X_test)))
else:
    print(classification_report(y_test, log_reg.predict(X_test_scaled)))

# ==========================
#  Save Best Model and Scaler
# ==========================

joblib.dump(best_model, "model/heart_model.joblib")
joblib.dump(scaler, "model/scaler.joblib")

print("\n💾 Model saved successfully")
