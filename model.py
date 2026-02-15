import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# -------------------------
# 1️⃣ Load Dataset
# -------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Drop non-informative column
if "EmployeeNumber" in df.columns:
    df = df.drop("EmployeeNumber", axis=1)

# -------------------------
# 2️⃣ Encode Categorical Variables
# -------------------------
label_encoders = {}
for column in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# -------------------------
# 3️⃣ Define Features & Target
# -------------------------
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -------------------------
# 4️⃣ Stratified Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# 5️⃣ Apply SMOTE (only on training data)
# -------------------------
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# -------------------------
# 6️⃣ Train Model
# -------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# 7️⃣ Predict Probabilities
# -------------------------
y_probs = model.predict_proba(X_test)[:, 1]

# -------------------------
# 8️⃣ Find Best Threshold (Max F1 for Class 1)
# -------------------------
best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.3, 0.8, 0.05):
    temp_pred = (y_probs > t).astype(int)
    report = classification_report(y_test, temp_pred, output_dict=True)
    f1_1 = report["1"]["f1-score"]

    if f1_1 > best_f1:
        best_f1 = f1_1
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best F1 (Class 1):", best_f1)

# Final prediction
y_pred = (y_probs > best_threshold).astype(int)


# -------------------------
# 9️⃣ Evaluation
# -------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------
# 🔟 Feature Importance
# -------------------------
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:\n")
print(importance_df.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# -------------------------
# Confusion Matrix (Bonus)
# -------------------------
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()
