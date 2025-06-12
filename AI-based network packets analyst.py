import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("KDDTest-21_PRJECT.csv")

# --- Data Cleaning ---
# Fill numeric NaNs with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill object NaNs with mode
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical columns
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# --- Feature & Target split ---
label_col = df.columns[-1]  # Assuming last column is the target
X = df.drop(label_col, axis=1)
y = df[label_col]

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Prediction & Evaluation ---
y_pred = model.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["Normal", "Attack"]

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Predicted {label}" for label in labels],
            yticklabels=[f"Actual {label}" for label in labels])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# --- Feature Importance Plot ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10  # Show top 10 important features

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices][:top_n], y=X.columns[indices][:top_n])
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
