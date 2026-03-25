import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load dataset
df = pd.read_csv("data/Fraud_Analysis_Dataset.csv")

# 2. Remove columns not needed for prediction
df = df.drop(["nameOrig", "nameDest"], axis=1)

# 3. Create extra helpful features
df["amount_vs_oldbalance_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1e-5)
df["amount_vs_newbalance_orig"] = df["amount"] / (df["newbalanceOrig"] + 1e-5)
df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

# 4. Replace infinity values
df.replace([np.inf, -np.inf], 0, inplace=True)

# 5. Split into input and output
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# 6. Tell Python which columns are text and which are numbers
categorical_features = ["type"]
numeric_features = [col for col in X.columns if col not in categorical_features]

# 7. Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

# 8. Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# 9. Full pipeline = preprocessing + model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# 10. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 11. Train model
pipeline.fit(X_train, y_train)

# 12. Test model
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# 13. Save trained pipeline
joblib.dump(pipeline, "model/fraud_pipeline.pkl")
print("Model saved successfully in model/fraud_pipeline.pkl")