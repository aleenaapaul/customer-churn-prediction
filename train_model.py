import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("Telco-Customer-Churn.csv")


# ============================================================
# 2. DATA CLEANING
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(
    subset=["TotalCharges"]
).copy()


# Remove customer ID
df_model = df.drop(
    "customerID",
    axis=1
)


# ============================================================
# 3. TARGET ENCODING
# ============================================================

df_model["Churn"] = df_model["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 4. FEATURES / TARGET
# ============================================================

X = df_model.drop(
    "Churn",
    axis=1
)

y = df_model["Churn"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 7. FINAL TUNED RANDOM FOREST
# ============================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "churn_model.pkl"
)

print("Model saved as churn_model.pkl")