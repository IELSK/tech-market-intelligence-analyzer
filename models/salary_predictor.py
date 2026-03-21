import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_PROCESSED, DATA_ANALYSIS, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load
df = pd.read_parquet(DATA_PROCESSED / "dev_dataset.parquet")
print(f"Loaded dataset: {df.shape}")

# Clean
df = df.dropna(subset=["YearsCode", "DevType", "Country"])
print(f"Shape after dropping NaN: {df.shape}")

# Encode languages 
df["LanguageList"] = df["LanguageHaveWorkedWith"].str.split(";")
df["LanguageList"] = df["LanguageList"].apply(
    lambda langs: [l.strip() for l in langs]
)

mlb_languages = MultiLabelBinarizer()
language_encoded = pd.DataFrame(
    mlb_languages.fit_transform(df["LanguageList"]),
    columns=mlb_languages.classes_,
    index=df.index,
)
print(f"Languages encoded: {len(mlb_languages.classes_)} columns")

# Encode DevType
# DevType also has multiple values separated by ";"
df["DevTypeList"] = df["DevType"].str.split(";").apply(
    lambda types: [t.strip() for t in types]
)

mlb_devtype = MultiLabelBinarizer()
devtype_encoded = pd.DataFrame(
    mlb_devtype.fit_transform(df["DevTypeList"]),
    columns=mlb_devtype.classes_,
    index=df.index,
)
print(f"Dev types encoded: {len(mlb_devtype.classes_)} columns")

# Encode Country 
df["Country_encoded"] = df["Country"].astype("category").cat.codes

# Build feature matrix
X = pd.concat(
    [
        df[["YearsCode", "Country_encoded"]].reset_index(drop=True),
        devtype_encoded.reset_index(drop=True),
        language_encoded.reset_index(drop=True),
    ],
    axis=1,
)
y = df["ConvertedCompYearly"].reset_index(drop=True)

print(f"Feature matrix shape: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Train models
models = {
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    results[name] = {"mae": mae, "r2": r2}
    print(f"  MAE: ${mae:,.0f}")
    print(f"  R²:  {r2:.4f}")

    joblib.dump(model, MODELS_DIR / f"{name}.pkl")
    print(f"  Saved to: {MODELS_DIR / f'{name}.pkl'}")

# Save encoders
joblib.dump(mlb_languages, MODELS_DIR / "mlb_languages.pkl")
joblib.dump(mlb_devtype, MODELS_DIR / "mlb_devtype.pkl")
joblib.dump(df["Country"].astype("category").cat.categories.tolist(), MODELS_DIR / "country_categories.pkl")
joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_columns.pkl")
print(f"\nEncoders saved to: {MODELS_DIR}")

# Summary
print("\nModel comparison:")
for name, metrics in results.items():
    print(f"  {name}: MAE=${metrics['mae']:,.0f} | R²={metrics['r2']:.4f}")