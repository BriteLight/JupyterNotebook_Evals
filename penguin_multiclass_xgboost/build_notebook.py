"""Generate the companion Jupyter notebook using only the Python standard library."""

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md("""# Public-data multiclass classification: Palmer Penguins

This project transfers the original fraud-cancellation notebook's core methodology to a completely different domain: predicting penguin species from field measurements. It retains pandas/NumPy, a scikit-learn `ColumnTransformer` and `Pipeline`, imputation, scaling, one-hot encoding, multiclass XGBoost probabilities, ROC AUC, confusion tables, artifact persistence, and SHAP explanations.

## Why this dataset

The curated Palmer Penguins data has 344 observations, three target species, numeric and categorical predictors, genuine missing values, and observations from 2007–2009. That last field enables a meaningful out-of-time (OOT) evaluation: develop with 2007–2008 and score 2009 without training on it. The data is CC0 and small enough to study interactively.

Source and documentation: https://github.com/allisonhorst/palmerpenguins

## Other strong next studies

1. **UCI Steel Plates Faults** — 1,941 rows, 27 predictors, seven fault classes. Closest operational analogue to risk/fault classification; good next step for imbalance, per-class AUC, and reason codes.
2. **UCI Dry Bean** — 13,611 rows, 16 image-derived numeric features, seven varieties. Best for deeper multiclass tuning and stable SHAP comparisons, but it does not naturally exercise categorical preprocessing or OOT validation.
3. **Forest CoverType** — 581,012 rows and seven classes, available through `sklearn.datasets.fetch_covtype`. Best for scale, sparse indicators, runtime/memory profiling, and XGBoost tuning; less convenient for a first clean study.

Palmer Penguins is selected because it exercises the widest range of the familiar tools in a compact, inspectable notebook. Treat this as a learning workflow, not a biological identification system."""),
    md("""## Study design and leakage controls

- Target: `species` (Adelie, Chinstrap, Gentoo).
- Development period: 2007–2008. A stratified split creates train and in-time test sets.
- OOT period: 2009, untouched until final evaluation.
- `year` is used only for splitting and is excluded from the predictors.
- The preprocessor is fit on training rows only. Test and OOT rows use `transform`, never `fit_transform`.
- `OneHotEncoder(handle_unknown='ignore')` makes unseen OOT categories safe.
- Macro OvR ROC AUC gives each species equal weight; accuracy and confusion tables make hard predictions easy to inspect."""),
    code("""# If needed, run once in a fresh environment:
# %pip install pandas numpy scikit-learn xgboost shap matplotlib

from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgb.__version__)
print("shap:", shap.__version__)"""),
    md("""## 1. Load and audit public data

The URL is pinned to a repository commit for reproducibility. If internet access is unavailable, download the CSV once, place it at `data/penguins.csv`, and the loader will use it automatically."""),
    code("""DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv"
)
LOCAL_DATA = Path("data/penguins.csv")

if LOCAL_DATA.exists():
    df = pd.read_csv(LOCAL_DATA)
else:
    df = pd.read_csv(DATA_URL)

print("shape:", df.shape)
display(df.head())
display(pd.DataFrame({"dtype": df.dtypes, "missing": df.isna().sum()}))
display(df["species"].value_counts(dropna=False).rename("count"))
display(pd.crosstab(df["year"], df["species"], margins=True))"""),
    md("""## 2. Define development and out-of-time populations

Rows without a target cannot be supervised examples. Predictor missingness is deliberately retained for the imputers."""),
    code("""TARGET = "species"
TIME_COLUMN = "year"
OOT_YEAR = 2009

model_df = df.dropna(subset=[TARGET, TIME_COLUMN]).copy()
feature_columns = [
    "island", "bill_length_mm", "bill_depth_mm",
    "flipper_length_mm", "body_mass_g", "sex",
]

development_df = model_df.loc[model_df[TIME_COLUMN] < OOT_YEAR].copy()
oot_df = model_df.loc[model_df[TIME_COLUMN] == OOT_YEAR].copy()

label_encoder = LabelEncoder()
label_encoder.fit(model_df[TARGET])
class_names = label_encoder.classes_.tolist()

X_development = development_df[feature_columns]
y_development = label_encoder.transform(development_df[TARGET])
X_oot = oot_df[feature_columns]
y_oot = label_encoder.transform(oot_df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X_development,
    y_development,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_development,
)

print("classes:", dict(enumerate(class_names)))
print("train/test/OOT:", X_train.shape, X_test.shape, X_oot.shape)"""),
    md("""## 3. Build and fit the preprocessing pipeline

This expands the original numeric-only transformer. Numeric columns use median imputation and standardization; categorical columns use most-frequent imputation and one-hot encoding. (Scaling is not required by tree models, but is retained to practice a reusable preprocessing pattern.)"""),
    code("""numeric_features = [
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"
]
categorical_features = ["island", "sex"]

numeric_transformer = Pipeline(steps=[
    ("num_imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("cat_imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess_steps = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_train_array = preprocess_steps.fit_transform(X_train)
X_test_array = preprocess_steps.transform(X_test)
X_oot_array = preprocess_steps.transform(X_oot)
transformed_features = preprocess_steps.get_feature_names_out()

X_train_df = pd.DataFrame(X_train_array, columns=transformed_features, index=X_train.index)
X_test_df = pd.DataFrame(X_test_array, columns=transformed_features, index=X_test.index)
X_oot_df = pd.DataFrame(X_oot_array, columns=transformed_features, index=X_oot.index)

display(X_train_df.head())
print("transformed shapes:", X_train_df.shape, X_test_df.shape, X_oot_df.shape)"""),
    md("""## 4. Train multiclass XGBoost

`multi:softprob` is XGBoost's probability-producing multiclass objective. The test set is used as the validation set; the OOT set remains outside fitting and tuning."""),
    code("""xgb_clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    eval_metric=["merror", "mlogloss"],
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

xgb_clf.fit(
    X_train_df,
    y_train,
    eval_set=[(X_train_df, y_train), (X_test_df, y_test)],
    verbose=False,
)

evaluation_history = xgb_clf.evals_result()
print("final validation mlogloss:", evaluation_history["validation_1"]["mlogloss"][-1])"""),
    md("""## 5. Probability predictions and evaluation"""),
    code("""def evaluate_multiclass(name, y_true, probabilities, labels):
    predicted = np.argmax(probabilities, axis=1)
    metrics = {
        "dataset": name,
        "accuracy": accuracy_score(y_true, predicted),
        "auc_ovr_macro": roc_auc_score(
            y_true, probabilities, labels=np.arange(len(labels)),
            multi_class="ovr", average="macro"
        ),
        "auc_ovo_macro": roc_auc_score(
            y_true, probabilities, labels=np.arange(len(labels)),
            multi_class="ovo", average="macro"
        ),
    }
    confusion = pd.crosstab(
        pd.Categorical.from_codes(y_true, labels),
        pd.Categorical.from_codes(predicted, labels),
        rownames=["Actual"], colnames=["Predicted"], dropna=False,
    )
    return metrics, confusion, predicted

y_pred = xgb_clf.predict_proba(X_test_df)
y_pred_oot = xgb_clf.predict_proba(X_oot_df)

test_metrics, test_confusion, y_pred_abs = evaluate_multiclass(
    "in-time test", y_test, y_pred, class_names
)
oot_metrics, oot_confusion, y_pred_oot_abs = evaluate_multiclass(
    "2009 OOT", y_oot, y_pred_oot, class_names
)

display(pd.DataFrame([test_metrics, oot_metrics]).set_index("dataset"))
display(test_confusion)
display(oot_confusion)
print("OOT classification report:")
print(classification_report(y_oot, y_pred_oot_abs, target_names=class_names, zero_division=0))"""),
    md("""## 6. Inspect class probabilities and one-vs-one AUC

The pairwise table mirrors the original notebook's diagnostic: for each ordered class pair, use the first class's probability as its score."""),
    code("""oot_predictions = pd.DataFrame(y_pred_oot, columns=class_names, index=X_oot.index)
oot_predictions["predicted_species"] = np.array(class_names)[y_pred_oot_abs]
oot_predictions["actual_species"] = np.array(class_names)[y_oot]
oot_predictions["confidence"] = y_pred_oot.max(axis=1)
display(oot_predictions.head(10))

pairwise_auc = []
for first_idx, first_class in enumerate(class_names):
    for second_idx, second_class in enumerate(class_names):
        if first_idx == second_idx:
            continue
        mask = np.isin(y_oot, [first_idx, second_idx])
        binary_actual = (y_oot[mask] == first_idx).astype(int)
        pairwise_auc.append({
            "comparison": f"{first_class} vs {second_class}",
            "auc": roc_auc_score(binary_actual, y_pred_oot[mask, first_idx]),
            "n": int(mask.sum()),
        })

display(pd.DataFrame(pairwise_auc).sort_values("auc"))"""),
    md("""## 7. Learning curves and feature importance"""),
    code("""results = xgb_clf.evals_result()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for metric, axis in zip(["merror", "mlogloss"], axes):
    axis.plot(results["validation_0"][metric], label="train")
    axis.plot(results["validation_1"][metric], label="test")
    axis.set(title=metric, xlabel="boosting round", ylabel=metric)
    axis.legend()
plt.tight_layout()
plt.show()

feature_importance = (
    pd.DataFrame({"feature": transformed_features, "importance": xgb_clf.feature_importances_})
    .sort_values("importance", ascending=False)
)
display(feature_importance)

feature_importance.head(15).sort_values("importance").plot.barh(
    x="feature", y="importance", figsize=(8, 5), legend=False, title="XGBoost feature importance"
)
plt.tight_layout()
plt.show()"""),
    md("""## 8. SHAP global and local explanations

Recent SHAP versions typically return multiclass values shaped `(rows, features, classes)`; older versions may return one matrix per class. The helper normalizes both forms."""),
    code("""explainer = shap.TreeExplainer(xgb_clf)
raw_shap_values = explainer.shap_values(X_test_df)

def shap_by_class(values, n_classes):
    if isinstance(values, list):
        return values
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == n_classes:
        return [values[:, :, i] for i in range(n_classes)]
    if values.ndim == 3 and values.shape[0] == n_classes:
        return [values[i] for i in range(n_classes)]
    raise ValueError(f"Unexpected multiclass SHAP shape: {values.shape}")

shap_classes = shap_by_class(raw_shap_values, len(class_names))

mean_abs_shap = pd.DataFrame(
    {name: np.abs(shap_classes[i]).mean(axis=0) for i, name in enumerate(class_names)},
    index=transformed_features,
)
display(mean_abs_shap.assign(overall=mean_abs_shap.mean(axis=1)).sort_values("overall", ascending=False))

mean_abs_shap.head(15).sort_index().plot.barh(figsize=(9, 6))
plt.title("Mean absolute SHAP value by class")
plt.xlabel("mean |SHAP value|")
plt.tight_layout()
plt.show()

for class_index, class_name in enumerate(class_names):
    print(f"SHAP summary for {class_name}")
    shap.summary_plot(
        shap_classes[class_index], X_test_df,
        feature_names=transformed_features, show=True
    )"""),
    md("""### Local explanation and reusable reason codes

For each OOT row, the reason code lists the five features with the largest positive contributions toward the model's predicted class."""),
    code("""oot_shap_raw = explainer.shap_values(X_oot_df)
oot_shap_classes = shap_by_class(oot_shap_raw, len(class_names))

reason_codes = []
for row_position, predicted_class in enumerate(y_pred_oot_abs):
    contributions = pd.Series(
        oot_shap_classes[predicted_class][row_position],
        index=transformed_features,
    )
    positive = contributions[contributions > 0].sort_values(ascending=False).head(5)
    reason_codes.append(";".join(positive.index))

oot_predictions["reason_code_for_predicted_class"] = reason_codes
display(oot_predictions.sort_values("confidence", ascending=False).head(10))

row_position = int(np.argmax(y_pred_oot.max(axis=1)))
predicted_class = y_pred_oot_abs[row_position]
base_values = np.asarray(explainer.expected_value).reshape(-1)
local_explanation = shap.Explanation(
    values=oot_shap_classes[predicted_class][row_position],
    base_values=base_values[predicted_class],
    data=X_oot_df.iloc[row_position].values,
    feature_names=transformed_features.tolist(),
)
shap.plots.waterfall(local_explanation, max_display=12)"""),
    md("""## 9. Save and reload artifacts

Pickle is convenient when Python/library versions are controlled. Never unpickle an untrusted file. For longer-lived or cross-environment deployment, also retain dependency versions and consider XGBoost's native model format."""),
    code("""with open(ARTIFACT_DIR / "preprocessor.pkl", "wb") as file:
    pickle.dump(preprocess_steps, file)
with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)
with open(ARTIFACT_DIR / "shap_explainer.pkl", "wb") as file:
    pickle.dump(explainer, file)

xgb_clf.save_model(ARTIFACT_DIR / "penguin_xgb_model.json")
oot_predictions.to_csv(ARTIFACT_DIR / "oot_predictions.csv", index=True)

with open(ARTIFACT_DIR / "preprocessor.pkl", "rb") as file:
    loaded_preprocessor = pickle.load(file)
loaded_model = xgb.XGBClassifier()
loaded_model.load_model(ARTIFACT_DIR / "penguin_xgb_model.json")

reloaded_probabilities = loaded_model.predict_proba(
    pd.DataFrame(
        loaded_preprocessor.transform(X_oot),
        columns=loaded_preprocessor.get_feature_names_out(),
        index=X_oot.index,
    )
)
np.testing.assert_allclose(y_pred_oot, reloaded_probabilities, rtol=1e-6, atol=1e-7)
print("Reload check passed. Artifacts written to:", ARTIFACT_DIR.resolve())"""),
    md("""## 10. Suggested experiments

1. Compare the chronological OOT result with repeated stratified cross-validation; explain why they answer different questions.
2. Remove `island` and measure the change. It may be a strong geographic proxy for species, making this a useful leakage/proxy discussion.
3. Tune `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, and `learning_rate` using development data only.
4. Compare constant-zero versus median imputation and inspect how their SHAP explanations differ.
5. Add per-class calibration curves and Brier scores; multiclass AUC alone does not test probability calibration.
6. Graduate to Steel Plates Faults, then implement class-weight experiments, gains/lift tables, and threshold-based review queues.

## Methodology improvements over the source notebook

- Separate executable cells and narrative sections instead of one very large cell.
- Use `multi:softprob`, explicit class ordering, and both OvR/OvO macro AUC.
- Fit preprocessing only on training data and preserve a truly untouched OOT slice.
- Include categorical preprocessing rather than silently dropping categorical fields.
- Avoid hard-coded local paths, deprecated pandas aggregation patterns, and fixed SHAP array assumptions.
- Save a native XGBoost JSON model and verify round-trip predictions after reload."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).with_name("palmer_penguins_multiclass_xgboost.ipynb")
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(output)
