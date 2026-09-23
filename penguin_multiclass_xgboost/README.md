# Palmer Penguins multiclass XGBoost study

Open `palmer_penguins_multiclass_xgboost.ipynb` and run it from top to bottom.

The notebook downloads a commit-pinned CC0 Palmer Penguins CSV on first use. To run offline, place the CSV at `data/penguins.csv`. Generated models, preprocessors, explanations, and predictions are written to `artifacts/`.

Suggested environment:

```bash
python -m pip install pandas numpy scikit-learn xgboost shap matplotlib jupyter
```

The notebook intentionally keeps 2009 as an untouched out-of-time sample and uses 2007–2008 for development.
