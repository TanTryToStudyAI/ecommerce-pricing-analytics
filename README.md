# E-Commerce Pricing Analytics

An end-to-end data science project using the Books to Scrape sandbox dataset to build and evaluate regression models for book price prediction.

## Project Status

- Phase 1 — Data Scraping: **Completed**
- Phase 2 — Data Preprocessing & EDA: **Completed**
- Phase 3 — Model Engineering: **Completed**
- Final phase — Documentation & GitHub preparation: **Completed locally; ready for repository upload**

## Project Structure

```text
ecommerce_pricing_analytics/
├── data/
│   ├── raw/
│   └── processed/
│       └── books_processed_data.csv
├── models/
│   └── final_price_model.joblib
├── notebooks/
│   └── phase3_model_engineering_final.ipynb
├── reports/
│   ├── phase3_cv_results.csv
│   ├── phase3_tuning_results.csv
│   └── phase3_final_test_results.csv
├── docs/
│   └── phase3_modeling_decisions.md
├── src/
│   └── models/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── train.py
│       └── evaluate.py
├── .gitignore
├── project_specification.md
├── README.md
└── requirements.txt
```

## Modelling Workflow

The Phase 3 model predicts `price` from `availability`, `rating`, and `category`. `title` and `upc` are excluded as specified in the project methodology.

The workflow uses an 80/20 train-test split with `random_state=42`, 5-fold K-Fold cross-validation, preprocessing inside scikit-learn pipelines, and RMSE as the primary tuning and model-selection metric. Candidate models are a mean `DummyRegressor`, Multiple Linear Regression, tuned Random Forest Regression, and tuned SVR.

The final model selected under the specified lowest mean cross-validated RMSE rule is the `DummyRegressor`. Its untouched test-set results are:

| Metric | Test result |
|---|---:|
| RMSE | 14.530465 |
| MAE | 12.661919 |
| R² | -0.005098 |

See `project_specification.md` and `docs/phase3_modeling_decisions.md` for the full methodology and executed results.

## Reproducing Phase 3

From the project root:

```powershell
pip install -r requirements.txt
python -m src.models.train
```

The training command reads `data/processed/books_processed_data.csv` and writes the selected fitted pipeline to `models/final_price_model.joblib`.

To inspect the full modelling workflow, open:

```text
notebooks/phase3_model_engineering_final.ipynb
```

## GitHub

This repository is intended as a personal data-science portfolio project. Do not commit secrets, local virtual environments, Python caches, notebook checkpoints, or raw scraped files whose redistribution has not been verified.
