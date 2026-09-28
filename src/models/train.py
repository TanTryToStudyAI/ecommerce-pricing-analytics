from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from .pipeline import FEATURES, RANDOM_STATE, make_baseline, make_linear, make_random_forest, make_svr
from .evaluate import regression_metrics

TARGET = "price"
CV_FOLDS = 5

RF_GRID = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
}

SVR_GRID = {
    "model__kernel": ["rbf", "linear"],
    "model__C": [1, 10, 100],
    "model__epsilon": [0.01, 0.1, 0.5],
    "model__gamma": ["scale", "auto"],
}


def load_data(path):
    df = pd.read_csv(path)
    required = ["title", "price", "availability", "rating", "category", "upc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def train_models(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    category_levels = sorted(df["category"].dropna().unique().tolist())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    baseline = make_baseline(category_levels)
    linear = make_linear(category_levels)
    rf_search = GridSearchCV(
        make_random_forest(category_levels), RF_GRID, scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, refit=True, return_train_score=False
    )
    svr_search = GridSearchCV(
        make_svr(category_levels), SVR_GRID, scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=-1, refit=True, return_train_score=False
    )

    # Fit all models only on the training set.
    baseline.fit(X_train, y_train)
    linear.fit(X_train, y_train)
    rf_search.fit(X_train, y_train)
    svr_search.fit(X_train, y_train)

    fitted = {
        "DummyRegressor": baseline,
        "LinearRegression": linear,
        "RandomForestRegressor": rf_search.best_estimator_,
        "SVR": svr_search.best_estimator_,
    }

    # Model selection uses training-set CV RMSE only. The test set is untouched here.
    cv_rmse = {}
    for name, model in [("DummyRegressor", make_baseline(category_levels)), ("LinearRegression", make_linear(category_levels))]:
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_rmse[name] = -scores.mean()
    cv_rmse["RandomForestRegressor"] = -rf_search.best_score_
    cv_rmse["SVR"] = -svr_search.best_score_

    cv_summary = (
        pd.DataFrame([{"Model": k, "Mean_CV_RMSE": v} for k, v in cv_rmse.items()])
        .sort_values("Mean_CV_RMSE", ascending=True)
        .reset_index(drop=True)
    )
    selected_name = cv_summary.iloc[0]["Model"]
    selected_model = fitted[selected_name]

    tuning_summary = pd.DataFrame([
        {"Model": "RandomForestRegressor", "Best_CV_RMSE": -rf_search.best_score_, "Best_Params": str(rf_search.best_params_)},
        {"Model": "SVR", "Best_CV_RMSE": -svr_search.best_score_, "Best_Params": str(svr_search.best_params_)},
    ])

    # The untouched test set is evaluated exactly once, after selection.
    selected_test_metrics = regression_metrics(y_test, selected_model.predict(X_test))
    test_summary = pd.DataFrame([{"Model": selected_name, **selected_test_metrics}])

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "cv": cv, "models": fitted, "cv_summary": cv_summary,
        "test_summary": test_summary, "selected_name": selected_name,
        "selected_model": selected_model, "selected_test_metrics": selected_test_metrics,
        "tuning_summary": tuning_summary, "rf_search": rf_search, "svr_search": svr_search,
    }


def main(data_path="data/processed/books_processed_data.csv", model_path="models/final_price_model.joblib"):
    df = load_data(data_path)
    result = train_models(df)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["selected_model"], model_path)
    return result


if __name__ == "__main__":
    result = main()
    print("Selected model:", result["selected_name"])
    print("\nCV comparison:")
    print(result["cv_summary"].to_string(index=False))
    print("\nFinal test evaluation (selected model only):")
    print(result["test_summary"].to_string(index=False))
    print("\nTuning summary:")
    print(result["tuning_summary"].to_string(index=False))
