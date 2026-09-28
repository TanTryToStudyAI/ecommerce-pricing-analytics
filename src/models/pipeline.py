"""Reusable Phase 3 preprocessing and model-pipeline definitions."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

NUMERIC_FEATURES = ["availability", "rating"]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def make_preprocessor(category_levels=None):
    encoder_kwargs = {"drop": "first", "handle_unknown": "ignore"}
    if category_levels is not None:
        encoder_kwargs["categories"] = [list(category_levels)]
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(**encoder_kwargs), CATEGORICAL_FEATURES),
    ])


def make_pipeline(model, category_levels=None):
    return Pipeline([
        ("preprocessor", make_preprocessor(category_levels)),
        ("model", model),
    ])


def make_baseline(category_levels=None):
    return make_pipeline(DummyRegressor(strategy="mean"), category_levels)


def make_linear(category_levels=None):
    return make_pipeline(LinearRegression(), category_levels)


def make_random_forest(category_levels=None):
    return make_pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), category_levels)


def make_svr(category_levels=None):
    return make_pipeline(SVR(), category_levels)
