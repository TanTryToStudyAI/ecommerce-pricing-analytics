# Project Specification: E-Commerce Pricing Analytics

## 1. Project Overview
This project aims to build a predictive pricing model for e-commerce products using machine learning techniques. We will scrape product data from a sandbox environment (Books to Scrape), clean and normalize the features, and train models to predict product prices based on their attributes. This serves as a comprehensive portfolio project demonstrating data acquisition, preprocessing, and predictive modeling capabilities.

## 2. Project Architecture (Folder Structure)
The project follows a standard Data Science structure:
```text
ecommerce_pricing_analytics/
│
├── data/
│   ├── raw/           # Raw HTML or JSON data scraped from the website
│   └── processed/     # Cleaned, normalized data ready for modeling
│
├── models/            # Saved trained models (e.g., Random Forest .pkl files)
├── notebooks/         # Jupyter notebooks for EDA, visualizations, and model experimentation
├── src/               # Source code for the pipeline
│   ├── data/          # Scripts for web scraping (scraper.py) and data cleaning (preprocess.py)
│   └── models/        # Scripts for model training (train.py) and evaluation (evaluate.py)
│
├── requirements.txt   # List of dependencies (e.g., requests, beautifulsoup4, pandas, scikit-learn)
└── project_specification.md # This document
```

## 3. Phase 1: Data Scraping (Completed)

### 3.1 Objective
Develop a web scraper to extract product information from the target sandbox website and store it as a structured raw dataset.

### 3.2 Target Source
*   **Website:** Books to Scrape (`http://books.toscrape.com/`)
*   **Scope:** All available product pages across all categories.

### 3.3 Data Points to Extract
For every single product, extract the following features:
1.  `title`: The full name of the book.
2.  `price`: The listed price (numeric value only, strip the currency symbol).
3.  `availability`: The stock status (e.g., "In stock (19 available)") -> extract the integer indicating the quantity available.
4.  `rating`: The star rating (convert text like "Three" to integer `3`).
5.  `category`: The genre/category of the book.
6.  `product_description`: The text description of the book.
7.  `upc`: Universal Product Code.

### 3.4 Technical Requirements
*   **Language:** Python 3.x
*   **Libraries:** `requests` (with `Session()` implementation), `BeautifulSoup4` (for HTML parsing).
*   **Pagination:** The scraper must automatically navigate through all pagination links to extract the entire catalog.
*   **Output:** The script must save the scraped data into `data/raw/books_raw_data.csv`.
*   **Robustness:** Implement polite scraping (e.g., `time.sleep()`) and handle potential network timeouts or missing data fields gracefully.

## 4. Project Progress and Future Phases

### 4.1 Phase 2: Data Preprocessing & EDA (Completed)

Phase 2 data cleaning and quality assessment have been completed using the raw dataset produced in Phase 1.

The completed preprocessing work includes:

* **Dataset structure validation:** The raw dataset contains 1,000 observations and 7 variables.
* **Missing-value assessment:** Two `product_description` values are missing. The remaining retained variables contain no missing values.
* **Duplicate assessment:** There are no exact duplicate rows and no duplicate UPCs. One duplicated title was retained because the corresponding records have different product information and different UPCs.
* **Numeric validation:** `price`, `availability`, and `rating` contain valid values. No invalid price, availability, or rating observations were identified.
* **UPC validation:** UPC values are complete, unique, 16-character identifiers and are retained as product identifiers rather than numerical measurements.
* **Category validation:** All 50 source categories were retained, including `Default` and `Add a comment`.
* **Text-quality assessment:** The `product_description` field contains missing values, UTF-8/Latin-1 mojibake, and repeated-prefix/source-format artefacts. The investigation identified possible mojibake in 9 titles and 562 product descriptions, and repeated-prefix patterns in 925 product descriptions.
* **Processed dataset construction:** `product_description` was excluded from the modelling dataset because of its text-quality issues and its long free-text structure. No observations were removed.
* **Final processed dataset:** The resulting dataset contains 1,000 observations and 6 variables:
  `title`, `price`, `availability`, `rating`, `category`, and `upc`.
* **Output:** The processed dataset is saved as `data/processed/books_processed_data.csv`.
* **Data preservation:** The Phase 1 raw dataset remains unchanged.

Categorical encoding has not been applied during the cleaning stage. The `category` variable remains available in its validated categorical form for transformation within the modelling pipeline in Phase 3.

### 4.2 Phase 3: Model Engineering (Completed)

Phase 3 will use the processed dataset produced in Phase 2 to develop and evaluate regression models for predicting `price`. The modelling workflow is designed to provide a reproducible comparison of a linear model, a tree-based nonlinear model, and a support-vector regression model while preventing data leakage during preprocessing and model selection.

#### 4.2.1 Modelling Objective

The primary objective is to predict the continuous `price` of a book using validated product attributes from the Phase 2 processed dataset.

The target variable is:

* `price`

The modelling features are:

* `availability`
* `rating`
* `category`

The following variables will not be used as predictors:

* `title`: excluded because it is free-text information and is not required for the planned structured-feature regression analysis.
* `upc`: excluded because it is a product identifier rather than a meaningful numerical or categorical predictor.

The processed dataset will not be modified for the purpose of removing observations. Phase 3 will use the validated observations produced by Phase 2.

#### 4.2.2 Data Splitting and Validation Strategy

The dataset will be divided into:

* **80% training data**
* **20% test data**

A fixed `random_state=42` will be used to make the train-test split reproducible.

The test set will remain untouched during model training, hyperparameter tuning, and model selection. It will be used only once the modelling procedure has been finalised for the final performance evaluation.

Within the training data, **5-fold K-Fold cross-validation** will be used for model assessment and hyperparameter tuning. Cross-validation will allow the model-selection process to use multiple training/validation partitions while preserving the independence of the final test set.

#### 4.2.3 Feature Preprocessing

Preprocessing will be performed inside scikit-learn modelling pipelines so that transformations are fitted only on the relevant training data and are consistently applied to validation and test data.

The preprocessing strategy is:

* **Numeric features:** `availability` and `rating`
  * `StandardScaler` will be applied.
* **Categorical feature:** `category`
  * `OneHotEncoder` will be applied.
  * All 50 validated source categories will be retained.
  * `drop="first"` will be used to avoid redundant dummy variables for linear regression.
  * `handle_unknown="ignore"` will be used so that previously unseen categories do not cause prediction-time failures.

`rating` will be treated as a numeric ordinal variable on its validated 1–5 scale rather than being one-hot encoded.

No rare-category aggregation will be performed.

#### 4.2.4 Target Transformation and Outlier Treatment

The original `price` scale will be retained.

No log transformation of `price` will be applied because the Phase 3 modelling-readiness assessment found no strong skewness requiring such a transformation. The observed price distribution has a mean of approximately 35.07, a median of approximately 35.98, and skewness of approximately -0.037.

No price observations will be removed or winsorised as an outlier treatment. The modelling-readiness assessment identified no price observations outside the IQR-based outlier boundaries.

#### 4.2.5 Baseline Model

A `DummyRegressor` using the mean prediction strategy will be included as a baseline.

The baseline establishes the predictive performance that can be achieved by always predicting the mean training-set price. The trained regression models must be evaluated against this baseline to determine whether they provide useful predictive improvement.

#### 4.2.6 Regression Models

Three primary regression approaches will be developed:

1. **Multiple Linear Regression**
   * Used as an interpretable linear reference model.
   * No hyperparameter tuning will be performed because standard multiple linear regression does not expose the type of model-complexity hyperparameters targeted by the planned grid search.

2. **Random Forest Regression**
   * Used to model nonlinear relationships and interactions between predictors.
   * Hyperparameters will be tuned using `GridSearchCV`.

3. **Support Vector Regression (SVR)**
   * Used as an alternative regression approach capable of modelling nonlinear relationships depending on the selected kernel.
   * Hyperparameters will be tuned using `GridSearchCV`.

#### 4.2.7 Hyperparameter Tuning

`GridSearchCV` with **5-fold K-Fold cross-validation** will be used to tune the Random Forest Regression and SVR models.

The primary `GridSearchCV` scoring criterion will be **RMSE**, implemented using the appropriate negative-RMSE scoring convention required by scikit-learn.

The best hyperparameter configuration for each tuned model will therefore be the configuration with the lowest mean cross-validated RMSE on the training data.

The Linear Regression model will be fitted without hyperparameter grid search and will serve as the untuned linear reference model.

#### 4.2.8 Evaluation Metrics

Three complementary regression metrics will be reported:

* **RMSE (Root Mean Squared Error)** — the primary performance metric. RMSE is expressed in the same units as `price` and gives greater influence to larger prediction errors.
* **MAE (Mean Absolute Error)** — added as a complementary error metric because it expresses the typical absolute prediction error in the original price units and is less sensitive to individual large errors than RMSE. Reporting both RMSE and MAE provides a more complete view of prediction error.
* **R² (Coefficient of Determination)** — reported as a complementary measure of the proportion of variation in `price` explained by the model.

RMSE will be the primary metric for hyperparameter tuning and model selection. MAE and R² will be reported alongside RMSE to provide complementary information about model performance.

#### 4.2.9 Model Selection and Final Evaluation

Model selection will be based on the **lowest mean cross-validated RMSE on the training data**. This prevents the untouched test set from influencing model selection.

After the modelling and selection procedure has been finalised, the selected model will be evaluated once on the untouched 20% test set using:

* Test RMSE
* Test MAE
* Test R²

The final test-set evaluation will therefore provide an out-of-sample estimate of the selected model's predictive performance.

Model comparison will include the DummyRegressor baseline, Multiple Linear Regression, tuned Random Forest Regression, and tuned SVR.

No model-performance results will be assumed before the models are actually trained and evaluated.

#### 4.2.10 Leakage Control and Reproducibility

The modelling workflow will apply preprocessing through scikit-learn pipelines and will keep the test set isolated from model fitting, preprocessing fitting, hyperparameter tuning, and model selection.

The following reproducibility settings will be used where applicable:

* `random_state=42`
* 5-fold K-Fold cross-validation
* fixed 80/20 train-test split

This ensures that preprocessing, cross-validation, tuning, and final evaluation follow a consistent and reproducible procedure.

### 4.2.11 Final Model Output

After model selection and final test evaluation, the final fitted preprocessing-and-model pipeline is saved under the existing `models/` directory defined in the project folder structure.

The saved pipeline contains the preprocessing required to transform new observations consistently with the training data, together with the selected regression model.

Phase 3 produced:

* trained regression models and their evaluation results;
* a comparison of baseline and candidate models;
* the selected final modelling pipeline;
* documented modelling decisions and methodological rationale; and
* a saved final model/pipeline for future prediction use.

#### 4.2.12 Phase 3 Completion Status

Phase 3 model engineering has been completed using the validated dataset produced in Phase 2. The planned modelling workflow was implemented and executed, including the preprocessing pipeline, baseline model, linear regression reference model, tuned Random Forest Regression, and tuned Support Vector Regression.

The completed Phase 3 work includes:

* **Data splitting:** The processed dataset was divided into 80% training data and 20% test data using `random_state=42`.
* **Preprocessing pipeline:** `StandardScaler` was applied to `availability` and `rating`, while `OneHotEncoder` was applied to `category` with `drop="first"` and `handle_unknown="ignore"`. The validated 50-category schema was retained for categorical encoding.
* **Baseline model:** A `DummyRegressor` using the mean prediction strategy was fitted as the baseline.
* **Linear model:** Multiple Linear Regression was fitted as the untuned linear reference model.
* **Random Forest model:** Random Forest Regression was tuned using `GridSearchCV` with 5-fold cross-validation and negative RMSE scoring.
* **SVR model:** Support Vector Regression was tuned using `GridSearchCV` with 5-fold cross-validation and negative RMSE scoring.
* **Model selection:** Model selection was based on the lowest mean cross-validated RMSE from the training data, without using the test set during model selection.
* **Final model selection result:** Based on the completed cross-validation comparison, the `DummyRegressor` achieved the lowest mean cross-validated RMSE and was therefore selected under the specified model-selection rule.
* **Final test evaluation:** The selected model was evaluated on the untouched 20% test set after model selection. Test RMSE, MAE, and R² were calculated and recorded.
* **Reproducibility:** The Phase 3 workflow uses the specified fixed random state and cross-validation configuration and is implemented through reusable Python source code and a Jupyter notebook.
* **Model artifact:** The final fitted preprocessing-and-model pipeline was saved as `models/final_price_model.joblib`.
* **Results and documentation:** Cross-validation results, tuning results, final test results, and Phase 3 modelling decisions were saved under the project `reports/` and `docs/` directories.

No Phase 1 raw data or Phase 2 processed observations were modified as part of Phase 3.
