# Credit Risk Modelling: Lauki Finance

An end-to-end machine learning project that predicts a loan applicant's probability of default from demographic, loan, and credit bureau inputs, converts it into a 300 to 900 credit score with a Poor / Average / Good / Excellent rating, and serves it through an interactive web app for loan officers.

[**Live demo**](ADD-STREAMLIT-APP-URL) (hosted on Streamlit Community Cloud; if the app is asleep, click "Wake up" and allow about 30 seconds)

## Key result

Only about 8.6% of the loans in the data default, and models trained on the raw data missed roughly a quarter of those defaults (72 to 75% recall for the default models). The client asked for **recall above 90%** and **precision above 50%** on the default class, so I tested four modeling strategies. Resampling the training data with SMOTE-Tomek fixed the imbalance, and a tuned Logistic Regression on top of it met both targets while staying fully interpretable, which the SOW also requires.

| Metric (hold-out test set, 12,497 loans) | Value | Client target |
|------------------------------------------|-------|---------------|
| Recall (default class) | **95%** | above 90% |
| Precision (default class) | **55%** | above 50% |
| F1 (default class) | 0.69 | n/a |
| Accuracy | 93% | n/a |
| AUC | 0.984 | n/a |
| Gini | 0.967 | n/a |
| KS statistic | 0.880 | n/a |

The model also rank-orders risk well: the riskiest 20% of applicants contain **98.6%** of all defaults (1,059 of 1,074).

## Project background

The project is built against a case-study scope of work (SOW): as a Data Scientist/ML Engineer I was engaged by Lauki Finance, an India-based Non-Banking Financial Company (NBFC), to build a credit risk model and a scorecard modeled on the CIBIL scoring approach. It categorizes loan applications into Poor, Average, Good, and Excellent.

The SOW defines two phases:

1. **Phase 1 (development and implementation):** build the predictive model, the scorecard, and a Streamlit application. *This repository covers Phase 1.*
2. **Phase 2 (monitoring and MLOps):** performance monitoring and straight-through processing (STP) for high-confidence applications after a 2-month production trial (out of scope here).

### Phase 1 objectives

- A predictive model trained on historical loan data and default indicators
- A scorecard that maps model output to Poor, Average, Good, and Excellent
- A model with **high explainability**, so the business can interpret its behavior
- **Recall above 90% and precision above 50%** on the default class (agreed with the client during the project)
- A user-friendly Streamlit app where loan officers enter borrower demographics, loan details, and bureau information to get a default probability and credit rating in real time

### How the deliverables map to this repo

| SOW deliverable | Where it lives |
|-----------------|----------------|
| Trained model | `artifacts/model_data.joblib` (model, scaler, feature list) |
| Model development notebook | `notebooks/MLProject_Credit-Risk-Model.ipynb` |
| Streamlit app | `main.py` |
| Prediction and scorecard logic | `prediction_helper.py` |
| Deployed model | [Live demo](ADD-STREAMLIT-APP-URL) |
| Documentation | This README |

## How the problem was approached, stage by stage

### 1. Business understanding

Loan officers need a fast, consistent way to judge whether an applicant is likely to default, and the business needs to understand *why* the model scores someone the way it does. Two things shaped the project:

- **Recall matters most.** Missing a future defaulter is costlier than flagging a safe applicant for review, so the client set a recall target above 90%, with precision above 50% so that the flags stay useful.
- **Explainability is a deliverable.** I favored a model whose behavior can be read directly from its coefficients, and I translated its output into a familiar score and rating band rather than a raw probability alone.

### 2. Data understanding

Three tables of 50,000 records each, joined on `cust_id` into a single dataset of **50,000 rows and 33 columns**:

| Table | Columns | Content |
|-------|---------|---------|
| Customers | 12 | Age, gender, marital and employment status, income, dependants, residence type, years at address, city, state, zipcode |
| Loans | 15 | Purpose, type, sanctioned and loan amounts, fees, GST, net disbursement, tenure, principal outstanding, bank balance, dates, **default flag** |
| Bureau | 8 | Open and closed accounts, total loan months, delinquent months, total days past due (DPD), enquiry count, credit utilization ratio |

The target is imbalanced: **4,297 defaults (8.6%)** against 45,703 non-defaults.

To prevent data leakage, I split the data **before** any cleaning, imputation, scaling, or feature engineering: a stratified 75/25 split gave 37,500 training and 12,500 test rows. Every statistic used downstream (imputation mode, scaler, selection decisions) comes from the training set only.

### 3. Data cleaning

| Issue found | Action and reasoning |
|-------------|----------------------|
| 62 missing `residence_type` values (47 in training) | Imputed with the training-set mode ("Owned"), applied to both sets, since this is a categorical column with very few gaps |
| Duplicate rows | None found |
| `processing_fee` greater than loan amount (5 training rows, up to 5.3 million on a 3.6 million loan) | The business contact confirmed the fee is about 3% of the loan amount, so rows above that were treated as errors and dropped (12 training rows and 3 test rows in total) |
| `loan_purpose` typo: "Personaal" | Corrected to "Personal" in both sets |

After cleaning, the training set has **37,488 rows** and the test set **12,497 rows**.

### 4. Exploratory data analysis

Using **matplotlib and seaborn**, I used box plots to look for outliers and histograms for distributions, then plotted KDE curves of every numeric column split by default status to see which features separate defaulters from non-defaulters:

- Younger borrowers were slightly more likely to default
- Higher loan tenure, delinquent months, total DPD, and credit utilization pointed to higher default risk
- Loan amount and income showed no clear signal on their own, which suggested their ratio might, and the loan-to-income ratio confirmed it (next stage)

### 5. Feature engineering

- **Loan-to-income ratio (LTI):** `loan_amount / income`. Defaulters clearly concentrate at higher values
- **Delinquency ratio:** `delinquent_months × 100 / total_loan_months`, the share of a borrower's credit history spent delinquent
- **Average DPD per delinquency:** `total_dpd / delinquent_months`, set to 0 when there were no delinquent months, which avoids division by zero
- The raw columns behind these ratios (`loan_amount`, `income`, `total_loan_months`, `delinquent_months`, `total_dpd`) and the two date columns were removed at the business contact's request
- **Min-max scaling** of the numeric features, fitted on the training set only
- **One-hot encoding** (`drop_first=True`) for the categorical features

### 6. Feature selection and multicollinearity

**Numeric features: Variance Inflation Factor (VIF).** The loan fee columns were redundant copies of each other. `processing_fee`, `gst`, and `net_disbursement` had infinite VIF, `sanction_amount` was 101, and `principal_outstanding` was 16. I dropped all five. Every remaining numeric feature has a VIF below 6.5.

**All features: Information Value (IV).** I computed Weight of Evidence and IV for every feature (numeric features binned into 10 groups) and kept those with IV above 0.02:

| Feature | IV | Kept |
|---------|----|------|
| `credit_utilization_ratio` | 2.353 | Yes |
| `delinquency_ratio` | 0.717 | Yes |
| `loan_to_income` | 0.476 | Yes |
| `avg_dpd_per_delinquency` | 0.402 | Yes |
| `loan_purpose` | 0.369 | Yes |
| `residence_type` | 0.247 | Yes |
| `loan_tenure_months` | 0.219 | Yes |
| `loan_type` | 0.163 | Yes |
| `age` | 0.089 | Yes |
| `number_of_open_accounts` | 0.085 | Yes |
| Everything else (`enquiry_count`, `bank_balance_at_application`, `employment_status`, `gender`, `city`, `state`, and others) | below 0.01 | No |

That leaves **10 features, which become 13 model inputs** after one-hot encoding.

### 7. Modeling: four attempts

The default class is only 8.6% of the data, so I worked through the problem in stages. All results below are on the same untouched test set (1,074 defaults out of 12,497 loans), and the metrics are for the default class.

| Attempt | Approach | Model | Precision | Recall | F1 |
|---------|----------|-------|-----------|--------|----|
| 1 | No imbalance handling | Logistic Regression | 0.85 | 0.72 | 0.78 |
| 1 | No imbalance handling | Random Forest | 0.85 | 0.72 | 0.78 |
| 1 | No imbalance handling | XGBoost | 0.82 | 0.75 | 0.78 |
| 1 | No imbalance handling, tuned (RandomizedSearchCV) | Logistic Regression | 0.83 | 0.74 | 0.78 |
| 1 | No imbalance handling, tuned (RandomizedSearchCV) | XGBoost | 0.77 | 0.83 | 0.80 |
| 2 | Random undersampling | Logistic Regression | 0.51 | 0.95 | 0.67 |
| 2 | Random undersampling | XGBoost | 0.53 | 0.98 | 0.69 |
| 3 | SMOTE-Tomek | Logistic Regression | 0.55 | 0.94 | 0.70 |
| 3 | **SMOTE-Tomek, tuned (Optuna)** | **Logistic Regression (selected)** | **0.55** | **0.95** | **0.69** |
| 4 | SMOTE-Tomek, tuned (Optuna) | XGBoost | 0.72 | 0.83 | 0.77 |

What I learned from each attempt:

1. **Without imbalance handling,** every model reached about 0.78 F1 but recall stayed between 72% and 83%, short of the 90% target. Tuning XGBoost raised recall to 83% at the cost of precision.
2. **Undersampling** pushed recall above 90% but threw away most of the non-default records (34,265 down to 3,223) and left precision barely above the 50% floor.
3. **SMOTE-Tomek** (oversampling the minority class, then removing noisy borderline pairs) balanced the training set at 34,194 per class without discarding data, and gave Logistic Regression 94 to 95% recall with 55% precision. Optuna tuning (50 trials, `C` ≈ 304, `saga` solver) left the scores essentially unchanged, so the default model was already close to optimal.
4. **XGBoost with SMOTE-Tomek and tuning** did not beat Logistic Regression on the client's targets: recall only reached 83%.

**Why Logistic Regression.** It was the only family that met both client targets with the highest precision (55%), and, unlike a boosted tree ensemble, its coefficients can be explained directly to the business. Undersampled XGBoost had slightly higher recall (98%) but lower precision (53%) and is harder to interpret.

### 8. Evaluation of the selected model

| Metric | Value |
|--------|-------|
| AUC | 0.984 |
| Gini (2 × AUC - 1) | 0.967 |
| KS statistic | 0.880 |

**Rank ordering.** Test loans were sorted by predicted default probability and split into 10 equal groups (decile 1 is the riskiest):

| Decile | Loans | Defaults | Actual default rate |
|--------|-------|----------|---------------------|
| 1 | 1,250 | 897 | 71.8% |
| 2 | 1,250 | 162 | 13.0% |
| 3 | 1,249 | 10 | 0.8% |
| 4 | 1,250 | 5 | 0.4% |
| 5 to 10 | 7,498 | 0 | 0.0% |

The default rate falls steadily as predicted risk falls, and the top two deciles hold 1,059 of the 1,074 defaults (98.6%). That is exactly the behavior a scorecard needs.

### 9. Explainability: what drives default risk

Because the final model is a Logistic Regression on scaled features, its coefficients are directly comparable. A positive coefficient raises the default probability; a negative one lowers it.

| Feature | Coefficient | Effect on default risk |
|---------|-------------|------------------------|
| `loan_to_income` | 18.59 | Strongly increases |
| `credit_utilization_ratio` | 16.48 | Strongly increases |
| `delinquency_ratio` | 14.16 | Strongly increases |
| `loan_purpose_Home` | -3.81 | Decreases (vs. Auto) |
| `avg_dpd_per_delinquency` | 1.99 | Increases |
| `residence_type_Rented` | 1.97 | Increases (vs. Mortgage) |
| `residence_type_Owned` | -1.88 | Decreases (vs. Mortgage) |
| `number_of_open_accounts` | 1.30 | Increases |
| `loan_purpose_Personal` | 1.13 | Increases (vs. Auto) |
| `loan_type_Unsecured` | 1.13 | Increases (vs. Secured) |
| `loan_purpose_Education` | 0.99 | Increases (vs. Auto) |
| `loan_tenure_months` | 0.67 | Slightly increases |
| `age` | 0.09 | Negligible |

Intercept: -21.57. Three features dominate the model: the loan-to-income ratio, credit utilization, and the delinquency ratio.

### 10. From probability to credit score

The model's raw output is converted to a score on a 300 to 900 scale, in the spirit of bureau scores:

```
default_probability = 1 / (1 + exp(-(w·x + b)))
credit_score        = 300 + (1 - default_probability) × 600
```

| Credit score | Rating |
|--------------|--------|
| 300 to 499 | Poor |
| 500 to 649 | Average |
| 650 to 749 | Good |
| 750 to 900 | Excellent |

### 11. Deployment

1. The loan officer enters the applicant's details in the Streamlit form: age, income, loan amount, loan tenure, average days past due, delinquency ratio, credit utilization ratio, open loan accounts, residence type, loan purpose, and loan type. The loan-to-income ratio is calculated and displayed automatically.
2. `prediction_helper.py` converts the inputs into the model's 13 features: the loan-to-income ratio and one-hot encoded categories.
3. Numeric features are scaled with the saved `MinMaxScaler`. The scaler was fitted on a wider set of columns than the model uses, so the unused ones are filled with placeholder values to let it run; they are dropped before prediction and do not affect the result.
4. The Logistic Regression model returns the default probability, which is converted into the credit score and rating.
5. The app displays the default probability, the credit score, and the rating.

The model, scaler, feature list, and columns to scale are saved together with joblib in `artifacts/model_data.joblib`, and the app is deployed on Streamlit Community Cloud.

## Results against the acceptance criteria

| Requirement | Target | Result |
|-------------|--------|--------|
| Recall on the default class | above 90% | 95% (met) |
| Precision on the default class | above 50% | 55% (met) |
| Scorecard with Poor / Average / Good / Excellent categories | n/a | Met: 300 to 900 score with four rating bands |
| High explainability | n/a | Met: coefficient-level interpretation (section 9) |
| Streamlit app for real-time assessment | n/a | Met: [live demo](ADD-STREAMLIT-APP-URL) |
| Documentation | n/a | Met: this README |

The recall and precision targets were agreed with the client during the project; the SOW itself sets no numeric thresholds.

## Tech stack

| Area | Tools |
|------|-------|
| Language | Python |
| Data handling | pandas, NumPy |
| Visualization and EDA | matplotlib, seaborn |
| Statistical checks | statsmodels (`variance_inflation_factor`), SciPy (KS statistic), Weight of Evidence / Information Value |
| Modeling | scikit-learn (Logistic Regression, Random Forest, MinMaxScaler, RandomizedSearchCV), XGBoost |
| Class imbalance | imbalanced-learn (RandomUnderSampler, SMOTE-Tomek) |
| Hyperparameter tuning | Optuna |
| Model persistence | joblib |
| App and deployment | Streamlit, Streamlit Community Cloud |
| Version control | Git, GitHub |

## Repository structure

```
├── artifacts/              # Saved model, scaler, and feature list (.joblib)
│   └── model_data.joblib
├── assets/                 # README media
│   └── app_screenshot.png      # App screenshot
├── .gitignore
├── LICENSE                 
├── README.md
├── main.py                 # Streamlit UI
├── prediction_helper.py    # Preprocessing, scoring, rating
└── requirements.txt
```

## Author

**Gires Kibwana Folomo**, Data Scientist, Johannesburg, South Africa
