# 🏠 Real Estate Investment Advisor

Predicting Property Profitability & Future Value using Machine Learning

## Overview
An ML-powered application that helps real estate investors:
- **Classify** whether a property is a "Good Investment"
- **Predict** the estimated property price after 5 years

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Streamlit, MLflow, Matplotlib, Seaborn

## Project Structure

real-estate-investment-advisor/
├── data/ # Dataset (india_housing_prices.csv)
├── notebooks/ # EDA, feature engineering, model training
├── models/ # Saved model artifacts (.pkl) — generated locally, not in repo
├── src/ # Source scripts
├── str_app.py # Streamlit application
├── requirements.txt # Python dependencies
└── README.md


## Setup & Run

1. Clone this repository
```bash
git clone https://github.com/deepsingh95/real-estate-investment-advisor.git
cd real-estate-investment-advisor
```

2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the notebook (`notebooks/01_data_loading_cleaning.ipynb`) once to regenerate the `models/` folder — it is excluded from the repo due to file size limits.

5. Run the Streamlit app
```bash
streamlit run str_app.py
```

## Model Results

**Classification (Good_Investment)** — Best: Random Forest (Accuracy: 80.96%, ROC-AUC: 0.927)

**Regression (Future_Price_5Y)** — Best: Linear Regression (R² = 1.0000)

## MLflow Tracking
Experiments logged under `Real_Estate_Classification` and `Real_Estate_Regression`.
Best models registered as:
- `RealEstate_GoodInvestment_Classifier`
- `RealEstate_FuturePrice_Regressor`

View locally:
```bash
cd notebooks
mlflow ui
```

## Documentation
See `Real_Estate_Investment_Advisor_Documentation.docx` for full methodology, EDA findings, and results.
