# Week 08 · Monday — Time Series Analysis

## E-Commerce Sales Forecasting & Equipment Failure Prediction

PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar  
Date: April 13, 2026

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels prophet scikit-learn

# 2. Place data files in the data/ directory (see structure below)

# 3. Run the analysis
python time_series_analysis.py
```

## Python Version

- **Python 3.10+** recommended (tested on 3.14)

## Packages Required

| Package       | Version  | Purpose                        |
|---------------|----------|--------------------------------|
| pandas        | ≥ 1.5    | Data manipulation              |
| numpy         | ≥ 1.24   | Numerical computation          |
| matplotlib    | ≥ 3.7    | Plotting                       |
| seaborn       | ≥ 0.12   | Statistical visualization      |
| statsmodels   | ≥ 0.14   | ARIMA, SARIMA, ADF, ACF/PACF   |
| prophet       | ≥ 1.1    | Facebook Prophet model         |
| scikit-learn  | ≥ 1.3    | Random Forest, metrics, scaler |

## Project Structure

```
week-08/monday/
├── time_series_analysis.py   # Main analysis script (all 7 sub-steps)
├── README.md                 # This file
├── prompts.md                # AI prompts + critique
├── data/
│   ├── ecommerce/
│   │   ├── olist_orders_dataset.csv
│   │   └── olist_order_payments_dataset.csv
│   └── sensor/
│       └── sensor.csv
└── plots/                    # Generated plots (auto-created)
    ├── substep1_decomposition.png
    ├── substep1_acf_pacf.png
    ├── substep1_overview.png
    ├── substep3_arima_forecast.png
    ├── substep4_model_comparison.png
    ├── substep5_failure_model.png
    └── substep7_fleet_cost.png
```

## Data Sources

- **E-Commerce**: [Brazilian E-Commerce (Olist)](https://kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Sensor**: [Pump Sensor Data](https://kaggle.com/datasets/nphantawee/pump-sensor-data)

## Sub-steps Covered

| #   | Title                          | Level  | Status |
|-----|--------------------------------|--------|--------|
| 1   | E-Commerce Data Characterisation | Easy   | ✅     |
| 2   | Sensor Data Cleaning           | Easy   | ✅     |
| 3   | ARIMA Modeling                 | Medium | ✅     |
| 4   | SARIMA / Prophet Comparison    | Medium | ✅     |
| 5   | Sensor Failure Prediction      | Medium | ✅     |
| 6   | Rule-Based vs ML Comparison    | Hard   | ✅     |
| 7   | Fleet-Wide Deployment Cost     | Hard   | ✅     |
