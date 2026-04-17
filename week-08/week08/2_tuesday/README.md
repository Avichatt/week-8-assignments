# Hospital Readmission Optimization Project

## Scenario
Dr. Priya Anand, Head of Analytics at a hospital chain, needs a 30-day readmission prediction system. Real-world hospital datasets are often messy, containing errors in patient age, BMI, and vital records. This project provides a principled approach to cleaning such data, building a neural network from scratch using NumPy, and optimizing clinical intervention costs.

## Sub-steps Overview
- **Sub-steps 1 & 2 (Easy)**: Data quality audit and principled cleaning strategy for 2,000 patient records.
- **Sub-steps 3 & 4 (Medium)**: Building a 3-layer NumPy Neural Network (Forward/Backprop) and training it with weighted loss to handle class imbalance.
- **Sub-step 5 (Medium)**: Clinical cost optimization and administrative recommendation.
- **Sub-step 6 (Hard)**: Reproducing misleading "94% accuracy" results via data leakage analysis.

## Installation
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## How to Run
1. **Generate Messy Data**: `python generate_data.py`
2. **Execute Full Analysis**: Open `hospital_readmission_analysis.ipynb` and run all cells.
3. **Run Individual Modules**:
   - `python audit_data.py`: Identifies data quality issues.
   - `python clean_data.py`: Prepares a modeling-ready dataset.
   - `python modeling.py`: Trains the NumPy neural network.
   - `python clinical_analysis.py`: Finds the optimal clinical threshold.

## Expected Output
- **Audit Logs**: Summary of age/BMI/BP inconsistencies.
- **Visuals**: Training loss curve (`loss_curve.png`).
- **Clinical Recommendation**: Formal brief for Dr. Anand regarding cost-optimized thresholds.
