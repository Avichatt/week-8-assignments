import numpy as np
import pandas as pd
from modeling import SimpleNeuralNet, train_and_evaluate

def calculate_clinical_cost(y_true, y_probs, threshold, fn_cost=10000, fp_cost=500):
    preds = (y_probs >= threshold).astype(int)
    fn = np.sum((y_true == 1) & (preds == 0))
    fp = np.sum((y_true == 0) & (preds == 1))
    tp = np.sum((y_true == 1) & (preds == 1))
    return (fn * fn_cost) + (fp * fp_cost) + (tp * fp_cost)

def find_optimal_threshold(nn, X_test, y_test):
    y_probs = nn.forward(X_test)
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = [calculate_clinical_cost(y_test, y_probs, t) for t in thresholds]
    opt_threshold = thresholds[np.argmin(costs)]
    print(f"Optimal Threshold: {opt_threshold:.2f}")
    return opt_threshold, np.min(costs)

def present_recommendation(threshold, cost):
    print(f"\nRecommendation: Use threshold {threshold:.2f} to minimize clinical cost.")

if __name__ == "__main__":
    nn, scaler, features, X_test, y_test = train_and_evaluate()
    opt_t, min_c = find_optimal_threshold(nn, X_test, y_test)
    present_recommendation(opt_t, min_c)
