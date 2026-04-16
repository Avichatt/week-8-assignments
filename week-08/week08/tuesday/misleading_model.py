import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def reproduce_misleading_94_percent():
    df = pd.read_csv("clean_hospital_records.csv")
    df['leaky_feature'] = df['readmitted'] * np.random.normal(10, 0.5, len(df)) + (1-df['readmitted']) * np.random.normal(0, 0.5, len(df))
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    X, y = df[['age', 'leaky_feature']], df['readmitted']
    model.fit(X, y)
    preds = model.predict(X)
    print(f"Misleading Accuracy: {accuracy_score(y, preds)*100:.1f}%")
    print("Confusion Matrix:\n", confusion_matrix(y, preds))

if __name__ == "__main__":
    reproduce_misleading_94_percent()
