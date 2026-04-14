import pandas as pd
import numpy as np
import os

def generate_messy_data(output_path):
    np.random.seed(42)
    n_records = 2000
    data = {
        'patient_id': range(1000, 1000 + n_records),
        'age': np.random.randint(20, 90, n_records),
        'bmi': np.random.normal(27, 5, n_records).astype(object),
        'gender': np.random.choice(['Male', 'Female'], n_records),
        'blood_pressure': np.random.normal(120, 15, n_records),
        'cholesterol': np.random.normal(200, 30, n_records),
        'readmitted': np.random.choice([0, 1], n_records, p=[0.85, 0.15])
    }
    df = pd.DataFrame(data)
    df.loc[df.sample(10).index, 'age'] = -5
    df.loc[df.sample(5).index, 'age'] = 250
    df.loc[df.sample(50).index, 'bmi'] = df.sample(50)['bmi'].apply(lambda x: f"{x:.1f} kg/m2")
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path}")

if __name__ == "__main__":
    generate_messy_data("hospital_records.csv")
