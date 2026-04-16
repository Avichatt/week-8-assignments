import pandas as pd
import numpy as np

def perform_audit(file_path):
    print(f"--- Data Quality Audit for {file_path} ---")
    df = pd.read_csv(file_path)
    
    
    print("\n[Audit: Age]")
    print(f"Missing values: {df['age'].isnull().sum()}")
    print(f"Min age: {df['age'].min()}")
    print(f"Max age: {df['age'].max()}")
    
    
    print("\n[Audit: BMI]")
    print(f"Missing values: {df['bmi'].isnull().sum()}")
    
   
    print("\n[Audit: Gender]")
    print(f"Unique values: {df['gender'].unique()}")
    
    
    print("\n[Audit: Blood Pressure]")
    print(f"Zero values: {(df['blood_pressure'] == 0).sum()}")
    

    print("\n[Audit: Readmission]")
    print(df['readmitted'].value_counts(normalize=True))

if __name__ == "__main__":
    perform_audit("hospital_records.csv")
