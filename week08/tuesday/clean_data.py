import pandas as pd
import numpy as np
import os

def clean_hospital_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
   
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'] = df['age'].fillna(df['age'].median()).clip(0, 110)
    
   
    def parse_bmi(val):
        if pd.isna(val): return np.nan
        if isinstance(val, str): val = val.replace(' kg/m2', '').strip()
        try: return float(val)
        except: return np.nan
    df['bmi'] = df['bmi'].apply(parse_bmi)
    df['bmi'] = df['bmi'].fillna(df['bmi'].median()).clip(10, 60)
    
    
    df['gender'] = df['gender'].astype(str).str.upper().map({
        'MALE': 'Male', 'M': 'Male', 'FEMALE': 'Female', 'F': 'Female'
    }).fillna('Unknown')
    
   
    df['blood_pressure'] = df['blood_pressure'].replace(0, np.nan).fillna(df['blood_pressure'].median())
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    clean_hospital_data("hospital_records.csv", "clean_hospital_records.csv")
