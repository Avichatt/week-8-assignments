import pandas as pd
import numpy as np
import os
import shutil
import random
from datetime import datetime, timedelta

def prep_datasets():
    base_dir = r"C:\Users\Avi\.gemini\antigravity\scratch\week-08\thursday\data"
    archive1 = r"C:\Users\Avi\Downloads\archive\RELIANCE.csv"
    archive2 = r"C:\Users\Avi\Downloads\archive (1)\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    
    # 1. Stock Prices
    print("Preparing stock_prices.csv...")
    if os.path.exists(archive1):
        shutil.copy(archive1, os.path.join(base_dir, "stock_prices.csv"))
    else:
        print(f"Error: {archive1} not found.")

    # 2. Chat Logs
    print("Preparing chat_logs.csv...")
    if os.path.exists(archive2):
        df = pd.read_csv(archive2)
        # Take a subset to make processing faster, 5000 rows
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        
        # Add synthetic customer_ids (average 3 interactions per customer)
        n_customers = 5000 // 3
        customer_ids = np.random.randint(1, n_customers + 1, size=5000)
        df['customer_id'] = [f"CUST_{i:04d}" for i in customer_ids]
        
        # Sort by customer_id
        df = df.sort_values(by=['customer_id']).reset_index(drop=True)
        
        def random_date(start, end):
            delta = end - start
            int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
            random_second = random.randrange(int_delta)
            return start + timedelta(seconds=random_second)

        d1 = datetime.strptime('1/1/2023 1:30 PM', '%m/%d/%Y %I:%M %p')
        d2 = datetime.strptime('12/31/2023 4:50 AM', '%m/%d/%Y %I:%M %p')
        
        timestamps = []
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%m-%d-%Y %H:%M',
            '%Y/%m/%d %H:%M:%S'
        ]
        
        for _ in range(len(df)):
            dt = random_date(d1, d2)
            fmt = random.choice(formats)
            timestamps.append(dt.strftime(fmt))
            
        df['timestamp'] = timestamps
        
        # Add churn logic
        churn_labels = []
        for _, group in df.groupby('customer_id'):
            churn_prob = 0.1
            if any('cancel' in str(intent).lower() for intent in group['intent']):
                churn_prob = 0.6
            if len(group) > 4:
                churn_prob += 0.2
            churn = 1 if random.random() < churn_prob else 0
            churn_labels.extend([churn] * len(group))
            
        df['churn'] = churn_labels
        
        df.to_csv(os.path.join(base_dir, "chat_logs.csv"), index=False)
        print("Done.")
    else:
        print(f"Error: {archive2} not found.")

if __name__ == '__main__':
    prep_datasets()
