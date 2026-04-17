import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# 1. Stock Data Prep
def stock_prep(filepath):
    df = pd.read_csv(filepath)
    # Using 'Close' column
    close_prices = df['Close'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    
    window_size = 14
    X, y = create_sequences(scaled_data, window_size)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# 2. Chat Logs Prep
def chat_logs_prep(filepath):
    df = pd.read_csv(filepath)
    # the timestamp column will not parse with standard datetime call due to mixed formats
    # Use pandas to_datetime with mixed format handling or format='mixed'
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    
    # Check EDA: basic aggregation
    df['is_cancel'] = df['intent'].apply(lambda x: 1 if 'cancel' in str(x).lower() else 0)
    agg_df = df.groupby('customer_id').agg({
        'timestamp': ['count', 'min', 'max'],
        'is_cancel': 'sum',
        'churn': 'max'
    }).reset_index()
    agg_df.columns = ['customer_id', 'msg_count', 'first_msg', 'last_msg', 'cancel_count', 'churn']
    agg_df['duration_days'] = (agg_df['last_msg'] - agg_df['first_msg']).dt.total_seconds() / 86400.0
    
    return df, agg_df

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, scaler = stock_prep("data/stock_prices.csv")
    print(X_tr.shape)
    
    df, agg_df = chat_logs_prep("data/chat_logs.csv")
    print(agg_df.head())
