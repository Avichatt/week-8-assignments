import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

class SimpleNeuralNet:
    def __init__(self, input_size, hidden1, hidden2):
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2/hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2/hidden2)
        self.b3 = np.zeros((1, 1))
        
    def relu(self, z): return np.maximum(0, z)
    def relu_derivative(self, z): return (z > 0).astype(float)
    def sigmoid(self, z): return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3
    
    def backward(self, X, y, output, lr, pos_weight=5.0):
        m = y.shape[0]
        dz3 = (output - y)
        dz3[y == 1] *= pos_weight
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W3 -= lr * dW3; self.b3 -= lr * db3

def train_and_evaluate():
    df = pd.read_csv("clean_hospital_records.csv")
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Unknown': 0.5})
    features = ['age', 'bmi', 'gender', 'blood_pressure', 'cholesterol']
    X, y = df[features].values, df['readmitted'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)
    nn = SimpleNeuralNet(5, 16, 8)
    lr, epochs, losses = 0.1, 1000, []
    for epoch in range(epochs):
        output = nn.forward(X_train)
        loss = -np.mean(y_train * np.log(output + 1e-8) * 5 + (1 - y_train) * np.log(1 - output + 1e-8))
        losses.append(loss)
        nn.backward(X_train, y_train, output, lr)
    
    predictions = (nn.forward(X_test) > 0.5).astype(int)
    print("\n--- Neural Network Performance ---\n", classification_report(y_test, predictions))
    return nn, scaler, features, X_test, y_test

if __name__ == "__main__":
    train_and_evaluate()
