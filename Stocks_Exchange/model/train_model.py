import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(data):
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    

    joblib.dump(model, os.path.join('../model/stock_model.pkl'))
    print("Model trained and saved as 'stock_model.pkl'")

if __name__ == "__main__":
    stock_data = pd.read_csv('C:/Users/MUDIT/OneDrive/Desktop/DM_Project/Stock_Price_Predictor/data/stock_data.csv')
    train_model(stock_data)
