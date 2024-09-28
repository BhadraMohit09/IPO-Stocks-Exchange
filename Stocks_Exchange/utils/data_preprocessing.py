import pandas as pd

def preprocess_data(stock_data):
    # Preprocessing steps (e.g., feature selection, normalization)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)
    return stock_data[['Open', 'High', 'Low', 'Volume']]
