import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.data_preprocessing import preprocess_data

# Load the model
model = joblib.load('model/stock_model.pkl')

# Streamlit app title
st.title("Stock Price Prediction App")

# User input for stock symbol and date range
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOG):", "AAPL")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# Button to predict stock price
if st.button("Predict"):
    # Load and preprocess the data
    stock_data = pd.read_csv('data/stock_data.csv')  # Update with your dataset path
    processed_data = preprocess_data(stock_data)

    # Ensure that processed_data is correctly shaped for predictions
    # Make predictions
    predictions = model.predict(processed_data)

    # Create a DataFrame for predicted prices
    predicted_prices = pd.DataFrame(predictions, columns=['Predicted Close'], index=stock_data.index)

    # Create a figure for historical stock prices
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])

    # Update the layout
    fig.update_layout(title='Historical Stock Prices',
                      xaxis_title='Date',
                      yaxis_title='Price (USD)')

    # Display historical stock prices
    st.subheader("Historical Stock Prices")
    st.plotly_chart(fig, use_container_width=True)

    # Create a figure for predicted stock prices
    fig2 = go.Figure(data=[go.Scatter(x=predicted_prices.index, y=predicted_prices['Predicted Close'])])

    # Update the layout
    fig2.update_layout(title='Predicted Stock Prices',
                       xaxis_title='Date',
                       yaxis_title='Price (USD)')

    # Display predicted stock prices
    st.subheader("Predicted Stock Prices")
    st.plotly_chart(fig2, use_container_width=True)