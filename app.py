import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from datetime import datetime, timedelta

# Load models
btc_model = load_model('btc_lstm.h5')
eth_model = load_model('eth_lstm.h5')

# Load scalers
btc_scaler = MinMaxScaler()
eth_scaler = MinMaxScaler()

# Load BTC training data 
btc_data = pd.read_csv('btc_training_data.csv')  # Change to your actual BTC training data file
btc_scaler.fit(btc_data[['Close']].values.reshape(-1, 1))

# Load ETH training data 
eth_data = pd.read_csv('eth_training_data.csv')  # Change to your actual BNB training data file
eth_scaler.fit(eth_data[['Close']].values.reshape(-1, 1))

def get_prediction_data(model, scaler, training_data, days_to_predict):
    # Calculate prediction dates
    last_training_date = datetime.strptime(training_data['Date'].max(), '%Y-%m-%d')
    prediction_dates = [(last_training_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_to_predict + 1)]

    # Create input data for the model
    last_sequence = training_data['Close'].tail(15).values.reshape(-1, 1)
    input_sequence = scaler.transform(last_sequence).reshape(1, 15, 1)

    # Predict values
    predicted_values_scaled = []
    for _ in range(days_to_predict):
        prediction = model.predict(input_sequence, verbose=0)
        predicted_values_scaled.append(prediction[0, 0])
        input_sequence = np.roll(input_sequence, -1)
        input_sequence[0, -1, 0] = prediction[0, 0]

    # Convert predicted values back to the original scale
    predicted_values = scaler.inverse_transform(np.array(predicted_values_scaled).reshape(-1, 1)).reshape(-1)

    return prediction_dates, predicted_values

# Streamlit app
st.title('Cryptocurrency Price Prediction')

# Sidebar with user input
crypto_choice = st.sidebar.selectbox('Select Cryptocurrency', ['BTC-USD', 'ETH-USD'])
days_to_predict = st.sidebar.slider('Select Number of Days to Predict', 1, 30, 7)

# Fetch data and models based on user input
if crypto_choice == 'BTC-USD':
    model = btc_model
    scaler = btc_scaler
    training_data = btc_data
elif crypto_choice == 'ETH-USD':
    model = eth_model
    scaler = eth_scaler
    training_data = eth_data

# Get prediction data
prediction_dates, predicted_values = get_prediction_data(model, scaler, training_data, days_to_predict)

# Display predicted prices
st.subheader(f'Predicted Prices for {crypto_choice} in the Next {days_to_predict} Days')
st.line_chart(pd.Series(predicted_values, index=prediction_dates))
