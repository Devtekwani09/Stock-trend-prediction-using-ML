import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import yfinance as yf




# Specify the start date as '2010-01-01'
start_date = "2010-01-01"


st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
 # Create a ticker object for AAPL
aapl = yf.Ticker(user_input)
# Fetch historical data for AAPL starting from '2010-01-01'
df = aapl.history(period="max", start=start_date)

#describing data
st.subheader('Data From 2010')
st.write(df.describe())

#visualizations

st.subheader('closing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('closing Price Vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

# MA = moving average
st.subheader('closing Price Vs Time Chart with 100MA & 200 MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200, label='200-day MA')
plt.plot(ma100, label='100-day MA')
plt.plot(df.Close, label='Closing Price')

plt.legend()
st.pyplot(fig)


#spliting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

#scaling data from 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


#load model

model = load_model('keras_model.h5')


# testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final Graph

st.subheader('Predictions Vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'original price')
plt.plot(y_predicted, 'r', label = 'predicted price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)