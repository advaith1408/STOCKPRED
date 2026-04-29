import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# -------------------------------
# UI CONFIGURATION & STYLING
# -------------------------------
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="centered")

# Use a visually pleasing style for Matplotlib charts
plt.style.use('ggplot')

st.title('📈 Stock Market Predictor')
st.markdown("Predict the future price of your favorite stocks using AI and Machine Learning.")

# -------------------------------
# INPUTS
# -------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    stock = st.text_input('Enter Stock Symbol', 'GOOG')
with col2:
    start = st.text_input('Start Date', '2012-01-01')
with col3:
    end = st.text_input('End Date', '2022-12-31')

selected_model = st.selectbox("🧠 Select Prediction Model", ['LSTM (Pre-trained)', 'Linear Regression', 'Random Forest', 'KNN'])

st.markdown("---")

# -------------------------------
# FETCH DATA
# -------------------------------
with st.spinner('Fetching stock data...'):
    try:
        data = yf.download(stock, start=start, end=end, progress=False)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

if data.empty:
    st.warning("No data found. Using fallback data.")
    data = pd.DataFrame({
        'Close': np.random.rand(1000) * 100
    })

st.subheader('Stock Data')
st.write(data.tail())

# -------------------------------
# PREPARE DATA & FIX SCALING
# -------------------------------
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0,1))

# Fit scaler ONLY on training data to prevent data leakage
data_train_scale = scaler.fit_transform(data_train)

# For testing data, we append the last 100 days of train data to create the first test sequence
past_100_days = data_train.tail(100)
data_test_extended = pd.concat([past_100_days, data_test], ignore_index=True)
# Transform test data using the fitted scaler
data_test_scale = scaler.transform(data_test_extended)

# -------------------------------
# MOVING AVERAGES (Visually Enhanced)
# -------------------------------
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()

fig1 = plt.figure(figsize=(10,5))
plt.plot(data.Close, label='Price', color='#1f77b4', linewidth=1.5)
plt.plot(ma_50_days, label='MA50', color='#ff7f0e', linewidth=2)
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()

fig2 = plt.figure(figsize=(10,5))
plt.plot(data.Close, label='Price', color='#1f77b4', linewidth=1.5)
plt.plot(ma_50_days, label='MA50', color='#ff7f0e', linewidth=2)
plt.plot(ma_100_days, label='MA100', color='#2ca02c', linewidth=2)
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()

fig3 = plt.figure(figsize=(10,5))
plt.plot(data.Close, label='Price', color='#1f77b4', linewidth=1.5)
plt.plot(ma_100_days, label='MA100', color='#2ca02c', linewidth=2)
plt.plot(ma_200_days, label='MA200', color='#d62728', linewidth=2)
plt.legend()
st.pyplot(fig3)

st.markdown("---")

# -------------------------------
# PREPARE MODEL INPUTS (TRAIN & TEST)
# -------------------------------
# 1. Training Data (Used for on-the-fly academic models)
x_train = []
y_train = []
for i in range(100, data_train_scale.shape[0]):
    x_train.append(data_train_scale[i-100:i])
    y_train.append(data_train_scale[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Flatten X for Classical ML models (from 3D to 2D)
x_train_flat = x_train.reshape(x_train.shape[0], x_train.shape[1])

# 2. Testing Data (Used for all models)
x_test = []
y_test = []
for i in range(100, data_test_scale.shape[0]):
    x_test.append(data_test_scale[i-100:i])
    y_test.append(data_test_scale[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1])

# -------------------------------
# MODEL CONTROL
# -------------------------------
st.subheader('Prediction')

run_model = st.button(f"Run {selected_model}", use_container_width=True)

if run_model:
    try:
        with st.spinner(f'Training & Running {selected_model}...'):
            if selected_model == 'LSTM (Pre-trained)':
                import keras
                model = keras.saving.load_model('Stock Predictions Model.keras')
                predict = model.predict(x_test)
            elif selected_model == 'Linear Regression':
                model = LinearRegression()
                model.fit(x_train_flat, y_train)
                predict = model.predict(x_test_flat)
                predict = predict.reshape(-1, 1) # reshape to match LSTM output format
            elif selected_model == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(x_train_flat, y_train)
                predict = model.predict(x_test_flat)
                predict = predict.reshape(-1, 1)
            elif selected_model == 'KNN':
                model = KNeighborsRegressor(n_neighbors=5)
                model.fit(x_train_flat, y_train)
                predict = model.predict(x_test_flat)
                predict = predict.reshape(-1, 1)

    except Exception as e:
        st.error(f"Model execution failed: {e}")
        predict = y_test.copy()
else:
    predict = y_test.copy()

# -------------------------------
# REVERSE SCALING
# -------------------------------
scale = 1 / scaler.scale_
predict = predict * scale
y_test_unscaled = y_test * scale

# -------------------------------
# ACCURACY METRICS & PLOT RESULTS
# -------------------------------
if run_model:
    st.markdown(f"### 🏆 {selected_model} Accuracy")
    
    # Safely flatten to 1D arrays for metric calculation
    y_true = np.squeeze(y_test_unscaled)
    y_pred = np.squeeze(predict)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    accuracy_percentage = max(0, 100 - mape)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Average Error)", f"${mae:.2f}")
    col2.metric("RMSE", f"${rmse:.2f}")
    col3.metric("Estimated Accuracy", f"{accuracy_percentage:.2f}%")

st.subheader(f'Original vs Predicted Price ({selected_model})')

fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test_unscaled, label='Original Price', color='#2ca02c', linewidth=2)
plt.plot(predict, label='Predicted Price', color='#d62728', linewidth=2, linestyle='--')
plt.xlabel('Time (Days)')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)