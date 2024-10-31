#this is the code that actually ran the neural network
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional  # type: ignore
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Load the stock data
userTicker = 'AMZN'
print(f"Downloading stock data for {userTicker}...")
data = yf.download(userTicker, start='2010-01-01', end='2024-01-01')

# Keep only the essential columns and calculate percent change
data['Volume Change'] = data['Volume'].pct_change()
combinedData = data[['Close', 'Volume Change']].dropna()

# Scale each feature using StandardScaler
scalerClose = StandardScaler()
scalerVolume = StandardScaler()

closeScaledData = scalerClose.fit_transform(combinedData[['Close']])
volumeScaledData = scalerVolume.fit_transform(combinedData[['Volume Change']])

# Prepare X and y, with weekly time steps
steps = 7  # Predict one week ahead
X, y = [], []

for i in range(steps, len(closeScaledData)):
    X.append(np.column_stack((closeScaledData[i-steps:i, 0], volumeScaledData[i-steps:i, 0])))
    y.append(closeScaledData[i, 0])  # Target is the next week's stock price

X, y = np.array(X), np.array(y)

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
validationScores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50, return_sequences=True),
        Dropout(0.20),
        LSTM(units=50),
        Dropout(0.20),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)  # Reduced epochs to speed up cross-validation

    # Validate and calculate MSE for each fold
    y_pred = model.predict(X_test)
    y_pred_rescaled = scalerClose.inverse_transform(y_pred)
    y_test_rescaled = scalerClose.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    validationScores.append(mse)

print("Average MSE across folds:", np.mean(validationScores))

# Predict the next week's closing price for the stock based on the last available data
last_sequence_combined = np.column_stack((closeScaledData[-steps:], volumeScaledData[-steps:]))
last_sequence_combined = last_sequence_combined.reshape(1, steps, 2)  # Reshape for LSTM

next_week_scaled_stock = model.predict(last_sequence_combined)
next_week_price_stock = scalerClose.inverse_transform(next_week_scaled_stock)[0, 0]  # Rescale to original price

# Get the current price and calculate percent change
current_price_stock = float(combinedData['Close'].values[-1])
percent_change_stock = (((next_week_price_stock - current_price_stock) / current_price_stock) * 100)

# Calculate the date for the next week's prediction
next_week_date = combinedData.index[-1] + timedelta(days=7)  # Date one week from the last date in the dataset

# Plot the actual price and predicted price
# Scatter plot for actual vs predicted prices
plt.figure(figsize=(12, 6))

# Plot actual prices as blue scatter points
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Prices', s=50, alpha=0.6)

# Plot predicted prices as red scatter points
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Prices', s=50, alpha=0.6)

# Add title and labels
plt.title(f'{userTicker} Stock Price Prediction: Actual vs Predicted', fontsize=16)
plt.xlabel('Test Data Points', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Display the current price and next predicted price with the specific prediction date
print(f"Current Price for {userTicker} on {combinedData.index[-1].date()}: ${current_price_stock:.2f}")
print(f"Next Predicted Price for {userTicker} on {next_week_date.date()}: ${next_week_price_stock:.2f}")
print(f"Predicted Percent Change for {userTicker}: {percent_change_stock:.2f}%")
plt.show()