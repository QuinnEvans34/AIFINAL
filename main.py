import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Loads the stock the user is looking up
userTicker = 'AMZN'
print(f"Downloading stock data for {userTicker}...")
data = yf.download(userTicker, start='2014-01-01', end='2024-01-01')

# loads SP 500 to compare for market
sp500 = '^GSPC'
sp500data = yf.download(sp500, start='2014-01-01', end='2024-01-01')

# use pct_change to look at amount change between quarters / quarters 90 days
data['Quarterly Change'] = data['Close'].pct_change(90)  
sp500data['Quarterly Change'] = sp500data['Close'].pct_change(90)

# concat data so it aligns
combinedData = pd.concat([data[['Close', 'Quarterly Change']], sp500data['Quarterly Change']], axis=1)
combinedData.columns = ['Close_AAPL', 'Quarterly Change_AAPL', 'Quarterly Change_SP500']
combinedData = combinedData.dropna()  # Drop rows with NaN values due to pct_change calculation

# Use MinMaxScaler for scaling
scalerStock = MinMaxScaler(feature_range=(0, 1))
scaler500 = MinMaxScaler(feature_range=(0, 1))

# Scale the stock and S&P 500 quarterly changes
stockScaledData = scalerStock.fit_transform(combinedData[['Close_AAPL']])
sp500ScaledData = scaler500.fit_transform(combinedData[['Quarterly Change_SP500']])

# Define features (X) and labels (y)

time_steps = 40  # this will change with the user input
X, y = [], []

# Prepares the data for LSTM
for i in range(time_steps, len(stockScaledData)):
    X.append(np.column_stack((stockScaledData[i-time_steps:i, 0], sp500ScaledData[i-time_steps:i, 0])))  # Combine stock and S&P 500
    y.append(stockScaledData[i, 0])  # Target is the next quarter's stock price

# Convert X and y to numpy arrays and reshape X for LSTM samples, time_steps, features
X, y = np.array(X), np.array(y)

# split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model with two features
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)  # Output layer for users predicted price
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Predict on test set and rescale predictions back to original values
y_pred = model.predict(X_test)
y_pred_rescaled = scalerStock.inverse_transform(y_pred)  # Rescale predictions to original stock prices
y_test_rescaled = scalerStock.inverse_transform(y_test.reshape(-1, 1))  # Rescale actual prices

# Predict the next quarter's closing price for the stock based on the last available data
last_sequence_stock = stockScaledData[-time_steps:]  # Last time_steps quarters of the stock
last_sequence_sp500 = sp500ScaledData[-time_steps:]  # Last time_steps quarters of S&P 500
last_sequence_combined = np.column_stack((last_sequence_stock, last_sequence_sp500))
last_sequence_combined = last_sequence_combined.reshape(1, time_steps, 2)  # Reshape for LSTM

next_quarter_scaled_stock = model.predict(last_sequence_combined)
next_quarter_price_stock = scalerStock.inverse_transform(next_quarter_scaled_stock)[0, 0]  # Rescale to original price

# Get the current prices and calculate percent change for stock
current_price_stock = combinedData['Close_AAPL'].values[-1].item()
percent_change_stock = ((next_quarter_price_stock - current_price_stock) / current_price_stock) * 100

# Calculate the date for the next quarter
last_date = combinedData.index[-1]  # Last date in the dataset
next_quarter_date = last_date + timedelta(days=90)  # Approximate date for the next quarter

# Display the current price, next predicted price, percent change, and comparison to S&P 500
print("\n============================================")
print(f"Current Price for {userTicker} on {last_date.date()}: ${current_price_stock:.2f}")
print(f"Next Predicted Price for {userTicker} on {next_quarter_date.date()}: ${next_quarter_price_stock:.2f}")
print(f"Predicted Percent Change for {userTicker}: {percent_change_stock:.2f}%\n")

# Comparison to current S&P 500 performance
current_sp500_change = combinedData['Quarterly Change_SP500'].values[-1] * 100  # Convert to percentage
print(f"S&P 500 Quarterly Change as of {last_date.date()}: {current_sp500_change:.2f}%")
print("============================================")
