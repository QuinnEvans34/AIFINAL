import yfinance as yf
import pandas as pd
import numpy as np
# Load stock data for a specific ticker
ticker = 'AAPL'
data = yf.download(ticker, start='2022-01-01', end='2023-01-01')

# Display the first few rows
#print(data.head()) 

# Calculate the percentage change in closing price from quarter to quarter
data['Quarterly Change'] = data['Close'].pct_change(90)  # approx. 90 trading days in a quarter
data = data.dropna()  # Drop rows with NaN values due to pct_change calculation

# Features (e.g., quarterly change)
X = data[['Quarterly Change']].values  # You could add more features here like volume, moving averages, etc.
# Label (we are predicting the closing price)
y = data['Close'].values

# Display first few rows of processed data
print(data[['Close', 'Quarterly Change']].head())

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
