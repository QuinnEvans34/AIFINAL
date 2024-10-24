import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

def fetch_and_preprocess_data(ticker, start='2010-01-01', end='2023-01-01'):
    # Fetching stock data
    
    data = yf.download(ticker, start=start, end=end)
    
    # Preprocessing
    close_prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    
    return data, scaler, scaled_data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def build_and_train_model(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=0)
    return model

# List of tickers
tickers = ['^GSPC', 'AAPL', 'GOOGL']  # Example tickers: S&P 500, Apple, Google

# Dictionary to store results
results = {}

for ticker in tickers:
    print(f"Processing {ticker}...")
    data, scaler, scaled_data = fetch_and_preprocess_data(ticker)
    
    look_back = 60
    X, y = create_dataset(scaled_data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_and_train_model(X_train, y_train)
    
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"{ticker} Model Loss: {loss}")
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Store results
    results[ticker] = {
        'actual': y_test,
        'predicted': predictions,
        'dates': data.index[len(data) - len(y_test) - look_back: -look_back]
    }

# Plotting the results for each ticker
fig, axes = plt.subplots(len(tickers), 1, figsize=(16, 4 * len(tickers)))
fig.suptitle('LSTM Model: Actual vs Predicted Prices', fontsize=16)

for i, (ticker, result) in enumerate(results.items()):
    ax = axes[i] if len(tickers) > 1 else axes
    
    # Fetch recent data for plotting actual prices
    recent_data = yf.download(ticker, start='2022-01-01', end='2023-01-01')
    actual_prices = recent_data['Close'].values[-len(result['actual']):]
    
    ax.plot(result['dates'][-len(actual_prices):], actual_prices, label='Actual Price', color='blue')
    ax.plot(result['dates'][-len(result['predicted']):], result['predicted'].flatten(), label='Predicted Price', color='red')
    ax.set_title(f'{ticker} Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Rotate x-axis labels for better readability if they overlap
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# Debug print to check data values
for ticker, result in results.items():
    print(f"\nTicker Data Check for {ticker}:")
    print(f"Sample actual data: {result['actual'][:5]}")
    print(f"Sample predicted data: {result['predicted'].flatten()[:5]}")
    print(f"Sample dates: {result['dates'][:5]}")


# testestestestestestsetes pen15