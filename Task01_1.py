import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

ticker = 'AAPL'
data = yf.download(ticker, start='2023-01-01', end='2026-01-01')

data['Next_Close'] = data['Close'].shift(-1)

data = data.dropna()

X = data[['Open', 'High', 'Low', 'Volume', 'Close']]
y = data['Next_Close']

split_index = int(len(X) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: ${rmse:.2f}")
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Close', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Close', color='orange', linestyle='--')
plt.title(f'{ticker} Stock Price Prediction - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()