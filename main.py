import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('dataset.csv')
df = df[df["crypto_name"] == "Bitcoin"].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 2. Feature Engineering
df["volatility"] = (df["high"] - df["low"]) / df["close"]
df["return"] = df["close"].pct_change()
df["volume_change"] = df["volume"].pct_change()

# RSI Calculation
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss.replace(0, np.nan)
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].fillna(50)

# Lag Features
df['vol_lag_1'] = df['volatility'].shift(1)
df['vol_lag_2'] = df['volatility'].shift(2)

# 3. Data Cleaning
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 4. Model Training
features = ["return", "volume_change", "RSI", "vol_lag_1", "vol_lag_2", "marketCap"]
X = df[features]
y = df["volatility"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. Output Results
preds = model.predict(X_test)
print(f"Model Results -> R2 Score: {r2_score(y_test, preds):.4f}")

# Plot Prediction
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label='Actual', color='blue')
plt.plot(preds[:100], label='Predicted', color='red', linestyle='--')
plt.title('Bitcoin Volatility Prediction')
plt.legend()
plt.show()