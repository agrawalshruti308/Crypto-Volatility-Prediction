import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(page_title="Crypto Volatility Predictor", layout="wide")
st.title("📊 Bitcoin Volatility Prediction System")

# 2. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    df = df[df["crypto_name"] == "Bitcoin"].copy()
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# 3. Prepare Model with strict cleaning
def prepare_model(df_input):
    temp_df = df_input.copy()
    temp_df["volatility"] = (temp_df["high"] - temp_df["low"]) / temp_df["close"]
    temp_df["return"] = temp_df["close"].pct_change()
    temp_df["volume_change"] = temp_df["volume"].pct_change()
    
    # RSI
    delta = temp_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    temp_df['RSI'] = 100 - (100 / (1 + rs))
    
    temp_df['vol_lag_1'] = temp_df['volatility'].shift(1)
    
    # --- ERROR FIX START ---
    # Infinity aur NaN values ko hatana zaroori hai
    temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
    temp_df = temp_df.dropna(subset=["return", "volume_change", "RSI", "vol_lag_1", "volatility"])
    # --- ERROR FIX END ---
    
    features = ["return", "volume_change", "RSI", "vol_lag_1", "marketCap"]
    X = temp_df[features]
    y = temp_df["volatility"]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, temp_df

model, cleaned_df = prepare_model(df)

# 4. Sidebar Inputs
st.sidebar.header("User Input Features")
user_return = st.sidebar.slider("Current Return %", -0.1, 0.1, 0.01)
user_rsi = st.sidebar.slider("RSI Value", 0, 100, 50)
user_vol = st.sidebar.slider("Last Day Volatility", 0.0, 0.2, 0.05)

# 5. Prediction
if st.button("Predict Volatility"):
    # Last available market cap and volume change for context
    input_data = np.array([[user_return, 0.0, user_rsi, user_vol, cleaned_df['marketCap'].iloc[-1]]])
    prediction = model.predict(input_data)
    
    st.subheader("Results")
    st.metric(label="Predicted Volatility Level", value=f"{prediction[0]:.4f}")
    
    if prediction[0] > 0.05:
        st.warning("High Risk: Market might be unstable!")
    else:
        st.success("Low Risk: Market looks stable.")

st.subheader("Historical Price Chart")
st.line_chart(cleaned_df.set_index('date')['close'])