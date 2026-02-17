import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Dataset
df = pd.read_csv('dataset.csv')
df['date'] = pd.to_datetime(df['date'])

print("--- EDA: Dataset Overview ---")
print(df.info())

# 2. Summary Statistics (PDF Requirement)
print("\n--- Summary Statistics ---")
summary = df.describe()
summary.to_csv('data_summary.csv') 
print(summary)

# 3. Visualizing Trends (Bitcoin Price)
plt.figure(figsize=(12, 6))
btc_data = df[df['crypto_name'] == 'Bitcoin']
plt.plot(btc_data['date'], btc_data['close'], color='orange')
plt.title('Bitcoin Price Trend')
plt.grid(True)
plt.savefig('price_trend.png') 
plt.show()

# 4. Correlation Matrix (PDF Requirement)
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

print("\n✅ EDA Completed! Graphs and Summary saved.")