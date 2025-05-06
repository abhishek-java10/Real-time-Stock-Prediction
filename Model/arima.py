import pandas as pd
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv('final_dataset.csv')

def get_tickers(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

def train_arima(df, stock, save_dir='models/arima'):
    os.makedirs(save_dir, exist_ok=True)
    target_col = f'{stock}_Daily_Return'
    if target_col not in df.columns:
        print(f"{stock}: Target column not found.")
        return
    series = df[target_col].dropna()
    model = ARIMA(series, order=(5, 0, 2))
    model_fit = model.fit()
    joblib.dump(model_fit, f"{save_dir}/arima_{stock}.pkl")
    print(f"{stock}: ARIMA model saved.")

if __name__ == "__main__":
    for stock in get_tickers(df):
        train_arima(df, stock)
