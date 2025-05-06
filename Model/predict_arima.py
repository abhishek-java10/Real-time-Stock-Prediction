import pandas as pd
import joblib
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Load dataset
df = pd.read_csv('final_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

def get_tickers(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

# Function to get next 5 valid trading days
def get_next_trading_days(df, num_days=5):
    future_dates = []
    last_date = df['Date'].max()
    while len(future_dates) < num_days:
        last_date += timedelta(days=1)
        if last_date.weekday() < 5:
            future_dates.append(last_date)
    return [d.strftime('%Y-%m-%d') for d in future_dates]

def main():
    # Get prediction dates
    date_cols = get_next_trading_days(df, num_days=5)

    # Run prediction
    results = []
    for stock in get_tickers(df):
        model_path = f"models/arima/arima_{stock}.pkl"
        open_col = f"Open_{stock}"

        if not os.path.exists(model_path):
            print(f"{stock}: Model not found.")
            continue
        if open_col not in df.columns:
            print(f"{stock}: Missing Open column.")
            continue

        try:
            model = joblib.load(model_path)
            forecast_returns = model.forecast(steps=5)

            if len(forecast_returns) != 5:
                raise ValueError("Forecast did not return 5 steps.")

            latest_open = df[open_col].iloc[-1]
            predicted_closes = [(latest_open * (1 + r)) for r in forecast_returns]

            row = {'Stock': stock, **{date_cols[i]: predicted_closes[i] for i in range(5)}}
            results.append(row)
            print(f"{stock}: {['{:.2f}'.format(c) for c in predicted_closes]}")

        except Exception as e:
            print(f"{stock}: Forecast failed ({type(e).__name__}: {e})")

    # Save results
    if results:
        pd.DataFrame(results).to_csv("arima_predicted_closes_5days.csv", index=False)
        print("\nSaved to arima_predicted_closes_5days.csv")
    else:
        print("\nNo ARIMA predictions saved.")
    
if __name__ == "__main__":
    main()
