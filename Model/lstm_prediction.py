import pandas as pd
import torch
import joblib
import numpy as np
import os
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

# === Model Definition ===
class StockLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super(StockLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

# === Utility Functions ===
def get_tickers(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

def get_next_trading_days(df, num_days=5):
    future_dates = []
    last_date = df['Date'].max()
    while len(future_dates) < num_days:
        last_date += timedelta(days=1)
        if last_date.weekday() < 5:
            future_dates.append(last_date)
    return [d.strftime('%Y-%m-%d') for d in future_dates]

# === Prediction Function ===
def predict_lstm_5days(df, stock, model_dir='models/lstm', window_size=7):
    try:
        feature_cols = joblib.load(f"{model_dir}/features_{stock}.pkl")
        scaler = joblib.load(f"{model_dir}/scaler_{stock}.pkl")
        target_scaler = joblib.load(f"{model_dir}/target_scaler_{stock}.pkl")

        model = StockLSTM(input_dim=len(feature_cols))
        model.load_state_dict(torch.load(f"{model_dir}/lstm_{stock}.pt", map_location='cpu'))
        model.eval()

        recent = df[feature_cols].tail(window_size).copy()
        if recent.shape[0] < window_size:
            return None, "Not enough data"

        latest_open_col = f"Open_{stock}"
        if latest_open_col not in df.columns:
            return None, "Missing Open column"

        latest_open = df[latest_open_col].iloc[-1]
        predicted_closes = []

        for _ in range(5):
            x_input = scaler.transform(recent)
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred_scaled = model(x_tensor).item()
                predicted_return = target_scaler.inverse_transform([[pred_scaled]])[0][0]

            predicted_close = latest_open * (1 + predicted_return)
            predicted_closes.append(predicted_close)

            new_row = recent.iloc[-1].copy()
            if f'{stock}_Daily_Return' in recent.columns:
                new_row[f'{stock}_Daily_Return'] = predicted_return
            recent = pd.concat([recent, pd.DataFrame([new_row])], ignore_index=True).tail(window_size)

            latest_open = predicted_close

        return predicted_closes, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# === Main Runner ===
def main():
    df = pd.read_csv("final_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    tickers = get_tickers(df)
    future_dates = get_next_trading_days(df, num_days=5)

    results = []
    for stock in tickers:
        preds, error = predict_lstm_5days(df, stock)
        if error:
            print(f"{stock}: Skipped ({error})")
        else:
            print(f"{stock}: {[f'{p:.2f}' for p in preds]}")
            results.append({'Stock': stock, **{future_dates[i]: preds[i] for i in range(5)}})

    if results:
        pd.DataFrame(results).to_csv("lstm_predicted_closes_5days.csv", index=False)
        print("Predictions saved to lstm_predicted_closes_5days.csv")
    else:
        print("No predictions saved.")

if __name__ == "__main__":
    main()
