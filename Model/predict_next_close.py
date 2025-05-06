import pandas as pd
import torch
import joblib
import numpy as np
import os
from torch import nn
from datetime import timedelta

# === Model Definition ===
class AdvancedStockTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, dropout=0.2):
        super(AdvancedStockTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_layer(x).squeeze(1)

# === Utility Functions ===
def get_stocks(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

def get_next_trading_days(df, num_days=5):
    df['Date'] = pd.to_datetime(df['Date'])
    future_dates = []
    last_date = df['Date'].max()
    while len(future_dates) < num_days:
        last_date += timedelta(days=1)
        if last_date.weekday() < 5:  # Skip weekends (Mon-Fri = 0-4)
            future_dates.append(last_date)
    return [d.strftime('%Y-%m-%d') for d in future_dates]

# === Prediction Function ===
def predict_next_5_closes(df, stock, model_dir='models', window_size=7):
    try:
        feature_cols = joblib.load(f'{model_dir}/features_{stock}.pkl')
        scaler = joblib.load(f'{model_dir}/scaler_{stock}.pkl')
        target_scaler = joblib.load(f'{model_dir}/target_scaler_{stock}.pkl')

        input_dim = len(feature_cols)
        model = AdvancedStockTransformer(input_dim=input_dim)
        model.load_state_dict(torch.load(f'{model_dir}/transformer_{stock}.pt', map_location='cpu'))
        model.eval()

        recent = df[feature_cols].tail(window_size).copy()
        if recent.shape[0] < window_size:
            return None, "Not enough data"

        latest_open_col = f'Open_{stock}'
        if latest_open_col not in df.columns:
            return None, "Missing Open column"

        latest_open = df[latest_open_col].iloc[-1]
        predicted_closes = []

        for _ in range(5):
            x_input = scaler.transform(recent)
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                predicted_return_scaled = model(x_tensor).item()
                predicted_return = target_scaler.inverse_transform([[predicted_return_scaled]])[0][0]

            predicted_close = latest_open * (1 + predicted_return)
            predicted_closes.append(predicted_close)

            # Simulate next row input for rolling prediction
            new_row = recent.iloc[-1].copy()
            if f"{stock}_Daily_Return" in feature_cols:
                new_row[f"{stock}_Daily_Return"] = predicted_return
            recent = pd.concat([recent, pd.DataFrame([new_row])], ignore_index=True).tail(window_size)

            latest_open = predicted_close

        return predicted_closes, None

    except FileNotFoundError as e:
        return None, f"Missing file: {e.filename}"
    except Exception as e:
        return None, str(e)

# === Main Function ===
def main():
    df = pd.read_csv('final_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    stocks = get_stocks(df)
    date_cols = get_next_trading_days(df, num_days=5)
    results = []

    print("\nPredicted Close Prices for Next 5 Trading Days:")
    for stock in stocks:
        pred_list, error = predict_next_5_closes(df, stock)
        if error:
            print(f"{stock}: Skipped ({error})")
        else:
            print(f"{stock}: {['{:.2f}'.format(p) for p in pred_list]}")
            results.append({'Stock': stock, **{date_cols[i]: pred_list[i] for i in range(5)}})

    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv('predicted_closes_5days.csv', index=False)
        print("\n✅ Predictions saved to predicted_closes_5days.csv")
    else:
        print("\n❌ No predictions saved. All stocks failed.")

if __name__ == "__main__":
    main()