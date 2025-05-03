import pandas as pd
import torch
import joblib
import numpy as np
import os
from torch import nn

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

# === Utility ===
def get_stocks(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

# === Prediction Function ===
import pandas as pd
import torch
import joblib
import numpy as np
import os
from torch import nn

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

# === Utility ===
def get_stocks(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

# === Prediction Function ===
def predict_next_close(df, stock, model_dir='models', window_size=2):
    try:
        # Load saved feature list and scalers
        feature_cols = joblib.load(f'{model_dir}/features_{stock}.pkl')
        scaler = joblib.load(f'{model_dir}/scaler_{stock}.pkl')
        target_scaler = joblib.load(f'{model_dir}/target_scaler_{stock}.pkl')

        input_dim = len(feature_cols)
        model = AdvancedStockTransformer(input_dim=input_dim)
        model.load_state_dict(torch.load(f'{model_dir}/transformer_{stock}.pt'))
        model.eval()

        # Prepare last 7 days of features
        recent = df[feature_cols].tail(window_size)

        if recent.shape[0] < window_size:
            return None, "Not enough data"

        x_input = scaler.transform(recent)
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            predicted_return_scaled = model(x_tensor).item()
            predicted_return = target_scaler.inverse_transform([[predicted_return_scaled]])[0][0]

        latest_open_col = f'Open_{stock}'
        if latest_open_col not in df.columns:
            return None, "Missing Open column"

        latest_open = df[latest_open_col].iloc[-1]
        predicted_close = latest_open * (1 + predicted_return)

        return predicted_close, None

    except FileNotFoundError as e:
        return None, f"Missing file: {e.filename}"
    except Exception as e:
        return None, str(e)


def main():
    df = pd.read_csv('final_dataset.csv')
    stocks = get_stocks(df)
    results = []

    print("\n Predicted next Close Prices:")
    for stock in stocks:
        pred, error = predict_next_close(df, stock)
        if error:
            print(f"{stock}: Skipped ({error})")
        else:
            print(f"{stock}: {pred:.2f}")
            results.append({'Stock': stock, 'Predicted_Close': pred})

    # Save predictions
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv('predicted_closes.csv', index=False)
        print("\n Predictions saved to predicted_closes.csv")
    else:
        print("\n No predictions were saved. All stocks failed.")

if __name__ == "__main__":
    main()

def main():
    df = pd.read_csv('final_dataset.csv')
    stocks = get_stocks(df)
    results = []

    print("\n Predicted next Close Prices:")
    for stock in stocks:
        pred, error = predict_next_close(df, stock)
        if error:
            print(f"{stock}: Skipped ({error})")
        else:
            print(f"{stock}: {pred:.2f}")
            results.append({'Stock': stock, 'Predicted_Close': pred})

    # Save predictions
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv('predicted_closes.csv', index=False)
        print("\n Predictions saved to predicted_closes.csv")
    else:
        print("\n No predictions were saved. All stocks failed.")

if __name__ == "__main__":
    main()