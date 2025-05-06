import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

def get_tickers(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

def get_stock_dataset(df, stock, window_size=7):
    feature_cols = [
        col for col in df.columns
        if (col.endswith(f'_{stock}') or col.startswith(f'{stock}_')) and col != f'{stock}_Daily_Return'
    ]
    if 'Day_of_Week' in df.columns:
        feature_cols.append('Day_of_Week')

    target_col = f'{stock}_Daily_Return'
    df = df[feature_cols + [target_col]].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(df[[target_col]]).flatten()

    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(X_scaled[i:i+window_size])
        y.append(y_scaled[i+window_size])

    return np.array(X), np.array(y), scaler, target_scaler, feature_cols

def train_lstm(df, stock, window_size=7, model_dir='models/lstm'):
    os.makedirs(model_dir, exist_ok=True)
    try:
        X, y, scaler, target_scaler, feature_cols = get_stock_dataset(df, stock, window_size)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        model = StockLSTM(input_dim=X_train.shape[2]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        loss_fn = nn.SmoothL1Loss()

        best_val_loss = float('inf')
        patience = 20
        counter = 0

        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = loss_fn(preds, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = loss_fn(val_preds, y_val).item()

            scheduler.step(val_loss)

            print(f"{stock} | Epoch {epoch+1}/500 | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{model_dir}/lstm_{stock}.pt")
                joblib.dump(scaler, f"{model_dir}/scaler_{stock}.pkl")
                joblib.dump(target_scaler, f"{model_dir}/target_scaler_{stock}.pkl")
                joblib.dump(feature_cols, f"{model_dir}/features_{stock}.pkl")
                print(f"Model saved for {stock}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping for {stock} at epoch {epoch+1}")
                    break

    except Exception as e:
        print(f"{stock}: Training failed ({type(e).__name__}: {e})")

def main():
    df = pd.read_csv("final_dataset.csv")
    for stock in get_tickers(df):
        print(f"Training LSTM for: {stock}")
        train_lstm(df, stock)

    print("All LSTM models trained and saved.")

if __name__ == "__main__":
    main()
