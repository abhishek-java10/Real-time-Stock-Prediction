import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('final_dataset.csv')


def get_stocks(df):
    return sorted({col.split('_')[1] for col in df.columns if col.startswith('Close_')})

def get_stock_dataset(df, stock, window_size=2):
    raw_feature_cols = [
        col for col in df.columns
        if (
            (col.endswith(f'_{stock}') and not col.startswith(f'Close_{stock}'))
            or (col.startswith(f'{stock}_') and col != f'{stock}_Daily_Return')
        )
    ]

    if 'Day_of_Week' in df.columns:
        raw_feature_cols.append('Day_of_Week')

    target_col = f'{stock}_Daily_Return'

    feature_cols = raw_feature_cols.copy()

    # Drop NaNs BEFORE scaling
    df = df[feature_cols + [target_col]].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])

    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(df[[target_col]]).flatten()

    X, y = [], []
    for i in range(len(df) - window_size):
        window = features_scaled[i:i+window_size]
        if np.isnan(window).any():  
            continue
        X.append(window)
        y.append(target_scaled[i+window_size])

    return np.array(X), np.array(y), scaler, target_scaler, feature_cols

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


def save_model(model, stock, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/transformer_{stock}.pt')

def save_scaler(scaler, stock, save_dir='models', target=False):
    os.makedirs(save_dir, exist_ok=True)
    suffix = 'target_' if target else ''
    joblib.dump(scaler, f'{save_dir}/{suffix}scaler_{stock}.pkl')

def save_feature_list(feature_cols, stock, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(feature_cols, f'{save_dir}/features_{stock}.pkl')

def train_model_for_stock(df, stock, window_size=7):
    model_path = f'models/transformer_{stock}.pt'
    if os.path.exists(model_path):
        os.remove(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, scaler, target_scaler, feature_cols = get_stock_dataset(df, stock, window_size)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = AdvancedStockTransformer(input_dim=X_train.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    loss_fn = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience, counter = 20, 0
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

        print(f"Epoch {epoch+1}/500 | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, stock)
            save_scaler(scaler, stock)
            save_scaler(target_scaler, stock, target=True)
            save_feature_list(feature_cols, stock)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1} for {stock}")
                break

def main():
    stocks = get_stocks(df)
    for stock in stocks:
        print(f"\n Training model for: {stock}")
        train_model_for_stock(df, stock)

    print("\n All models trained and saved successfully!")

if __name__ == "__main__":
    main()
