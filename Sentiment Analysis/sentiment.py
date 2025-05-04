import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Merging final_dataset.csv and sentiment_score.csv
    final_df = pd.read_csv('final_dataset.csv')
    sentiment_df = pd.read_csv('sentiment_score.csv')

    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.date
    final_df["Date"] = pd.to_datetime(final_df["Date"]).dt.date

    avg_sentiment = sentiment_df.groupby(["Date", "Stock_symbol"])["sentiment_score"].mean().reset_index()
    sentiment_pivot = avg_sentiment.pivot(index="Date", columns="Stock_symbol", values="sentiment_score")
    sentiment_pivot.columns = [f"sentiment_{col}" for col in sentiment_pivot.columns]
    merged_df = pd.merge(final_df, sentiment_pivot, on="Date", how="left")

    stock = "AAPL"
    features = [f"Close_{stock}", f"sentiment_{stock}"]
    df = merged_df[["Date"] + features].dropna().copy()
    df.set_index("Date", inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    class StockDataset(Dataset):
        def __init__(self, data, window=4):
            self.X, self.y = [], []
            for i in range(len(data) - window):
                self.X.append(data[i:i+window])
                self.y.append(data[i+window][0])
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    window_size = 4
    train_data = scaled_data[:-1]
    test_input = torch.tensor(scaled_data[-window_size:], dtype=torch.float32).unsqueeze(0)

    train_dataset = StockDataset(train_data, window_size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    class TimeSeriesTransformer(nn.Module):
        def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_layer = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.transformer(x)
            return self.output_layer(x[:, -1])

    model = TimeSeriesTransformer(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(30):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"[Transformer] Epoch {epoch} - Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        pred_scaled = model(test_input).item()
        predicted_price = scaler.inverse_transform([[pred_scaled, 0]])[0][0]

    print(f"\nPredicted next close price for {stock}: ${predicted_price:.2f}")

    price_col = f"Close_{stock}"
    sentiment_col = f"sentiment_{stock}"
    df = merged_df[["Date", price_col, sentiment_col]].dropna().copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[price_col, sentiment_col]])
    df_scaled = pd.DataFrame(scaled, columns=[price_col, sentiment_col])
    df_scaled["Date"] = df["Date"].values
    df_scaled["Predicted"] = df_scaled[price_col].shift(1)
    df_plot = df_scaled.dropna().copy()

    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["Date"], df_plot[price_col], label="Actual Price", color="blue")
    plt.plot(df_plot["Date"], df_plot["Predicted"], label="Predicted Price", color="red", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.title(f"Actual vs Predicted Price of {stock}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    actual_prices = scaler.inverse_transform(df_plot[[price_col, sentiment_col]])[:, 0]
    predicted_prices = scaler.inverse_transform(
        pd.concat([df_plot[["Predicted"]], df_plot[[sentiment_col]]], axis=1)
    )[:, 0]

    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()