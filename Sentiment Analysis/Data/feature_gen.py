# pip install datasets

from datasets import load_dataset
import pandas as pd

def filter_sentiment_data():
    tickers = {
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
        "PLTR", "CRM", "CSCO", "IBM", "NOW"
    }

    print("Loading FNSPID dataset...")
    dataset = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

    small_sample = []
    count = 0

    print("Filtering data by ticker and date range...")
    for row in dataset:
        try:
            date_str = row["Date"][:10]
            if "2023-05-01" <= date_str <= "2025-04-28" and row["Stock_symbol"] in tickers:
                small_sample.append(row)
                count += 1
            if count >= 1000:
                break
        except KeyError:
            continue

    print(f"Collected {len(small_sample)} matching rows.")
    
    df = pd.DataFrame(small_sample)
    df.to_csv("sentiment_data.csv", index=False)
    print("Saved filtered data to 'sentiment_data.csv'")


if __name__ == "__main__":
    filter_sentiment_data()
