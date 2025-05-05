import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datasets import load_dataset

def collect_and_analyze_sentiment():
    tickers = {
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL",
        "PLTR", "CRM", "CSCO", "IBM", "NOW"
    }

    print("Loading FNSPID dataset...")
    dataset = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

    filtered_rows = []
    print("Filtering data by ticker and date range...")
    for row in dataset:
        try:
            date_str = row["Date"][:10]
            if "2023-05-01" <= date_str <= "2025-04-28" and row["Stock_symbol"] in tickers:
                filtered_rows.append(row)
        except KeyError:
            continue

    print(f"Collected {len(filtered_rows)} matching rows.")

    df = pd.DataFrame(filtered_rows)
    df.to_csv("sentiment_data.csv", index=False)
    print("Saved filtered data to 'sentiment_data.csv'")

    print("Loading: sentiment_data.csv")
    df = pd.read_csv("sentiment_data.csv")

    print("Initializing VADER sentiment analyzer...")
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if pd.isna(text):
            return 0.0
        return analyzer.polarity_scores(str(text))["compound"]

    def classify(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    print("Calculating sentiment scores...")
    df["sentiment_score"] = df["Article_title"].apply(get_sentiment)
    df["sentiment_label"] = df["sentiment_score"].apply(classify)

    df.to_csv("sentiment_score.csv", index=False)
    print("Saved with sentiment to 'sentiment_score.csv'")