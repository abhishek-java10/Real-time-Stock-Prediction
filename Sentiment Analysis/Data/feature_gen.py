# pip install datasets
# pip install vaderSentiment

from datasets import load_dataset
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def analyze_sentiment():
    input_path = "sentiment_data.csv"
    output_path = "sentiment_score.csv"

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

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

    df.to_csv(output_path, index=False)
    print(f"Saved with sentiment to '{output_path}'")


if __name__ == "__main__":
    filter_sentiment_data()
    analyze_sentiment()