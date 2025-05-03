# pip install vaderSentiment

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    analyze_sentiment()
