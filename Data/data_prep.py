import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os

# Returning list of 15 top companies in tech (manually defined for yfinance, as it has no 'Sector' method).
def get_tickerList():
    # Since yf.Sector().top_companies doesn't exist, use a hardcoded list of top tech tickers
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'ADBE', 'CRM', 'ORCL', 'INTC', 'CSCO', 'AMD', 'IBM', 'QCOM']

# Function to download data for our tickers list
def get_historic_data(ticker):
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=7)
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return data

# Function to get recommendations
def get_recommendations(ticker):
    return yf.Ticker(ticker).upgrades_downgrades

# Function to rename columns to avoid merge conflicts
def change_column_name(data, ticker):
    data.columns = [f"{col}_{ticker}" for col in data.columns]
    return data

# Function to add ticker as a column and merge recommendations
def merge_recommendations(data, recommendations, ticker):
    data = data.copy()
    data['Ticker'] = ticker
    merged = pd.merge(data, recommendations, how='left', left_index=True, right_index=True)
    return merged

# Save the full dataset to CSV
def save_data(data):
    data.to_csv('./finance_data.csv')

# Save a sample of the final data
def get_sample(data):
    os.makedirs('./samples', exist_ok=True)
    data.head(100).to_csv('./samples/sample_finance_data.csv')

# Main execution
if __name__ == "__main__":
    flag = False
    for ticker in get_tickerList():
        print(f"Processing {ticker}...")
        try:
            data = change_column_name(get_historic_data(ticker), ticker)
            rec = get_recommendations(ticker)
            combined = merge_recommendations(data, rec, ticker)
            if not flag:
                final_data = combined
                flag = True
            else:
                final_data = pd.concat([final_data, combined])
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    save_data(final_data)
    get_sample(final_data)
    print("Data saved successfully.")
