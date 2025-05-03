import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import os

# Returning list of 15 top companies in tech
def get_tickerList():
    return yf.Sector(key='technology').top_companies.index.str.lower().to_list()[:10]

# Function to download data for our tickers list
def get_historic_data(ticker):
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=730)
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return data

# Function to rename columns to avoid merge conflicts
def change_column_name(data):
    data.columns = ['_'.join(col) for col in data.columns]
    return data

# Function to get recommendations
def get_recommendations(ticker):
    return yf.Ticker(ticker).upgrades_downgrades

# Funciton to rename recommendations coulumns to track the tickers
def rename_recommendations_columns(recommendations_df, ticker):
    recommendations_df.index = recommendations_df.index.date
    recommendations_df.columns = [ f"{col}_{ticker}" for col in recommendations_df.columns]
    return recommendations_df

# Function to remove duplicate dates to avoid merge conflicts
def remove_duplicate_dates(recommendations_df):
    # Convert index to date only if it's a datetime
    if isinstance(recommendations_df.index, pd.DatetimeIndex):
        recommendations_df = recommendations_df[~recommendations_df.index.duplicated(keep='first')]
    else:
        # If index is not datetime, consider converting if needed
        recommendations_df.index = pd.to_datetime(recommendations_df.index, errors='coerce')
        recommendations_df = recommendations_df[~recommendations_df.index.duplicated(keep='first')]
    return recommendations_df

# Function to add ticker as a column and merge recommendations
def merge_recommendations(data, recommendations_df):
    merged = pd.merge(data, recommendations_df, how='left', left_index=True, right_index=True)
    return merged

def get_market_indices_data():
    end_date = dt.date.today() - dt.timedelta(days=7)
    start_date = end_date - dt.timedelta(days=730)

    # Fetch NASDAQ
    nasdaq = yf.download('^IXIC', start=start_date, end=end_date, interval='1d')
    if isinstance(nasdaq.columns, pd.MultiIndex):
        nasdaq.columns = ['_'.join(col).strip() for col in nasdaq.columns.values]

    close_cols = [col for col in nasdaq.columns if 'Close' in col]
    nasdaq = nasdaq[[close_cols[0]]]  # Pick the first match
    nasdaq.rename(columns={close_cols[0]: 'Close_NASDAQ'}, inplace=True)

    # Fetch S&P500
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = ['_'.join(col).strip() for col in sp500.columns.values]

    close_cols = [col for col in sp500.columns if 'Close' in col]
    sp500 = sp500[[close_cols[0]]]
    sp500.rename(columns={close_cols[0]: 'Close_SP500'}, inplace=True)

    # Merge them on Date
    indices_data = nasdaq.merge(sp500, left_index=True, right_index=True, how='outer')
    return indices_data


# Save the full dataset to CSV
def save_data(data):
    data.to_csv('./finance_data.csv')

# Save a sample of the final data
def get_sample(data):
    os.makedirs('./samples', exist_ok=True)
    data.head(100).to_csv('./samples/sample_finance_data.csv')

# Main execution
def main():
    flag = False
    final_data = []
    for ticker in get_tickerList():
        print(f"Processing {ticker}...")
        try:
            data = change_column_name(get_historic_data(ticker))
            rec = remove_duplicate_dates(rename_recommendations_columns(get_recommendations(ticker), ticker))
            combined = merge_recommendations(data, rec)
            final_data.append(combined)
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")
    if final_data:
        final_data = pd.concat(final_data, axis=1)

        # 👉 Add NASDAQ + SP500
        print("Fetching NASDAQ and S&P500 data...")
        market_indices = get_market_indices_data()
        print("Merging market indices...")
        final_data = final_data.merge(market_indices, left_index=True, right_index=True, how='left')

        save_data(final_data)
        get_sample(final_data)
        print("Data saved successfully.")
    else:
        print("No data was processed.")



if __name__ == "__main__":
    main()