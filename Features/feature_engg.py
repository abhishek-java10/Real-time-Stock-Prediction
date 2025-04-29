import pandas as pd
import numpy as np

# Load dataset globally
data = pd.read_csv('cleaned_data.csv')

# Dynamically extract stock tickers from column names that follow the format 'Close_<TICKER>'.
def get_unique_stocks():
    return sorted({col.split('_')[1] for col in data.columns if col.startswith('Close_')})

# Converts the 'Date' column to datetime format and extracts additional temporal features.
def convert_date_and_extract_features():
    data['Date'] = pd.to_datetime(data['Date'])
    data['Day_of_Week'] = data['Date'].dt.dayofweek

# For each stock, calculate daily return, high-low range, and volatility.
def engineer_price_features(stocks):
    for stock in stocks:
        data[f'{stock}_Daily_Return'] = (data[f'Close_{stock}'] - data[f'Open_{stock}']) / data[f'Open_{stock}']
        data[f'{stock}_High_Low_Range'] = (data[f'High_{stock}'] - data[f'Low_{stock}']) / data[f'Open_{stock}']
        data[f'{stock}_Volatility'] = data[f'High_{stock}'] - data[f'Low_{stock}']

# Creates a binary feature for analyst rating change for each stock.
def engineer_rating_change(stocks):
    for stock in stocks:
        to_col = f'ToGrade_{stock.lower()}'
        from_col = f'FromGrade_{stock.lower()}'
        if to_col in data.columns and from_col in data.columns:
            data[f'{stock}_Rating_Change'] = np.where(data[to_col] == data[from_col], 0, 1)

# Drops textual and irrelevant columns like analyst firm names, grades, and actions.
def drop_textual_and_irrelevant_columns():
    drop_keywords = ['firm', 'tograde', 'fromgrade', 'action', 'unnamed']
    cols_to_drop = [col for col in data.columns if any(key in col.lower() for key in drop_keywords)]
    return data.drop(columns=cols_to_drop)

# Saves the final DataFrame with new features to CSV.
def save_dataset_with_new_feature(df_final):
    df_final.to_csv('./final_dataset.csv', index=False)

# Full pipeline: extract stocks, engineer features, drop unnecessary columns, save result.
def run_feature_engineering_pipeline():
    convert_date_and_extract_features()
    stocks = get_unique_stocks()
    engineer_price_features(stocks)
    engineer_rating_change(stocks)
    df_final = drop_textual_and_irrelevant_columns()
    save_dataset_with_new_feature(df_final)

# Run the full feature engineering pipeline
run_feature_engineering_pipeline()
