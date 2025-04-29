import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the dataset and making it a global variable so that every funtion can use it.
data = pd.read_csv('cleaned_data.csv')

# Reason for taking log of the amount is to normalize the amount.
def log_amount():
    for col in data.columns:
        if 'Volume' in col:
            data[f'log_{col}'] = np.log1p(data[col])

# Engineering Volatality feature.
def volatality(window=5):
    for col in data.columns:
        if 'Close' in col:
            data[f'{col}_volatility_{window}d'] = data[col].rolling(window=window).std()

# moving averages feature. (Create short/long moving averages (SMA))
def moving_averages(short_window=5, long_window=20):
    for col in data.columns:
        if 'Close' in col:
            data[f'{col}_sma_{short_window}'] = data[col].rolling(window=short_window).mean()
            data[f'{col}_sma_{long_window}'] = data[col].rolling(window=long_window).mean()

# price momentum feature. (Difference between current and previous day close)
def price_momentum():
    for col in data.columns:
        if 'Close' in col:
            data[f'{col}_momentum'] = data[col].diff()

# daily_returns. (Percentage change)
def daily_returns():
    for col in data.columns:
        if 'Close' in col:
            data[f'{col}_returns'] = data[col].pct_change()

# recommendation_flags. (Binary encoding of recommendation presence)
def recommendation_flags():
    for col in data.columns:
        if 'ToGrade' in col or 'Action' in col:
            data[f'{col}_flag'] = data[col].apply(lambda x: 0 if x == 'None' else 1)


# Valocity Features like minute wise features
def save_sample_with_new_feature():
    data.to_csv('./feature_sample.csv')

if __name__ == '__main__':
    log_amount()
    volatality()
    moving_averages()
    price_momentum()
    daily_returns()
    recommendation_flags()
    save_sample_with_new_feature()
