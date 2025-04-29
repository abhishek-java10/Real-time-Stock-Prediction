import pandas as pd
import numpy as np

# Load dataset globally
data = pd.read_csv('cleaned_data.csv')

# Dynamically extract stock tickers from column names that follow the format 'Close_<TICKER>'.
def get_unique_stocks():
    return sorted({
        col.split('_')[1]
        for col in data.columns
        if col.startswith('Close_') and not col.startswith(('Close_SP500', 'Close_NASDAQ'))
    })


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
    print('Saving the Final Dataset with new features.')
    df_final.to_csv('./final_dataset.csv', index=False)

def engineer_lagged_returns(stocks):
    for stock in stocks:
        daily_return_col = f'{stock}_Daily_Return'
        if daily_return_col in data.columns:
            for lag in range(1, 6):  # Last 5 days lag
                data[f'{stock}_Return_Lag{lag}'] = data[daily_return_col].shift(lag)

def engineer_future_averages(stocks):
    for stock in stocks:
        close_col = f'Close_{stock}'
        if close_col in data.columns:
            data[f'{stock}_Future_3day_Avg_Close'] = data[close_col].shift(-1).rolling(window=3).mean()

def engineer_recent_volatility(stocks):
    for stock in stocks:
        close_col = f'Close_{stock}'
        if close_col in data.columns:
            data[f'{stock}_5d_Volatility'] = data[close_col].pct_change().rolling(window=5).std()


# Additional momentum and trend features
def engineer_momentum_features(stocks):
    for stock in stocks:
        close_col = f'Close_{stock}'
        if close_col in data.columns:
            # 5-day Moving Average Return
            data[f'{stock}_MA5_Return'] = data[close_col].pct_change(periods=5)
            
            # 10-day Moving Average Return
            data[f'{stock}_MA10_Return'] = data[close_col].pct_change(periods=10)
            
            # 5-day Volatility (Rolling Standard Deviation)
            data[f'{stock}_Volatility_5d'] = data[close_col].pct_change().rolling(window=5).std()

            # RSI Calculation (Relative Strength Index)
            delta = data[close_col].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(window=14, min_periods=14).mean()
            ma_down = down.rolling(window=14, min_periods=14).mean()
            rs = ma_up / ma_down
            data[f'{stock}_RSI'] = 100 - (100 / (1 + rs))

# Engineer market index features (SP500 and NASDAQ returns + relative ratios)
def engineer_market_indices_features():
    if 'Close_SP500' in data.columns and 'Close_NASDAQ' in data.columns:
        # SP500 and NASDAQ daily returns
        data['SP500_Return'] = data['Close_SP500'].pct_change()
        data['NASDAQ_Return'] = data['Close_NASDAQ'].pct_change()

        # Relative strength ratios for each stock
        for stock in get_unique_stocks():
            close_col = f'Close_{stock}'
            if close_col in data.columns:
                data[f'{stock}_vs_SP500_Ratio'] = data[close_col] / data['Close_SP500']
                data[f'{stock}_vs_NASDAQ_Ratio'] = data[close_col] / data['Close_NASDAQ']

        # Drop raw Close_SP500 and Close_NASDAQ after using them
        drop_cols = ['Close_SP500', 'Close_NASDAQ']
        data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)


# Full pipeline: extract stocks, engineer features, drop unnecessary columns, save result.
def run_feature_engineering_pipeline():
    convert_date_and_extract_features()
    stocks = get_unique_stocks()
    engineer_price_features(stocks)
    engineer_rating_change(stocks)
    engineer_momentum_features(stocks)  
    engineer_lagged_returns(stocks)      
    engineer_recent_volatility(stocks)   
    # engineer_future_averages(stocks)
    engineer_market_indices_features()    
    df_final = drop_textual_and_irrelevant_columns()
    df_final = df_final.fillna(method='ffill')
    df_final = df_final.dropna().reset_index(drop=True)
    save_dataset_with_new_feature(df_final)


# Run the full feature engineering pipeline
def main():
    run_feature_engineering_pipeline()

if __name__ == '__main__':
    main()