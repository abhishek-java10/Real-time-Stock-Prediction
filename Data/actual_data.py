import yfinance as yf
import pandas as pd
import datetime as dt

def download_actual_closes(prediction_csv='predicted_closes_5days.csv'):
    pred_df = pd.read_csv(prediction_csv)
    ticker_list = pred_df['Stock'].unique().tolist()

    forecast_dates = pd.to_datetime(pred_df.columns[1:]).strftime('%Y-%m-%d').tolist()

    end_date = pd.to_datetime(forecast_dates[-1]) + dt.timedelta(days=1)
    actual_data = yf.download(
        tickers=ticker_list,
        start=forecast_dates[0],
        end=end_date,
        interval='1d'
        )['Close']
    
    return actual_data

def clean_data():
    df = pd.read_csv("./actual_closes.csv")
    df.index = df.Date
    df.drop(['Date'], inplace=True, axis = 1)
    new_df = df.transpose()
    new_df.columns = new_df.columns.to_list()
    new_df.index.name = 'Stock'
    return new_df

def main():
    data = download_actual_closes()
    data.to_csv('./actual_closes.csv')
    clean_df = clean_data()
    clean_df.to_csv('./cleaned_actual_closes.csv')

if __name__ == "__main__":
    main()