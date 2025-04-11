import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

# Returning list of 15 top companies in tech.
def get_tickerList():
    return yf.Sector(key='technology').top_companies.index.str.lower().to_list()[:15]

# Fuction to download data for our tickers list return the same.
def get_historic_data():
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days = 7)
    data = yf.download(get_tickerList(),start= start_date, end= end_date, interval='1h')
    return data

# Funtion to get recommondations.
def get_recommondations(ticker):
    tick_data = yf.Ticker(ticker)
    tick_rec = tick_data.get_upgrades_downgrades()

# Adding recommondations to the dataset
def merge_recommondations(data, recommondations):
    return data

# Saving the final dataset to the csv
def save_data(data):
    data.to_csv('./finance_data.csv')

# Saving the sample of the final data
def get_sample(data):
    data.head(100).to_csv('./samples/sample_finance_data.csv')


if __name__ == "__main__":
    data = get_historic_data()
    save_data(data)