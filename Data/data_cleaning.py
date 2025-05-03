import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

# Creating a global variable accessible in every method.
data = pd.read_csv('finance_data.csv')

# Function to print some basic stats about the dataset
def data_description():
    print(data.head(5))
    print(data.info())
    print(data.describe())
    print(data.columns)
    print(data.isna().sum())

# Function remove null values if any in target varibale
def removing_null_values_of_targets():
    target_cols = [col for col in data.columns if 'Close' in col]
    data.dropna(subset=target_cols, how='all', axis=0, inplace=True)

# Function to fill null values of recommendations
def fill_na_recommendation():
    # selecting recommendation columns to avoid fill wrong values
    mat_key = ['Firm','ToGrade', 'FromGrade', 'Action']
    values = {}
    for col in data.columns:
        for key in mat_key:
            if key in col:
                values[col] = 'None'      
    data.fillna(values, inplace=True)

def save_data():
    data.to_csv("cleaned_data.csv")
    print("Cleaned Dataset Saved Sucessfully.")

def main():
    data_description()
    removing_null_values_of_targets()
    fill_na_recommendation()
    save_data()

if __name__ == "__main__":
    main()