
# 📈 Real-Time Stock Prediction

This project implements a real-time stock price prediction system using state-of-the-art machine learning models: **Transformer**, **LSTM (Long Short-Term Memory)**, and **ARIMA (AutoRegressive Integrated Moving Average)**. It forecasts future stock prices based on historical OHLCV data, helping investors and analysts make informed decisions.

---

## 🔍 Features

- **Data Preprocessing**: Cleans and prepares historical stock data for modeling.
- **Model Training**:
  - **Transformer** for capturing long-term dependencies in time series.
  - **LSTM** for sequential pattern learning.
  - **ARIMA** for traditional statistical forecasting.
- **Prediction**: Forecasts next-day stock prices for multiple tickers.
- **Evaluation**: Calculates RMSE and confidence scores for model comparison.
- **Visualization**: Interactive charts (using Plotly) for prediction insights and performance metrics.

---

## 📁 Project Structure

```
Real-time-Stock-Prediction/
├── Data/
│   ├── actual_closes.csv
│   ├── cleaned_data.csv
│   └── finance_data.csv
├── Features/
│   └── feature_engineering.py
├── Model/
│   ├── lstm_model.py
│   ├── arima_model.py
│   └── transformer_model.py
├── charts/
│   ├── confidence_score_plotly.html
│   └── rmse_comparison_plotly.html
├── models/
│   ├── lstm_model.h5
│   ├── arima_model.pkl
│   └── transformer_model.pt
├── samples/
│   └── sample_predictions.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abhishek-java10/Real-time-Stock-Prediction.git
   cd Real-time-Stock-Prediction
   ```

2. **Create a virtual environment (optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Usage

1. **Add your stock data** to the `Data/` directory (`.csv` format).
2. **Run the main pipeline**:
   ```bash
   python main.py
   ```
3. **Output**: Predictions, saved models, and visualizations are automatically generated.

---

## 📊 Visualizations

- **Confidence Score Plot**: Shows model prediction confidence (Plotly interactive).
- **RMSE Comparison Chart**: Highlights model accuracy across LSTM, Transformer, and ARIMA.
- **Prediction Charts**: Actual vs. predicted price visualizations per ticker.

---

## 📌 Notes

- All trained models are saved in the `models/` directory.
- Visuals are exported as interactive HTML files in `charts/`.
- Sample predictions can be found in the `samples/` folder.

---

## 🤖 Models Summary

| Model       | Strengths                                          | Format            |
|-------------|----------------------------------------------------|-------------------|
| Transformer | Captures long-range dependencies, fast inference  | `.pt` (PyTorch)   |
| LSTM        | Good for short-term sequential dependencies        | `.h5` (Keras)     |
| ARIMA       | Strong statistical baseline for linear trends      | `.pkl` (Pickle)   |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

For questions or contributions, feel free to open an issue or connect with [Abhishek Teotia](https://github.com/abhishek-java10).
