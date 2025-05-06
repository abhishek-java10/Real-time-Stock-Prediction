import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# === File paths
transformer_path = "predicted_closes_5days.csv"
lstm_path = "lstm_predicted_closes_5days.csv"
arima_path = "arima_predicted_closes_5days.csv"
actual_path = "cleaned_actual_closes.csv"
charts_dir = "charts/charts_plotly"

# === Load datasets
transformer_df = pd.read_csv(transformer_path)
lstm_df = pd.read_csv(lstm_path)
arima_df = pd.read_csv(arima_path)
actual_df = pd.read_csv(actual_path)

# === Ensure output directory
os.makedirs(charts_dir, exist_ok=True)

# === Forecast dates
forecast_dates = transformer_df.columns[1:]
first_date = forecast_dates[0]

# === Common stocks
common_stocks = set(transformer_df['Stock']) & set(lstm_df['Stock']) & set(arima_df['Stock']) & set(actual_df['Stock'])

# === Collect metrics
metrics = []

for stock in sorted(common_stocks):
    try:
        actual_vals = actual_df[actual_df['Stock'] == stock][forecast_dates].values.flatten()
        transformer_vals = transformer_df[transformer_df['Stock'] == stock][forecast_dates].values.flatten()
        lstm_vals = lstm_df[lstm_df['Stock'] == stock][forecast_dates].values.flatten()
        arima_vals = arima_df[arima_df['Stock'] == stock][forecast_dates].values.flatten()

        # === Calculate metrics for first forecast day
        actual_first = actual_vals[0]
        for model_name, pred in zip(
            ["Transformer", "LSTM", "ARIMA"],
            [transformer_vals[0], lstm_vals[0], arima_vals[0]]
        ):
            rmse = np.sqrt((pred - actual_first) ** 2)
            confidence = 1 - abs(pred - actual_first) / actual_first
            metrics.append({
                'Stock': stock,
                'Model': model_name,
                'Date': first_date,
                'Prediction': pred,
                'Actual': actual_first,
                'RMSE': round(rmse, 4),
                'ConfidenceScore (%)': round(confidence * 100, 2)
            })

        # === Annotation text
        metrics_text = f"Transformer RMSE: {np.sqrt((transformer_vals[0] - actual_first) ** 2):.2f}<br>" \
                       f"LSTM RMSE: {np.sqrt((lstm_vals[0] - actual_first) ** 2):.2f}<br>" \
                       f"ARIMA RMSE: {np.sqrt((arima_vals[0] - actual_first) ** 2):.2f}"

        # === Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_dates, y=actual_vals, mode='lines+markers', name='Actual', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=transformer_vals, mode='lines+markers', name='Transformer'))
        fig.add_trace(go.Scatter(x=forecast_dates, y=lstm_vals, mode='lines+markers', name='LSTM'))
        fig.add_trace(go.Scatter(x=forecast_dates, y=arima_vals, mode='lines+markers', name='ARIMA'))

        fig.add_annotation(
            x=forecast_dates[-1],
            y=max(actual_vals.max(), transformer_vals.max(), lstm_vals.max(), arima_vals.max()),
            text=metrics_text,
            showarrow=False,
            align="left",
            bordercolor="black",
            borderwidth=1,
            bgcolor="lightyellow",
            font=dict(size=12),
        )

        fig.update_layout(
            title=f"{stock} - Predicted vs Actual Close Prices",
            xaxis_title="Date",
            yaxis_title="Close Price",
            template="plotly_white",
            hovermode="x unified"
        )

        fig.write_html(f"{charts_dir}/{stock}.html")
        print(f"Chart saved: {stock}")

    except Exception as e:
        print(f"Skipped {stock}: {e}")

# === Save metrics to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("next_day_model_metrics.csv", index=False)

# === Plotly RMSE Chart
fig_rmse = px.bar(
    metrics_df,
    x="Stock",
    y="RMSE",
    color="Model",
    title="Next-Day RMSE Comparison per Model",
    barmode="group"
)
fig_rmse.write_html("rmse_comparison_plotly.html")

# === Plotly Confidence Chart
fig_conf = px.bar(
    metrics_df,
    x="Stock",
    y="ConfidenceScore (%)",
    color="Model",
    title="Next-Day Confidence Score (%) per Model",
    barmode="group"
)
fig_conf.write_html("confidence_score_plotly.html")

print("All Plotly charts and metrics saved successfully.")
