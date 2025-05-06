# === main.py ===
# Data Prep
print("\nData Prep")
import Data.data_prep as data_prep
data_prep.main()

# Data Cleaning
print("\nData Cleaning")
import Data.data_cleaning as data_cleaning
data_cleaning.main()

# Feature Engineering
print("\nFeature Engineering")
import Features.feature_engg as feature_engg
feature_engg.main()

# Train models
print("\nTraining Models")
print("\nTraning Transformer")
import Model.model_building_DL as model_building
model_building.main()

print("\nTraning Arima")
import Model.arima as arima
arima.main()

print("Training LSTM")
import Model.lstm as lstm
lstm.main()

# Predict next close prices
print("\nPredicting Next Close Prices")
from Model import predict_next_close, predict_arima, lstm_prediction
predict_next_close.main()
predict_arima.main()
lstm_prediction.main()


print("\n Full Pipeline Completed Successfully!")
