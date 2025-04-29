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
import Model.model_building_DL as model_building  # <- your training script as train_models.py
model_building.main()

# Predict next close prices
print("\nPredicting Next Close Prices")
import Model.predict_next_close as predict_next  # <- your prediction script as predict_next_close.py
predict_next.main()

# # 3. (Optional) Download actual closes & evaluate
# print("\n🟣 Step 3: Evaluating Model Predictions")
# import evaluate_predictions   # <- your evaluation script
# evaluate_predictions.main()

print("\n Full Pipeline Completed Successfully!")
