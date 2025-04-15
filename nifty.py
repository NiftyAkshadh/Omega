import pickle
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load selected features
with open('selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)
print("final_selected_features:\n", selected_features)

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("\nScaler Loaded Successfully")

# Load XGBoost model
xgb_model = joblib.load('xgb_model.pkl')
print("\nXGBoost Model Loaded Successfully")

# Optional: View model parameters
print("\nModel Parameters:\n", xgb_model.get_params())

# View saved plots
plt.figure()
img1 = plt.imread('feature_importance.png')
plt.imshow(img1)
plt.axis('off')
plt.title('Feature Importance')
plt.show()

plt.figure()
img2 = plt.imread('future_prediction.png')
plt.imshow(img2)
plt.axis('off')
plt.title('Future Prediction')
plt.show()

plt.figure()
img3 = plt.imread('profitability_simulation.png')
plt.imshow(img3)
plt.axis('off')
plt.title('Profitability Simulation')
plt.show()
