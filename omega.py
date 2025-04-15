import sys, os
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppresses all TensorFlow INFO/WARNING/ERROR messages

# Suppress stdout and stderr during import
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    from sigma import df, models
except Exception as e:
    pass

# Restore stdout and stderr
sys.stdout = old_stdout
sys.stderr = old_stderr

import pandas as pd
from sigma import df, models
import numpy as np
from sklearn.base import clone

def validate_last_date():
    # Ensure proper datetime handling
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # Debug: Show date ranges
    print(f"\nDate Range in DataFrame: {df.index.min()} to {df.index.max()}")
    
    # Get the last date
    last_date = df.index.max()
    
    # Train on all data except the last date
    train = df[df.index < last_date]
    test = df[df.index == last_date]
    
    if test.empty:
        print("\nERROR: Could not find the last date for validation")
        return pd.DataFrame()
    
    # Prepare features
    features = ["Open", "High", "Low", "Volume"]
    target = "Close"
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    print(f"\n=== Last Date Validation ===")
    print(f"Training Period: {train.index.min().date()} to {train.index[-2].date()}")
    print(f"Test Date: {last_date.date()}")
    
    results = []
    for name, model in models.items():
        # Clone model to prevent contamination
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        
        # Calculate metrics
        actual = y_test.iloc[0]
        predicted = y_pred[0]
        absolute_error = abs(actual - predicted)
        percentage_error = (absolute_error / actual) * 100
        accuracy = 100 - percentage_error
        
        results.append({
            "Model": name,
            "Actual": round(actual, 2),
            "Predicted": round(predicted, 2),
            "Absolute Error": round(absolute_error, 2),
            "Accuracy": round(accuracy, 2)
        })
        
        print(f"\n{name}:")
        print(f"  Test Date: {last_date.date()}")
        print(f"  Actual Close: {actual:.2f}")
        print(f"  Predicted Close: {predicted:.2f}")
        print(f"  Absolute Error: {absolute_error:.2f}")
        print(f"  Accuracy: {accuracy:.2f}%")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    results = validate_last_date()
    if not results.empty:
        print("\n=== Final Report ===")
        print(results)
        results.to_csv("last_date_validation_report.csv", index=False)
