import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

print("Retraining the Isolation Forest model...")

# Load the dataset
try:
    data = pd.read_csv("data/transaction_anomalies_dataset.csv")
    print(f"Dataset loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Use the three main features
    features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
    
    if all(col in data.columns for col in features):
        X = data[features].copy()
        
        print(f"\nFeature statistics:")
        print(X.describe())
        
        # Check for any obvious issues
        print(f"\nData info:")
        print(f"Any null values: {X.isnull().sum().sum()}")
        print(f"Any infinite values: {np.isinf(X).sum().sum()}")
        
        # Clean the data
        X = X.dropna()
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"Clean dataset shape: {X.shape}")
        
        # Train a new model with better parameters
        # Lower contamination rate since most transactions should be normal
        model = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies (more realistic)
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )
        
        print("Training model...")
        model.fit(X)
        
        # Test the model on some sample data
        print("\n=== Testing trained model ===")
        
        # Normal transaction
        normal_test = pd.DataFrame([[150, 180, 5]], columns=features)
        normal_pred = model.predict(normal_test)
        print(f"Normal transaction (150, 180, 5): {normal_pred[0]} ({'Normal' if normal_pred[0] == 1 else 'Anomaly'})")
        
        # Suspicious transaction  
        suspicious_test = pd.DataFrame([[15000, 200, 1]], columns=features)
        suspicious_pred = model.predict(suspicious_test)
        print(f"Suspicious transaction (15000, 200, 1): {suspicious_pred[0]} ({'Normal' if suspicious_pred[0] == 1 else 'Anomaly'})")
        
        # Another normal transaction
        normal_test2 = pd.DataFrame([[200, 180, 8]], columns=features)
        normal_pred2 = model.predict(normal_test2)
        print(f"Normal transaction (200, 180, 8): {normal_pred2[0]} ({'Normal' if normal_pred2[0] == 1 else 'Anomaly'})")
        
        # Save the model
        joblib.dump(model, "models/isolation_forest.pkl")
        print("\n✅ Model saved successfully!")
        
        # Show some statistics about predictions on the dataset
        all_predictions = model.predict(X)
        anomaly_count = sum(1 for pred in all_predictions if pred == -1)
        normal_count = sum(1 for pred in all_predictions if pred == 1)
        
        print(f"\nDataset predictions:")
        print(f"Normal transactions: {normal_count} ({normal_count/len(all_predictions)*100:.1f}%)")
        print(f"Anomalous transactions: {anomaly_count} ({anomaly_count/len(all_predictions)*100:.1f}%)")
        
    else:
        print(f"❌ Required features not found. Available: {list(data.columns)}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
