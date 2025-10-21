import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detect_anomalies import AnomalyDetector
import joblib

# Test the model directly
print("Testing the anomaly detection model...")

try:
    detector = AnomalyDetector()
    
    # Test normal transaction (should be False/Normal) - values within training range
    normal_transaction = {
        'Transaction_Amount': 1000.0,
        'Average_Transaction_Amount': 1020.0,
        'Frequency_of_Transactions': 12
    }
    
    # Test suspicious transaction (should be True/Anomaly) - high amount, low frequency
    suspicious_transaction = {
        'Transaction_Amount': 3200.0,
        'Average_Transaction_Amount': 1000.0,
        'Frequency_of_Transactions': 5
    }
    
    # Load model directly to check raw predictions
    model = joblib.load("models/isolation_forest.pkl")
    
    print("\n=== RAW MODEL PREDICTIONS ===")
    import pandas as pd
    
    normal_df = pd.DataFrame([normal_transaction])
    suspicious_df = pd.DataFrame([suspicious_transaction])
    
    normal_pred = model.predict(normal_df)
    suspicious_pred = model.predict(suspicious_df)
    
    print(f"Normal transaction raw prediction: {normal_pred[0]} (should be 1 for normal)")
    print(f"Suspicious transaction raw prediction: {suspicious_pred[0]} (should be -1 for anomaly)")
    
    print("\n=== DETECTOR CLASS PREDICTIONS ===")
    normal_result = detector.predict(normal_transaction)
    suspicious_result = detector.predict(suspicious_transaction)
    
    print(f"Normal transaction: {'ANOMALY' if normal_result else 'NORMAL'}")
    print(f"Suspicious transaction: {'ANOMALY' if suspicious_result else 'NORMAL'}")
    
    print("\n=== MODEL INFO ===")
    print(f"Contamination rate: {model.contamination}")
    print(f"Number of estimators: {model.n_estimators}")
    
except Exception as e:
    print(f"Error: {e}")
