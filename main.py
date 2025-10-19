import pandas as pd
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detect_anomalies import AnomalyDetector
    from utils import load_data, evaluate_model
    from data_preprocessing import DataPreprocessor
    from train_model import train_and_evaluate
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install -r Requirements.txt")
    sys.exit(1)

def main():
    print("🚨 Anomaly Detection in Transactions 🚨")
    print("="*50)
    
    while True:
        print("\nChoose an option:")
        print("1. Detect anomaly in a single transaction")
        print("2. Analyze dataset for anomalies")
        print("3. Retrain model")
        print("4. View dataset statistics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            detect_single_transaction()
        elif choice == '2':
            analyze_dataset()
        elif choice == '3':
            retrain_model()
        elif choice == '4':
            view_statistics()
        elif choice == '5':
            print("Thank you for using Anomaly Detection System!")
            break
        else:
            print("Invalid choice. Please try again.")

def detect_single_transaction():
    """Detect anomaly in a single transaction"""
    try:
        print("\n--- Single Transaction Anomaly Detection ---")
        
        # Get user input
        amount = float(input("Enter transaction amount: $"))
        avg_amount = float(input("Enter average transaction amount: $"))
        frequency = int(input("Enter frequency of transactions: "))
        
        # Create detector and predict
        detector = AnomalyDetector()
        user_input = {
            'Transaction_Amount': amount,
            'Average_Transaction_Amount': avg_amount,
            'Frequency_of_Transactions': frequency
        }
        
        result = detector.predict(user_input)
        
        if result:
            print("\n🚨 ANOMALY DETECTED! This transaction appears suspicious.")
        else:
            print("\n✅ Normal transaction. No anomaly detected.")
            
    except FileNotFoundError:
        print("\n❌ Model not found. Please train the model first (option 3).")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

def analyze_dataset():
    """Analyze the entire dataset for anomalies"""
    try:
        print("\n--- Dataset Anomaly Analysis ---")
        
        # Load data
        data_path = "data/transaction_anomalies_dataset.csv"
        data = load_data(data_path)
        
        # Load model and detect anomalies
        detector = AnomalyDetector()
        
        # Prepare features
        features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
        
        if all(col in data.columns for col in features):
            predictions = []
            print("Analyzing transactions...")
            for _, row in data.iterrows():
                pred = detector.predict(row[features].to_dict())
                predictions.append(pred)
            
            anomaly_count = sum(predictions)
            total_count = len(predictions)
            
            print(f"\nAnalysis Results:")
            print(f"Total transactions: {total_count}")
            print(f"Anomalies detected: {anomaly_count}")
            print(f"Anomaly rate: {(anomaly_count/total_count)*100:.2f}%")
            
            # Show some anomalous transactions
            if anomaly_count > 0:
                anomaly_indices = [i for i, pred in enumerate(predictions) if pred == 1]
                print(f"\nFirst 5 anomalous transactions:")
                for i in anomaly_indices[:5]:
                    row = data.iloc[i]
                    print(f"Transaction {i+1}: Amount=${row['Transaction_Amount']:.2f}, Avg=${row['Average_Transaction_Amount']:.2f}, Freq={row['Frequency_of_Transactions']}")
        else:
            print(f"❌ Required columns not found. Available columns: {list(data.columns)}")
            
    except Exception as e:
        print(f"\n❌ Error analyzing dataset: {str(e)}")

def retrain_model():
    """Retrain the anomaly detection model"""
    try:
        print("\n--- Model Retraining ---")
        
        # Load data
        data_path = "data/transaction_anomalies_dataset.csv"
        data = load_data(data_path)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        # Prepare features and labels
        features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
        
        if all(col in processed_data.columns for col in features):
            X = processed_data[features]
            y = processed_data.get('Is_Anomaly', processed_data.get('Class', [0]*len(X)))
            
            # Split data (simple split for demo)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            print("Training model...")
            model = train_and_evaluate(X_train, X_test, y_test)
            print("\n✅ Model retrained successfully!")
        else:
            print(f"❌ Required columns not found. Available columns: {list(processed_data.columns)}")
            
    except Exception as e:
        print(f"\n❌ Error retraining model: {str(e)}")

def view_statistics():
    """View dataset statistics"""
    try:
        print("\n--- Dataset Statistics ---")
        
        # Load data
        data_path = "data/transaction_anomalies_dataset.csv"
        data = load_data(data_path)
        
        print(f"\nDataset Shape: {data.shape}")
        print(f"\nColumn Information:")
        print(data.info())
        
        print(f"\nStatistical Summary:")
        print(data.describe())
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            print(f"\nMissing Values:")
            print(missing_values[missing_values > 0])
        else:
            print("\n✅ No missing values found.")
            
    except Exception as e:
        print(f"\n❌ Error viewing statistics: {str(e)}")

if __name__ == "__main__":
    main()
