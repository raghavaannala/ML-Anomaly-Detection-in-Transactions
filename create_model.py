import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def create_model():
    """Create and save a proper isolation forest model"""
    
    # Load the dataset
    try:
        data = pd.read_csv('data/transaction_anomalies_dataset.csv')
        print(f"Dataset loaded successfully with {data.shape[0]} records")
        print(f"Columns: {list(data.columns)}")
        
        # Check what columns are available
        required_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
        available_features = []
        
        for feature in required_features:
            if feature in data.columns:
                available_features.append(feature)
            else:
                # Try to find similar column names
                similar_cols = [col for col in data.columns if feature.lower().replace('_', '').replace(' ', '') in col.lower().replace('_', '').replace(' ', '')]
                if similar_cols:
                    print(f"Using {similar_cols[0]} instead of {feature}")
                    available_features.append(similar_cols[0])
        
        if len(available_features) < 2:
            print("Not enough matching features found. Using first 3 numeric columns...")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            available_features = list(numeric_cols[:3])
        
        print(f"Using features: {available_features}")
        
        # Prepare training data
        X = data[available_features].fillna(data[available_features].mean())
        
        # Create and train the model
        model = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42,
            max_samples='auto'
        )
        
        print("Training model...")
        model.fit(X)
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        joblib.dump(model, 'models/isolation_forest.pkl')
        print("✅ Model saved successfully to models/isolation_forest.pkl")
        
        # Test the model
        predictions = model.predict(X[:10])
        print(f"Sample predictions: {predictions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return False

if __name__ == "__main__":
    create_model()
