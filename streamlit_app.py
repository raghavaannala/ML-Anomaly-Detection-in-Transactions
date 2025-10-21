import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detect_anomalies import AnomalyDetector
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    from train_model import train_and_evaluate
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🚨 ML Anomaly Detection Suite",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 ML Anomaly Detection Suite</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning for Financial Fraud Detection</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Single Detection", "📊 Dataset Analysis", "🎯 Model Training", "📈 Statistics", "ℹ️ About"]
    )
    
    # Status indicator
    current_time = datetime.now().strftime("%H:%M:%S")
    st.sidebar.success(f"🟢 System Ready • {current_time}")
    
    # Page routing
    if page == "🔍 Single Detection":
        single_detection_page()
    elif page == "📊 Dataset Analysis":
        dataset_analysis_page()
    elif page == "🎯 Model Training":
        model_training_page()
    elif page == "📈 Statistics":
        statistics_page()
    elif page == "ℹ️ About":
        about_page()

def single_detection_page():
    st.header("🔍 Single Transaction Analysis")
    st.markdown("Analyze individual transactions for suspicious patterns using ML algorithms")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Transaction Details")
        
        # Initialize session state if not exists
        if 'amount' not in st.session_state:
            st.session_state.amount = 0.0
        if 'avg_amount' not in st.session_state:
            st.session_state.avg_amount = 0.0
        if 'frequency' not in st.session_state:
            st.session_state.frequency = 0
        
        # Input fields with better styling using session state values
        amount = st.number_input(
            "💵 Transaction Amount ($)",
            min_value=0.0,
            value=st.session_state.amount,
            step=0.01,
            help="Enter the transaction amount in dollars",
            key="amount_input"
        )
        
        avg_amount = st.number_input(
            "📊 Average Transaction Amount ($)",
            min_value=0.0,
            value=st.session_state.avg_amount,
            step=0.01,
            help="Enter the user's average transaction amount",
            key="avg_amount_input"
        )
        
        frequency = st.number_input(
            "🔄 Frequency of Transactions",
            min_value=0,
            value=st.session_state.frequency,
            step=1,
            help="Number of transactions in the recent period",
            key="frequency_input"
        )
        
        # Update session state with current input values
        st.session_state.amount = amount
        st.session_state.avg_amount = avg_amount
        st.session_state.frequency = frequency
        
        # Quick test buttons
        st.subheader("🚀 Quick Test Scenarios")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✅ Normal Transaction", key="normal_btn"):
                st.session_state.amount = 1000.0
                st.session_state.avg_amount = 1020.0
                st.session_state.frequency = 12
                st.rerun()
        
        with col_btn2:
            if st.button("⚠️ Suspicious Transaction", key="suspicious_btn"):
                st.session_state.amount = 3500.0
                st.session_state.avg_amount = 1000.0
                st.session_state.frequency = 5
                st.rerun()
    
    with col2:
        st.subheader("📊 Analysis Result")
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            if amount > 0 and avg_amount > 0 and frequency > 0:
                with st.spinner("🔄 Analyzing transaction..."):
                    try:
                        # Create detector and predict
                        detector = AnomalyDetector()
                        user_input = {
                            'Transaction_Amount': amount,
                            'Average_Transaction_Amount': avg_amount,
                            'Frequency_of_Transactions': frequency
                        }
                        
                        result = detector.predict(user_input)
                        
                        # Calculate risk score
                        risk_ratio = amount / max(avg_amount, 1)
                        risk_score = min(risk_ratio * frequency, 100)
                        
                        if result:
                            st.markdown(f"""
                            <div class="danger-card">
                                <h3>🚨 ANOMALY DETECTED!</h3>
                                <p>This transaction shows suspicious patterns</p>
                                <hr style="border-color: rgba(255,255,255,0.3);">
                                <p><strong>Risk Score:</strong> {risk_score:.1f}/100</p>
                                <p><strong>Amount Deviation:</strong> {risk_ratio:.2f}x average</p>
                                <p><strong>Recommendation:</strong> Review manually</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-card">
                                <h3>✅ NORMAL TRANSACTION</h3>
                                <p>Transaction appears legitimate</p>
                                <hr style="border-color: rgba(255,255,255,0.3);">
                                <p><strong>Risk Score:</strong> {risk_score:.1f}/100</p>
                                <p><strong>Amount Deviation:</strong> {risk_ratio:.2f}x average</p>
                                <p><strong>Status:</strong> Approved</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Risk score gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Risk Score"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgray"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except FileNotFoundError:
                        st.error("❌ Model not found. Please train the model first.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please fill in all fields with positive values.")

def dataset_analysis_page():
    st.header("📊 Complete Dataset Analysis")
    st.markdown("Analyze the entire transaction dataset to identify patterns and anomalies")
    
    if st.button("📊 Start Analysis", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load data
            status_text.text("Loading dataset...")
            progress_bar.progress(20)
            data_path = "data/transaction_anomalies_dataset.csv"
            data = load_data(data_path)
            
            st.success(f"✅ Dataset loaded: {data.shape[0]} records, {data.shape[1]} features")
            
            # Analyze anomalies
            status_text.text("Analyzing transactions for anomalies...")
            progress_bar.progress(40)
            
            detector = AnomalyDetector()
            features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
            
            if all(col in data.columns for col in features):
                predictions = []
                for i, (_, row) in enumerate(data.iterrows()):
                    if i % 100 == 0:
                        progress_bar.progress(40 + int((i / len(data)) * 40))
                    pred = detector.predict(row[features].to_dict())
                    predictions.append(pred)
                
                progress_bar.progress(80)
                status_text.text("Generating visualizations...")
                
                anomaly_count = sum(predictions)
                total_count = len(predictions)
                anomaly_rate = (anomaly_count / total_count) * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", f"{total_count:,}")
                
                with col2:
                    st.metric("Anomalies Detected", f"{anomaly_count:,}")
                
                with col3:
                    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
                
                with col4:
                    st.metric("Normal Transactions", f"{total_count - anomaly_count:,}")
                
                # Create visualizations
                create_analysis_visualizations(data, predictions)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis completed!")
                
                # Show sample anomalous transactions
                if anomaly_count > 0:
                    st.subheader("🚨 Sample Anomalous Transactions")
                    anomaly_indices = [i for i, pred in enumerate(predictions) if pred == 1]
                    sample_anomalies = data.iloc[anomaly_indices[:10]]
                    st.dataframe(sample_anomalies[features], use_container_width=True)
            
            else:
                st.error("❌ Required columns not found in dataset")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def create_analysis_visualizations(data, predictions):
    """Create comprehensive visualizations for analysis results"""
    
    # Convert predictions to pandas Series for easier filtering
    pred_series = pd.Series(predictions, dtype=bool)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Anomaly Distribution', 'Transaction Amount Distribution', 
                       'Normal vs Anomaly Amounts', 'Risk Score Distribution'),
        specs=[[{"type": "pie"}, {"type": "histogram"}],
               [{"type": "box"}, {"type": "scatter"}]]
    )
    
    # 1. Anomaly Distribution Pie Chart
    anomaly_counts = [sum(predictions), len(predictions) - sum(predictions)]
    labels = ['Anomalies', 'Normal']
    colors = ['#ff6b6b', '#4ecdc4']
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=anomaly_counts,
        marker_colors=colors,
        textinfo='label+percent'
    ), row=1, col=1)
    
    # 2. Transaction Amount Distribution
    fig.add_trace(go.Histogram(
        x=data['Transaction_Amount'],
        nbinsx=30,
        marker_color='#667eea',
        opacity=0.7
    ), row=1, col=2)
    
    # 3. Box plot comparison
    normal_amounts = data[~pred_series]['Transaction_Amount']
    anomaly_amounts = data[pred_series]['Transaction_Amount']
    
    fig.add_trace(go.Box(
        y=normal_amounts,
        name='Normal',
        marker_color='#4ecdc4'
    ), row=2, col=1)
    
    fig.add_trace(go.Box(
        y=anomaly_amounts,
        name='Anomalies',
        marker_color='#ff6b6b'
    ), row=2, col=1)
    
    # 4. Risk Score Scatter
    risk_scores = []
    for _, row in data.iterrows():
        risk_ratio = row['Transaction_Amount'] / max(row['Average_Transaction_Amount'], 1)
        risk_score = min(risk_ratio * row['Frequency_of_Transactions'], 100)
        risk_scores.append(risk_score)
    
    colors_scatter = ['#ff6b6b' if pred else '#4ecdc4' for pred in predictions]
    
    fig.add_trace(go.Scatter(
        x=list(range(len(risk_scores))),
        y=risk_scores,
        mode='markers',
        marker=dict(color=colors_scatter, size=6, opacity=0.7),
        name='Risk Scores'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Comprehensive Analysis Dashboard",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

def model_training_page():
    st.header("🎯 Model Training & Optimization")
    st.markdown("Retrain the Isolation Forest model with the latest data for improved accuracy")
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <h4>🔬 Training Process</h4>
        <p>The training process includes the following steps:</p>
        <ul>
            <li>Data preprocessing and feature engineering</li>
            <li>Model training with optimized Isolation Forest parameters</li>
            <li>Performance evaluation and metrics calculation</li>
            <li>Model validation and cross-validation</li>
            <li>Automatic model saving for future predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 Start Training", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load data
            status_text.text("Loading training data...")
            progress_bar.progress(20)
            data_path = "data/transaction_anomalies_dataset.csv"
            data = load_data(data_path)
            
            st.success(f"✅ Data loaded: {data.shape}")
            
            # Preprocess data
            status_text.text("Preprocessing data...")
            progress_bar.progress(40)
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.fit_transform(data)
            
            # Prepare features
            features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
            
            if all(col in processed_data.columns for col in features):
                X = processed_data[features]
                y = processed_data.get('Is_Anomaly', processed_data.get('Class', [0]*len(X)))
                
                # Split data
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                st.info(f"📊 Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
                
                # Train model
                status_text.text("Training model...")
                progress_bar.progress(70)
                
                # Capture training output
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    model = train_and_evaluate(X_train, X_test, y_test)
                
                output = f.getvalue()
                
                progress_bar.progress(100)
                status_text.text("✅ Training completed!")
                
                st.success("🎉 Model retrained successfully!")
                
                # Display training output
                with st.expander("📊 Training Details"):
                    st.text(output)
            
            else:
                st.error("❌ Required columns not found in dataset")
                
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")

def statistics_page():
    st.header("📈 Dataset Statistics & Insights")
    st.markdown("Comprehensive statistical analysis of the transaction dataset")
    
    if st.button("📈 Generate Statistics", type="primary"):
        try:
            # Load data
            data_path = "data/transaction_anomalies_dataset.csv"
            data = load_data(data_path)
            
            # Basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Dataset Shape", f"{data.shape[0]} × {data.shape[1]}")
            
            with col2:
                st.metric("Number of Records", f"{len(data):,}")
            
            with col3:
                st.metric("Number of Features", len(data.columns))
            
            # Column information
            st.subheader("📋 Dataset Columns")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            numeric_data = data.select_dtypes(include=[np.number])
            st.dataframe(numeric_data.describe(), use_container_width=True)
            
            # Missing values visualization
            if data.isnull().sum().any():
                st.subheader("❌ Missing Values")
                missing_data = data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if not missing_data.empty:
                    fig = px.bar(
                        x=missing_data.index,
                        y=missing_data.values,
                        title="Missing Values by Column",
                        labels={'x': 'Columns', 'y': 'Missing Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ No missing values found!")
            
            # Feature distributions
            st.subheader("📈 Feature Distributions")
            numeric_cols = numeric_data.columns[:4]  # Show first 4 numeric columns
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=numeric_cols
            )
            
            for i, col in enumerate(numeric_cols):
                row = (i // 2) + 1
                col_pos = (i % 2) + 1
                
                fig.add_trace(
                    go.Histogram(x=data[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=600, title_text="Feature Distribution Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error generating statistics: {str(e)}")

def about_page():
    st.header("ℹ️ About ML Anomaly Detection Suite")
    
    st.markdown("""
    ## 🎯 Overview
    
    This application provides a comprehensive solution for detecting anomalies in financial transactions using advanced machine learning techniques.
    
    ## 🔬 Technology Stack
    
    - **Algorithm**: Isolation Forest (Unsupervised Learning)
    - **Framework**: Streamlit for modern web interface
    - **Visualization**: Plotly for interactive charts
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn
    
    ## 📊 Features
    
    - **Single Transaction Analysis**: Analyze individual transactions in real-time
    - **Batch Dataset Analysis**: Process entire datasets with comprehensive reporting
    - **Model Training**: Retrain models with new data for improved accuracy
    - **Statistical Insights**: Detailed statistical analysis and visualizations
    - **Interactive Dashboards**: Modern, responsive user interface
    
    ## 🚀 Model Specifications
    
    - **Algorithm**: Isolation Forest
    - **Parameters**: 100 estimators, 2% contamination rate
    - **Features**: Transaction Amount, Average Amount, Frequency
    - **Performance**: Optimized for financial fraud detection
    
    ## 📈 Use Cases
    
    - Credit card fraud detection
    - Banking transaction monitoring
    - E-commerce payment security
    - Financial compliance and auditing
    """)
    
    # Model info from file
    try:
        with open("model.md", "r") as f:
            model_info = f.read()
        
        with st.expander("🔧 Detailed Model Information"):
            st.markdown(model_info)
    except FileNotFoundError:
        st.info("Model documentation not found.")

if __name__ == "__main__":
    main()
