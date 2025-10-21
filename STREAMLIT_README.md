# 🚨 ML Anomaly Detection Suite - Streamlit Edition

A modern, interactive web application for detecting anomalies in financial transactions using advanced machine learning techniques.

## ✨ Features

### 🔍 **Single Transaction Analysis**
- Real-time anomaly detection for individual transactions
- Interactive risk score visualization with gauge charts
- Quick test scenarios with pre-configured data
- Detailed analysis results with recommendations

### 📊 **Dataset Analysis**
- Comprehensive batch processing of transaction datasets
- Interactive visualizations with Plotly charts:
  - Anomaly distribution pie charts
  - Transaction amount histograms
  - Box plots comparing normal vs anomalous transactions
  - Risk score scatter plots with color coding
- Progress tracking for large datasets
- Sample anomalous transaction display

### 🎯 **Model Training**
- Interactive model retraining interface
- Real-time progress tracking
- Training metrics and performance evaluation
- Automatic model saving and validation

### 📈 **Statistics Dashboard**
- Comprehensive dataset statistics
- Missing value analysis
- Feature distribution visualizations
- Interactive data exploration tools

### ℹ️ **About & Documentation**
- Detailed model specifications
- Technology stack information
- Use case examples
- Performance metrics

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r Requirements.txt
```

### Running the Application

**Option 1: Command Line**
```bash
streamlit run streamlit_app.py
```

**Option 2: Batch File (Windows)**
```bash
run_streamlit.bat
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 🎨 User Interface

### Modern Design Features
- **Responsive Layout**: Adapts to different screen sizes
- **Dark Theme Support**: Professional dark mode interface
- **Interactive Charts**: Plotly-powered visualizations
- **Gradient Styling**: Modern CSS with gradient backgrounds
- **Progress Indicators**: Real-time feedback for operations
- **Sidebar Navigation**: Easy page switching
- **Status Indicators**: System status with timestamps

### Color Scheme
- **Primary**: Linear gradient from #667eea to #764ba2
- **Success**: Gradient from #4facfe to #00f2fe
- **Warning/Danger**: Gradient from #fa709a to #fee140
- **Background**: Clean white with subtle gray accents

## 🔧 Technical Architecture

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **Visualization**: Plotly (Interactive charts)
- **ML Framework**: Scikit-learn (Isolation Forest)
- **Data Processing**: Pandas, NumPy
- **Styling**: Custom CSS with modern gradients

### Model Specifications
- **Algorithm**: Isolation Forest (Unsupervised Learning)
- **Parameters**: 
  - 100 estimators
  - 2% contamination rate
  - Random state: 42
- **Features**: Transaction Amount, Average Amount, Frequency
- **Performance**: Optimized for financial fraud detection

## 📱 Page Overview

### 🔍 Single Detection
- Input fields for transaction details
- Real-time analysis with risk scoring
- Interactive gauge chart for risk visualization
- Quick test buttons for demo scenarios

### 📊 Dataset Analysis
- File upload and processing
- Multi-panel visualization dashboard
- Progress tracking with status updates
- Exportable results and insights

### 🎯 Model Training
- Interactive training interface
- Real-time progress monitoring
- Training metrics display
- Model performance evaluation

### 📈 Statistics
- Dataset overview and metrics
- Column information and data types
- Statistical summaries and distributions
- Missing value analysis

### ℹ️ About
- Application overview and features
- Technical specifications
- Model documentation
- Use case examples

## 🎯 Use Cases

### Financial Services
- **Credit Card Fraud Detection**: Real-time transaction monitoring
- **Banking Security**: Suspicious activity identification
- **Payment Processing**: E-commerce fraud prevention
- **Compliance Monitoring**: Regulatory requirement fulfillment

### Business Applications
- **Risk Management**: Transaction risk assessment
- **Audit Support**: Automated anomaly flagging
- **Customer Protection**: Fraud prevention measures
- **Data Analysis**: Pattern recognition and insights

## 🔒 Security Features

- **Input Validation**: Comprehensive data validation
- **Error Handling**: Graceful error management
- **Session Management**: Secure state handling
- **Data Privacy**: Local processing without external data transmission

## 📊 Performance Metrics

- **Processing Speed**: Real-time analysis for single transactions
- **Batch Processing**: Efficient handling of large datasets
- **Memory Usage**: Optimized for standard hardware
- **Accuracy**: High precision anomaly detection

## 🛠️ Customization

### Styling
The application uses custom CSS for modern styling. You can modify the appearance by editing the CSS in the `st.markdown()` sections of `streamlit_app.py`.

### Model Parameters
Model parameters can be adjusted in the source modules located in the `src/` directory:
- `detect_anomalies.py`: Detection logic
- `train_model.py`: Training parameters
- `data_preprocessing.py`: Data processing steps

### Visualizations
Charts and visualizations can be customized by modifying the Plotly configurations in the visualization functions.

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r Requirements.txt
```

**Model Not Found**
- Run the model training page to create a new model
- Ensure `isolation_forest.pkl` exists in the `models/` directory

**Data Loading Issues**
- Verify `transaction_anomalies_dataset.csv` exists in the `data/` directory
- Check file permissions and format

### Performance Optimization
- For large datasets, consider processing in smaller batches
- Close unused browser tabs to free memory
- Restart the application if performance degrades

## 📞 Support

For technical support or feature requests, please refer to the project documentation or contact the development team.

## 🔄 Updates

The application supports hot reloading during development. Changes to the code will automatically refresh the interface when saved.

---

**Enjoy using the ML Anomaly Detection Suite! 🚀**
