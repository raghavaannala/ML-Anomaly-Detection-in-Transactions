import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detect_anomalies import AnomalyDetector
    from utils import load_data
    from data_preprocessing import DataPreprocessor
    from train_model import train_and_evaluate
except ImportError as e:
    print(f"Import error: {e}")

class AnomalyDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚨 Anomaly Detection in Transactions")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'light': '#ecf0f1',
            'dark': '#34495e'
        }
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        """Configure custom styles for widgets"""
        self.style.configure('Title.TLabel', 
                           font=('Arial', 16, 'bold'),
                           foreground=self.colors['primary'])
        
        self.style.configure('Header.TLabel',
                           font=('Arial', 12, 'bold'),
                           foreground=self.colors['dark'])
        
        self.style.configure('Success.TLabel',
                           font=('Arial', 11),
                           foreground=self.colors['success'])
        
        self.style.configure('Danger.TLabel',
                           font=('Arial', 11),
                           foreground=self.colors['danger'])
        
        self.style.configure('Primary.TButton',
                           font=('Arial', 10, 'bold'))
        
    def create_widgets(self):
        """Create and arrange all GUI widgets"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = ttk.Label(title_frame, 
                               text="🚨 Anomaly Detection in Transactions", 
                               style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame,
                                 text="Advanced Machine Learning for Fraud Detection",
                                 font=('Arial', 10, 'italic'))
        subtitle_label.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs
        self.create_detection_tab()
        self.create_analysis_tab()
        self.create_training_tab()
        self.create_statistics_tab()
        
    def create_detection_tab(self):
        """Create single transaction detection tab"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="🔍 Single Detection")
        
        # Header
        header_label = ttk.Label(detection_frame,
                                text="Single Transaction Anomaly Detection",
                                style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Input frame
        input_frame = ttk.LabelFrame(detection_frame, text="Transaction Details", padding=20)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        # Transaction Amount
        ttk.Label(input_frame, text="Transaction Amount ($):").grid(row=0, column=0, sticky='w', pady=5)
        self.amount_var = tk.StringVar()
        amount_entry = ttk.Entry(input_frame, textvariable=self.amount_var, width=20)
        amount_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Average Transaction Amount
        ttk.Label(input_frame, text="Average Transaction Amount ($):").grid(row=1, column=0, sticky='w', pady=5)
        self.avg_amount_var = tk.StringVar()
        avg_amount_entry = ttk.Entry(input_frame, textvariable=self.avg_amount_var, width=20)
        avg_amount_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Frequency
        ttk.Label(input_frame, text="Frequency of Transactions:").grid(row=2, column=0, sticky='w', pady=5)
        self.frequency_var = tk.StringVar()
        frequency_entry = ttk.Entry(input_frame, textvariable=self.frequency_var, width=20)
        frequency_entry.grid(row=2, column=1, padx=10, pady=5)
        
        # Detect button
        detect_btn = ttk.Button(input_frame, 
                               text="🔍 Detect Anomaly",
                               command=self.detect_single_anomaly,
                               style='Primary.TButton')
        detect_btn.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Result frame
        result_frame = ttk.LabelFrame(detection_frame, text="Detection Result", padding=20)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.result_label = ttk.Label(result_frame, 
                                     text="Enter transaction details and click 'Detect Anomaly'",
                                     font=('Arial', 12))
        self.result_label.pack(pady=20)
        
        # Quick test buttons
        quick_frame = ttk.Frame(detection_frame)
        quick_frame.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(quick_frame, text="Quick Tests:").pack(side='left')
        
        normal_btn = ttk.Button(quick_frame, text="Normal Transaction", 
                               command=lambda: self.load_sample_data(False))
        normal_btn.pack(side='left', padx=5)
        
        anomaly_btn = ttk.Button(quick_frame, text="Suspicious Transaction",
                                command=lambda: self.load_sample_data(True))
        anomaly_btn.pack(side='left', padx=5)
        
    def create_analysis_tab(self):
        """Create dataset analysis tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Dataset Analysis")
        
        # Header
        header_label = ttk.Label(analysis_frame,
                                text="Complete Dataset Anomaly Analysis",
                                style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Control frame
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(fill='x', padx=20, pady=10)
        
        analyze_btn = ttk.Button(control_frame,
                                text="📊 Analyze Dataset",
                                command=self.analyze_dataset,
                                style='Primary.TButton')
        analyze_btn.pack(side='left', padx=5)
        
        # Results frame with scrollable text
        results_frame = ttk.LabelFrame(analysis_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.analysis_text = scrolledtext.ScrolledText(results_frame, 
                                                      height=15, 
                                                      font=('Consolas', 10))
        self.analysis_text.pack(fill='both', expand=True)
        
    def create_training_tab(self):
        """Create model training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🎯 Model Training")
        
        # Header
        header_label = ttk.Label(training_frame,
                                text="Retrain Anomaly Detection Model",
                                style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Info frame
        info_frame = ttk.LabelFrame(training_frame, text="Training Information", padding=20)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """
        This will retrain the Isolation Forest model using the current dataset.
        The process includes:
        • Data preprocessing and feature engineering
        • Model training with optimized parameters
        • Performance evaluation and metrics calculation
        • Model saving for future predictions
        """
        
        ttk.Label(info_frame, text=info_text, justify='left').pack()
        
        # Training controls
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(fill='x', padx=20, pady=10)
        
        train_btn = ttk.Button(control_frame,
                              text="🎯 Start Training",
                              command=self.retrain_model,
                              style='Primary.TButton')
        train_btn.pack(side='left', padx=5)
        
        # Training output
        output_frame = ttk.LabelFrame(training_frame, text="Training Output", padding=10)
        output_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.training_text = scrolledtext.ScrolledText(output_frame,
                                                      height=12,
                                                      font=('Consolas', 10))
        self.training_text.pack(fill='both', expand=True)
        
    def create_statistics_tab(self):
        """Create dataset statistics tab"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📈 Statistics")
        
        # Header
        header_label = ttk.Label(stats_frame,
                                text="Dataset Statistics and Insights",
                                style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Control frame
        control_frame = ttk.Frame(stats_frame)
        control_frame.pack(fill='x', padx=20, pady=10)
        
        stats_btn = ttk.Button(control_frame,
                              text="📈 Load Statistics",
                              command=self.show_statistics,
                              style='Primary.TButton')
        stats_btn.pack(side='left', padx=5)
        
        # Statistics display
        stats_display_frame = ttk.LabelFrame(stats_frame, text="Dataset Overview", padding=10)
        stats_display_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.stats_text = scrolledtext.ScrolledText(stats_display_frame,
                                                   height=20,
                                                   font=('Consolas', 10))
        self.stats_text.pack(fill='both', expand=True)
        
    def load_sample_data(self, is_anomaly=False):
        """Load sample data for quick testing"""
        if is_anomaly:
            # Suspicious transaction
            self.amount_var.set("15000")
            self.avg_amount_var.set("200")
            self.frequency_var.set("1")
        else:
            # Normal transaction
            self.amount_var.set("150")
            self.avg_amount_var.set("180")
            self.frequency_var.set("5")
    
    def detect_single_anomaly(self):
        """Detect anomaly in a single transaction"""
        try:
            # Validate inputs
            amount = float(self.amount_var.get())
            avg_amount = float(self.avg_amount_var.get())
            frequency = int(self.frequency_var.get())
            
            # Create detector and predict
            detector = AnomalyDetector()
            user_input = {
                'Transaction_Amount': amount,
                'Average_Transaction_Amount': avg_amount,
                'Frequency_of_Transactions': frequency
            }
            
            result = detector.predict(user_input)
            
            if result:
                self.result_label.config(text="🚨 ANOMALY DETECTED!\nThis transaction appears suspicious.",
                                       style='Danger.TLabel')
            else:
                self.result_label.config(text="✅ NORMAL TRANSACTION\nNo anomaly detected.",
                                       style='Success.TLabel')
                
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
        except FileNotFoundError:
            messagebox.showerror("Model Error", "Model not found. Please train the model first.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def analyze_dataset(self):
        """Analyze the entire dataset for anomalies"""
        try:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, "Starting dataset analysis...\n\n")
            self.root.update()
            
            # Load data
            data_path = "data/transaction_anomalies_dataset.csv"
            data = load_data(data_path)
            
            self.analysis_text.insert(tk.END, f"Dataset loaded: {data.shape[0]} records\n")
            self.analysis_text.insert(tk.END, f"Features: {list(data.columns)}\n\n")
            
            # Load model and detect anomalies
            detector = AnomalyDetector()
            features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
            
            if all(col in data.columns for col in features):
                self.analysis_text.insert(tk.END, "Analyzing transactions for anomalies...\n")
                self.root.update()
                
                predictions = []
                for i, (_, row) in enumerate(data.iterrows()):
                    if i % 100 == 0:  # Update progress
                        self.analysis_text.insert(tk.END, f"Processed {i}/{len(data)} transactions\n")
                        self.root.update()
                    
                    pred = detector.predict(row[features].to_dict())
                    predictions.append(pred)
                
                anomaly_count = sum(predictions)
                total_count = len(predictions)
                
                self.analysis_text.insert(tk.END, f"\n{'='*50}\n")
                self.analysis_text.insert(tk.END, f"ANALYSIS RESULTS\n")
                self.analysis_text.insert(tk.END, f"{'='*50}\n")
                self.analysis_text.insert(tk.END, f"Total transactions: {total_count}\n")
                self.analysis_text.insert(tk.END, f"Anomalies detected: {anomaly_count}\n")
                self.analysis_text.insert(tk.END, f"Anomaly rate: {(anomaly_count/total_count)*100:.2f}%\n\n")
                
                # Show anomalous transactions
                if anomaly_count > 0:
                    anomaly_indices = [i for i, pred in enumerate(predictions) if pred == 1]
                    self.analysis_text.insert(tk.END, f"Sample anomalous transactions:\n")
                    self.analysis_text.insert(tk.END, f"{'-'*60}\n")
                    
                    for i in anomaly_indices[:10]:  # Show first 10
                        row = data.iloc[i]
                        self.analysis_text.insert(tk.END, 
                            f"Transaction {i+1}: Amount=${row['Transaction_Amount']:.2f}, "
                            f"Avg=${row['Average_Transaction_Amount']:.2f}, "
                            f"Freq={row['Frequency_of_Transactions']}\n")
                
            else:
                self.analysis_text.insert(tk.END, f"❌ Required columns not found.\n")
                self.analysis_text.insert(tk.END, f"Available: {list(data.columns)}\n")
                
        except Exception as e:
            self.analysis_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
    
    def retrain_model(self):
        """Retrain the anomaly detection model"""
        try:
            self.training_text.delete(1.0, tk.END)
            self.training_text.insert(tk.END, "Starting model retraining...\n\n")
            self.root.update()
            
            # Load data
            data_path = "data/transaction_anomalies_dataset.csv"
            data = load_data(data_path)
            
            self.training_text.insert(tk.END, f"Data loaded: {data.shape}\n")
            
            # Preprocess data
            preprocessor = DataPreprocessor()
            processed_data = preprocessor.fit_transform(data)
            
            self.training_text.insert(tk.END, "Data preprocessing completed\n")
            
            # Prepare features
            features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
            
            if all(col in processed_data.columns for col in features):
                X = processed_data[features]
                y = processed_data.get('Is_Anomaly', processed_data.get('Class', [0]*len(X)))
                
                # Split data
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                self.training_text.insert(tk.END, f"Training set: {len(X_train)} samples\n")
                self.training_text.insert(tk.END, f"Test set: {len(X_test)} samples\n\n")
                self.training_text.insert(tk.END, "Training model...\n")
                self.root.update()
                
                # Train model (redirect output)
                import io
                import contextlib
                
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    model = train_and_evaluate(X_train, X_test, y_test)
                
                output = f.getvalue()
                self.training_text.insert(tk.END, output)
                self.training_text.insert(tk.END, "\n✅ Model retrained successfully!\n")
                
            else:
                self.training_text.insert(tk.END, f"❌ Required columns not found.\n")
                
        except Exception as e:
            self.training_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
    
    def show_statistics(self):
        """Display dataset statistics"""
        try:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "Loading dataset statistics...\n\n")
            self.root.update()
            
            # Load data
            data_path = "data/transaction_anomalies_dataset.csv"
            data = load_data(data_path)
            
            self.stats_text.insert(tk.END, f"{'='*60}\n")
            self.stats_text.insert(tk.END, f"DATASET OVERVIEW\n")
            self.stats_text.insert(tk.END, f"{'='*60}\n")
            self.stats_text.insert(tk.END, f"Dataset Shape: {data.shape}\n")
            self.stats_text.insert(tk.END, f"Number of Records: {len(data)}\n")
            self.stats_text.insert(tk.END, f"Number of Features: {len(data.columns)}\n\n")
            
            self.stats_text.insert(tk.END, f"COLUMNS:\n")
            self.stats_text.insert(tk.END, f"{'-'*40}\n")
            for i, col in enumerate(data.columns, 1):
                self.stats_text.insert(tk.END, f"{i:2d}. {col}\n")
            
            self.stats_text.insert(tk.END, f"\n{'='*60}\n")
            self.stats_text.insert(tk.END, f"STATISTICAL SUMMARY\n")
            self.stats_text.insert(tk.END, f"{'='*60}\n")
            
            # Get numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            stats_summary = numeric_data.describe()
            
            self.stats_text.insert(tk.END, str(stats_summary))
            
            # Missing values
            self.stats_text.insert(tk.END, f"\n\n{'='*60}\n")
            self.stats_text.insert(tk.END, f"MISSING VALUES\n")
            self.stats_text.insert(tk.END, f"{'='*60}\n")
            
            missing_values = data.isnull().sum()
            if missing_values.any():
                for col, missing in missing_values.items():
                    if missing > 0:
                        self.stats_text.insert(tk.END, f"{col}: {missing} ({missing/len(data)*100:.2f}%)\n")
            else:
                self.stats_text.insert(tk.END, "✅ No missing values found.\n")
                
        except Exception as e:
            self.stats_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
