UPI Fraud Detection 🔐

This project detects fraudulent transactions in UPI (Unified Payments Interface) systems using a mix of Machine Learning, Deep Learning, Anomaly Detection, and Rule-based techniques.

📂 Project Files
- `ML_DATABASE_FINAL.csv` → Dataset containing transaction records  
- `tempCodeRunnerFile.py` → End-to-end fraud detection pipeline (ML/DL + anomaly detection + behavioral analysis + rule engine)  
- `upi_algo_test_finding_Fraud_amt_with_type_graph.py` → Extended fraud detection with visualizations (ROC curves, fraud-by-type graphs, feature importance, etc.)  

⚡ Features
- ✅ Deep Learning model (Keras Sequential)  
- ✅ Random Forest Classifier for fraud detection  
- ✅ Isolation Forest for anomaly detection  
- ✅ Rule-based fraud checks (amount thresholds, suspicious balances)  
- ✅ Network analysis (fraud ring detection using graph theory)  
- ✅ Behavioral analytics (detects abnormal transaction behavior)  
- ✅ Data visualization (confusion matrices, ROC curves, fraud amount by type, feature importance)  

🛠️ Tech Stack
- Programming Language: Python  
- Libraries:
  - Pandas, NumPy  
  - Scikit-learn  
  - TensorFlow / Keras  
  - Matplotlib, Seaborn  
  - NetworkX  

🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/UPI_FRAUD_DETECTION.git
   cd UPI_FRAUD_DETECTION
Install dependencies:
pip install -r requirements.txt

Run a script:
python upi_algo_test_finding_Fraud_amt_with_type_graph.py

📊 Dataset
The dataset (ML_DATABASE_FINAL.csv) includes:
Transaction step, type, and amount
Origin & destination balances
Fraud label (isFraud)

⚠️ Note: Always anonymize sensitive financial data before using in real-world systems.

📌 Future Enhancements
Deploy as a real-time fraud detection API/dashboard
Use advanced architectures (LSTMs, Transformers) for sequential fraud detection
Add adaptive rule-based detection with dynamic thresholds





📜 License
This project is licensed under the MIT License – free to use, modify, and share.
