import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
import networkx as nx
from sklearn.metrics import pairwise_distances

# Load and prepare the datasets
data_deep_learning = pd.read_csv(r'C:\Users\Lenovo\Downloads\ML_DATABASE.csv')
data_random_forest = pd.read_csv(r'C:\Users\Lenovo\Downloads\ML_DATABASE.csv')
data_random_forest.fillna(0, inplace=True)

# Deep Learning Part
# Encode the 'type' column
le = LabelEncoder()
data_deep_learning['type_encoded'] = le.fit_transform(data_deep_learning['type'])

# Create new features for balance changes
data_deep_learning['balanceOrigChange'] = data_deep_learning['newbalanceOrig'] - data_deep_learning['oldbalanceOrg']
data_deep_learning['balanceDestChange'] = data_deep_learning['newbalanceDest'] - data_deep_learning['oldbalanceDest']

# Apply logarithmic transformation to the 'amount' column to reduce skewness
data_deep_learning['log_amount'] = np.log1p(data_deep_learning['amount'])

# Features and target variable
X_dl = data_deep_learning[['type_encoded', 'log_amount', 'balanceOrigChange', 'balanceDestChange']]
y_dl = data_deep_learning['isFraud']

# Splitting the dataset into training and testing sets
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, random_state=42, stratify=y_dl)

# Standardize the features
scaler_dl = StandardScaler()
X_train_dl_scaled = scaler_dl.fit_transform(X_train_dl)
X_test_dl_scaled = scaler_dl.transform(X_test_dl)

# Building the deep learning model
model_dl = Sequential()
model_dl.add(Dense(64, input_dim=X_train_dl_scaled.shape[1], activation='relu'))
model_dl.add(Dense(32, activation='relu'))
model_dl.add(Dense(1, activation='sigmoid'))

# Compile the model
model_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_dl.fit(X_train_dl_scaled, y_train_dl, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred_dl = (model_dl.predict(X_test_dl_scaled) > 0.5).astype("int32")
y_pred_proba_dl = model_dl.predict(X_test_dl_scaled)

print("Deep Learning Model Evaluation")
print("Confusion Matrix:")
print(confusion_matrix(y_test_dl, y_pred_dl))
print("Classification Report:")
print(classification_report(y_test_dl, y_pred_dl))
print(f"Accuracy: {accuracy_score(y_test_dl, y_pred_dl):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test_dl, y_pred_proba_dl):.4f}")

# Analyze patterns
fraud_cases_dl = data_deep_learning[data_deep_learning['isFraud'] == 1]
fraud_amount_by_type_dl = fraud_cases_dl.groupby('type')['amount'].sum()
print(fraud_amount_by_type_dl)

# Identify and print the highest fraud type
highest_fraud_type_dl = fraud_amount_by_type_dl.idxmax()
highest_fraud_amount_dl = fraud_amount_by_type_dl.max()
print(f"The transaction type with the highest total fraud amount is '{highest_fraud_type_dl}' with a total loss of {highest_fraud_amount_dl}.")

# Random Forest and Other Techniques Part
data_random_forest = pd.get_dummies(data_random_forest, columns=['type'])
data_random_forest['diffOrig'] = data_random_forest['oldbalanceOrg'] - data_random_forest['newbalanceOrig']
data_random_forest['diffDest'] = data_random_forest['oldbalanceDest'] - data_random_forest['newbalanceDest']

feature_columns_rf = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'diffOrig', 'diffDest'
] + [col for col in data_random_forest.columns if col.startswith('type_')]
target_column_rf = 'isFraud'

X_rf = data_random_forest[feature_columns_rf]
y_rf = data_random_forest[target_column_rf]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)
scaler_rf = StandardScaler()
X_train_rf = scaler_rf.fit_transform(X_train_rf)
X_test_rf = scaler_rf.transform(X_test_rf)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_rf, y_train_rf)

y_pred_rf = model_rf.predict(X_test_rf)
print("Random Forest Model Evaluation")
print("Confusion Matrix:")
print(confusion_matrix(y_test_rf, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test_rf, y_pred_rf))
print("\nAccuracy Score:")
print(accuracy_score(y_test_rf, y_pred_rf))

# Batch Fraud Detection
def batch_fraud_detection(transactions):
    transactions.fillna(0, inplace=True)
    transactions = pd.get_dummies(transactions, columns=['type'])
    
    for col in [col for col in data_random_forest.columns if col.startswith('type_')]:
        if col not in transactions.columns:
            transactions[col] = 0
    
    transactions['diffOrig'] = transactions['oldbalanceOrg'] - transactions['newbalanceOrig']
    transactions['diffDest'] = transactions['oldbalanceDest'] - transactions['newbalanceDest']
    transactions = transactions[feature_columns_rf]
    transactions_scaled = scaler_rf.transform(transactions)
    
    predictions = model_rf.predict(transactions_scaled)
    
    return predictions

# Anomaly Detection
isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
isolation_forest.fit(X_train_rf)
anomalies = isolation_forest.predict(X_test_rf)
anomalies = [1 if x == -1 else 0 for x in anomalies]

print("Confusion Matrix for Anomaly Detection:")
print(confusion_matrix(y_test_rf, anomalies))
print("\nClassification Report for Anomaly Detection:")
print(classification_report(y_test_rf, anomalies))
print("\nAccuracy Score for Anomaly Detection:")
print(accuracy_score(y_test_rf, anomalies))

# Rule-based Fraud Detection
def rule_based_fraud_detection(transaction):
    if transaction['amount'] > 10000:
        return 1
    if transaction['oldbalanceOrg'] == 0 and transaction['newbalanceOrig'] == 0 and transaction['amount'] > 0:
        return 1
    if transaction['oldbalanceDest'] == 0 and transaction['newbalanceDest'] == transaction['amount']:
        return 1
    return 0

new_transaction = {
    'step': 1,
    'type': 'PAYMENT',
    'amount': 15000.0,
    'nameOrig': 'C1234567890',
    'oldbalanceOrg': 5000.0,
    'newbalanceOrig': 4000.0,
    'nameDest': 'M1234567890',
    'oldbalanceDest': 1000.0,
    'newbalanceDest': 2000.0
}

is_fraud = rule_based_fraud_detection(new_transaction)
print(f"Is the transaction fraudulent? {'Yes' if is_fraud else 'No'}")

# Real-Time Scoring Function
def calculate_risk_score(transaction):
    # Preprocess the transaction
    transaction['type_encoded'] = le.transform([transaction['type']])[0]
    transaction['balanceOrigChange'] = transaction['newbalanceOrig'] - transaction['oldbalanceOrg']
    transaction['balanceDestChange'] = transaction['newbalanceDest'] - transaction['oldbalanceDest']
    transaction['log_amount'] = np.log1p(transaction['amount'])
    
    # Extract features
    features = np.array([[
        transaction['type_encoded'],
        transaction['log_amount'],
        transaction['balanceOrigChange'],
        transaction['balanceDestChange']
    ]])
    
    # Scale features
    features_scaled = scaler_dl.transform(features)
    
    # Predict risk score
    risk_score = model_dl.predict(features_scaled)[0][0]
    return risk_score

# Reporting and Blocking
reported_frauds = {}

def report_fraud(transaction_id):
    if transaction_id not in reported_frauds:
        reported_frauds[transaction_id] = 0
    reported_frauds[transaction_id] += 1

    if reported_frauds[transaction_id] > 50:
        block_account(transaction_id)

def block_account(account_id):
    print(f"Account {account_id} has been blocked due to multiple fraud reports.")

transaction_id = 'C1234567890'
report_fraud(transaction_id)

for _ in range(51):
    report_fraud(transaction_id)

print(reported_frauds)

# Network Analysis
def create_transaction_graph(transactions):
    G = nx.Graph()
    for _, row in transactions.iterrows():
        G.add_node(row['nameOrig'], type='customer')
        G.add_node(row['nameDest'], type='merchant')
        G.add_edge(row['nameOrig'], row['nameDest'], weight=row['amount'])
    
    return G

def detect_fraud_rings(graph, threshold=10000):
    fraud_rings = []
    for subgraph in nx.connected_components(graph):
        subgraph_nodes = list(subgraph)
        if len(subgraph_nodes) > 1:
            for node in subgraph_nodes:
                neighbors = list(graph.neighbors(node))
                if any(graph[node][neighbor]['weight'] > threshold for neighbor in neighbors):
                    fraud_rings.append(subgraph_nodes)
                    break
    return fraud_rings

G = create_transaction_graph(data_random_forest)
fraud_rings = detect_fraud_rings(G)
print(f"Detected {len(fraud_rings)} potential fraud rings.")

# Behavioral Analytics
def detect_behavioral_anomalies(transactions, historical_data, threshold=0.5):
    numeric_historical_data = historical_data[feature_columns_rf].select_dtypes(include=[np.number])
    historical_data_mean = numeric_historical_data.groupby(historical_data['nameOrig']).mean().reset_index()
    anomalies = []

    for _, transaction in transactions.iterrows():
        user_history = historical_data_mean[historical_data_mean['nameOrig'] == transaction['nameOrig']]
        if not user_history.empty:
            distance = pairwise_distances([transaction[feature_columns_rf].values], user_history[feature_columns_rf].values)[0][0]
            if distance > threshold:
                anomalies.append(transaction['nameOrig'])

    return anomalies

historical_data = data_random_forest[data_random_forest['step'] < 720]
recent_transactions = data_random_forest[data_random_forest['step'] >= 720]
behavioral_anomalies = detect_behavioral_anomalies(recent_transactions, historical_data)
print(f"Detected {len(behavioral_anomalies)} behavioral anomalies.")
