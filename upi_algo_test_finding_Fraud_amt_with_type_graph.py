import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import networkx as nx
from sklearn.metrics import pairwise_distances

# Load the dataset
data = pd.read_csv(r'C:\Users\Lenovo\Downloads\ML_DATABASE.csv')

# Encode the 'type' column
le = LabelEncoder()
data['type_encoded'] = le.fit_transform(data['type'])

# Create new features for balance changes
data['balanceOrigChange'] = data['newbalanceOrig'] - data['oldbalanceOrg']
data['balanceDestChange'] = data['newbalanceDest'] - data['oldbalanceDest']

# Apply logarithmic transformation to the 'amount' column to reduce skewness
data['log_amount'] = np.log1p(data['amount'])

# Features and target variable
X = data[['type_encoded', 'log_amount', 'balanceOrigChange', 'balanceDestChange']]
y = data['isFraud']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the deep learning model
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
y_pred_proba = model.predict(X_test_scaled)

# Confusion Matrix and Classification Report for Deep Learning Model
print("Deep Learning Model Evaluation")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Plotting Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, 'Confusion Matrix - Deep Learning Model')

# Plotting ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label='Deep Learning Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Analyze patterns
fraud_cases = data[data['isFraud'] == 1]
fraud_amount_by_type = fraud_cases.groupby('type')['amount'].sum()
print(fraud_amount_by_type)

# Identify and print the highest fraud type
highest_fraud_type = fraud_amount_by_type.idxmax()
highest_fraud_amount = fraud_amount_by_type.max()
print(f"The transaction type with the highest total fraud amount is '{highest_fraud_type}' with a total loss of {highest_fraud_amount}.")

# Plot Fraud Amount by Type
fraud_amount_by_type.plot(kind='bar', figsize=(10, 6), title='Fraud Amount by Type')
plt.xlabel('Transaction Type')
plt.ylabel('Total Fraud Amount')
plt.show()

# Random Forest Model
data = pd.read_csv(r'C:\Users\Lenovo\Downloads\ML_DATABASE.csv')
data.fillna(0, inplace=True)
data = pd.get_dummies(data, columns=['type'])
data['diffOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['diffDest'] = data['oldbalanceDest'] - data['newbalanceDest']

feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'diffOrig', 'diffDest'
] + [col for col in data.columns if col.startswith('type_')]
target_column = 'isFraud'

X = data[feature_columns]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Model Evaluation")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred_rf))

# Plotting Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, 'Confusion Matrix - Random Forest Model')

# Feature Importance for Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.title("Feature Importances - Random Forest")
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [feature_columns[i] for i in indices], rotation=90)
plt.show()

# Anomaly Detection
isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
isolation_forest.fit(X_train)
anomalies = isolation_forest.predict(X_test)
anomalies = [1 if x == -1 else 0 for x in anomalies]

print("Confusion Matrix for Anomaly Detection:")
print(confusion_matrix(y_test, anomalies))
print("\nClassification Report for Anomaly Detection:")
print(classification_report(y_test, anomalies))
print("\nAccuracy Score for Anomaly Detection:")
print(accuracy_score(y_test, anomalies))

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
    features_scaled = scaler.transform(features)
    
    # Predict risk score
    risk_score = model.predict(features_scaled)[0][0]
    return risk_score

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
edges = data[data['isFraud'] == 1][['nameOrig', 'nameDest']]
G = nx.from_pandas_edgelist(edges, source='nameOrig', target='nameDest')

def connected_components_subgraphs(G):
    return (G.subgraph(c).copy() for c in nx.connected_components(G))

fraud_rings = list(connected_components_subgraphs(G))
print(f"Detected {len(fraud_rings)} potential fraud rings.")

# Behavioral Analytics
historical_data = data[data['nameOrig'] == new_transaction['nameOrig']]
historical_data = historical_data.groupby('nameOrig').mean().reset_index()
new_transaction_df = pd.DataFrame([new_transaction])
combined_data = pd.concat([historical_data, new_transaction_df])

def detect_behavioral_anomalies(data):
    # Keep only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Handle NaNs (fill with 0 or column mean)
    numeric_data = numeric_data.fillna(0)   # option 1: replace NaN with 0
    # numeric_data = numeric_data.fillna(numeric_data.mean())  # option 2: replace with mean
    
    # Compute pairwise distances
    distances = pairwise_distances(numeric_data, numeric_data)
    anomalies = np.where(distances > np.percentile(distances, 95), 1, 0)
    return anomalies

behavioral_anomalies = detect_behavioral_anomalies(combined_data)
print(f"Detected {sum(behavioral_anomalies[0])} behavioral anomalies.")

# Plotting behavior of new transaction against historical data
plt.figure(figsize=(10, 6))
plt.plot(combined_data.columns[1:], combined_data.iloc[0, 1:], label='Historical Data')
plt.plot(combined_data.columns[1:], combined_data.iloc[1, 1:], label='New Transaction')
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Behavioral Analysis of Transactions')
plt.legend()
plt.show()

