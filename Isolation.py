import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load your CSV dataset
data = pd.read_csv('capture_preprocessed.csv')

# Select features relevant for anomaly detection
features = [
    'frame.len', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto',
    'tcp.srcport', 'tcp.dstport', 'tcp.flags',
    'udp.srcport', 'udp.dstport',
    'tls.handshake.extensions_server_name', 'tls.record.version',
    'tls.handshake.type', 'quic.long.packet_type', 'quic.version'
]

# Preprocessing

# Fill missing values (if any)
data.fillna(-1, inplace=True)

# Encode categorical/text features to numerical codes
label_encoders = {}
categorical_features = [
    'ip.src', 'ip.dst', 'ip.proto',
    'tls.handshake.extensions_server_name', 'tls.record.version',
    'tls.handshake.type', 'quic.long.packet_type', 'quic.version'
]

for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Prepare feature matrix for model
X = data[features].astype(float)

# Initialize and train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# Predict anomalies (-1 for anomaly, 1 for normal)
data['anomaly'] = model.predict(X)

# Filter out detected anomalies as potential threats
anomalies = data[data['anomaly'] == -1]

# Output summary
print(f"Total records: {len(data)}")
print(f"Detected potential threats (anomalies): {len(anomalies)}")
print(anomalies.head())

# Optional: Save anomalies to a file
anomalies.to_csv('detected_threats.csv', index=False)
