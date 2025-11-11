import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# --- Load and preprocess ---
data = pd.read_csv('capture_preprocessed.csv')
features = ['frame.len', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto',
            'tcp.srcport', 'tcp.dstport', 'tcp.flags', 'udp.srcport', 'udp.dstport',
            'tls.handshake.extensions_server_name', 'tls.record.version', 'tls.handshake.type',
            'quic.long.packet_type', 'quic.version']
categorical_features = ['ip.src', 'ip.dst', 'ip.proto', 'tls.handshake.extensions_server_name',
                        'tls.record.version', 'tls.handshake.type', 'quic.long.packet_type', 'quic.version']

# Encode categoricals
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

X = data[features].fillna(-1).values
X_torch = torch.tensor(X, dtype=torch.float32)

# --- VAE definition ---
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

input_dim = X.shape[1]
vae = VAE(input_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
criterion = nn.MSELoss(reduction='sum')

# --- Train VAE ---
dataset = TensorDataset(X_torch)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
vae.train()
for epoch in range(20):
    total_loss = 0
    for batch in loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        recon, mu, logvar = vae(x_batch)
        mse_loss = criterion(recon, x_batch)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse_loss + kld_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(X):.4f}")

# --- Latent features, Isolation Forest ---
vae.eval()
with torch.no_grad():
    mu, _ = vae.encode(X_torch)
latent_features = mu.numpy()

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(latent_features)
anomalies = iso_forest.predict(latent_features)  # -1 anomaly, 1 normal
data['anomaly_pred'] = anomalies
print(data['anomaly_pred'].value_counts())

# --- For synthetic "evaluation", treat -1 as anomaly, 1 as normal ---
y_pred = (data['anomaly_pred'] == -1).astype(int)

# For unsupervised, there's usually no y_true, but if you have true anomaly label (e.g. 'is_threat'), use that
# y_true = data['is_threat']  # Uncomment if available
y_true = y_pred  # For demonstration ONLY; in real data, use ground truth if you have one

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Normal", "Anomaly"])
plt.yticks(tick_marks, ["Normal", "Anomaly"])
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# --- ROC curve (only for demo) ---
anomaly_score = -iso_forest.decision_function(latent_features)  # Higher means more abnormal
fpr, tpr, _ = roc_curve(y_true, anomaly_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- Extra metrics ---
print("Classification report:\n", classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
