import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest


# --- If starting from scratch, run preprocessing as before ---
def preprocess(input_csv):
    # Read your comma-separated CSV file
    df = pd.read_csv(input_csv, sep=',', quotechar='"')
    numeric_cols = ['frame.len', 'frame.time_epoch',
                    'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.flags']
    categorical_cols = ['ip.src', 'ip.dst', 'ip.proto', 'tls.handshake.extensions_server_name',
                        'tls.record.version', 'tls.handshake.type', 'quic.long.packet_type', 'quic.version']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# --- Your VAE model class here ---
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
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
        return self.fc3(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- Load, preprocess and run VAE + Isolation Forest ---
# 1. Load preprocessed data
df = preprocess('capture_preprocessed.csv')
data = df.values.astype('float32')
tensor_data = torch.tensor(data)

# 2. Load/training your VAE
input_dim = data.shape[1]
vae = VAE(input_dim)
# If you've trained and saved your VAE weights:
# vae.load_state_dict(torch.load('vae_weights.pth'))
vae.eval()

# 3. Extract latent features (mean vectors)
with torch.no_grad():
    mu, logvar = vae.encode(tensor_data)
latent_features = mu.numpy()

# 4. Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(latent_features)
anomaly_labels = iso_forest.predict(latent_features)  # -1=anomaly, 1=normal

# 5. Create labeled dataframe and save to CSV
df['label'] = anomaly_labels
df.to_csv("vae_iso_labeled.csv", index=False)
print("Saved VAE-Isolation Forest labeled CSV as 'vae_iso_labeled.csv'")
