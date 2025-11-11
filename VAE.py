import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
# Load your CSV dataset
df = pd.read_csv('capture_preprocessed.csv')
print(df.columns)


# Load and preprocess your CSV (normalized, categorical encoded)
def preprocess(input_csv):
    df = pd.read_csv(input_csv, sep='\t', quotechar='"', encoding='utf-8', engine='python')

    # Numeric and categorical columns
    numeric_cols = ['frame.number', 'frame.len', 'frame.time_epoch',
                    'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.flags']
    
    print("Columns before scaling:", df.columns)

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

    # Normalize numerical data between 0 and 1
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df[numeric_cols + categorical_cols].values.astype(np.float32)

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
        return self.fc3(h)  # no sigmoid for regression type reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # MSE reconstruction loss
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

# Main training routine
def train_vae(data_path, epochs=50, batch_size=64, lr=1e-3):
    data = preprocess(data_path)
    tensor_data = torch.tensor(data)

    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = data.shape[1]
    vae = VAE(input_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    for epoch in range(epochs):
        running_loss = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(x_batch)
            loss = loss_function(recon_batch, x_batch, mu, logvar)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return vae, torch.tensor(data)

# Example usage
vae, data_tensor = train_vae('capture_preprocessed.csv')

# After training, extract latent features and use Isolation Forest or other methods.

