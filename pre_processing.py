import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np


def preprocess(input_csv):
    # Read CSV with comma separator and default ASCII-compatible encoding
    df = pd.read_csv(input_csv, sep=',', quotechar='"')

    # Define numeric and categorical columns
    numeric_cols = ['frame.len', 'frame.time_epoch',
                    'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport', 'tcp.flags']
    categorical_cols = ['ip.src', 'ip.dst', 'ip.proto', 'tls.handshake.extensions_server_name',
                        'tls.record.version', 'tls.handshake.type', 'quic.long.packet_type', 'quic.version']

    # Convert numeric columns, fill missing with -1
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

    # Fill missing categorical values, convert to string for encoding
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)

    # Encode categorical columns
    le_dict = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

    # Normalize numeric columns between 0 and 1
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Preprocessed data sample:")
    print(df.head())

    return df, le_dict, scaler


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
        return self.fc3(h)  # no sigmoid for regression-type output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def train_vae(data_csv, epochs=50, batch_size=64, lr=1e-3):
    df, le_dict, scaler = preprocess(data_csv)
    data = df.values.astype(np.float32)
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

    return vae, tensor_data, le_dict, scaler


# Usage:
vae_model, data_tensor, label_encoders, scaler = train_vae('capture_preprocessed.csv')
