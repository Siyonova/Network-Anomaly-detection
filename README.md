# AI-Based Real-Time Threat Analysis for Networks

---

## Project Overview

This project implements a full pipeline for **network traffic anomaly (threat) detection** using a hybrid AI approach. The pipeline combines deep learning and tree-based models to identify potential malicious traffic in real-time network data.

- **Unsupervised feature learning:** A Variational Autoencoder (VAE) compresses network features into a latent space.
- **Unsupervised anomaly detection:** Isolation Forest models detect anomalies based on latent features.
- **Supervised threat classification:** LightGBM classifier is trained on pseudo-labeled data for high accuracy and low latency.

The system aims to identify real-world network threats such as DDoS, Man-in-the-Middle, Phishing, Ransomware, Trojans, and SQL injection attacks.

---

## Setup Instructions

1. Clone the repository:
      git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2. Create and activate a Python virtual environment:
    python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

3. Install required packages:
    pip install -r requirements.txt

4. Place your raw Wireshark exported CSV files under the `data/` directory.

5. Run preprocessing, model training, and evaluation:
   python pre_processing.py
    python VAE.py
    python Isolation.py
    python bulb.py


---

## Usage Examples

- To preprocess raw CSV:
   python pre_processing.py --input data/capture_raw.csv --output data/capture_preprocessed.csv

- To train VAE and extract latent features:
python VAE.py --input data/capture_preprocessed.csv


- To run Isolation Forest anomaly detection:
python Isolation.py --input data/vae_latent_features.csv --output data/vae_iso_labeled.csv


- To train and evaluate supervised classification with LightGBM:
python bulb.py --input data/vae_iso_labeled.csv


---

## Dataset Notes

- Dataset consists of 10,635 total records comprising both normal and malicious network traffic.
- Malicious traffic examples include DDoS, Man-in-the-Middle, Phishing, Ransomware, Trojan Horse, SQL Injection, and Rootkit behaviors.
- Features include packet-level info like IP address, protocol, ports, flags, and TLS/QUIC handshake details.
- Raw data extracted from Wireshark captures, then cleaned, normalized, and encoded for ML modeling.

---

## Model Descriptions & Results Summary

| Model                 | Purpose                                      | Results                                      |
|-----------------------|----------------------------------------------|----------------------------------------------|
| **Variational Autoencoder (VAE)**       | Compress features to latent space for unsupervised representation | 20 epochs, latent dimension 16, effective encoding |
| **Isolation Forest**  | Unsupervised outlier and anomaly detection on latent features       | Detected 532 anomalies (~5% contamination)   |
| **LightGBM Classifier** | Supervised classification on pseudo-labeled data                 | Accuracy: 99.86%, F1: 0.9856, ROC AUC: 0.9998 |

- Latency: Pipeline prediction time roughly 0.0167 seconds for all 10,635 samples (LightGBM)
- Plots: Confusion matrix, ROC curve and feature importance generated

---

## Documentation

- Code and scripts organized in folders:  
- `data/`: Raw and processed datasets  
- `notebooks/`: Jupyter notebooks for EDA, experiments  
- `src/`: Model definitions and training scripts
- Detailed report available in `report/` folder


---

Thanks for checking out the project! Feel free to open issues or pull requests for feedback and improvements.

