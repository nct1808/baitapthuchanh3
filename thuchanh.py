import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

path= (r"D:\thuchanh\trans.csv")
df = pd.read_csv(path)
df

# --- Tiền xử lý ---
features = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
data = df[features].values
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- Dataset dạng sliding window ---
SEQ_LEN = 24

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]

dataset = TimeSeriesDataset(data_scaled, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# --- Mô hình Transformer Encoder ---
class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.linear_in = nn.Linear(feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # lấy trung bình theo chiều thời gian
        return self.linear_out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerEncoder(feature_dim=4).to(device)

# --- Trích xuất đặc trưng ---
model.eval()
features_list = []

with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        out = model(batch)
        features_list.append(out.cpu().numpy())

features_array = np.vstack(features_list)

# --- PCA để giảm chiều và trực quan ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features_array)

# --- Gaussian Mixture Model ---
# --- Gaussian Mixture Model ---
gmm = GaussianMixture(n_components=3, random_state=42, verbose=2)
clusters = gmm.fit_predict(features_array)

# --- Lấy tọa độ trọng tâm cụm sau khi PCA ---
centroids_pca = pca.transform(gmm.means_)

# --- Vẽ kết quả phân cụm + trọng tâm ---
plt.figure(figsize=(10,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="viridis", s=15)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='red', marker='X', s=150, label='Cluster Centers')
plt.title("Phân cụm GMM sau khi trích đặc trưng bằng Transformer")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.tight_layout()
plt.show()