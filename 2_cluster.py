import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F
import os
import os.path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
from torch.nn.parameter import Parameter
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time
from datetime import datetime

class Encoder(nn.Module):
    def __init__(self, input_dim=512, encoded_dim=34):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, encoded_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(input_dim=512, encoded_dim=34).to(device)

    file_path = 'your_features_path.npy'
    data = np.load(file_path, allow_pickle=True)

    encoder.load_state_dict(torch.load('state/ae_encoder/encoder_best_epoch.pth'))

    encoded_data_list = []
    with torch.no_grad():
        encoder.eval()
        for sample in data:
            sample_tensor = torch.tensor(sample).float().to(device)
            encoded_sample = encoder(sample_tensor).cpu().numpy()
            encoded_data_list.append(encoded_sample)
    encoded_data = np.vstack(encoded_data_list).astype(np.float32)

    n_clusters = 500

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    cluster_labels = kmeans.fit_predict(encoded_data)

    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)

    plt.bar(range(n_clusters), cluster_counts, color='blue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points in Cluster')
    plt.title('Distribution of Points among Clusters')
    plt.show()

    cluster_centers_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    torch.save(cluster_centers_tensor, 'state/clusters/cluster_centers_tensor.pth')
