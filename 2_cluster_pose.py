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
from tqdm import tqdm
import torch.optim as optim
from torch.nn.parameter import Parameter
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import time
from datetime import datetime


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = 'your_poses_path.npy'
    data = np.load(file_path, allow_pickle=True)

    data_filtered = [seq for seq in data if seq.shape[0] > 0]
    data_list = []

    with torch.no_grad():
        for sample in data_filtered:
            sample_tensor = torch.tensor(sample).float().to(device)
            data_list.append(sample_tensor.cpu())

    data = np.vstack(data_list)

    n_clusters = 500

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data)

    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)

    cluster_centers = np.array([data[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
    cluster_centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
    torch.save(cluster_centers_tensor, 'state/clusters/cluster_centers_tensor_pose.pth')
