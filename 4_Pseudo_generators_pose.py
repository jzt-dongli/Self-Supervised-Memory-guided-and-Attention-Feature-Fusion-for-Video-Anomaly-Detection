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

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.m_items = Parameter(F.normalize(torch.rand((memory_size, key_dim), dtype=torch.float), dim=1),
                                 requires_grad=True)


    def get_score(self, mem, query):
        score = torch.matmul(query, torch.t(mem))
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory

    def forward(self, query):
        # read
        top_half_output, bottom_half_output = self.read(query, self.m_items)

        return top_half_output, bottom_half_output

    def z_score_normalize(self, tensor):
        eps = 1e-9
        mean = tensor.mean()
        std = tensor.std()
        normalized = (tensor - mean) / (std + eps)
        return normalized

    def read(self, query, keys, temperature_1=1,temperature_2=1):
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        pseudo_normal_outputs = []
        pseudo_abnormal_outputs = []
        all_top_indices = []
        all_bottom_indices = []

        for i in range(softmax_score_memory.shape[0]):
            scores = softmax_score_memory[i]
            _, sorted_indices = torch.sort(scores, descending=True)

            top_indices = sorted_indices[0:5]
            bottom_indices = sorted_indices[5:10]
            all_top_indices.append(top_indices)
            all_bottom_indices.append(bottom_indices)

            softmax_top_scores = F.softmax(scores[top_indices] / temperature_1, dim=0)
            softmax_bottom_scores = F.softmax(scores[bottom_indices] / temperature_2, dim=0)

            top_output = torch.matmul(softmax_top_scores, keys[top_indices])
            bottom_output = torch.matmul(softmax_bottom_scores, keys[bottom_indices])

            original_feature = query[i].unsqueeze(0)

            pseudo_normal_output = 0.5 * original_feature + 0.5 * top_output
            pseudo_abnormal_output = 0.01 * original_feature + 0.99 * bottom_output

            pseudo_normal_outputs.append(pseudo_normal_output)
            pseudo_abnormal_outputs.append(pseudo_abnormal_output)

        pseudo_normal_outputs = torch.cat(pseudo_normal_outputs, dim=0)
        pseudo_abnormal_outputs = torch.cat(pseudo_abnormal_outputs, dim=0)

        with open('state/indices/top_indices.txt', 'a') as f:
            for indices_list in all_top_indices:
                row_values = []
                for indices in indices_list:
                    if indices.dim() == 0:
                        indices = indices.view(1)
                    row_values.extend([str(item.item()) for item in indices])
                f.write(' '.join(row_values) + '\n')

        with open('state/indices/bottom_indices.txt', 'a') as f:
            for indices_list in all_bottom_indices:
                row_values = []
                for indices in indices_list:
                    if indices.dim() == 0:
                        indices = indices.view(1)
                    row_values.extend([str(item.item()) for item in indices])
                f.write(' '.join(row_values) + '\n')
        return pseudo_normal_outputs, pseudo_abnormal_outputs

class convAE(nn.Module):
    def __init__(self, encoded_dim, key_dim, memory_size):
        super(convAE, self).__init__()
        self.memory = Memory(memory_size, encoded_dim, key_dim)

    def forward(self, x):
        top_half_output, bottom_half_output = self.memory(
                x)

        return top_half_output, bottom_half_output, x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = 'your_pose_features_train.npy'
    # file_path = 'your_pose_features_test.npy'
    data = np.load(file_path, allow_pickle=True)
    data_filtered = [seq for seq in data if seq.shape[0] > 0]

    n_clusters = 500
    lamda = 1
    model = convAE(encoded_dim=34, key_dim=34, memory_size=n_clusters)

    with torch.no_grad():
        model.memory.m_items.data.copy_(cluster_centers_tensor)
    if os.path.isfile('your_model_state.pth'):
        model.load_state_dict(torch.load('your_model_state.pth'))

    mse_loss = nn.MSELoss()
    model.to(device)
    model.eval()
    pseudo_normal_list = []
    pseudo_abnormal_list = []
    for inputs in data_filtered:
        inputs_tensor = torch.from_numpy(inputs).float().to(device)
        top_half_output, bottom_half_output, fea = model(inputs_tensor)
        top_loss = mse_loss(top_half_output, fea)
        bottom_loss = mse_loss(bottom_half_output, fea)
        print(f'top_loss: {top_loss.item():.8f},',f'bottom_loss: {bottom_loss.item():.8f}')
        pseudo_normal_list.append(top_half_output)
        pseudo_abnormal_list.append(bottom_half_output)

    torch.save(pseudo_normal_list, 'state/pseudo_list/pseudo_normal_list_pose.pth')
    torch.save(pseudo_abnormal_list, 'state/pseudo_list/pseudo_abnormal_list_pose.pth')
    # torch.save(pseudo_normal_list, 'state/pseudo_list/test_pose_memorized.pth')
