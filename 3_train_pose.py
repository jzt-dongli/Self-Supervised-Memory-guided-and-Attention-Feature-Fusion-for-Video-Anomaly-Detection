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
        # Constants
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
        # gathering loss
        gathering_loss = self.gather_loss(query, self.m_items)
        # spreading_loss
        spreading_loss = self.spread_loss(query, self.m_items)
        # read
        updated_query, softmax_score_query, softmax_score_memory = self.read(query, self.m_items)
        mem_mse_loss = nn.MSELoss()
        mem_reconstruction_loss = mem_mse_loss(query, updated_query)

        return updated_query, self.m_items, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss, mem_reconstruction_loss

    def spread_loss(self, query, keys, k=5):
        loss = torch.nn.TripletMarginLoss(margin=1.0)
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        _, gathering_indices = torch.topk(softmax_score_memory, k+1, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, k]]

        spreading_loss = loss(query, pos, neg)

        return spreading_loss

    def gather_loss(self, query, keys, k=5):
        # loss_huber = torch.nn.SmoothL1Loss()
        loss = nn.MSELoss()
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        topk_scores, gathering_indices = torch.topk(softmax_score_memory, k, dim=1)
        topk_scores = F.softmax(topk_scores, dim=1)

        gathered_keys = keys[gathering_indices]  # [4096, 5, 34]

        gather = torch.einsum('ijk,ij->ik', gathered_keys, topk_scores)  # [4096, 34]

        gather_loss = loss(gather, query)
        return gather_loss

    def read(self, query, keys):
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        updated_query = torch.matmul(softmax_score_memory, keys)

        return updated_query, softmax_score_query, softmax_score_memory

class convAE(nn.Module):
    def __init__(self, input_dim, key_dim, memory_size):
        super(convAE, self).__init__()
        self.memory = Memory(memory_size, input_dim, key_dim)

    def forward(self, x, train=True):
        fea = x
        if train:
            updated_query, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss, mem_reconstruction_loss = self.memory(
                fea)
            output = updated_query

            return output, updated_query, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss, mem_reconstruction_loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = 'your_pose_features_train.npy'
    data = np.load(file_path, allow_pickle=True)
    test_file_path = 'your_pose_features_test.npy'
    test_data = np.load(test_file_path, allow_pickle=True)

    data_filtered = [seq for seq in data if seq.shape[0] > 0]

    stacked_data = np.vstack(data_filtered)
    data_tensor = torch.tensor(stacked_data, dtype=torch.float32)

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    test_data_filtered = [seq for seq in test_data if seq.shape[0] > 0]
    test_stacked_data = np.vstack(test_data_filtered)
    test_data_tensor = torch.tensor(test_stacked_data, dtype=torch.float32)

    test_dataset = TensorDataset(test_data_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=4096)

    n_clusters = 500
    model = convAE(input_dim=34, key_dim=34, memory_size=n_clusters)

    cluster_centers_tensor_pose = torch.load('state/clusters/cluster_centers_tensor_pose.pth')
    with torch.no_grad():
        model.memory.m_items.data.copy_(cluster_centers_tensor_pose)
    if os.path.isfile('your_model_state.pth'):
        model.load_state_dict(torch.load('your_model_state.pth'))
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.memory.parameters(), lr=learning_rate)

    model.to(device)
    mse_loss = nn.MSELoss()
    lamda1 = 1
    lamda2 = 1
    epoch = 5000
    save_interval = 100
    train_mem_reconstruction_losses = []
    test_mem_reconstruction_losses = []
    for e in tqdm(range(epoch)):
        model.train()
        avg_total = 0
        avg_gather = 0
        avg_spread = 0
        avg_reconstruction = 0
        avg_mem_reconstruction = 0
        for inputs_tensor in dataloader:
            inputs_tensor = inputs_tensor[0].to(device)
            outputs, updated_query, m_items, _, _, gather_loss, spread_loss, mem_reconstruction_loss = model(
                x=inputs_tensor)

            reconstruction_loss = mse_loss(inputs_tensor, outputs)
            total_loss = gather_loss*5 + spread_loss + mem_reconstruction_loss
            avg_total = avg_total + total_loss
            avg_gather = avg_gather + gather_loss
            avg_spread = avg_spread + spread_loss
            avg_reconstruction = avg_reconstruction + reconstruction_loss
            avg_mem_reconstruction = avg_mem_reconstruction + mem_reconstruction_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print('\nlr:', optimizer.param_groups[0]['lr'])

        optimizer.step()
        avg_total = avg_total / len(dataloader)
        avg_gather = avg_gather / len(dataloader)
        avg_spread = avg_spread / len(dataloader)
        avg_reconstruction = avg_reconstruction / len(dataloader)
        avg_mem_reconstruction = avg_mem_reconstruction / len(dataloader)

        print('\n')
        print(f'total_loss: {avg_total.item():.8f}\n',
              f'gather_loss: {avg_gather.item():.8f}\n',
              f'spread_loss: {avg_spread.item():.8f}\n',
              f'reconstruction_loss: {avg_reconstruction.item():.8f}\n',
              f'mem_reconstruction_loss: {avg_mem_reconstruction.item():.8f}')

        if (e + 1) % save_interval == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = f'state/memory_state/model_state_dict_{e + 1}_{timestamp}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"model state is saved in {save_path} (Epoch: {e + 1}, time: {timestamp})")

        model.eval()

        total_test_loss = 0
        total_gather_loss = 0
        total_spread_loss = 0
        total_mem_reconstruction_loss = 0
        with torch.no_grad():
            for inputs_tensor in test_dataloader:
                inputs_tensor = inputs_tensor[0].to(device)
                outputs, updated_query, m_items, _, _, gather_loss, spread_loss, test_mem_reconstruction_loss = model(
                    x=inputs_tensor)

                reconstruction_loss = mse_loss(inputs_tensor, outputs)

                total_test_loss += reconstruction_loss.item()
                total_gather_loss += gather_loss.item()
                total_spread_loss += spread_loss.item()
                total_mem_reconstruction_loss += test_mem_reconstruction_loss.item()
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_gather_loss = total_gather_loss / len(test_dataloader)
        avg_spread_loss = total_spread_loss / len(test_dataloader)
        avg_mem_reconstruction_loss = total_mem_reconstruction_loss / len(test_dataloader)

        print('\n')
        print('total_loss: {:.6f}'.format(avg_test_loss))
        print('gather_loss: {:.6f}'.format(avg_gather_loss))
        print('spread_loss: {:.6f}'.format(avg_spread_loss))
        print('mem_reconstruction_loss: {:.6f}'.format(avg_mem_reconstruction_loss))
