import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve ,auc
from scipy.ndimage import gaussian_filter1d
import torch.nn.init as init
import matplotlib.pyplot as plt

class CrossModalAttention(nn.Module):

    def __init__(self, d_model, dropout=0.5):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model

        self.w_qs1 = nn.Linear(d_model, d_model, bias=False)
        self.w_ks1 = nn.Linear(d_model, d_model, bias=False)
        self.w_vs1 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs2 = nn.Linear(d_model, d_model, bias=False)
        self.w_ks2 = nn.Linear(d_model, d_model, bias=False)
        self.w_vs2 = nn.Linear(d_model, d_model, bias=False)
        self.w_qs3 = nn.Linear(d_model, d_model, bias=False)
        self.w_ks3 = nn.Linear(d_model, d_model, bias=False)
        self.w_vs3 = nn.Linear(d_model, d_model, bias=False)

        self.preprocess_f1 = nn.Linear(d_model, d_model)
        self.preprocess_f2 = nn.Linear(d_model, d_model)
        self.preprocess_f3 = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):

        init.kaiming_uniform_(self.preprocess_f1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.preprocess_f2.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.preprocess_f3.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.w_qs1.weight)
        init.kaiming_uniform_(self.w_ks1.weight)
        init.kaiming_uniform_(self.w_vs1.weight)
        init.kaiming_uniform_(self.w_qs2.weight)
        init.kaiming_uniform_(self.w_ks2.weight)
        init.kaiming_uniform_(self.w_vs2.weight)
        init.kaiming_uniform_(self.w_qs3.weight)
        init.kaiming_uniform_(self.w_ks3.weight)
        init.kaiming_uniform_(self.w_vs3.weight)

    def forward(self, f1, f2, f3, mask1=None, mask2=None, mask3=None):

        f1_pre = F.relu(self.preprocess_f1(f1))
        f2_pre = F.relu(self.preprocess_f2(f2))
        f3_pre = F.relu(self.preprocess_f3(f3))

        f1_pre = self.dropout(f1_pre)
        f2_pre = self.dropout(f2_pre)
        f3_pre = self.dropout(f3_pre)

        q1, k1, v1 = self.w_qs1(f1_pre), self.w_ks1(f1_pre), self.w_vs1(f1_pre)
        q2, k2, v2 = self.w_qs2(f2_pre), self.w_ks2(f2_pre), self.w_vs2(f2_pre)
        q3, k3, v3 = self.w_qs3(f3_pre), self.w_ks3(f3_pre), self.w_vs3(f3_pre)

        if mask1 is not None:
            k1 = k1.masked_fill(mask1 == True, -1e9)
            v1 = v1.masked_fill(mask1 == True, -1e9)
        if mask2 is not None:
            k2 = k2.masked_fill(mask2 == True, -1e9)
            v2 = v2.masked_fill(mask2 == True, -1e9)
        if mask3 is not None:
            k3 = k3.masked_fill(mask3 == 0, -1e9)
            v3 = v3.masked_fill(mask3 == 0, -1e9)

        f1_updated = torch.matmul(self.softmax(torch.matmul(q2, k1.transpose(-2, -1)) + torch.matmul(q3, k1.transpose(-2, -1))), v1)
        f2_updated = torch.matmul(self.softmax(torch.matmul(q1, k3.transpose(-2, -1)) + torch.matmul(q3, k2.transpose(-2, -1))), v2)
        f3_updated = torch.matmul(self.softmax(torch.matmul(q1, k3.transpose(-2, -1)) + torch.matmul(q2, k3.transpose(-2, -1))), v3)
        # f2_updated = torch.matmul(self.softmax(torch.matmul(q3, k2.transpose(-2, -1))), v2)
        # f3_updated = torch.matmul(self.softmax(torch.matmul(q2, k3.transpose(-2, -1))), v3)
        f_concatenated = torch.cat([f1_updated, f2_updated, f3_updated, f1_pre, f2_pre, f3_pre], dim=-1)
        # f_concatenated = torch.cat([f2_updated, f3_updated, f2_pre, f3_pre], dim=-1)
        return f_concatenated

def smooth_predictions(scores, sigma=3):
    smoothed_scores = gaussian_filter1d(scores, sigma=sigma)

    return smoothed_scores
class AnomalyScoreClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.5):
        super(AnomalyScoreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):

        if self.fc1.bias is not None:
            self.fc1.bias.data.fill_(0.01)
        if self.fc2.bias is not None:
            self.fc2.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        score = torch.sigmoid(self.fc2(x))
        return score


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pseudo_normal_list_padded_path = 'state/pseudo_list/pseudo_normal_list_padded.pth'
    pseudo_normal_list_masks_path = 'state/pseudo_list/pseudo_normal_list_masks.pth'
    pseudo_normal_list_pose_padded_path = 'state/pseudo_list/pseudo_normal_list_pose_padded.pth'
    pseudo_normal_list_pose_masks_path = 'state/pseudo_list/pseudo_normal_list_pose_masks.pth'
    pseudo_abnormal_list_padded_path = 'state/pseudo_list/pseudo_abnormal_list_padded.pth'
    pseudo_abnormal_list_masks_path = 'state/pseudo_list/pseudo_abnormal_list_masks.pth'
    pseudo_abnormal_list_pose_padded_path = 'state/pseudo_list/pseudo_abnormal_list_pose_padded.pth'
    pseudo_abnormal_list_pose_masks_path = 'state/pseudo_list/pseudo_abnormal_list_pose_masks.pth'
    velocity_padded_path = 'state/pseudo_list/velocity_padded.pth'
    velocity_masks_path = 'state/pseudo_list/velocity_masks.pth'

    pseudo_normal_list_padded = torch.stack(torch.load(pseudo_normal_list_padded_path)).to(device)
    pseudo_normal_list_masks = torch.stack(torch.load(pseudo_normal_list_masks_path)).to(device)
    pseudo_normal_list_pose_padded = torch.stack(torch.load(pseudo_normal_list_pose_padded_path)).to(device)
    pseudo_normal_list_pose_masks = torch.stack(torch.load(pseudo_normal_list_pose_masks_path)).to(device)
    pseudo_abnormal_list_padded = torch.stack(torch.load(pseudo_abnormal_list_padded_path)).to(device)
    pseudo_abnormal_list_masks = torch.stack(torch.load(pseudo_abnormal_list_masks_path)).to(device)
    pseudo_abnormal_list_pose_padded = torch.stack(torch.load(pseudo_abnormal_list_pose_padded_path)).to(device)
    pseudo_abnormal_list_pose_masks = torch.stack(torch.load(pseudo_abnormal_list_pose_masks_path)).to(device)
    velocity_padded = torch.load(velocity_padded_path).to(device)
    velocity_masks = torch.load(velocity_masks_path).to(device)

    num_normal = len(pseudo_normal_list_padded)
    num_abnormal = len(pseudo_abnormal_list_padded)
    label_normal = torch.zeros(num_normal).to(device)
    label_abnormal = torch.ones(num_abnormal).to(device)
    train_label = torch.cat([label_normal, label_abnormal], dim=0).to(device)
    train_label = train_label.view(-1,1)
    label_normal = label_normal.view(-1,1)
    label_abnormal = label_abnormal.view(-1,1)

    test_label = np.load('frame_labels/frame_labels_shanghai.npy')
    test_label = torch.tensor(test_label, dtype=torch.float32).to(device)
    test_label = test_label.view(-1, 1)

    test_deep_feature_memorized_padded = torch.stack(torch.load('state/pseudo_list/test_deep_feature_memorized_padded.pth')).to(device)
    test_deep_feature_memorized_masks = torch.stack(torch.load('state/pseudo_list/test_deep_feature_memorized_masks.pth')).to(device)
    test_pose_memorized_padded = torch.stack(torch.load('state/pseudo_list/test_pose_memorized_padded.pth')).to(device)
    test_pose_memorized_masks = torch.stack(torch.load('state/pseudo_list/test_pose_memorized_masks.pth')).to(device)
    test_velocity_padded = torch.load('state/pseudo_list/test_velocity_padded.pth').to(device)
    test_velocity_masks = torch.load('state/pseudo_list/test_velocity_masks.pth').to(device)

    attn = CrossModalAttention(34).to(device)
    cls = AnomalyScoreClassifier(4080).to(device)
    optimizer = optim.Adam(list(attn.parameters()) + list(cls.parameters()), lr=1e-4)

    # attn.load_state_dict(torch.load('state/fusion/attn_state_dict_final.pth'))
    # cls.load_state_dict(torch.load('state/fusion/cls_state_dict_final.pth'))

    num_params_optimized = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print(f"Total number of parameters being optimized: {num_params_optimized}")


    num_epochs = 2000
    criterion = nn.BCELoss()
    max_auc_result = 0

    for epoch in range(num_epochs):
        attn.train()
        cls.train()

        normal_fused_features = attn(pseudo_normal_list_padded.to(device),
                              pseudo_normal_list_pose_padded.to(device),
                              velocity_padded.to(device),
                              pseudo_normal_list_masks.to(device),
                              pseudo_normal_list_pose_masks.to(device),
                              velocity_masks.to(device)
                              )
        abnormal_fused_features = attn(pseudo_abnormal_list_padded.to(device),
                              pseudo_abnormal_list_pose_padded.to(device),
                              velocity_padded.to(device),
                              pseudo_abnormal_list_masks.to(device),
                              pseudo_abnormal_list_pose_masks.to(device),
                              velocity_masks.to(device)
                              )
        half_app_abnormal_fused_features = attn(pseudo_abnormal_list_padded.to(device),
                                            pseudo_normal_list_pose_padded.to(device),
                                            velocity_padded.to(device),
                                            pseudo_abnormal_list_masks.to(device),
                                            pseudo_normal_list_pose_masks.to(device),
                                            velocity_masks.to(device)
                                            )
        half_pose_abnormal_fused_features = attn(pseudo_normal_list_padded.to(device),
                                            pseudo_abnormal_list_pose_padded.to(device),
                                            velocity_padded.to(device),
                                            pseudo_normal_list_masks.to(device),
                                            pseudo_abnormal_list_pose_masks.to(device),
                                            velocity_masks.to(device)
                                            )


        normal_outputs = cls(normal_fused_features)
        abnormal_outputs = cls(abnormal_fused_features)
        half_app_outputs = cls(half_app_abnormal_fused_features)
        half_pose_outputs = cls(half_pose_abnormal_fused_features)

        loss1 = criterion(normal_outputs, label_normal)
        loss2 = criterion(abnormal_outputs, label_abnormal)
        loss3 = criterion(half_app_outputs, label_abnormal)
        loss4 = criterion(half_pose_outputs, label_abnormal)

        # """.................."""
        # test_fused_features = attn(test_deep_feature_memorized_padded,
        #                            test_pose_memorized_padded,
        #                            test_velocity_padded,
        #                            test_deep_feature_memorized_masks,
        #                            test_pose_memorized_masks,
        #                            test_velocity_masks)
        # test_outputs = cls(test_fused_features)
        # loss5 = criterion(test_outputs,test_label)
        # """.................."""

        loss = (loss1 + loss3 + loss4 + loss2)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, Loss1: {loss1.item():.6f}, Loss2: {loss2.item():.6f}, Loss3: {loss3.item():.6f}, Loss4: {loss4.item():.6f}', end='\t\t')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, Loss1: {loss1.item():.6f}, Loss3: {loss3.item():.6f}, Loss4: {loss4.item():.6f}', end='\t\t')

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        attn.eval()
        cls.eval()
        with torch.no_grad():
            normal_outputs = normal_outputs.cpu().detach().numpy().flatten() if isinstance(normal_outputs,torch.Tensor) else np.array(normal_outputs).flatten()
            abnormal_outputs = abnormal_outputs.cpu().detach().numpy().flatten() if isinstance(abnormal_outputs,torch.Tensor) else np.array(abnormal_outputs).flatten()
            label_norm = label_normal.cpu().numpy().flatten() if isinstance(label_normal,torch.Tensor) else np.array(label_normal).flatten()
            label_abnorm = label_abnormal.cpu().numpy().flatten() if isinstance(label_abnormal,torch.Tensor) else np.array(label_abnormal).flatten()

            label_norm = label_norm.astype(int)
            label_abnorm = label_abnorm.astype(int)

            all_outputs = np.concatenate((normal_outputs, abnormal_outputs), axis=0)
            all_labels = np.concatenate((label_norm, label_abnorm), axis=0)

            # all_outputs = smooth_predictions(all_outputs)
            train_auc = roc_auc_score(all_labels, all_outputs)
            print(f"train auc score:{train_auc:.6f}",end='\t')

            test_fused_features = attn(test_deep_feature_memorized_padded,
                                       test_pose_memorized_padded,
                                       test_velocity_padded,
                                       test_deep_feature_memorized_masks,
                                       test_pose_memorized_masks,
                                       test_velocity_masks)
            test_outputs = cls(test_fused_features)


            test_outputs_np = test_outputs.cpu().detach().numpy().flatten() if isinstance(test_outputs,
                                                                                          torch.Tensor) else np.array(
                test_outputs).flatten()



            smoothed_test_outputs = smooth_predictions(test_outputs_np)

            test_label_np = test_label.cpu().numpy().flatten() if isinstance(test_label, torch.Tensor) else np.array(
                test_label).flatten()

            test_auc_score = roc_auc_score(test_label_np, test_outputs_np)
            print(f"test AUC Score: {test_auc_score:6f}",end='\t')
            smoothed_auc_score = roc_auc_score(test_label_np, smoothed_test_outputs)
            print(f"Smoothed AUC Score: {smoothed_auc_score:6f}")

            # ROC
            # fpr, tpr, _ = roc_curve(test_label_np, smoothed_test_outputs)
            # roc_auc = auc(fpr, tpr)
            #
            # plt.figure()
            # lw = 2
            # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic Curve')
            # plt.legend(loc="lower right")
            # plt.show()

            # EER
            # fpr, tpr, thresholds = roc_curve(test_label_np, test_outputs_np)
            # fnr = 1 - tpr

            # eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
            # EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            # print(f"EER: {EER:6f}")
            #
            # smoothed_fpr, smoothed_tpr, smoothed_thresholds = roc_curve(test_label_np, smoothed_test_outputs)
            # smoothed_fnr = 1 - smoothed_tpr
            # smoothed_eer_threshold = smoothed_thresholds[np.nanargmin(np.absolute((smoothed_fnr - smoothed_fpr)))]
            # smoothed_EER = smoothed_fpr[np.nanargmin(np.absolute((smoothed_fnr - smoothed_fpr)))]
            # print(f"Smoothed EER: {smoothed_EER:6f}")

            if max_auc_result < smoothed_auc_score:
                max_auc_result = smoothed_auc_score
                torch.save(attn.state_dict(), 'state/fusion/attn_state_dict_final.pth')
                torch.save(cls.state_dict(), 'state/fusion/cls_state_dict_final.pth')
                with open('state/fusion/predictions.txt', 'w') as f:
                    for pred in smoothed_test_outputs:
                        f.write(f"{pred}\n")
                with open('state/fusion/predictions_un_smoothed.txt', 'w') as f:
                    for pred in test_outputs_np:
                        f.write(f"{pred}\n")



