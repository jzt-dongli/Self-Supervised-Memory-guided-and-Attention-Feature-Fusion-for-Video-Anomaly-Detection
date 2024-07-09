import torch
import numpy as np

def pad_and_create_mask(data_list, target_shape=(30, 34)):    #ped2
# def pad_and_create_mask(data_list, target_shape=(20, 34)):      #avenue
# def pad_and_create_mask(data_list, target_shape=(42, 34)):      #shanghaitech
    padded_tensors = []
    masks = []
    for tensor in data_list:
        pad_bottom = target_shape[0] - tensor.shape[0]

        padded_tensor = torch.zeros(*target_shape)

        padded_tensor[:tensor.shape[0], :] = tensor

        padded_tensors.append(padded_tensor)

        mask = torch.zeros(*target_shape)
        if pad_bottom > 0:
            mask[-pad_bottom:, :] = -1e9
        masks.append(mask)

    return padded_tensors, masks

pseudo_abnormal_list = torch.load('state/pseudo_list/pseudo_abnormal_list.pth')
pseudo_abnormal_list_pose = torch.load('state/pseudo_list/pseudo_abnormal_list_pose.pth')
pseudo_normal_list = torch.load('state/pseudo_list/pseudo_normal_list.pth')
pseudo_normal_list_pose = torch.load('state/pseudo_list/pseudo_normal_list_pose.pth')

test_deep_feature_memorized = torch.load('state/pseudo_list/test_deep_feature_memorized.pth')
test_pose_memorized = torch.load('state/pseudo_list/test_pose_memorized.pth')

# ped2
missing_indices = [1291, 1362, 1443, 1523]
# avenue
# missing_indices = avenue_missing_indices  #Data is stored in the same level folder
# shanghaitech
# missing_indices = shanghaitech_missing_indices  #Data is stored in the same level folder


insert_feature = torch.zeros(1, 34)
insert_feature = insert_feature.cuda()

def insert_missing_features(data_list, missing_indices, insert_feature):
    for index in missing_indices:
        data_list.insert(index, insert_feature)
    return data_list

pseudo_normal_list_pose = insert_missing_features(pseudo_normal_list_pose, missing_indices, insert_feature)
pseudo_abnormal_list_pose = insert_missing_features(pseudo_abnormal_list_pose, missing_indices, insert_feature)

test_pose_memorized = insert_missing_features(test_pose_memorized, missing_indices_test, insert_feature)

pseudo_abnormal_list_padded, pseudo_abnormal_list_masks = pad_and_create_mask(pseudo_abnormal_list)
pseudo_abnormal_list_pose_padded, pseudo_abnormal_list_pose_masks = pad_and_create_mask(pseudo_abnormal_list_pose)
pseudo_normal_list_padded, pseudo_normal_list_masks = pad_and_create_mask(pseudo_normal_list)
pseudo_normal_list_pose_padded, pseudo_normal_list_pose_masks = pad_and_create_mask(pseudo_normal_list_pose)

test_deep_feature_memorized_padded, test_deep_feature_memorized_masks = pad_and_create_mask(test_deep_feature_memorized)
test_pose_memorized_padded, test_pose_memorized_masks = pad_and_create_mask(test_pose_memorized)

torch.save(pseudo_abnormal_list_padded, 'state/pseudo_list/pseudo_abnormal_list_padded.pth')
torch.save(pseudo_abnormal_list_pose_padded, 'state/pseudo_list/pseudo_abnormal_list_pose_padded.pth')
torch.save(pseudo_normal_list_padded, 'state/pseudo_list/pseudo_normal_list_padded.pth')
torch.save(pseudo_normal_list_pose_padded, 'state/pseudo_list/pseudo_normal_list_pose_padded.pth')

torch.save(pseudo_abnormal_list_masks, 'state/pseudo_list/pseudo_abnormal_list_masks.pth')
torch.save(pseudo_abnormal_list_pose_masks, 'state/pseudo_list/pseudo_abnormal_list_pose_masks.pth')
torch.save(pseudo_normal_list_masks, 'state/pseudo_list/pseudo_normal_list_masks.pth')
torch.save(pseudo_normal_list_pose_masks, 'state/pseudo_list/pseudo_normal_list_pose_masks.pth')

torch.save(test_deep_feature_memorized_padded, 'state/pseudo_list/test_deep_feature_memorized_padded.pth')
torch.save(test_deep_feature_memorized_masks, 'state/pseudo_list/test_deep_feature_memorized_masks.pth')
torch.save(test_pose_memorized_padded, 'state/pseudo_list/test_pose_memorized_padded.pth')
torch.save(test_pose_memorized_masks, 'state/pseudo_list/test_pose_memorized_masks.pth')

velocity_path = 'your_velocity.npy'
velocity = np.load(velocity_path, allow_pickle=True)

test_velocity_path = '/root/pycharm/project/【SOTA2022】Accurate-Interpretable-VAD-master/extracted_features/shanghaitech/test/velocity.npy'
test_velocity = np.load(test_velocity_path, allow_pickle=True)


target_shape_n, target_shape_m = (30, 34)   #ped2
# target_shape_n, target_shape_m = (20, 34)   #avenue
# target_shape_n, target_shape_m = (42, 34)   #shanghaitech

extended_velocity_list = []
mask_list = []

for vel in velocity:
    n, m = vel.shape
    mean_vel = np.mean(vel, axis=1, keepdims=True)
    vel_extended = np.repeat(mean_vel, 34, axis=1)
    mask = np.ones_like(vel_extended)

    if n < target_shape_n:
        padding_needed = target_shape_n - n
        vel_padded = np.pad(vel_extended, ((0, padding_needed), (0, 0)), 'constant', constant_values=0)
        mask_padded = np.pad(mask, ((0, padding_needed), (0, 0)), 'constant', constant_values=-1e9)
    else:
        vel_padded = vel_extended
        mask_padded = mask

    extended_velocity_list.append(vel_padded)
    mask_list.append(mask_padded)

extended_velocity_array = np.array(extended_velocity_list, dtype=np.float32)
mask_array = np.array(mask_list, dtype=np.float32)

extended_velocity_tensor = torch.from_numpy(extended_velocity_array)
mask_tensor = torch.from_numpy(mask_array)

torch.save(extended_velocity_tensor, 'state/pseudo_list/velocity_padded.pth')
torch.save(mask_tensor, 'state/pseudo_list/velocity_masks.pth')

extended_velocity_list = []
mask_list = []

for vel in test_velocity:
    n, m = vel.shape
    mean_vel = np.mean(vel, axis=1, keepdims=True)
    vel_extended = np.repeat(mean_vel, 34, axis=1)
    mask = np.ones_like(vel_extended)

    if n < target_shape_n:
        padding_needed = target_shape_n - n
        vel_padded = np.pad(vel_extended, ((0, padding_needed), (0, 0)), 'constant', constant_values=0)
        mask_padded = np.pad(mask, ((0, padding_needed), (0, 0)), 'constant', constant_values=-1e9)
    else:
        vel_padded = vel_extended
        mask_padded = mask

    extended_velocity_list.append(vel_padded)
    mask_list.append(mask_padded)

extended_velocity_array = np.array(extended_velocity_list, dtype=np.float32)
mask_array = np.array(mask_list, dtype=np.float32)

extended_velocity_tensor = torch.from_numpy(extended_velocity_array)
mask_tensor = torch.from_numpy(mask_array)

torch.save(extended_velocity_tensor, 'state/pseudo_list/test_velocity_padded.pth')
torch.save(mask_tensor, 'state/pseudo_list/test_velocity_masks.pth')

