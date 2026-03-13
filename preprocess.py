import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# 1659-1889 files -> squats
# 1348-1558 files -> pushups
# class PennActionDataset(Dataset):
#     def __init__(self, annotation_dir, seq_len=128):
#         # get files in range of squats and pushups
#         self.files = [f for f in os.listdir(annotation_dir) if f.endswith('.mat')]
#         self.annotation_dir = annotation_dir
#         self.seq_len = seq_len

#         self.action_to_label = {
#             'squat': 0,
#             'pushup': 1,
#             'bench_press': 2,
#             'pullup': 3,
#             'jumping_jacks': 4,
#             'situp': 5,
#             'tennis_serve': 6,
#             'bowl': 7,
#             'jump_rope': 8,
#             'baseball_pitch': 9,
#             'clean_and_jerk': 10,
#             'strum_guitar': 11,
#             'baseball_swing': 12,
#             'golf_swing': 13,
#             'tennis_forehand': 14
#         }

#         # Precompute labels for stratification
#         self.labels = []
#         for f in self.files:
#             path = os.path.join(self.annotation_dir, f)
#             _, _, _, label = load_mat_file(path)
#             self.labels.append(self.action_to_label[label])

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = os.path.join(self.annotation_dir, self.files[idx])
#         x, y, visibility, label = load_mat_file(path)

#         # create skeleton representation
#         skeleton = stack_joints(x, y)
#         skeleton = root_center(skeleton)
#         skeleton = scale_normalize(skeleton)

#         # sample or pad frames to fixed length
#         skeleton, rep_count = sample_frames(skeleton, self.seq_len)

#         tensor = to_tensor(skeleton)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)

#         return tensor, label, rep_count

class AugmentedPennActionDataset(Dataset):
    def __init__(self, annotation_dir, seq_len=240):
        # get files in range of squats and pushups
        self.files = [f for f in os.listdir(annotation_dir) if f.endswith('.mat')]
        self.annotation_dir = annotation_dir
        self.seq_len = seq_len

        self.action_to_label = {
            'squat': 0,
            'pushup': 1,
            'bench_press': 2,
            'pullup': 3,
            'jumping_jacks': 4,
            'situp': 5,
            'tennis_serve': 6,
            'bowl': 7,
            'jump_rope': 8,
            'baseball_pitch': 9,
            'clean_and_jerk': 10,
            'strum_guitar': 11,
            'baseball_swing': 12,
            'golf_swing': 13,
            'tennis_forehand': 14
        }

        # Precompute labels for stratification
        self.labels = []
        for f in self.files:
            path = os.path.join(self.annotation_dir, f)
            _, _, _, label, _, _ = load_mat_file(path)
            self.labels.append(self.action_to_label[label])
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.annotation_dir, self.files[idx])
        x, y, visibility, label, rep_count, nframes = load_mat_file(path)

        # create skeleton representation
        skeleton = stack_joints(x, y)
        skeleton = root_center(skeleton)
        skeleton = scale_normalize(skeleton)

        # sample or pad frames to fixed length
        skeleton = sample_frames(skeleton, self.seq_len)

        tensor = to_tensor(skeleton)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        rep_count = torch.tensor(rep_count, dtype=torch.float32)

        return tensor, label, rep_count
    
# load .mat file and extract x, y, visibility, and action
def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    x = mat["x"] # shape: (num_frames, num_joints)
    y = mat["y"] # shape: (num_frames, num_joints)
    visibility = mat["visibility"] # shape: (num_frames, num_joints)
    action = mat["action"][0]
    rep_count = mat["rep_count"][0][0]
    nframes = mat["nframes"][0][0]

    return x, y, visibility, action, rep_count, nframes

# stack x and y to create a skeleton representation
def stack_joints(x, y):
    skeleton = np.stack((x, y), axis=2)
    return skeleton

# stack x, y, and visibility to create a skeleton representation with visibility
def stack_joints_with_visibility(x, y, visibility):
    skeleton = np.stack((x, y, visibility), axis=2)
    return skeleton

# Root-Centered Normalization Pick pelvis/hip joint as root.
def root_center(skeleton, root_index=7):
    root = skeleton[:, root_index:root_index+1, :] # shape: (num_frames, 2)
    skeleton -= root
    return skeleton

# BBox Normalization Use bbox for scale invariance
def bbox_normalize(skeleton, bbox):
    x_min = bbox[:, 0:1]
    y_min = bbox[:, 1:2]
    w = bbox[:, 2:3]
    h = bbox[:, 3:4]

    skeleton[:, :, 0] = (skeleton[:, :, 0] - x_min) / w
    skeleton[:, :, 1] = (skeleton[:, :, 1] - y_min) / h
    return skeleton

# Root-Centered + scale by body size
def scale_normalize(skeleton):
    max_val = np.max(np.abs(skeleton))
    skeleton = skeleton / (max_val + 1e-6)
    return skeleton

# Sample or pad frames to a fixed length
def sample_frames(skeleton, target_len=240):
    T = skeleton.shape[0]

    # If there are more frames than target_len, sample uniformly. If fewer, pad with 0.
    if T >= target_len:
        indices = np.linspace(0, T-1, target_len).astype(int)
        skeleton = skeleton[indices]
    else:
        pad_len = target_len - T
        padding = np.zeros((pad_len, skeleton.shape[1], skeleton.shape[2]))
        skeleton = np.concatenate((skeleton, padding), axis=0)

    return skeleton

# convert to tensor format (C, T, V, M) where C is the number of channels (x, y), 
# T is the number of frames, V is the number of joints, and M is the number of people (always 1 for penn action dataset)
def to_tensor(skeleton):
    tensor = torch.tensor(skeleton, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)   # (C, T, V)
    tensor = tensor.unsqueeze(-1)     # (C, T, V, 1)
    return tensor

# if __name__ == "__main__":
#     dataset = AugmentedPennActionDataset(annotation_dir="augmentation/augmented_penn/labels")
#     print(f"Dataset size: {len(dataset)}")
#     tensor, label, rep_count, nframes = dataset[0]
#     print(f"Tensor shape: {tensor.shape}, Label: {label}, Repetition count: {rep_count}")

#     # print top 5 highest counts in the dataset and min max frames
#     rep_counts = []
#     nframes_list = []
#     for i in range(len(dataset)):
#         _, label, rep_count, nframes = dataset[i]
#         rep_counts.append(rep_count.item())
#         nframes_list.append(int(nframes))

#     print(f"Min frames: {min(nframes_list)}, Max frames: {max(nframes_list)}")
#     print(f"Unique repetition counts in dataset: {set(rep_counts)}")
#     # Print top 5 highest, mean, median nframes
#     top_frames = sorted(nframes_list, reverse=True)[:5]
#     print(f"Top 5 highest frame counts: {top_frames}")
#     print(f"Mean frames: {np.mean(nframes_list)}, Median frames: {np.median(nframes_list)}")
#     # count how many nframes are above 300
#     above_300 = sum(1 for n in nframes_list if n > 240)
#     print(f"Number of samples with more than 240 frames: {above_300}")
    
