import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class AugmentedPennActionDataset(Dataset):

    def __init__(self, annotation_dir, window_size=32, stride=8):

        self.annotation_dir = annotation_dir
        self.window_size = window_size
        self.stride = stride

        self.allowed_actions = ["squat", "pushup"]

        self.files = []
        self.file_labels = []
        self.samples = []
        self.sample_labels = []

        for f in os.listdir(annotation_dir):

            if not f.endswith(".mat"):
                continue

            path = os.path.join(annotation_dir, f)
            mat = scipy.io.loadmat(path)

            action = mat["action"][0]

            if action not in self.allowed_actions:
                continue

            self.files.append(f)
            self.file_labels.append(self.allowed_actions.index(action))

            T = mat["x"].shape[0]

            for start in range(0, T - window_size + 1, stride):
                self.samples.append((f, start))
                self.sample_labels.append(self.allowed_actions.index(action))



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        file_name, start = self.samples[idx]

        path = os.path.join(self.annotation_dir, file_name)

        x, y, visibility, action, phase = load_mat_file(path)

        skeleton = stack_joints(x, y, visibility)
        skeleton = root_center(skeleton)
        skeleton = scale_normalize(skeleton)

        end = start + self.window_size

        skeleton_window = skeleton[start:end]
        phase_window = phase[start:end]

        tensor = to_tensor(skeleton_window)

        phase = phase_window.squeeze(-1) * 2 * np.pi  # convert to radians
        phase_sin = np.sin(phase)
        phase_cos = np.cos(phase)

        phase_tensor = torch.tensor(
            np.stack([phase_sin, phase_cos], axis=1),
            dtype=torch.float32
        )

        label = torch.tensor(
            self.allowed_actions.index(action),
            dtype=torch.long
        )

        return tensor, phase_tensor, label
    
# load .mat file and extract x, y, visibility, and action
def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    x = mat["x"] # shape: (num_frames, num_joints)
    y = mat["y"] # shape: (num_frames, num_joints)
    visibility = mat["visibility"] # shape: (num_frames, num_joints)
    phase = mat["phase"] # shape: (num_frames, 1)
    action = mat["action"][0]

    return x, y, visibility, action, phase

# stack x, y, and visibility to create a skeleton representation
def stack_joints(x, y, visibility):
    skeleton = np.stack((x, y, visibility), axis=2)
    return skeleton

# Root-Centered Normalization Pick pelvis/hip joint as root.
def root_center(skeleton, left_hip_index=7, right_hip_index=8):
    hip_midpoint = (skeleton[:, left_hip_index:left_hip_index+1, :2] + skeleton[:, right_hip_index:right_hip_index+1, :2]) / 2.0 # only x and y channels
    skeleton[:, :, :2] -= hip_midpoint
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
    max_val = np.max(np.abs(skeleton[:, :, :2]))  # only consider x and y channels for scaling
    if max_val < 1e-6:
        print("Warning: max_val is very small, skipping normalization to avoid division by zero.")
        return skeleton
    skeleton[:, :, :2] /= (max_val + 1e-6)
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
    
