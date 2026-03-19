import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class AugmentedPennActionDataset(Dataset):

    def __init__(self, annotation_dir, window_size=16, stride=1):

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
            self.file_labels.append(0 if action == "squat" else 1)

            T = mat["x"].shape[0]

            for start in range(0, T - window_size + 1, stride):
                self.samples.append((f, start))
                self.sample_labels.append(0 if action == "squat" else 1)



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        file_name, start = self.samples[idx]

        path = os.path.join(self.annotation_dir, file_name)

        x, y, visibility, action, phase, nframes = load_mat_file(path)

        skeleton = stack_joints(x, y)
        skeleton = root_center(skeleton)
        skeleton = scale_normalize(skeleton)

        end = start + self.window_size

        skeleton_window = skeleton[start:end]
        phase_window = phase[start:end]

        tensor = to_tensor(skeleton_window)

        phase_tensor = torch.tensor(
            phase_window.squeeze(-1),
            dtype=torch.float32
        )

        label = torch.tensor(
            0 if action == "squat" else 1,
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
    nframes = mat["nframes"][0][0]

    return x, y, visibility, action, phase, nframes

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
    if max_val < 1e-6:
        print("Warning: max_val is very small, skipping normalization to avoid division by zero.")
        return skeleton
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
    
