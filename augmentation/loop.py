# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:24:49 2026

@author: RVT20
"""
import numpy as np
import scipy.io
import os
import shutil
import glob
from pathlib import Path
import random

augmentation_path = os.path.join('augmented_penn','labels')

for file in os.scandir(augmentation_path):
    if Path(file).suffix == '.mat':
        mat = scipy.io.loadmat(file)
        mat['rep_count'] = np.array([1])
        scipy.io.savemat(file.path, mat)
        
mat = scipy.io.loadmat("augmented_penn/labels/1700.mat")
for k, v in mat.items():
    if k.startswith("__"):
        continue
    if isinstance(v, np.ndarray):
        print(k, v.shape, v.dtype)
    else:
        print(k, type(v))
        
def loop_mat(src_path, dst_path, reps):
    mat_file = scipy.io.loadmat(src_path)
    data = {k: v for k, v in mat_file.items() if not k.startswith("__")}
    T = int(data['x'].shape[0])
    PER_FRAME_DATA = {"x", "y", "visibility", "bbox"}
    for k in PER_FRAME_DATA:
        if k in data:
            data[k] = np.concatenate([data[k]]*reps, axis=0)
    
    data["nframes"] = np.array([[T * reps]], dtype=np.int32)
    data["rep_count"] = np.array([[T * reps]], dtype=np.int32)
    scipy.io.savemat(dst_path, data)
    
def loop_frames(src_path, dst_path, reps):
    os.makedirs(dst_path, exist_ok=False)

    frames = sorted([
        f for f in os.listdir(src_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    T = len(frames)

    for r in range(reps):
        for i, fname in enumerate(frames):
            src = os.path.join(src_path, fname)

            
            new_idx = r * T + i + 1
            ext = os.path.splitext(fname)[1].lower()
            dst_name = f"{new_idx:06d}{ext}"
            dst = os.path.join(dst_path, dst_name)

            shutil.copy2(src, dst)

def make_random_loops(
    base_dir="augmented_penn",
    max_rep=6,
    min_rep=2,
    fraction=0.4,
    seed=13,
    keep_original=True
):
    labels_dir = os.path.join(base_dir, "labels")

    all_ids = []
    for fname in os.listdir(labels_dir):
        if fname.endswith(".mat") and "_loop" not in fname:
            all_ids.append(os.path.splitext(fname)[0])

    if seed is not None:
        random.seed(seed)

    k = max(1, int(len(all_ids) * fraction))
    chosen_ids = random.sample(all_ids, k)

    for old_id in chosen_ids:
        rep = random.randint(min_rep, max_rep)

        src_mat = os.path.join(base_dir, "labels", f"{old_id}.mat")
        src_frames = os.path.join(base_dir, "frames", old_id)

        new_id = f"{old_id}_loop{rep}"
        dst_mat = os.path.join(base_dir, "labels", f"{new_id}.mat")
        dst_frames = os.path.join(base_dir, "frames", new_id)

        if os.path.exists(dst_mat) or os.path.exists(dst_frames):
            continue

        if not os.path.isdir(src_frames):
            print(f"Skip {old_id}: no directory exists")
            continue

        loop_frames(src_frames, dst_frames, reps=rep)
        loop_mat(src_mat, dst_mat, reps=rep)

        print(f"Made {new_id}")

make_random_loops()



