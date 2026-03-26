# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:23:38 2026

@author: RVT20
"""
import numpy as np
import scipy.io
import os
import shutil
import glob
from pathlib import Path
import random
from PIL import Image

PER_FRAME_KEYS = {"x", "y", "visibility", "bbox"}

def load_clean_mat(path):
    m = scipy.io.loadmat(path)
    return {k: v for k, v in m.items() if not k.startswith("__")}

def save_mat(path, data):
    scipy.io.savemat(path, data)

def list_frames(frames_dir):
    frames = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return frames

def ensure_float(label_data):
    for k in ["x", "y", "bbox"]:
        if k in label_data and isinstance(label_data[k], np.ndarray):
            label_data[k] = label_data[k].astype(np.float64, copy=False)
    return label_data

def flip_horizontal(src_frames, dst_frames, label_data):
    os.makedirs(dst_frames, exist_ok=False)
    frames = list_frames(src_frames)
    W = None
    for fname in frames:
        I = Image.open(os.path.join(src_frames, fname))
        if W is None:
            W, H = I.size
        I = I.transpose(Image.FLIP_LEFT_RIGHT)
        I.save(os.path.join(dst_frames, fname))
    x = label_data["x"].copy()
    bbox = label_data["bbox"].copy()
    x = (W-1)-x
    
    x1 = bbox[:, 0].copy()
    x2 = bbox[:, 2].copy()
    bbox[:, 0] = (W - 1) - x2
    bbox[:, 2] = (W - 1) - x1

    label_data["x"] = x
    label_data["bbox"] = bbox
    return label_data

def translate_data(src_frames, dst_frames, label_data, dx, dy):
    os.makedirs(dst_frames, exist_ok=False)
    frames = list_frames(src_frames)
    H = W = None
    for fname in frames:
        I = Image.open(os.path.join(src_frames, fname))
        if W is None:
            W, H = I.size
        new_I = Image.new(I.mode, (W, H))
        new_I.paste(I, (dx, dy))
        new_I.save(os.path.join(dst_frames, fname))
    label_data["x"] = label_data["x"] + dx
    label_data["y"] = label_data["y"] + dy
    bbox = label_data["bbox"].copy()
    bbox[:, [0, 2]] += dx
    bbox[:, [1, 3]] += dy
    label_data["bbox"] = bbox
    return label_data

def scale_data(src_frames, dst_frames, label_data, scale):
    os.makedirs(dst_frames, exist_ok=False)
    frames = list_frames(src_frames)
    H = W = None
    for fname in frames:
        I = Image.open(os.path.join(src_frames, fname))
        if W is None:
            W, H = I.size
        newW, newH = int(W*scale), int(H*scale)
        I2 = I.resize((newW, newH), resample=Image.BILINEAR)
        new_I = Image.new(I.mode, (W, H))
        x0 = (W - newW)//2
        y0 = (H - newH)//2
        new_I.paste(I2, (x0, y0))
        new_I.save(os.path.join(dst_frames, fname))
    cx, cy = (W-1) / 2.0, (H-1) / 2.0
    x = label_data["x"].copy()
    y = label_data["y"].copy()
    x = (x - cx) * scale + cx
    y = (y - cy) * scale + cy
    bbox = label_data["bbox"].copy()
    for i in [0, 2]:
        bbox[:, i] = (bbox[:, i] - cx) * scale + cx
    for i in [1, 3]:
        bbox[:, i] = (bbox[:, i] - cy) * scale + cy
    label_data["x"] = x
    label_data["y"] = y
    label_data["bbox"] = bbox
    return label_data

def make_aug_variant(base_dir, old_id, suffix, aug_kind, params):
    labels_dir = os.path.join(base_dir, "labels")
    frames_dir = os.path.join(base_dir, "frames")

    src_mat = os.path.join(labels_dir, f"{old_id}.mat")
    src_frames = os.path.join(frames_dir, old_id)

    new_id = f"{old_id}_{suffix}"
    dst_mat = os.path.join(labels_dir, f"{new_id}.mat")
    dst_frames = os.path.join(frames_dir, new_id)

    if os.path.exists(dst_mat) or os.path.exists(dst_frames):
        return new_id

    label_data = load_clean_mat(src_mat)
    label_data = ensure_float(label_data)

    if aug_kind == "flip":
        data = flip_horizontal(src_frames, dst_frames, label_data)

    elif aug_kind == "translate":
        dx, dy = params["dx"], params["dy"]
        data = translate_data(src_frames, dst_frames, label_data, dx, dy)

    elif aug_kind == "scale":
        scale = params["scale"]
        data = scale_data(src_frames, dst_frames, label_data, scale)

    else:
        raise ValueError("Unknown aug_kind")
        
    if "x" in data:
        T = int(data["x"].shape[0])
        data["nframes"] = np.array([[T]], dtype=np.int32)

    save_mat(dst_mat, data)
    return new_id    

def augment_random_subset(base_dir="augmented_penn", fraction=0.4, seed=13):
    labels_dir = os.path.join(base_dir, "labels")
    all_ids = [
        os.path.splitext(f)[0]
        for f in os.listdir(labels_dir)
        if f.endswith(".mat") and "_flip" not in f and "_tr" not in f and "_sc" not in f
    ]

    random.seed(seed)
    k = max(1, int(len(all_ids) * fraction))
    chosen = random.sample(all_ids, k)

    for old_id in chosen:
        kind = random.choice(["flip", "translate", "scale"])

        if kind == "flip":
            make_aug_variant(base_dir, old_id, "flip", "flip", {})

        elif kind == "translate":
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            make_aug_variant(base_dir, old_id, f"tr{dx}_{dy}", "translate", {"dx": dx, "dy": dy})

        else:  # scale
            scale = random.uniform(0.9, 1.1)
            make_aug_variant(base_dir, old_id, f"sc{scale:.2f}", "scale", {"scale": scale})

    print("Augmentation done")

augment_random_subset()





