import os
import numpy as np
import torch
import torch.nn as nn
from preprocess import AugmentedPennActionDataset
from model.model import STGCN
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    annotation_dir = "./augmentation/augmented_penn/labels"
    
    # load dataset
    dataset = AugmentedPennActionDataset(annotation_dir)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"dataset file labels distribution: {dataset.file_labels.count(0)} squats, {dataset.file_labels.count(1)} pushups, {dataset.file_labels.count(2)} situps, {dataset.file_labels.count(3)} bench_press")

    # split dataset by file to avoid data leakage between train and test sets
    train_file_idx, temp_file_idx = train_test_split(
        range(len(dataset.files)),
        test_size=0.3,
        stratify=dataset.file_labels,
        random_state=42
    )

    val_file_idx, test_file_idx = train_test_split(
        temp_file_idx,
        test_size=0.5,
        stratify=[dataset.file_labels[i] for i in temp_file_idx],
        random_state=42
    )

    train_files = [dataset.files[i] for i in train_file_idx]
    val_files = [dataset.files[i] for i in val_file_idx]
    test_files = [dataset.files[i] for i in test_file_idx]

    train_indices = []
    val_indices = []
    test_indices = []
    for i, (file_name, start) in enumerate(dataset.samples):
        if file_name in train_files:
            train_indices.append(i)
        elif file_name in val_files:
            val_indices.append(i)
        elif file_name in test_files:
            test_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Dataset size: {len(dataset)}")

    num_classes = len(dataset.allowed_actions)
    num_joints = 13

    model = STGCN(num_class=num_classes, num_point=num_joints)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    action_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=3,
        #verbose=True
    )

    num_epochs = 150
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    min_delta = 1e-4 # minimum improvement to count
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_action = 0
        total_action = 0

        for batch_x, batch_phase, batch_action in train_loader:
            batch_x = batch_x.to(device)
            batch_action = batch_action.to(device)

            optimizer.zero_grad()
            action_logits = model(batch_x)

            loss = action_loss(action_logits, batch_action)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

            _, predicted_action = torch.max(action_logits, 1)
            correct_action += (predicted_action == batch_action).sum().item()
            total_action += batch_action.size(0)

        train_loss = running_loss / total_action
        train_acc = correct_action / total_action

        

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss_total = 0.0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_phase, batch_action in val_loader:
                batch_x = batch_x.to(device)
                batch_action = batch_action.to(device)

                action_logits = model(batch_x)

                loss = action_loss(action_logits, batch_action)

                val_loss_total += loss.item() * batch_x.size(0)
                total_val += batch_x.size(0)

        val_loss = val_loss_total / total_val

        # Adaptive LR step
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # ----------------- TESTING ----------------
    model.eval()
    correct_action = 0
    total_action = 0

    with torch.no_grad():
        for batch_x, batch_phase, batch_action in test_loader:
            batch_x = batch_x.to(device)
            batch_action = batch_action.to(device)

            action_logits = model(batch_x)

            # Action accuracy
            _, predicted_action = torch.max(action_logits, 1)
            correct_action += (predicted_action == batch_action).sum().item()
            total_action += batch_action.size(0)

    # Final metrics
    test_acc = correct_action / total_action

    print(f"Test Accuracy: {test_acc:.4f}")
        

        



               