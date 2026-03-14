import os
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
    print(f"dataset file labels distribution: {dataset.file_labels.count(0)} squats, {dataset.file_labels.count(1)} pushups")

    # split dataset by file to avoid data leakage between train and test sets
    train_file_idx, test_file_idx = train_test_split(
        range(len(dataset.files)),
        test_size=0.2,
        stratify=dataset.file_labels,
        random_state=42
    )

    train_files = [dataset.files[i] for i in train_file_idx]
    test_files = [dataset.files[i] for i in test_file_idx]

    train_indices = []
    test_indices = []
    for i, (file_name, start) in enumerate(dataset.samples):
        if file_name in train_files:
            train_indices.append(i)
        elif file_name in test_files:
            test_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Dataset size: {len(dataset)}")

    num_classes = len(dataset.allowed_actions)
    num_joints = 13

    model = STGCN(num_class=num_classes, num_point=num_joints)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    action_loss = nn.CrossEntropyLoss()
    phase_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    num_epochs = 150
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_action = 0
        total_action = 0
        total_count_error = 0
        total_count_samples = 0

        for batch_x, batch_phase, batch_action in train_loader:
            batch_x = batch_x.to(device)  # (N, C, T, V, M)
            batch_action = batch_action.to(device)
            batch_phase = batch_phase.to(device)
            batch_phase = batch_phase[:, ::4]  # downsample phase labels to match model output (every 4 frames)

            optimizer.zero_grad()
            action_logits, phase = model(batch_x)
            loss_action = action_loss(action_logits, batch_action)
            loss_phase = phase_loss(phase, batch_phase.float())
            loss = loss_action + loss_phase  # combine losses (weighthed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

            # action accuracy
            _, predicted_action = torch.max(action_logits, 1)
            correct_action += (predicted_action == batch_action).sum().item()
            total_action += batch_action.size(0)

            # phase prediction mae
            total_count_error += torch.abs(phase.squeeze() - batch_phase.float()).sum().item()
            total_count_samples += batch_phase.size(0)
        
        scheduler.step()

        epoch_loss = running_loss / total_action
        acc = correct_action / total_action if total_action > 0 else 0
        mae = total_count_error / total_count_samples if total_count_samples > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f} | MAE: {mae:.4f}")

    # Evaluation on test set
    model.eval()

    total_count_error = 0
    total_count_samples = 0

    with torch.no_grad():
        correct_action = 0
        total_action = 0

        for batch_x, batch_phase, batch_action in test_loader:
            batch_x = batch_x.to(device)
            batch_action = batch_action.long().to(device)
            batch_phase = batch_phase.to(device)
            batch_phase = batch_phase[:, ::4]  # downsample phase labels to match model output

            action_logits, phase = model(batch_x)
            _, predicted_action = torch.max(action_logits, 1)

            # Action classification metrics
            correct_action += (predicted_action == batch_action).sum().item()
            total_action += batch_action.size(0)

            # Phase prediction metrics (MAE example)
            total_count_error += torch.abs(phase.squeeze() - batch_phase.float()).sum().item()
            total_count_samples += batch_phase.size(0)

        accuracy = correct_action / total_action
        mae = total_count_error / total_count_samples
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Average Rep Count Error: {mae:.4f}")
    
    # save model
    torch.save(model.state_dict(), "stgcn_model.pth")
    print("Model saved.")



               