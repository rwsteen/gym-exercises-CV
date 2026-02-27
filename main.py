import os
import torch
import torch.nn as nn
from preprocess import PennActionDataset
from model.model import STGCN
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    annotation_dir = "../Penn_Action/labels/"
    
    # load dataset
    dataset = PennActionDataset(annotation_dir)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Action label distribution: {dataset.labels.count(0)} squats, {dataset.labels.count(1)} pushups")
    print(f"Unique action labels: {set(dataset.labels)}")
    print(f"Example tensor shape: {dataset[0][0].shape}, Example label: {dataset[0][1]}, Example rep count: {dataset[0][2]}")
    print(f"Amount of samples per repetition count: {torch.bincount(torch.tensor([sample[2] for sample in dataset]))}")
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"Dataset size: {len(dataset)}")

    num_classes = len(dataset.action_to_label)
    num_joints = 13

    model = STGCN(num_class=num_classes, num_point=num_joints)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)

    action_loss = nn.CrossEntropyLoss()
    count_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )

    num_epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_action = 0
        total_action = 0
        total_count_error = 0
        total_count_samples = 0

        for batch_x, batch_action, batch_count in train_loader:
            batch_x = batch_x.to(device)  # (N, C, T, V, M)
            batch_action = batch_action.to(device)
            batch_count = batch_count.to(device)

            optimizer.zero_grad()
            action_logits, rep_count = model(batch_x)
            loss_action = action_loss(action_logits, batch_action)
            loss_count = count_loss(rep_count.squeeze(), batch_count.float())
            loss = loss_action + 0.5 * loss_count  # combine losses (weighthed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

            # action accuracy
            _, predicted_action = torch.max(action_logits, 1)
            correct_action += (predicted_action == batch_action).sum().item()
            total_action += batch_action.size(0)

            # rep count mae
            total_count_error += torch.abs(rep_count.squeeze() - batch_count.float()).sum().item()
            total_count_samples += batch_count.size(0)
        
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

        for batch_x, batch_action, batch_count in test_loader:
            batch_x = batch_x.to(device)
            batch_action = batch_action.long().to(device)
            batch_count = batch_count.to(device)

            action_logits, rep_count = model(batch_x)
            _, predicted_action = torch.max(action_logits, 1)

            # Action classification metrics
            correct_action += (predicted_action == batch_action).sum().item()
            total_action += batch_action.size(0)

            # Rep count metrics (MAE example)
            total_count_error += torch.abs(rep_count.squeeze() - batch_count.float()).sum().item()
            total_count_samples += batch_count.size(0)

        accuracy = correct_action / total_action
        mae = total_count_error / total_count_samples
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Average Rep Count Error: {mae:.4f}")



               