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

    criterion = nn.CrossEntropyLoss()
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
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)  # (N, C, T, V, M)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)      # (N, num_classes)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()

        epoch_loss = running_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}")

    # Evaluation on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)

            outputs = model(batch_x)  # (N, num_classes)
            _, predicted = torch.max(outputs, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(batch_y.cpu())

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")



               