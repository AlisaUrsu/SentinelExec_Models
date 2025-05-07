import os
from scaled_dataset import ScaledDataset
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size=8192, val_split=0.15):
    # Split training into train and val
    val_len = int(val_split * len(X_train))
    train_len = len(X_train) - val_len

    X_val = X_train[-val_len:]
    y_val = y_train[-val_len:]
    X_train = X_train[:train_len]
    y_train = y_train[:train_len]

    # Fit scaler incrementally
    scaler = StandardScaler()
    batch_size_scaler = 10000
    for i in range(0, X_train.shape[0], batch_size_scaler):
        scaler.partial_fit(X_train[i:i+batch_size_scaler])

    # Wrap datasets
    train_dataset = ScaledDataset(X_train, y_train, scaler)
    val_dataset = ScaledDataset(X_val, y_val, scaler)
    test_dataset = ScaledDataset(X_test, y_test, scaler)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001, save_path="firstNN.pth"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss().to(device)

    early_stopping_threshold_count = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_acc_train = 0
        total_loss_train = 0
        total_samples_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss_train += loss.item() * inputs.size(0)
            predictions = (outputs >= 0.5).int()
            total_acc_train += (predictions == labels.int()).sum().item()
            total_samples_train += labels.size(0)

        model.eval()
        total_loss_val = 0
        total_acc_val = 0
        total_samples_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss_val += loss.item() * inputs.size(0)
                predictions = (outputs >= 0.5).int()
                total_acc_val += (predictions == labels.int()).sum().item()
                total_samples_val += labels.size(0)

        avg_train_loss = total_loss_train / total_samples_train
        avg_val_loss = total_loss_val / total_samples_val
        avg_train_acc = total_acc_train / total_samples_train
        avg_val_acc = total_acc_val / total_samples_val

        print(f'Epoch {epoch + 1} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Train Acc: {avg_train_acc:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'Val Acc: {avg_val_acc:.4f}')

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(avg_train_acc)
        val_accuracies.append(avg_val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print("Saved model")
            early_stopping_threshold_count = 0
        else:
            early_stopping_threshold_count += 1

        if early_stopping_threshold_count >= 1:
            print("Early stopping triggered.")
            break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "best_val_loss": best_val_loss, 
    }

def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, model_name="Model"):
    epochs = range(1, len(train_losses) + 1)

    print(model_name)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.tight_layout()
    plt.show()

def get_fpr(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mask_benign = y_true == 0
    nbenign = mask_benign.sum()
    nfalse = (y_pred[mask_benign] == 1).sum()
    
    return nfalse / float(nbenign)


def find_threshold(y_true, y_probs, fpr_target=0.01, step=0.0001):
    threshold = 0.0
    fpr = get_fpr(y_true, y_probs > threshold)
    while fpr > fpr_target and threshold < 1.0:
        threshold += step
        fpr = get_fpr(y_true, y_probs > threshold)
    return threshold, fpr

def evaluate_at_fpr_thresholds(y_true, y_probs, fpr_targets=[0.01, 0.001]):
    for fpr_target in fpr_targets:
        threshold, actual_fpr = find_threshold(y_true, y_probs, fpr_target)
        fnr = ((y_probs[y_true == 1] < threshold).sum()) / float((y_true == 1).sum())
        print(f"\nPerformance at {fpr_target*100:.1f}% FPR:")
        print(f"Threshold: {threshold:.4f}")
        print(f"False Positive Rate: {actual_fpr*100:.4f}%")
        print(f"False Negative Rate: {fnr*100:.4f}%")
        print(f"Detection Rate (TPR): {(1 - fnr)*100:.4f}%")


def evaluate_model_on_test(model, test_loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = model(xb).squeeze()  # probabilities from sigmoid
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Convert probabilities to binary predictions
    preds = (all_probs >= 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Test Accuracy: {accuracy:.6f}")
    print(f"Test Precision: {precision:.6f}")
    print(f"Test Recall: {recall:.6f}")
    print(f"Test F1 Score: {f1:.6f}")
    print(f"Test ROC AUC: {auc:.6f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "probs": all_probs,
        "labels": all_labels
    }


def plot_roc_curve(all_labels, all_probs):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)

    print(f"AUC Score: {auc_score:.6f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.6f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(all_labels, all_probs, threshold=0.5):
    y_pred = (all_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()