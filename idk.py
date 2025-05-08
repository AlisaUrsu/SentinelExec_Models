import os
from indexed_scaled_dataset import IndexedScaledDataset
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
from torch.utils.data import DataLoader, Dataset

def stratified_indices(y, val_ratio=0.15, seed=19):
    np.random.seed(seed)
    y = np.asarray(y)

    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]

    np.random.shuffle(class0_idx)
    np.random.shuffle(class1_idx)

    val_size0 = int(len(class0_idx) * val_ratio)
    val_size1 = int(len(class1_idx) * val_ratio)

    val_idx = np.concatenate([class0_idx[:val_size0], class1_idx[:val_size1]])
    train_idx = np.concatenate([class0_idx[val_size0:], class1_idx[val_size1:]])

    return train_idx, val_idx


def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size=8192, val_split=0.15):
    train_idx, val_idx = stratified_indices(y_train, val_ratio=val_split)

    # Fit scaler incrementally
    scaler = StandardScaler()
    for i in range(0, len(train_idx), 10000):
        batch_indices = train_idx[i:i+10000]
        scaler.partial_fit(X_train[batch_indices])

    train_dataset = IndexedScaledDataset(X_train, y_train, train_idx, scaler)
    val_dataset = IndexedScaledDataset(X_train, y_train, val_idx, scaler)
    test_dataset = ScaledDataset(X_test, y_test, scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler

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
    nbenign = (y_true == 0).sum()
    nfalse = (y_pred[y_true == 0] == 1).sum()
    return nfalse / float(nbenign)


def find_threshold(y_true, y_probs, fpr_target=0.01, step=0.0001):
    threshold = 0.0
    fpr = get_fpr(y_true, y_probs > threshold)
    while fpr > fpr_target and threshold < 1.0:
        threshold += step
        fpr = get_fpr(y_true, y_probs > threshold)
    return threshold, fpr

def evaluate_at_fpr_thresholds(y_true, y_probs, fpr_targets=[0.01, 0.001]):
    y_true = np.asarray(y_true).flatten()
    y_probs = np.asarray(y_probs).flatten()
    for fpr_target in fpr_targets:
        threshold, actual_fpr = find_threshold(y_true, y_probs, fpr_target)
        fnr = ((y_probs[y_true == 1] < threshold).sum()) / float((y_true == 1).sum())
        tpr = 1 - fnr
        print(f"\nPerformance at {fpr_target*100:.1f}% FPR:")
        print(f"Threshold: {threshold:.4f}")
        print(f"False Positive Rate: {actual_fpr*100:.4f}%")
        print(f"False Negative Rate: {fnr*100:.4f}%")
        print(f"Detection Rate (TPR): {tpr*100:.4f}%")

    plt.figure(figsize=(8, 8))
    fpr_plot, tpr_plot, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr_plot, tpr_plot, lw=4, color='k')
    plt.gca().set_xscale("log")
    plt.yticks(np.arange(22) / 20.0)
    plt.xlim([4e-5, 1.0])
    plt.ylim([0.65, 1.01])
    plt.gca().grid(True)
    plt.vlines(actual_fpr, 0, tpr, color="r", lw=2)
    plt.hlines(tpr, 0, actual_fpr, color="r", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve at {fpr_target*100:.2f}% FPR")
    plt.show()


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