import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

class CNNModel(nn.Module):

    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(

            # BLOCK 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.10),

            # BLOCK 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.20),

            # BLOCK 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # BLOCK 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




# ─────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────

def prepare_tensors(*arrays, dtype=torch.float32):
    """Convert numpy arrays to float tensors and permute to (N, C, H, W)."""
    return [
        torch.tensor(a, dtype=dtype).permute(0, 3, 1, 2)
        for a in arrays
    ]


def prepare_labels(*arrays):
    """Convert numpy label arrays to long tensors."""
    return [torch.tensor(a, dtype=torch.long) for a in arrays]


def make_loaders(train_x, train_y, val_x, val_y, test_x, test_y,
                 batch_size=32):
    """Wrap tensor pairs in DataLoaders."""
    def _loader(x, y, shuffle):
        return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                          shuffle=shuffle)

    return (
        _loader(train_x, train_y, shuffle=True),
        _loader(val_x,   val_y,   shuffle=False),
        _loader(test_x,  test_y,  shuffle=False),
    )


def data_preparation(train_x, train_y, val_x, val_y, test_x, test_y,
                     batch_size=32):
    """Full data-prep pipeline; returns (loaders, num_classes, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    xs = prepare_tensors(train_x, val_x, test_x)
    ys = prepare_labels(train_y, val_y, test_y)

    print(f"train_x shape: {xs[0].shape}")
    assert xs[0].shape[1] == 1, f"Expected 1 channel, got {xs[0].shape[1]}"

    loaders = make_loaders(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], batch_size=batch_size)
    num_classes = len(torch.unique(ys[0]))
    print(f"Number of classes: {num_classes}")

    return loaders, num_classes, device


# ─────────────────────────────────────────
# 2. Training helpers
# ─────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    Run one epoch (train or eval).
    Pass optimizer=None for eval mode (no gradient update).
    Returns (avg_loss, accuracy).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, preds_all, targets_all = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds_all.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets_all.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(targets_all, preds_all)
    return avg_loss, acc


def check_early_stopping(val_loss, best_val_loss, counter, patience,
                         model, save_path="best_model.pth"):
    """
    Save model on improvement; increment counter otherwise.
    Returns (best_val_loss, counter, stop).
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print("  ✓ Best model saved")
        counter = 0
    else:
        counter += 1

    stop = counter >= patience
    return best_val_loss, counter, stop


# ─────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, patience, save_path="best_model.pth"):
    """Full training loop with early stopping. Returns history dict."""
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, counter = float("inf"), 0

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        best_val_loss, counter, stop = check_early_stopping(
            val_loss, best_val_loss, counter, patience, model, save_path
        )
        if stop:
            print("\nEarly stopping triggered!")
            break

    return history


# ─────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────

def evaluate(model, test_loader, criterion, device,
             save_path="best_model.pth"):
    """Load best weights, run test set, print metrics. Returns (preds, targets)."""
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )

    # Collect preds for detailed metrics
    preds_all, targets_all = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            preds_all.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets_all.extend(targets.numpy())

    precision = precision_score(targets_all, preds_all, average="weighted", zero_division=0)
    recall    = recall_score(targets_all, preds_all, average="weighted", zero_division=0)
    f1        = f1_score(targets_all, preds_all, average="weighted", zero_division=0)

    print("\n" + "=" * 40)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(targets_all, preds_all, zero_division=0))

    return preds_all, targets_all


# ─────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────

def plot_history(history, save_path="training_history.png"):
    """Plot loss and accuracy curves side-by-side."""
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss per Epoch", "Accuracy per Epoch"],
        ["Loss", "Accuracy"],
    ):
        ax.plot(epochs_ran, history[metric[0]], label=f"Train {ylabel}", color="royalblue")
        ax.plot(epochs_ran, history[metric[1]], label=f"Val {ylabel}",   color="tomato")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    folder = Path("images")
    folder.mkdir(
        parents=True,
        exist_ok=True
    )
    plt.savefig(folder/save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(targets, preds, labels,
                          save_path="confusion_matrix.png"):
    """Plot and save a labelled confusion-matrix heatmap."""
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    folder = Path("images")
    folder.mkdir(
        parents=True,
        exist_ok=True
    )
    plt.savefig(folder/save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────
# 6. Orchestrator
# ─────────────────────────────────────────

def build_training_components(num_classes, device):
    """Instantiate model, criterion, optimizer, and scheduler."""
    model     = CNNModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    print(model)
    return model, criterion, optimizer, scheduler


def run_pipeline(train_x, train_y, val_x, val_y, test_x, test_y,
                 labels, epochs=50, patience=10, batch_size=32, scenario="mel"):
    """End-to-end pipeline: prep → train → evaluate → visualise."""
    torch.manual_seed(42)
    # Data
    (train_loader, val_loader, test_loader), num_classes, device = data_preparation(
        train_x, train_y, val_x, val_y, test_x, test_y, batch_size
    )

    # Components
    model, criterion, optimizer, scheduler = build_training_components(
        num_classes, device
    )

    # Train
    history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=epochs, patience=patience, save_path=(f"best_{scenario}_model.pth")
    )

    # Evaluate
    preds, targets = evaluate(model, test_loader, criterion, device, save_path=f"best_{scenario}_model.pth")

    # Visualise
    plot_history(history, save_path=(f"{scenario}_training_history.png"))
    plot_confusion_matrix(targets, preds, labels, save_path=(f"{scenario}_confusion_matrix.png"))

