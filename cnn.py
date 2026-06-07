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

            # BLOK 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # BLOK 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # BLOK 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # BLOK 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
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
    """Mengonversi numpy array ke float tensor dan mengubah urutan dimensi ke (N, C, H, W)."""
    return [
        torch.tensor(a, dtype=dtype).permute(0, 3, 1, 2)
        for a in arrays
    ]


def prepare_labels(*arrays):
    """Mengonversi numpy array label ke long tensor."""
    return [torch.tensor(a, dtype=torch.long) for a in arrays]


def make_loaders(train_x, train_y, val_x, val_y, test_x, test_y,
                 batch_size=32):
    """Membungkus pasangan tensor ke dalam DataLoaders."""
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
    """Alur lengkap penyiapan data; mengembalikan (loaders, num_classes, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    xs = prepare_tensors(train_x, val_x, test_x)
    ys = prepare_labels(train_y, val_y, test_y)

    print(f"Ukuran train_x: {xs[0].shape}")
    assert xs[0].shape[1] == 1, f"Diharapkan 1 kanal (channel), mendapatkan {xs[0].shape[1]}"

    loaders = make_loaders(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2], batch_size=batch_size)
    num_classes = len(torch.unique(ys[0]))
    print(f"Jumlah kelas: {num_classes}")

    return loaders, num_classes, device


# ─────────────────────────────────────────
# 2. Pembantu Pelatihan (Training Helpers)
# ─────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    Menjalankan satu epoch (pelatihan atau evaluasi).
    Gunakan optimizer=None untuk mode evaluasi (tanpa pembaruan gradien).
    Mengembalikan (avg_loss, accuracy).
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
    Menyimpan model ketika ada peningkatan; meningkatkan penghitung (counter) jika tidak.
    Mengembalikan (best_val_loss, counter, stop).
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print("  ✓ Model terbaik disimpan")
        counter = 0
    else:
        counter += 1

    stop = counter >= patience
    return best_val_loss, counter, stop


# ─────────────────────────────────────────
# 3. Loop Pelatihan (Training Loop)
# ─────────────────────────────────────────

def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, patience, save_path="best_model.pth"):
    """Loop pelatihan lengkap dengan penghentian awal (early stopping). Mengembalikan kamus riwayat (history dict)."""
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
        print(f"  Latih — Loss: {train_loss:.4f}  Akurasi: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}  Akurasi: {val_acc:.4f}")

        best_val_loss, counter, stop = check_early_stopping(
            val_loss, best_val_loss, counter, patience, model, save_path
        )
        if stop:
            print("\nPenghentian awal (early stopping) terpicu!")
            break

    return history


# ─────────────────────────────────────────
# 4. Evaluasi (Evaluation)
# ─────────────────────────────────────────

def evaluate(model, test_loader, criterion, labels, device,
             save_path="best_model.pth"):
    """Memuat bobot terbaik, menjalankan set pengujian, dan menampilkan metrik. Mengembalikan (preds, targets)."""
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_acc = run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )

    # Mengumpulkan prediksi untuk metrik mendetail
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
    print(f"Akurasi Pengujian : {test_acc:.4f}")
    print(f"Loss Pengujian     : {test_loss:.4f}")
    print(f"Presisi           : {precision:.4f}")
    print(f"Sensitivitas (Recall) : {recall:.4f}")
    print(f"Skor F1            : {f1:.4f}")
    print("=" * 40)
    print("\nLaporan Klasifikasi:")
    print(classification_report(targets_all, preds_all, target_names=labels, zero_division=0))

    return preds_all, targets_all


# ─────────────────────────────────────────
# 5. Visualisasi (Visualisation)
# ─────────────────────────────────────────

def plot_history(history, save_path="training_history.png"):
    """Menggambar kurva loss dan akurasi secara berdampingan."""
    epochs_ran = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss per Epoch", "Akurasi per Epoch"],
        ["Loss", "Akurasi"],
    ):
        ax.plot(epochs_ran, history[metric[0]], label=f"Latih {ylabel}", color="royalblue")
        ax.plot(epochs_ran, history[metric[1]], label=f"Val {ylabel}",   color="tomato")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.suptitle("Riwayat Pelatihan", fontsize=14, fontweight="bold")
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
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(30, 30))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 8})
    plt.title("Matriks Kebingungan (Confusion Matrix)", fontsize=16, fontweight="bold")
    plt.xlabel("Label Prediksi", fontsize=12)
    plt.ylabel("Label Sebenarnya", fontsize=12)

    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    folder = Path("images")
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(folder / save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────
# 6. Orkestrator (Orchestrator)
# ─────────────────────────────────────────

def build_training_components(num_classes, device):
    """Menginstansiasi model, kriteria loss, optimizer, dan scheduler."""
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
    """Alur lengkap (end-to-end pipeline): penyiapan → pelatihan → evaluasi → visualisasi."""
    torch.manual_seed(42)
    # Penyiapan Data
    (train_loader, val_loader, test_loader), num_classes, device = data_preparation(
        train_x, train_y, val_x, val_y, test_x, test_y, batch_size
    )

    # Komponen
    model, criterion, optimizer, scheduler = build_training_components(
        num_classes, device
    )

    # Pelatihan
    history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=epochs, patience=patience, save_path=(f"best_{scenario}_model.pth")
    )

    # Evaluasi
    preds, targets = evaluate(model, test_loader, criterion, labels, device, save_path=f"best_{scenario}_model.pth")

    # Visualisasi
    plot_history(history, save_path=(f"{scenario}_training_history.png"))
    plot_confusion_matrix(targets, preds, labels, save_path=(f"{scenario}_confusion_matrix.png"))

