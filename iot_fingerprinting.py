"""
IoT Radio Fingerprinting – Environment & Node Classification
=============================================================
Standalone Python script version of iot_fingerprinting.ipynb

Project:  Identifying Deployment Environments and IoT Sensor Nodes
          Using Link Quality Fluctuations
Radio:    IEEE 802.15.4 (Adafruit Feather nRF52840)

Scenario I  – Classify deployment environment (5 classes)
Scenario II – Classify sensor node (3 classes)

Strategy 1  – 75/25 random split (seen data)
Strategy 2  – Leave-one-environment-out (unseen data)

Usage:
    python iot_fingerprinting.py              # uses ./data_folder as default
    python iot_fingerprinting.py ./my_data    # custom data directory
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             ConfusionMatrixDisplay)
from torchinfo import summary

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DATA_DIR     = sys.argv[1] if len(sys.argv) > 1 else "./data_folder"
FRAME_SIZES  = [500, 1000]      # as specified in project description
OVERLAPS     = [0.4, 0.5]       # 40% and 50% overlap
EPOCHS       = 100
SEED         = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════
# 2. PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def clean_and_differentiate(rssi_series):
    """Interpolate NaN gaps, then compute y[i] = x[i+1] - x[i]."""
    s = pd.Series(rssi_series).interpolate(method="linear").bfill().ffill()
    return np.diff(s.values)


def normalise(y):
    """Min-max normalise to [0, 1]."""
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return np.full_like(y, 0.5, dtype=np.float32)
    return ((y - y_min) / (y_max - y_min)).astype(np.float32)


def segment_frames(signal, frame_size, overlap_frac):
    """Segment into overlapping frames."""
    step = int(frame_size * (1 - overlap_frac))
    frames = []
    for start in range(0, len(signal) - frame_size + 1, step):
        frames.append(signal[start : start + frame_size])
    return np.array(frames, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    """3-layer 1-D CNN for time-series classification."""
    def __init__(self, n_classes, frame_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResBlock(nn.Module):
    """Single 1-D residual block."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet1D(nn.Module):
    """Lightweight 1-D ResNet with 3 residual blocks."""
    def __init__(self, n_classes, frame_size):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(64), ResBlock(64), ResBlock(64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.head(self.res_blocks(self.stem(x)))


# ══════════════════════════════════════════════════════════════════
# 4. TRAINING AND EVALUATION
# ══════════════════════════════════════════════════════════════════

def make_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Wrap numpy arrays into PyTorch DataLoaders."""
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    g = torch.Generator().manual_seed(SEED)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(test_ds,  batch_size=batch_size))


def train_model(model, train_loader, epochs=60, lr=1e-3, verbose=True,
                test_loader=None, n_classes=None):
    """Train with Adam + weight decay + cosine annealing LR.
    If test_loader is provided, tracks train/test loss and accuracy per epoch.
    Returns a history dict with keys: train_loss, test_loss, train_acc, test_acc.
    """
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
        scheduler.step()

        train_loss = total_loss / n
        train_acc = correct / n
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ── Test evaluation (if provided) ──
        if test_loader is not None:
            model.eval()
            t_loss, t_correct, t_n = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    t_loss += loss_fn(out, yb).item() * len(xb)
                    t_correct += (out.argmax(1) == yb).sum().item()
                    t_n += len(xb)
            history["test_loss"].append(t_loss / t_n)
            history["test_acc"].append(t_correct / t_n)

        if verbose and (epoch + 1) % 15 == 0:
            msg = f"    Epoch {epoch+1:3d}/{epochs}  loss={train_loss:.4f}  acc={train_acc:.3f}"
            if test_loader is not None:
                msg += f"  | val_loss={history['test_loss'][-1]:.4f}  val_acc={history['test_acc'][-1]:.3f}"
            msg += f"  lr={opt.param_groups[0]['lr']:.2e}"
            print(msg)

    return history


def evaluate(model, test_loader, n_classes):
    """Returns (accuracy, confusion_matrix, y_true, y_pred)."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.append(model(xb.to(device)).argmax(1).cpu())
            all_true.append(yb)
    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    return acc, cm, y_true, y_pred


# ══════════════════════════════════════════════════════════════════
# 5. DATASET BUILDER
# ══════════════════════════════════════════════════════════════════

def build_dataset(scenario, frame_size, overlap, series_dict, ENVS, NODES,
                  ENV2IDX, NODE2IDX):
    """
    Build (X, y, group_labels) for a given scenario.
      scenario='env'  -> y = environment index (5 classes)
      scenario='node' -> y = node index        (3 classes)
      group_labels = environment name per frame (for leave-one-out)
    """
    X_all, y_all, groups = [], [], []

    if scenario == "env":
        for env in ENVS:
            combined = np.concatenate([series_dict[(n, env)] for n in NODES])
            frames = segment_frames(combined, frame_size, overlap)
            X_all.append(frames)
            y_all.append(np.full(len(frames), ENV2IDX[env], dtype=np.int64))
            groups.extend([env] * len(frames))

    elif scenario == "node":
        for node in NODES:
            for env in ENVS:
                frames = segment_frames(series_dict[(node, env)], frame_size, overlap)
                X_all.append(frames)
                y_all.append(np.full(len(frames), NODE2IDX[node], dtype=np.int64))
                groups.extend([env] * len(frames))

    X = np.concatenate(X_all)[:, np.newaxis, :]
    y = np.concatenate(y_all)
    return X, y, np.array(groups)


# ══════════════════════════════════════════════════════════════════
# 6. PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def plot_training_curves(history_store, filename):
    """Plot train/test loss and accuracy: 2x2 grid (Model x Scenario).
    Each subplot overlays frame=500 (solid) and frame=1000 (dashed)
    at overlap=50%, keeping the figure clean and readable.
    """
    combos = [
        ("CNN1D",    "I (Environment)"),
        ("CNN1D",    "II (Node)"),
        ("ResNet1D", "I (Environment)"),
        ("ResNet1D", "II (Node)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Training Curves -- Strategy 1 (Seen Data)\n"
                 "Solid = frame 500  |  Dashed = frame 1000  |  overlap = 50%",
                 fontsize=13, fontweight="bold")

    for idx, (mname, scn) in enumerate(combos):
        row = idx // 2
        col_base = (idx % 2) * 2   # 0 or 2

        ax_loss = axes[row][col_base]
        ax_acc  = axes[row][col_base + 1]
        title = f"{mname} | {scn}"

        for fs, ls in [(500, "-"), (1000, "--")]:
            key = (mname, scn, fs, 0.5)
            if key not in history_store:
                continue
            hist = history_store[key]
            ep = range(1, len(hist["train_loss"]) + 1)

            ax_loss.plot(ep, hist["train_loss"], ls, color="#2196F3",
                         label=f"Train f={fs}", alpha=0.9)
            if hist["test_loss"]:
                ax_loss.plot(ep, hist["test_loss"], ls, color="#FF5722",
                             label=f"Test f={fs}", alpha=0.9)

            ax_acc.plot(ep, [a*100 for a in hist["train_acc"]], ls,
                        color="#2196F3", label=f"Train f={fs}", alpha=0.9)
            if hist["test_acc"]:
                ax_acc.plot(ep, [a*100 for a in hist["test_acc"]], ls,
                            color="#FF5722", label=f"Test f={fs}", alpha=0.9)

        ax_loss.set_title(f"{title}\nLoss", fontsize=10)
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7); ax_loss.grid(True, alpha=0.3)

        ax_acc.set_title(f"{title}\nAccuracy", fontsize=10)
        ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(fontsize=7); ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")

def plot_confusion_matrices(cm_store, strategy, frame_size, overlap,
                            plot_configs, filename):
    """Plot 2x2 confusion matrix grid and save to file."""
    cmap = "Blues" if "Seen" in strategy else "Oranges"
    acc_label = "Accuracy" if "Seen" in strategy else "Mean Accuracy"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Confusion Matrices -- Strategy {strategy}\n"
                 f"frame={frame_size}, overlap={int(overlap*100)}%",
                 fontsize=14, fontweight="bold")

    for row, col, mname, scn_label in plot_configs:
        cm, labels, acc = cm_store[(mname, scn_label, strategy, frame_size, overlap)]
        ax = axes[row][col]
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap=cmap, colorbar=False)
        ax.set_title(f"{mname} -- Scenario {scn_label}\n"
                     f"{acc_label}: {acc*100:.1f}%", fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


def plot_bar_chart(results_df, filename):
    """Plot bar chart: 2x2 grid (frame x scenario) with clean grouped bars."""
    df = results_df.copy()
    frames = sorted(df["Frame"].unique())
    scenarios = df["Scenario"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("All Experiment Results", fontsize=15, fontweight="bold")

    for fi, fs in enumerate(frames):
        for si, scn in enumerate(scenarios):
            ax = axes[fi][si]
            sub = df[(df["Frame"] == fs) & (df["Scenario"] == scn)]

            # Group: each model-overlap combo gets a pair of bars (S1, S2)
            groups = sub.groupby(["Model", "Overlap"]).agg(list).reset_index()
            n_groups = len(groups)
            x = np.arange(n_groups)
            width = 0.35

            s1_vals, s2_vals, labels = [], [], []
            for _, row in groups.iterrows():
                for j, strat in enumerate(row["Strategy"]):
                    if "Seen" in strat and "Un" not in strat:
                        s1_vals.append(row["Accuracy"][j])
                    else:
                        s2_vals.append(row["Accuracy"][j])
                labels.append(f"{row['Model']}\n{row['Overlap']}")

            ax.bar(x - width/2, s1_vals, width, label="Strategy 1 (Seen)",
                   color="#2196F3", edgecolor="white")
            ax.bar(x + width/2, s2_vals, width, label="Strategy 2 (Unseen)",
                   color="#FF9800", edgecolor="white")

            # Value labels
            for i_bar, v in enumerate(s1_vals):
                ax.text(i_bar - width/2, v + 1, f"{v:.0f}%", ha="center",
                        fontsize=7, color="#1565C0")
            for i_bar, v in enumerate(s2_vals):
                ax.text(i_bar + width/2, v + 1, f"{v:.0f}%", ha="center",
                        fontsize=7, color="#E65100")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Scenario {scn} | frame={fs}", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 110)
            chance = 20 if "Environment" in scn else 33.3
            ax.axhline(y=chance, color="gray", ls="--", lw=0.8,
                       label=f"chance ({chance:.0f}%)")
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════
# 7. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════

def main():
    set_seed()
    print(f"Using device: {device}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Frame sizes: {FRAME_SIZES}")
    print(f"Overlaps: {OVERLAPS}")
    print(f"Epochs: {EPOCHS}")

    # ── Step 1: Load data ──
    print("\n" + "=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)

    all_frames = []
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        fname = os.path.basename(fpath)
        parts = fname.split("_")
        node = parts[0][-1]
        env  = parts[1]
        df = pd.read_csv(fpath)
        df["node"] = node
        df["environment"] = env
        all_frames.append(df)

    if not all_frames:
        print(f"ERROR: No CSV files found in {DATA_DIR}")
        sys.exit(1)

    data = pd.concat(all_frames, ignore_index=True)
    print(f"  Total rows:   {len(data):,}")
    print(f"  Nodes:        {sorted(data['node'].unique())}")
    print(f"  Environments: {sorted(data['environment'].unique())}")
    print(f"  NaN RSSI:     {data['rssi'].isna().sum()} "
          f"({data['rssi'].isna().mean()*100:.2f}%)")

    # ── Step 2: Preprocess ──
    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing (differentiate -> normalise)")
    print("=" * 70)

    series_dict = {}
    for (node, env), grp in data.groupby(["node", "environment"]):
        rssi = grp.sort_values("timestamp")["rssi"].values
        diff = clean_and_differentiate(rssi)
        norm = normalise(diff)
        series_dict[(node, env)] = norm

    print(f"  Built {len(series_dict)} preprocessed series")
    for k, v in sorted(series_dict.items()):
        print(f"    {k}: {len(v):,} samples, range=[{v.min():.3f}, {v.max():.3f}]")

    # ── Step 3: Setup labels ──
    ENVS  = sorted(data["environment"].unique())
    NODES = sorted(data["node"].unique())
    ENV2IDX  = {e: i for i, e in enumerate(ENVS)}
    NODE2IDX = {n: i for i, n in enumerate(NODES)}
    print(f"\n  Environment labels: {ENV2IDX}")
    print(f"  Node labels:        {NODE2IDX}")

    # ── Step 4: Print model summaries ──
    print("\n" + "=" * 70)
    print("STEP 3: Model architectures")
    print("=" * 70)
    print("\nCNN1D (5-class, frame=500):")
    summary(CNN1D(5, 500), input_size=(1, 1, 500), verbose=1)
    print("\nResNet1D (5-class, frame=500):")
    summary(ResNet1D(5, 500), input_size=(1, 1, 500), verbose=1)

    # ── Step 5: Run all 32 experiments ──
    print("\n" + "=" * 70)
    print("STEP 4: Running all 32 experiments")
    print("=" * 70)

    SCENARIOS  = [("env", 5, ENVS, "I (Environment)"),
                  ("node", 3, NODES, "II (Node)")]
    MODEL_DEFS = [("CNN1D", CNN1D), ("ResNet1D", ResNet1D)]

    results = []
    cm_store = {}
    history_store = {}   # training curves for Strategy 1

    for scenario_key, n_classes, label_names, scenario_label in SCENARIOS:
        for frame_size in FRAME_SIZES:
            for overlap in OVERLAPS:
                X, y, groups = build_dataset(
                    scenario_key, frame_size, overlap,
                    series_dict, ENVS, NODES, ENV2IDX, NODE2IDX)
                unique_groups = sorted(set(groups))

                for model_name, ModelClass in MODEL_DEFS:

                    # ── Strategy 1: 75/25 split ──
                    set_seed(SEED)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=0.25, random_state=SEED, stratify=y)
                    train_ld, test_ld = make_loaders(X_tr, y_tr, X_te, y_te)

                    print(f"\n{'='*70}")
                    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

                    set_seed(SEED)
                    model = ModelClass(n_classes, frame_size)
                    hist = train_model(model, train_ld, epochs=EPOCHS,
                                       test_loader=test_ld, n_classes=n_classes)
                    acc, cm, _, _ = evaluate(model, test_ld, n_classes)

                    # Store training history for plotting
                    hist_key = (model_name, scenario_label, frame_size, overlap)
                    history_store[hist_key] = hist

                    print(f"  >> {model_name} | Strategy 1 (Seen) | "
                          f"Scenario {scenario_label} | frame={frame_size} | "
                          f"overlap={int(overlap*100)}% | Accuracy: {acc*100:.1f}%")

                    results.append(dict(
                        Model=model_name, Scenario=scenario_label,
                        Strategy="1 (Seen)", Frame=frame_size,
                        Overlap=f"{int(overlap*100)}%",
                        Accuracy=round(acc*100, 1)))
                    cm_store[(model_name, scenario_label, "1 (Seen)",
                              frame_size, overlap)] = (cm, label_names, acc)

                    # ── Strategy 2: leave-one-environment-out ──
                    fold_accs, all_yt, all_yp = [], [], []
                    for fold_idx, held_out in enumerate(unique_groups):
                        mask_test  = groups == held_out
                        mask_train = ~mask_test
                        X_tr2, y_tr2 = X[mask_train], y[mask_train]
                        X_te2, y_te2 = X[mask_test],  y[mask_test]

                        if len(X_te2) == 0:
                            continue

                        set_seed(SEED + fold_idx)
                        tr2, te2 = make_loaders(X_tr2, y_tr2, X_te2, y_te2)
                        m = ModelClass(n_classes, frame_size)
                        _ = train_model(m, tr2, epochs=EPOCHS, verbose=False)
                        a, _, yt2, yp2 = evaluate(m, te2, n_classes)
                        fold_accs.append(a)
                        all_yt.append(yt2); all_yp.append(yp2)

                    mean_acc = np.mean(fold_accs) if fold_accs else 0.0
                    agg_cm = (confusion_matrix(np.concatenate(all_yt),
                                               np.concatenate(all_yp),
                                               labels=range(n_classes))
                              if all_yt
                              else np.zeros((n_classes, n_classes), dtype=int))

                    print(f"  Fold accuracies: "
                          f"{[f'{a:.3f}' for a in fold_accs]}")
                    print(f"  >> {model_name} | Strategy 2 (Unseen) | "
                          f"Scenario {scenario_label} | frame={frame_size} | "
                          f"overlap={int(overlap*100)}% | "
                          f"Accuracy: {mean_acc*100:.1f}%")

                    results.append(dict(
                        Model=model_name, Scenario=scenario_label,
                        Strategy="2 (Unseen)", Frame=frame_size,
                        Overlap=f"{int(overlap*100)}%",
                        Accuracy=round(mean_acc*100, 1)))
                    cm_store[(model_name, scenario_label, "2 (Unseen)",
                              frame_size, overlap)] = (
                                  agg_cm, label_names, mean_acc)

    print(f"\n{'='*70}")
    print(f"All 32 experiments complete. {len(cm_store)} confusion matrices stored.")

    # ── Step 6: Generate outputs ──
    print("\n" + "=" * 70)
    print("STEP 5: Generating outputs")
    print("=" * 70)

    # Training curves (loss & accuracy per epoch)
    plot_training_curves(history_store, "training_curves.png")

    # Confusion matrix plots (for BOTH frame sizes)
    plot_configs = [
        (0, 0, "CNN1D",    "I (Environment)"),
        (0, 1, "ResNet1D", "I (Environment)"),
        (1, 0, "CNN1D",    "II (Node)"),
        (1, 1, "ResNet1D", "II (Node)"),
    ]
    for fs in FRAME_SIZES:
        plot_confusion_matrices(cm_store, "1 (Seen)", fs, 0.5,
                                plot_configs, f"confusion_matrices_strategy1_f{fs}.png")
        plot_confusion_matrices(cm_store, "2 (Unseen)", fs, 0.5,
                                plot_configs, f"confusion_matrices_strategy2_f{fs}.png")

    # Per-class accuracy (for BOTH frame sizes)
    for fs in FRAME_SIZES:
        print(f"\n  Per-class accuracy (frame={fs}, overlap=50%):")
        for strategy in ["1 (Seen)", "2 (Unseen)"]:
            print(f"\n  {'='*60}")
            print(f"    STRATEGY {strategy} | frame={fs}")
            print(f"  {'='*60}")
            for mname in ["CNN1D", "ResNet1D"]:
                for scn_label in ["I (Environment)", "II (Node)"]:
                    cm, labels, acc = cm_store[
                        (mname, scn_label, strategy, fs, 0.5)]
                    row_sums = cm.sum(axis=1)
                    per_class = np.where(row_sums > 0,
                                         cm.diagonal() / row_sums * 100, 0.0)
                    print(f"\n    {mname} | Scenario {scn_label} | "
                          f"Overall: {acc*100:.1f}%")
                    for lbl, pca, rs in zip(labels, per_class, row_sums):
                        print(f"      {lbl:>10s}: {pca:5.1f}%  (n={rs})")

        # Results summary table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("FULL RESULTS SUMMARY")
    print("=" * 90)
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv("results_summary.csv", index=False)
    print("\n  Saved: results_summary.csv")

    # Bar chart
    plot_bar_chart(results_df, "results_comparison.png")

    print("\n" + "=" * 70)
    print("DONE. All outputs saved to current directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
