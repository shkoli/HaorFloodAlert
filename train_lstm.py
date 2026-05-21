"""
train_lstm.py  —  HaorFloodAlert LSTM sequence model trainer.

Trains a 2-layer LSTM on synthetic sequences calibrated to Sunamganj Haor
physical feature distributions. The LSTM weight (0.20) is excluded from
the primary accuracy metric because it trains on synthetic data; the
RF + XGBoost ensemble (weights renormalized to sum=1) provides the
validated LOOCV performance reported in the paper.

Architecture
------------
    Input  : (batch, seq_len=5, n_features=13)
    LSTM1  : hidden=64, dropout=0.30
    LSTM2  : hidden=32, dropout=0.25
    Linear : 32 -> 1, sigmoid output

Usage
-----
    python train_lstm.py
"""

from pathlib import Path
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FEATURES, MODELS_DIR

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 5
SEED    = 42
EPOCHS  = 150
LR      = 3e-4
BATCH   = 64

torch.manual_seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

N_FEAT = len(FEATURES)

# Per-feature distributions calibrated to haor observations
FLOOD_MU  = [-18.5, -24.0,  0.77, 130.0, 46.0, 30.0, 17.0, 1.9,
              32.0,   0.25, 15.5, -18.5, 150.0]
FLOOD_STD = [  3.5,   3.2,  0.10,  65.0,  9.0,  2.5,  5.5, 0.4,
              18.0,   0.12,  2.2,   2.5,  65.0]
DRY_MU    = [-11.5, -18.5,  0.62,  28.0, 16.0, 22.0,  9.5, 1.9,
               7.0,  -0.22,  8.0, -10.5,  30.0]
DRY_STD   = [  2.8,   2.5,  0.10,  22.0,  7.0,  4.5,  3.5, 0.4,
               6.0,   0.10,  1.8,   2.0,  22.0]
DAY_NOISE = [1.0, 0.8, 0.05, 20.0, 3.5, 1.5, 3.0, 0.08,
             10.0, 0.04, 0.8,  1.2, 15.0]

BOUNDS = [
    (-30, -5), (-35, -10), (0.35, 1.3),
    (0, 480), (2, 70), (10, 40), (1, 55), (0.3, 6), (0, 120),
    (-0.5, 0.7), (4, 25), (-28, -5), (0, 500),
]


def _clip(val, i):
    """Clip value to physical bounds for feature i."""
    return float(np.clip(val, *BOUNDS[i]))


def make_sequence(label, hard=False):
    """
    Generate a synthetic time-series sequence of length SEQ_LEN.

    Parameters
    ----------
    label : int  — 1 = flood, 0 = dry
    hard  : bool — if True, shift means toward the overlap region to
                   create harder-to-classify examples

    Returns
    -------
    seq : np.ndarray, shape (SEQ_LEN, N_FEAT)
    """
    mu  = np.array(FLOOD_MU  if label == 1 else DRY_MU,  dtype=np.float32)
    std = np.array(FLOOD_STD if label == 1 else DRY_STD, dtype=np.float32)

    if hard:
        mid = (np.array(FLOOD_MU) + np.array(DRY_MU)) / 2
        mu  = mu + 0.40 * (mid - mu)

    base = np.array(
        [_clip(RNG.normal(mu[i], std[i]), i) for i in range(N_FEAT)],
        dtype=np.float32,
    )
    seq = np.zeros((SEQ_LEN, N_FEAT), dtype=np.float32)
    for t in range(SEQ_LEN):
        step = base.copy()
        for i in range(N_FEAT):
            step[i] = _clip(step[i] + RNG.normal(0, DAY_NOISE[i]), i)
        step[2] = step[0] / max(abs(step[1]), 0.001)   # recompute VV/VH ratio
        seq[t]  = step
    return seq


def build_dataset(n_flood=600, n_dry=600, hard_frac=0.25):
    """
    Build balanced synthetic dataset with hard-margin examples.

    Parameters
    ----------
    n_flood, n_dry : int — number of sequences per class
    hard_frac      : float — fraction of examples near the decision boundary

    Returns
    -------
    X : np.ndarray, shape (n_flood + n_dry, SEQ_LEN, N_FEAT)
    y : np.ndarray, shape (n_flood + n_dry,)
    """
    seqs, labels = [], []
    for label, n in [(1, n_flood), (0, n_dry)]:
        n_hard = int(n * hard_frac)
        for i in range(n):
            seqs.append(make_sequence(label, hard=(i < n_hard)))
            labels.append(label)
    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)


class HaorLSTM(nn.Module):
    """Two-layer LSTM classifier for haor flood sequences."""

    def __init__(self, input_size, hidden1=64, hidden2=32,
                 dropout1=0.30, dropout2=0.25):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout2)
        self.fc    = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return torch.sigmoid(self.fc(out)).squeeze(1)


def main():
    """Main LSTM training pipeline."""
    print("=" * 60)
    print("  HaorFloodAlert — LSTM Trainer (synthetic sequences)")
    print(f"  Device: {DEVICE}  |  Epochs: {EPOCHS}")
    print("=" * 60)

    X_raw, y = build_dataset(n_flood=600, n_dry=600)
    n, seq, feat = X_raw.shape

    # Normalize per feature over the time and sample axes
    X_flat = X_raw.reshape(-1, feat)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X_norm = X_flat.reshape(n, seq, feat).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_norm, y, test_size=0.20, random_state=SEED, stratify=y
    )

    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    )
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    model = HaorLSTM(input_size=feat).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te).to(DEVICE)
        probs   = model(X_te_t).cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    acc  = accuracy_score(y_te, preds) * 100
    auc  = roc_auc_score(y_te, probs)
    f1   = f1_score(y_te, preds, zero_division=0) * 100

    print(f"\n  Test Accuracy : {acc:.1f}%")
    print(f"  AUC-ROC       : {auc:.4f}")
    print(f"  F1 Score      : {f1:.1f}%")

    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pth")
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    joblib.dump(SEQ_LEN, MODELS_DIR / "lstm_window.pkl")
    print(f"\nLSTM artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
