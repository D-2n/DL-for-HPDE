from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS_DIR))

from bubble import preprocess, normalize
from hyperbolic_pde.models.deeponet import DeepONet
from hyperbolic_pde.models.fno import FNO2d

T_MIN = 0.0
T_MAX = 5.0e-4
OUT_DIR = Path("bubble/results")
N_PLOT = 3
SAVE_PLOTS = True


class DeepONetDataset(Dataset):
    def __init__(self, X_func: np.ndarray, y: np.ndarray) -> None:
        self.X_func = torch.tensor(X_func, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X_func.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X_func[idx], self.y[idx]


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class FNODataset(Dataset):
    def __init__(self, X_func: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> None:
        self.X_func = torch.tensor(X_func, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        t = torch.tensor(t_grid, dtype=torch.float32).unsqueeze(0)  
        self.X = torch.zeros_like(t)
        self.T = t

    def __len__(self) -> int:
        return self.X_func.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.X_func[idx].unsqueeze(0)  
        inp = torch.stack([self.X, self.T, p], dim=0)  
        out = self.y[idx].unsqueeze(0).unsqueeze(0)  
        return inp, out


class SimpleGRU(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.head(out)
        return out.squeeze(-1)


def build_par(X_func: np.ndarray, y: np.ndarray) -> dict:
    return {
        "p_mean": float(np.mean(X_func)),
        "p_std": float(np.std(X_func)),
        "r_mean": float(np.mean(y)),
        "r_std": float(np.std(y)),
    }


def denorm_y(y: torch.Tensor, par: dict) -> torch.Tensor:
    return y * par["r_std"] + par["r_mean"]


def plot_samples(tag: str, n: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    if not SAVE_PLOTS:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = np.linspace(T_MIN, T_MAX, n)
    yt = y_true.detach().cpu().numpy()
    yp = y_pred.detach().cpu().numpy()
    if yt.ndim == 4:
        yt = yt[:, 0, 0, :]
        yp = yp[:, 0, 0, :]
    n_plot = min(N_PLOT, yt.shape[0])
    fig, axes = plt.subplots(n_plot, 1, figsize=(6, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i in range(n_plot):
        axes[i].plot(t, yt[i], label="true", linewidth=2)
        axes[i].plot(t, yp[i], label="pred", linestyle="--")
        axes[i].set_ylabel("R")
        axes[i].legend(loc="best")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}_samples.png", dpi=150)
    plt.close(fig)


def prepare_data(train_npz: Path, test_npz: Path, m: int, n: int):
    train = np.load(train_npz)
    test = np.load(test_npz)
    X_func_tr, X_loc_tr, y_tr = preprocess(train, m, n)
    X_func_te, X_loc_te, y_te = preprocess(test, m, n, is_test=True)
    par = build_par(X_func_tr, y_tr)
    X_func_tr, X_loc_tr, y_tr = normalize(X_func_tr, X_loc_tr, y_tr, par)
    X_func_te, X_loc_te, y_te = normalize(X_func_te, X_loc_te, y_te, par)
    return (X_func_tr, X_loc_tr, y_tr), (X_func_te, X_loc_te, y_te), par


def train_deeponet(train_npz: Path, test_npz: Path, m: int, n: int, device: torch.device) -> None:
    (X_func_tr, X_loc_tr, y_tr), (X_func_te, X_loc_te, y_te), par = prepare_data(
        train_npz, test_npz, m, n
    )
    loader = DataLoader(DeepONetDataset(X_func_tr, y_tr), batch_size=32, shuffle=True)

    model = DeepONet(
        branch_in=m,
        trunk_in=1,
        hidden_width=256,
        branch_layers=4,
        trunk_layers=4,
        latent_dim=128,
        activation="tanh",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    trunk = torch.tensor(X_loc_tr, dtype=torch.float32, device=device)

    model.train()
    start = time.perf_counter()
    for _ in range(400):
        for b, y in loader:
            b = b.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(b, trunk)
            loss = mse(pred, y)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        b = torch.tensor(X_func_te, dtype=torch.float32, device=device)
        t = torch.tensor(X_loc_te, dtype=torch.float32, device=device)
        y = torch.tensor(y_te, dtype=torch.float32, device=device)
        pred = model(b, t)
        mse_norm = mse(pred, y).item()
        pred_phys = denorm_y(pred, par)
        y_phys = denorm_y(y, par)
        mse_phys = mse(pred_phys, y_phys).item()

    plot_samples(f"deeponet_m{m}_n{n}", n, y_phys, pred_phys)
    print(f"[DeepONet] m={m} n={n} mse_norm={mse_norm:.3e} mse_phys={mse_phys:.3e} time={elapsed:.1f}s")


def train_rnn(train_npz: Path, test_npz: Path, n: int, device: torch.device) -> None:
    (X_func_tr, _, y_tr), (X_func_te, _, y_te), par = prepare_data(train_npz, test_npz, n, n)
    x_tr = X_func_tr[:, :, None]
    x_te = X_func_te[:, :, None]
    loader = DataLoader(SequenceDataset(x_tr, y_tr), batch_size=32, shuffle=True)

    model = SimpleGRU(hidden=256, layers=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    model.train()
    start = time.perf_counter()
    for _ in range(400):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse(pred, y)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_te, dtype=torch.float32, device=device)
        y = torch.tensor(y_te, dtype=torch.float32, device=device)
        pred = model(x)
        mse_norm = mse(pred, y).item()
        pred_phys = denorm_y(pred, par)
        y_phys = denorm_y(y, par)
        mse_phys = mse(pred_phys, y_phys).item()

    plot_samples(f"rnn_n{n}", n, y_phys, pred_phys)
    print(f"[RNN] n={n} mse_norm={mse_norm:.3e} mse_phys={mse_phys:.3e} time={elapsed:.1f}s")


def train_fno(train_npz: Path, test_npz: Path, n: int, device: torch.device) -> None:
    (X_func_tr, X_loc_tr, y_tr), (X_func_te, X_loc_te, y_te), par = prepare_data(
        train_npz, test_npz, n, n
    )
    t_grid = X_loc_tr.squeeze()
    loader = DataLoader(FNODataset(X_func_tr, y_tr, t_grid), batch_size=16, shuffle=True)

    modes_x = 1
    modes_t = min(24, (n // 2) + 1)
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=64,
        modes_x=modes_x,
        modes_t=modes_t,
        layers=5,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    model.train()
    start = time.perf_counter()
    for _ in range(400):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse(pred, y)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        test_ds = FNODataset(X_func_te, y_te, X_loc_te.squeeze())
        x = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0).to(device)
        y = torch.stack([test_ds[i][1] for i in range(len(test_ds))], dim=0).to(device)
        pred = model(x)
        mse_norm = mse(pred, y).item()
        pred_phys = denorm_y(pred, par)
        y_phys = denorm_y(y, par)
        mse_phys = mse(pred_phys, y_phys).item()

    plot_samples(f"fno_n{n}", n, y_phys, pred_phys)
    print(f"[FNO] n={n} mse_norm={mse_norm:.3e} mse_phys={mse_phys:.3e} time={elapsed:.1f}s")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    train_npz = Path("bubble/res_1000_1.npz")
    test_npz = Path("bubble/res_1000_08.npz")

    # Sparse vs dense discretizations
    deeponet_configs = [(50, 200), (200, 500)]
    rnn_points = [200, 500]
    fno_points = [500]

    for m, n in deeponet_configs:
        train_deeponet(train_npz, test_npz, m, n, device)

    for n in rnn_points:
        train_rnn(train_npz, test_npz, n, device)

    for n in fno_points:
        train_fno(train_npz, test_npz, n, device)


if __name__ == "__main__":
    main()
