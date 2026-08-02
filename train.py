"""Minimal Fed-DVAE training pipeline for C-MAPSS."""

from __future__ import annotations

import argparse
import math
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


SENSORS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11",
           "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
ALL_SENSORS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"] + ALL_SENSORS
INPUT_DIM = len(SENSORS)
MAX_RUL = 125.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def add_rul(df: pd.DataFrame, final_rul: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    if final_rul is None:
        df["RUL"] = df.groupby("unit_nr")["time_cycles"].transform("max") - df["time_cycles"]
    else:
        rul_map = {i + 1: value for i, value in enumerate(final_rul["RUL"].values)}
        max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
        df["RUL"] = df["unit_nr"].map(rul_map) + max_cycles - df["time_cycles"]
    return df


def add_operating_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["setting_1"] = df["setting_1"].abs().round()
    df["setting_2"] = df["setting_2"].abs().round(2)
    df["op_cond"] = (
        df["setting_1"].astype(str)
        + "_"
        + df["setting_2"].astype(str)
        + "_"
        + df["setting_3"].astype(str)
    )
    return df


def scale_by_condition(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for condition in train["op_cond"].unique():
        train_mask = train["op_cond"] == condition
        test_mask = test["op_cond"] == condition
        scaler = MinMaxScaler().fit(train.loc[train_mask, SENSORS])
        train.loc[train_mask, SENSORS] = scaler.transform(train.loc[train_mask, SENSORS])
        if test_mask.any():
            test.loc[test_mask, SENSORS] = scaler.transform(test.loc[test_mask, SENSORS])
    return train, test


def smooth(df: pd.DataFrame, alpha: float = 0.8) -> pd.DataFrame:
    df = df.copy()
    df[SENSORS] = df.groupby("unit_nr")[SENSORS].transform(
        lambda values: values.ewm(alpha=alpha).mean()
    )
    return df


def preprocess_pair(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.core.groupby.DataFrameGroupBy, pd.core.groupby.DataFrameGroupBy]:
    unused_sensors = [name for name in ALL_SENSORS if name not in SENSORS]
    train = add_operating_condition(train.drop(columns=unused_sensors))
    test = add_operating_condition(test.drop(columns=unused_sensors))
    train, test = scale_by_condition(train, test)
    train, test = smooth(train), smooth(test)
    dropped = ["setting_1", "setting_2", "setting_3", "op_cond"]
    train = train.drop(columns=dropped)
    test = test.drop(columns=dropped)
    return train.groupby("unit_nr"), test.groupby("unit_nr")


def load_source(data_dir: Path, domain: str, seed: int):
    source = add_rul(read_table(data_dir / f"train_{domain}.txt"))
    source["RUL"] = source["RUL"].clip(upper=MAX_RUL) / MAX_RUL
    units = source["unit_nr"].unique()
    train_units, held_out_units = train_test_split(
        units, test_size=0.2, random_state=seed
    )
    source_train = source[source["unit_nr"].isin(train_units)]
    source_held_out = source[source["unit_nr"].isin(held_out_units)]
    train_group, _ = preprocess_pair(source_train, source_held_out)
    return train_group


def load_target(data_dir: Path, domain: str):
    train = add_rul(read_table(data_dir / f"train_{domain}.txt"))
    train["RUL"] = train["RUL"].clip(upper=MAX_RUL) / MAX_RUL
    test_rul = pd.read_csv(
        data_dir / f"RUL_{domain}.txt", sep=r"\s+", header=None, names=["RUL"]
    )
    test = add_rul(read_table(data_dir / f"test_{domain}.txt"), test_rul)
    train_group, test_group = preprocess_pair(train, test)
    return train_group, test_group, test_rul


class SequenceDataset(Dataset):
    def __init__(
        self,
        groups,
        sequence_length: int,
        mode: str,
        final_rul: pd.DataFrame | None = None,
    ) -> None:
        samples: List[np.ndarray] = []
        labels: List[float] = []

        if mode == "train":
            for unit in groups.groups:
                values = groups.get_group(unit).to_numpy()
                for end in range(sequence_length, len(values)):
                    samples.append(values[end - sequence_length:end, 2:2 + INPUT_DIM].astype(float))
                    labels.append(float(values[end - 1, -1]))
        elif mode == "test":
            if final_rul is None:
                raise ValueError("final_rul is required for test sequences")
            target_rul = final_rul["RUL"].to_numpy()
            for index, unit in enumerate(groups.groups):
                values = groups.get_group(unit).to_numpy()
                if len(values) < sequence_length:
                    padded = np.zeros((sequence_length, values.shape[1]))
                    for column in range(values.shape[1]):
                        if len(values) > 1:
                            old_x = np.arange(len(values), dtype=np.float64)
                            slope, intercept = np.polyfit(old_x, values[:, column], deg=1)
                            new_x = np.linspace(0, sequence_length - 1, sequence_length)
                            padded[:, column] = new_x * len(values) / sequence_length * slope + intercept
                        else:
                            padded[:, column] = values[0, column]
                    values = padded
                samples.append(values[-sequence_length:, 2:2 + INPUT_DIM].astype(float))
                labels.append(min(float(target_rul[index]), MAX_RUL) / MAX_RUL)
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")

        self.x = torch.tensor(np.asarray(samples), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_len, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer("pe", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim, latent_dim = 64, 32
        self.input_proj = nn.Linear(INPUT_DIM, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, INPUT_DIM)
        )
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.transformer_encoder(self.pos_encoder(self.input_proj(x)))
        hidden = encoded.mean(dim=1)
        mu, log_var = self.fc_mu(hidden), self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        repeated = self.decoder_input(z).unsqueeze(1).repeat(1, x.size(1), 1)
        reconstruction = self.decoder_net(repeated)
        prediction = self.rul_head(z if self.training else mu)
        return {
            "rul_pred": prediction,
            "recon_x": reconstruction,
            "mu": mu,
            "log_var": log_var,
        }


def kl_divergence(output: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
    value = -0.5 * torch.sum(
        1 + output["log_var"] - output["mu"].pow(2) - output["log_var"].exp()
    )
    return value / (batch_size * INPUT_DIM)


def daa_losses(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    source_mean: torch.Tensor,
    source_cov: torch.Tensor,
    source_log_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimension = mu.size(1)
    target_mean = mu.mean(dim=0)
    centered = mu - target_mean.unsqueeze(0)
    target_cov = centered.T @ centered / (mu.size(0) - 1)
    mean_loss = F.mse_loss(target_mean, source_mean)
    coral_loss = torch.sum((target_cov - source_cov) ** 2) / (4 * dimension * dimension)
    log_var_loss = F.mse_loss(log_var.mean(dim=0), source_log_var)
    return mean_loss, coral_loss, log_var_loss


class Server:
    def __init__(self, loader: DataLoader, device: torch.device, learning_rate: float) -> None:
        self.loader = loader
        self.device = device
        self.model = TransformerVAE().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.source_mean = torch.zeros(32, device=device)
        self.source_cov = torch.eye(32, device=device)
        self.source_log_var = torch.zeros(32, device=device)

    def supervised_train(self, epochs: int, recon_weight: float, beta: float) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = (
                    F.mse_loss(output["rul_pred"], y.view(-1, 1))
                    + recon_weight * F.mse_loss(output["recon_x"], x)
                    + beta * kl_divergence(output, x.size(0))
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        self.update_statistics()
        return total_loss / total_samples

    def update_statistics(self) -> None:
        means, log_vars = [], []
        self.model.eval()
        with torch.no_grad():
            for x, _ in self.loader:
                output = self.model(x.to(self.device))
                means.append(output["mu"])
                log_vars.append(output["log_var"])
        all_means = torch.cat(means)
        self.source_mean = all_means.mean(dim=0).detach()
        centered = all_means - self.source_mean.unsqueeze(0)
        self.source_cov = (centered.T @ centered / (len(all_means) - 1)).detach()
        self.source_log_var = torch.cat(log_vars).mean(dim=0).detach()

    def packet(self) -> Dict[str, object]:
        return {
            "weights": {name: value.detach().clone() for name, value in self.model.state_dict().items()},
            "mean": self.source_mean.detach().clone(),
            "cov": self.source_cov.detach().clone(),
            "log_var": self.source_log_var.detach().clone(),
        }

    def aggregate(self, updates: Sequence[Tuple[float, Dict[str, torch.Tensor]]]) -> None:
        total_weight = sum(weight for weight, _ in updates)
        if total_weight <= 0:
            raise ValueError("Client aggregation weights must sum to a positive value")
        normalized = [(weight / total_weight, state) for weight, state in updates]
        server_state = self.model.state_dict()
        merged = OrderedDict()
        for name, current in server_state.items():
            if name.startswith("rul_head."):
                merged[name] = current
            elif not torch.is_floating_point(current):
                merged[name] = normalized[0][1][name].to(self.device)
            else:
                value = torch.zeros_like(current, device=self.device)
                for weight, state in normalized:
                    value.add_(state[name].to(self.device), alpha=weight)
                merged[name] = value
        self.model.load_state_dict(merged)


class Client:
    def __init__(
        self,
        domain: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float,
    ) -> None:
        self.domain = domain
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sample_count = len(train_loader.dataset)
        self.device = device
        self.model = TransformerVAE().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.source_mean = self.source_cov = self.source_log_var = None

    def receive(self, packet: Dict[str, object]) -> None:
        self.model.load_state_dict(packet["weights"])
        self.source_mean = packet["mean"].to(self.device)
        self.source_cov = packet["cov"].to(self.device)
        self.source_log_var = packet["log_var"].to(self.device)

    def adapt(self, epochs: int, args: argparse.Namespace) -> float:
        self.model.train()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not name.startswith("rul_head.")
        total_loss = 0.0
        total_samples = 0
        for _ in range(epochs):
            for x, _ in self.train_loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                mean_loss, coral_loss, var_loss = daa_losses(
                    output["mu"],
                    output["log_var"],
                    self.source_mean,
                    self.source_cov,
                    self.source_log_var,
                )
                loss = (
                    args.lambda_recon * F.mse_loss(output["recon_x"], x)
                    + args.beta * kl_divergence(output, x.size(0))
                    + args.lambda_mean * mean_loss
                    + args.lambda_coral * coral_loss
                    + args.lambda_var * var_loss
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        return total_loss / total_samples

    def upload(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if not name.startswith("rul_head.")
        }

    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                output = self.model(x.to(self.device))
                predictions.append(output["rul_pred"].cpu() * MAX_RUL)
                labels.append(y * MAX_RUL)
        prediction = torch.cat(predictions).numpy().reshape(-1)
        label = torch.cat(labels).numpy().reshape(-1)
        return float(np.sqrt(mean_squared_error(label, prediction))), float(
            mean_absolute_error(label, prediction)
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def prepare(args: argparse.Namespace, device: torch.device):
    print(f"Loading source domain {args.source}")
    source_group = load_source(args.data_dir, args.source, args.seed)
    source_dataset = SequenceDataset(source_group, args.sequence_length, "train")
    source_loader = make_loader(
        source_dataset, args.batch_size, True, True, args.workers, device.type == "cuda"
    )

    clients = []
    for domain in args.targets:
        print(f"Loading target client {domain}")
        train_group, test_group, final_rul = load_target(args.data_dir, domain)
        train_dataset = SequenceDataset(train_group, args.sequence_length, "train")
        test_dataset = SequenceDataset(test_group, args.sequence_length, "test", final_rul)
        clients.append(
            Client(
                domain,
                make_loader(train_dataset, args.batch_size, True, True, args.workers, device.type == "cuda"),
                make_loader(test_dataset, args.batch_size, False, False, args.workers, device.type == "cuda"),
                device,
                args.client_lr,
            )
        )
    return source_loader, clients


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"Using device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_loader, clients = prepare(args, device)
    server = Server(source_loader, device, args.server_lr)
    print(f"Source-supervised pre-training for {args.pretrain_epochs} epochs")
    pretrain_loss = server.supervised_train(
        args.pretrain_epochs, args.lambda_recon, args.beta
    )
    print(f"Source pre-training loss: {pretrain_loss:.6f}")

    total_samples = sum(client.sample_count for client in clients)
    aggregation_weights = {
        client.domain: client.sample_count / total_samples for client in clients
    }
    print(
        "Aggregation weights: "
        + ", ".join(f"{name}={value:.6f}" for name, value in aggregation_weights.items())
    )

    history = []
    for round_index in range(1, args.rounds + 1):
        packet = server.packet()
        updates = []
        round_history = []
        for client in clients:
            client.receive(packet)
            train_loss = client.adapt(args.client_epochs, args)
            rmse, mae = client.evaluate()
            print(
                f"Round {round_index:03d} | {client.domain} | "
                f"Train Total Loss {train_loss:.6f} | "
                f"Test RMSE {rmse:.4f} | Test MAE {mae:.4f}"
            )
            round_history.append(
                {
                    "round": round_index,
                    "domain": client.domain,
                    "train_total_loss": train_loss,
                    "test_rmse": rmse,
                    "test_mae": mae,
                }
            )
            updates.append((aggregation_weights[client.domain], client.upload()))
        server.aggregate(updates)
        server_loss = server.supervised_train(
            args.server_epochs, args.lambda_recon, args.beta
        )
        print(
            f"Round {round_index:03d} | Server | "
            f"Source Train Total Loss {server_loss:.6f}"
        )
        for record in round_history:
            record["server_source_train_total_loss"] = server_loss
        history.extend(round_history)

    history_path = args.output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Training complete. Per-round history saved to {history_path}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Minimal Fed-DVAE training on C-MAPSS")
    parser.add_argument("--data-dir", type=Path, default=root / "CMAPSSData")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--source", default="FD004")
    parser.add_argument("--targets", nargs="+", default=["FD001", "FD002", "FD003"])
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--client-epochs", type=int, default=10)
    parser.add_argument("--server-epochs", type=int, default=10)
    parser.add_argument("--server-lr", type=float, default=1e-3)
    parser.add_argument("--client-lr", type=float, default=5e-5)
    parser.add_argument("--lambda-recon", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=5e-4)
    parser.add_argument("--lambda-mean", type=float, default=1.0)
    parser.add_argument("--lambda-coral", type=float, default=1.0)
    parser.add_argument("--lambda-var", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
