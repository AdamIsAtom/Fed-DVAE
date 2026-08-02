# Fed-DVAE

This repository provides the PyTorch implementation of the paper
**Fed-DVAE: Federated Distribution-Aware Variational Auto-Encoder for
Privacy-Preserving Cross-Domain RUL Prediction**.

## Training flow

The implementation retains the five operations required by Fed-DVAE.

1. The server performs supervised Transformer-VAE pre-training on the labeled
   source domain.
2. The server broadcasts the complete model and the source latent statistics.
3. Each target client freezes the RUL prediction head and adapts its encoder and
   decoder with unlabeled target data. The local objective contains the
   reconstruction loss, KL divergence, latent-mean alignment, CORAL covariance
   alignment, and posterior log-variance alignment.
4. Clients upload only their encoder-decoder parameters. The server aggregates
   these parameters with sample-proportional weights while retaining its RUL
   prediction head.
5. The server updates the complete model with labeled source data and refreshes
   the source latent statistics before the next communication round.

The default configuration uses FD004 as the source domain and FD001, FD002, and
FD003 as the three target clients.

## Directory layout

```text
Fed-DVAE/
├── train.py
├── requirements.txt
├── README.md
└── CMAPSSData/
    ├── train_FD001.txt ... train_FD004.txt
    ├── test_FD001.txt  ... test_FD004.txt
    └── RUL_FD001.txt   ... RUL_FD004.txt
```

## Environment

Python 3.9 or later is recommended. Install the dependencies with

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches the local CUDA version
if the default installation does not provide CUDA support.

## Run

Run the default experiment from this directory.

```bash
python train.py
```

The default training schedule uses 10 source pre-training epochs, 100 federated
rounds, 10 local client epochs, and 10 source-supervised server epochs after each
aggregation.

The main settings can be overridden from the command line. For example,

```bash
python train.py --source FD001 --targets FD002 FD003 FD004 --rounds 100
```

Use `python train.py --help` to list all available options.

## Outputs

During training, each round reports the total local training loss, test RMSE,
and test MAE for every target client. The source-supervised server loss after
aggregation is also reported. Training creates an `outputs` directory containing

- `training_history.csv` with the per-round training loss, test RMSE, and test
  MAE for every target client

The random seed is fixed to 2025 by default. The code automatically uses CUDA
when it is available and otherwise runs on the CPU.
