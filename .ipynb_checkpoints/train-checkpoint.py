"""Train the MolMerger AttentiveFP regression model.

This script uses the already prepared `trainset.csv` from the repository.
It does not rebuild BigSolDB/ESOL/BNNLabs data.
"""
from __future__ import annotations
import scipy.stats as st
if not hasattr(st, "gilbrat") and hasattr(st, "gibrat"):
    st.gilbrat = st.gibrat
import argparse
import csv
import json
import random
import time
from pathlib import Path

import deepchem as dc
import numpy as np
import torch
from tqdm.auto import tqdm

from attentivefp_model import AttentiveFPModel
from molmerger_utils import MolMergerFeaturizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MolMerger AttentiveFP on prepared solubility data."
    )
    parser.add_argument("--data", default="trainset.csv", help="Prepared CSV file.")
    parser.add_argument(
        "--smiles-col",
        default="Smiles_Merged",
        help="Column containing merged solute-solvent SMILES.",
    )
    parser.add_argument(
        "--target-col", default="LogS", help="Regression target column."
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-timesteps", type=int, default=3)
    parser.add_argument("--graph-feat-size", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation RMSE improvement. 0 disables early stopping.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/molmerger_train",
        help="Directory for logs and DeepChem model artifacts.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=8192,
        help="DeepChem CSV featurization shard size.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(args: argparse.Namespace) -> dc.data.Dataset:
    print(f"Featurizing {args.data} with '{args.smiles_col}' -> '{args.target_col}'")
    loader = dc.data.CSVLoader(
        tasks=[args.target_col],
        smiles_field=args.smiles_col,
        featurizer=MolMergerFeaturizer(use_edges=True),
    )
    return loader.featurize(args.data, shard_size=args.shard_size)


def make_model(args: argparse.Namespace) -> AttentiveFPModel:
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "working_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    return AttentiveFPModel(
        n_tasks=1,
        mode="regression",
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        graph_feat_size=args.graph_feat_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_dir=str(model_dir),
    )


def save_training_config(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.json"
    config = vars(args).copy()
    config["number_atom_features"] = 32
    config["number_bond_features"] = 19
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def make_log_writer(args: argparse.Namespace):
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.csv"

    fieldnames = [
        "epoch",
        "train_rmse",
        "valid_rmse",
        "train_r2",
        "valid_r2",
        "best_valid_rmse",
        "is_best",
        "elapsed_sec",
        "learning_rate",
        "batch_size",
        "num_layers",
        "num_timesteps",
        "graph_feat_size",
        "dropout",
        "seed",
    ]
    log_file = log_path.open("w", newline="")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    return log_path, log_file, writer


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    dataset = load_dataset(args)
    splitter = dc.splits.IndexSplitter()
    train_dataset, valid_dataset = splitter.train_test_split(dataset, seed=args.seed)

    metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    metric_rmse = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
    metrics = [metric_r2, metric_rmse]

    model = make_model(args)
    config_path = save_training_config(args)
    log_path, log_file, log_writer = make_log_writer(args)
    best_model_dir = Path(args.output_dir) / "checkpoints" / "best_model"
    last_model_dir = Path(args.output_dir) / "checkpoints" / "last_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    last_model_dir.mkdir(parents=True, exist_ok=True)
    best_valid_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(
        "Training samples: "
        f"{len(train_dataset)} | Validation samples: {len(valid_dataset)}"
    )
    print(
        "Device: "
        f"{'cuda' if torch.cuda.is_available() else 'cpu'} | "
        f"Epochs: {args.epochs} | Batch size: {args.batch_size}"
    )

    start_time = time.time()
    progress = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    try:
        for epoch in progress:
            model.fit(train_dataset, nb_epoch=1)

            train_score = model.evaluate(train_dataset, metrics)
            valid_score = model.evaluate(valid_dataset, metrics)

            train_rmse = train_score["mean-rms_score"]
            valid_rmse = valid_score["mean-rms_score"]
            train_r2 = train_score["mean-pearson_r2_score"]
            valid_r2 = valid_score["mean-pearson_r2_score"]
            is_best = valid_rmse < best_valid_rmse

            if is_best:
                best_valid_rmse = valid_rmse
                best_epoch = epoch
                epochs_without_improvement = 0
                model.save_checkpoint(
                    model_dir=str(best_model_dir), max_checkpoints_to_keep=1
                )
            else:
                epochs_without_improvement += 1
            model.save_checkpoint(
                model_dir=str(last_model_dir), max_checkpoints_to_keep=1
            )

            elapsed = time.time() - start_time
            log_writer.writerow(
                {
                    "epoch": epoch,
                    "train_rmse": train_rmse,
                    "valid_rmse": valid_rmse,
                    "train_r2": train_r2,
                    "valid_r2": valid_r2,
                    "best_valid_rmse": best_valid_rmse,
                    "is_best": int(is_best),
                    "elapsed_sec": round(elapsed, 3),
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "num_layers": args.num_layers,
                    "num_timesteps": args.num_timesteps,
                    "graph_feat_size": args.graph_feat_size,
                    "dropout": args.dropout,
                    "seed": args.seed,
                }
            )
            log_file.flush()

            progress.set_postfix(
                {
                    "train_rmse": f"{train_rmse:.4f}",
                    "valid_rmse": f"{valid_rmse:.4f}",
                    "best": f"{best_valid_rmse:.4f}",
                    "valid_r2": f"{valid_r2:.4f}",
                }
            )
            tqdm.write(
                "epoch "
                f"{epoch:03d} | "
                f"train_rmse={train_rmse:.4f} valid_rmse={valid_rmse:.4f} | "
                f"train_r2={train_r2:.4f} valid_r2={valid_r2:.4f} | "
                f"best_valid_rmse={best_valid_rmse:.4f}"
                f"{' *' if is_best else ''}"
            )
            if args.patience and epochs_without_improvement >= args.patience:
                tqdm.write(
                    "early stopping: "
                    f"no validation RMSE improvement for {args.patience} epochs"
                )
                break
    finally:
        log_file.close()

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed / 60:.2f} minutes.")
    print(f"Training config: {config_path}")
    print(f"Training log: {log_path}")
    print(f"Best epoch: {best_epoch} | Best validation RMSE: {best_valid_rmse:.4f}")
    print(f"Best checkpoint directory: {best_model_dir}")
    print(f"Last checkpoint directory: {last_model_dir}")
    print(f"Working model artifacts are under: {Path(args.output_dir) / 'working_model'}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
