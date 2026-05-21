"""Optional robust evaluation on EvalData.csv grouped by solvent."""

from __future__ import annotations
import scipy.stats as st
if not hasattr(st, "gilbrat") and hasattr(st, "gibrat"):
    st.gilbrat = st.gibrat
import argparse
import json
from pathlib import Path

import deepchem as dc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from attentivefp_model import AttentiveFPModel
from molmerger_utils import MolMergerFeaturizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MolMerger model on EvalData.csv by solvent."
    )
    parser.add_argument("--data", default="EvalData.csv")
    parser.add_argument("--results", default="Results.csv")
    parser.add_argument(
        "--run-dir",
        default="runs/molmerger_train",
        help="Training run directory containing training_config.json and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory. Defaults to <run-dir>/checkpoints/best_model.",
    )
    parser.add_argument(
        "--output-csv",
        default="runs/molmerger_train/eval/solvent_metrics.csv",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def make_model(config: dict, run_dir: Path, checkpoint_dir: Path) -> AttentiveFPModel:
    model = AttentiveFPModel(
        n_tasks=1,
        mode="regression",
        num_layers=config.get("num_layers", 3),
        num_timesteps=config.get("num_timesteps", 3),
        graph_feat_size=config.get("graph_feat_size", 200),
        dropout=config.get("dropout", 0.2),
        batch_size=config.get("batch_size", 128),
        learning_rate=config.get("learning_rate", 0.001),
        model_dir=str(run_dir / "eval_model"),
    )
    model.restore(model_dir=str(checkpoint_dir))
    return model


def featurize_csv(csv_path: Path) -> dc.data.Dataset:
    loader = dc.data.CSVLoader(
        tasks=["LogS"],
        smiles_field="Smiles_Merged",
        featurizer=MolMergerFeaturizer(use_edges=True),
    )
    return loader.featurize(str(csv_path), shard_size=8192)


def solvent_names(results_path: Path) -> dict:
    if not results_path.exists():
        return {}
    results = pd.read_csv(results_path)
    if {"Smiles_Solvent", "Solvent_Name"}.issubset(results.columns):
        return dict(zip(results["Smiles_Solvent"], results["Solvent_Name"]))
    return {}


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    run_dir = Path(args.run_dir)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else run_dir / "checkpoints" / "best_model"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {data_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    config = load_config(run_dir)
    model = make_model(config, run_dir, checkpoint_dir)
    dataset = featurize_csv(data_path)
    predictions = model.predict(dataset).reshape(-1)

    df = pd.read_csv(data_path).copy()
    df["Predicted_LogS"] = predictions
    name_lookup = solvent_names(Path(args.results))

    rows = []
    for solvent, group in df.groupby("Smiles_Solvent"):
        y_true = group["LogS"].to_numpy()
        y_pred = group["Predicted_LogS"].to_numpy()
        rows.append(
            {
                "Smiles_Solvent": solvent,
                "Solvent_Name": name_lookup.get(solvent, ""),
                "n": len(group),
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred, squared=False),
                "R2": r2_score(y_true, y_pred) if len(group) > 1 else np.nan,
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows).sort_values("MAE")
    metrics_df.to_csv(output_path, index=False)

    print(metrics_df.to_string(index=False))
    print(f"Solvent metrics written to: {output_path}")


if __name__ == "__main__":
    main()
