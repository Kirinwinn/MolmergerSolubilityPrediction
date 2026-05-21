"""Predict aqueous solubility LogS for solute SMILES using a trained MolMerger model."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import deepchem as dc
import pandas as pd

from attentivefp_model import AttentiveFPModel
from molmerger_utils import MolMerger, MolMergerFeaturizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict water LogS from solute SMILES using MolMerger."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", nargs="+", help="One or more solute SMILES.")
    input_group.add_argument(
        "--input-csv", help="CSV containing solute SMILES for batch prediction."
    )
    parser.add_argument(
        "--smiles-col",
        default="Smiles_Solute",
        help="Solute SMILES column when using --input-csv.",
    )
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
        default="water_predictions.csv",
        help="Where to write predictions.",
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
        model_dir=str(run_dir / "predict_model"),
    )
    model.restore(model_dir=str(checkpoint_dir))
    return model


def read_solutes(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if args.smiles_col not in df.columns:
            raise ValueError(f"Column '{args.smiles_col}' not found in {args.input_csv}")
        return df.copy()
    return pd.DataFrame({args.smiles_col: args.smiles})


def featurize_for_prediction(df: pd.DataFrame, smiles_col: str) -> dc.data.Dataset:
    rows = []
    for idx, smiles in enumerate(df[smiles_col].astype(str)):
        merged = MolMerger(smiles, "O")
        rows.append(
            {
                "row_id": idx,
                "Smiles_Solute": smiles,
                "Smiles_Solvent": "O",
                "Smiles_Merged": merged,
                "LogS": 0.0,
            }
        )

    temp_df = pd.DataFrame(rows)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        temp_path = Path(tmp.name)
        temp_df.to_csv(tmp, index=False)

    try:
        loader = dc.data.CSVLoader(
            tasks=["LogS"],
            smiles_field="Smiles_Merged",
            featurizer=MolMergerFeaturizer(use_edges=True),
        )
        dataset = loader.featurize(str(temp_path), shard_size=8192)
    finally:
        temp_path.unlink(missing_ok=True)

    return dataset, temp_df


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else run_dir / "checkpoints" / "best_model"
    )
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    config = load_config(run_dir)
    input_df = read_solutes(args)
    dataset, pred_df = featurize_for_prediction(input_df, args.smiles_col)
    model = make_model(config, run_dir, checkpoint_dir)
    predictions = model.predict(dataset).reshape(-1)

    output_df = input_df.copy()
    output_df["Smiles_Solvent"] = "O"
    output_df["Smiles_Merged"] = pred_df["Smiles_Merged"]
    output_df["Predicted_LogS"] = predictions
    output_df.to_csv(args.output_csv, index=False)

    print(output_df.to_string(index=False))
    print(f"Predictions written to: {args.output_csv}")


if __name__ == "__main__":
    main()
