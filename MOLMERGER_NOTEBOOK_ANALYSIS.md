# MolMerger Notebook Notes

This note summarizes which parts of `MolMergerModel.ipynb` are needed for
training and inference, and which parts are only for dataset reconstruction or
paper figures.

## Environment

For the AutoDL server described by the user:

- GPU: NVIDIA GeForce RTX 2080 Ti, 11 GB VRAM
- Driver: 550.90.07
- CUDA shown by `nvidia-smi`: 12.4
- CUDA Toolkit from `nvcc`: 12.1

Use the pinned pip environment in:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-autodl-cu121.txt
```

Python 3.10 is recommended. The original notebook was Colab-oriented and used
Python 3.10 wheels.

## Notebook Structure

### 1. Package installation

Notebook cells: 2-3

Purpose:

- Installs RDKit, PyTorch, PyTorch Geometric, DeepChem, DGL, DGL-LifeSci.

Recommendation:

- Do not run these cells on the server after creating the environment.
- Use `requirements-autodl-cu121.txt` instead.
- `torch-geometric` and `dglgo` are not needed by the current notebook code.

### 2. MolMerger graph construction

Notebook cell: 6

Purpose:

- Defines `MolMerger(smiles1, smiles2)`.
- Computes Gasteiger charges for solute and solvent.
- Finds the most positive/negative atoms in each molecule.
- Combines solute and solvent into one RDKit molecule.
- Adds two hydrogen-bond-like edges between oppositely charged atoms.
- Returns `Smiles_Merged`.

Recommendation:

- Keep this logic.
- For aqueous solubility prediction, call `MolMerger(solute_smiles, "O")`.

### 3. Dataset reconstruction from raw sources

Notebook cells: 10-53

Purpose:

- Reconstructs training and evaluation data from raw BigSolDB, ESOL, and BNN
  Labs files.
- Cleans data.
- Converts solubility to `LogS`.
- Builds `BigSolDBTrain.csv`, `ESOLTrain.csv`, `BNNLabsTrain.csv`, `Train.csv`,
  and `EvalData.csv`.
- Plots dataset distributions.

Recommendation:

- Skip this section if using the repository's existing `trainset.csv` and
  `EvalData.csv`.
- Re-run only if you want to rebuild the paper's dataset from raw sources.
- Several raw files referenced here are not present in the downloaded repo:
  `BigSolDB.csv`, `delaney-processed (2).csv`,
  `acetone_solubility_data.csv`, `benzene_solubility_data.csv`,
  `ethanol_solubility_data.csv`.

### 4. MolMergerFeaturizer

Notebook cell: 57

Purpose:

- Defines atom and bond features.
- Produces DeepChem `GraphData` objects from RDKit molecules.
- Atom features have length 32.
- Bond features include bond type, ring/conjugation/stereo, and graph-distance
  encoding.

Recommendation:

- Keep this section.
- It must be executed before training, evaluation, or prediction.

### 5. AttentiveFP model implementation

Notebook cells: 60, 62, 63, 64

Purpose:

- Defines a DGL-based AttentiveFP GNN.
- Wraps it in a DeepChem `TorchModel`.

Recommendation:

- Keep these cells for now.
- Later, these should be moved into a Python module so training and prediction
  can run from scripts instead of a notebook.

### 6. Training dataset loading

Notebook cell: 68

Purpose:

- Loads `trainset.csv`.
- Uses `Smiles_Merged` as the graph input and `LogS` as the regression target.
- Splits data with `IndexSplitter`.

Recommendation:

- Keep, but consider replacing `IndexSplitter` with a scaffold, random, or
  solvent-aware split depending on the experiment.
- For paper reproduction, keep `IndexSplitter(seed=42)`.

### 7. Model creation and metrics

Notebook cell: 69

Purpose:

- Defines R2 and RMSE metrics.
- Creates `AttentiveFPModel` with:
  - `n_tasks=1`
  - regression mode
  - 3 message-passing layers
  - dropout 0.2
  - 3 readout timesteps

Recommendation:

- Keep.
- The variable `batchsize = 128` is currently unused. If needed, pass it into
  `AttentiveFPModel(batch_size=128, ...)`.

### 8. Training loop

Notebook cell: 71

Purpose:

- Trains one epoch at a time.
- Evaluates train/test metrics after each epoch.
- Stops early if test RMSE jumps after epoch 30.
- Saves checkpoint to `/content/save2.pt`.

Recommendation:

- Modify before server training.
- Initialize `i = 0`; the notebook currently relies on `i` existing from a
  previous cell.
- Change `/content/save2.pt` to a local path such as `checkpoints/molmerger`.
- Set `batch_size` explicitly in the model constructor.
- Consider saving the best model by validation RMSE rather than the last model.

### 9. Restore pretrained checkpoint

Notebook cell: 72

Purpose:

- Restores `/content/FinalTrainedModel.pt`.

Recommendation:

- Skip this for now.
- The repository's `FinalTrainedModel.pt` is only 2 bytes and is not a usable
  model checkpoint.

### 10. Test-set evaluation plot

Notebook cell: 74

Purpose:

- Predicts on the internal test split.
- Makes a true-vs-predicted scatter plot.

Recommendation:

- Keep for checking training quality.
- This should be run after training or after restoring a real checkpoint.

### 11. Robust solvent evaluation

Notebook cells: 75-77

Purpose:

- Evaluates each solvent in `EvalData.csv`.
- Uses `Results.csv` only to define solvent order and names.
- Produces per-solvent plots and a frequency/loss plot.

Recommendation:

- Optional.
- Useful for reproducing the paper's diverse-solvent evaluation.
- Not necessary if your goal is only water solubility prediction.

### 12. SHAP-like feature importance

Notebook cells: 79-86

Purpose:

- Perturbs atom/bond features and observes changes in R2/RMSE.
- Produces feature-importance plots.

Recommendation:

- Skip during normal training and prediction.
- Run only if you need interpretability figures.

## Minimal Execution Path

For training from the existing repository data:

1. Install `requirements-autodl-cu121.txt`.
2. Run/import cell 6: `MolMerger`.
3. Run/import cell 57: `MolMergerFeaturizer`.
4. Run/import cells 60, 62, 63, 64: AttentiveFP implementation.
5. Run modified cell 68: load `trainset.csv`.
6. Run modified cell 69: create model and metrics.
7. Run modified cell 71: train and save checkpoint.
8. Run cell 74: evaluate.

For water prediction after training:

1. Set `Smiles_Solvent` to `O`.
2. Build `Smiles_Merged = MolMerger(Smiles_Solute, "O")`.
3. Featurize `Smiles_Merged`.
4. Restore the trained checkpoint.
5. Call `model.predict(dataset)`.
