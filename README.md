# dynoNet for Airfoil Dynamic-Stall Modelling

Code accompanying the master's thesis *INVESTIGATION OF THE DYNONET ARCHITECTURE
An application to pitching wing data*
(Luca da Silveira,  Vrije Universiteit Brussel], 2026).

This repository contains the pyton scripts used to identify the unsteady
lift response of a pitching airfoil during dynamic stall with **dynoNet** — a
block-oriented neural-network architecture — and to reproduce the results
reported in the thesis.


---

## ⚠️ Important — the scripts are configured for a single machine

**Read this before running anything.** The scripts were written and run on one
specific Windows computer, and the relevant file locations are **hard-coded as
absolute paths**. As they stand, the scripts will *not* run on any other
machine until those paths are changed to match your own system.

Near the top of the training script you will find three lines that need
editing:

```python
# 1) where Python looks for the dynoNet source package
sys.path.insert(0, r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\dynoNet\dynonet\src")

# 2) the training dataset
TRAIN_PATH = r"C:\Users\lucad\Downloads\DatasetCFD.mat"

# 3) where all results/outputs are written
BASE_DIR   = r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\Masterproef\dynoNet\ExperimentValidation"
```

To run the code elsewhere, replace those three values with the corresponding
locations on your machine.

**Recommended (portable) fix.** Rather than swapping in another set of absolute
paths, point everything at locations *relative to the script itself*, so the
repository runs anywhere it is cloned. For example:

```python
from pathlib import Path

HERE = Path(__file__).resolve().parent          # folder containing this script
sys.path.insert(0, str(HERE / "dynonet" / "src"))   # if dynoNet is vendored here
TRAIN_PATH = HERE / "data" / "DatasetCFD.mat"
BASE_DIR   = HERE / "ExperimentValidation"
```

This caveat applies to all scripts: check the top of
each file for hard-coded paths before running.

---

## Overview

The training script builds **parallel Wiener–Hammerstein dynoNet models** that
map angle of attack to lift coefficient ($\alpha \rightarrow C_l$) from CFD
data, and supports three topologies:

| Version  | Parallel branches | Mixing block |
|----------|-------------------|--------------|
| `V2`     | 2                 | 2 → 1        |
| `V4`     | 3                 | 3 → 1        |
| `custom` | *N* (your choice) | *N* → 1      |

Each branch is a `G1 → Fnl → G2` chain (linear dynamic operator → static
nonlinearity → linear dynamic operator); the branch outputs are combined by a
static mixing network and passed through an output IIR filter `G_out`.

The pipeline also includes:

- **Transient masking** — the first samples of each excitation sweep (where the
  amplitude or mean angle changes) are excluded from the loss and the reported
  fit, so transients do not distort the accuracy figures.
- **Z-score normalisation** of both input and output, with the statistics saved
  so that validation can reproduce them exactly.
- **Metrics:** masked/full fit index and relative RMSE.
- **Saved artefacts** for every run (weights, plots, a results table, an
  architecture summary).

---

## Requirements

- **Python 3** (the original dynoNet package targets Python 3.7+).
- `numpy`, `scipy`, `matplotlib`, `torch` (PyTorch).
- **dynoNet** — see the next section.

Install the Python dependencies (excluding dynoNet) with:

```bash
pip install numpy scipy matplotlib torch
```

---

## Installing dynoNet
**dynoNet is not my work.** It is the architecture by Marco Forgione and Dario
Piga, released under the **MIT licence**, available at
<https://github.com/forgi86/dynonet>. The files `lti.py`, `static.py`,
`functional.py`, `metrics.py` and `filtering.py` belong to that project.

dynoNet is available this way:

**Reference it.** Clone the original repository somewhere on
your machine and point line (1) above at its `src` folder:
```bash
   git clone https://github.com/forgi86/dynonet.git
   ```

---

## Data

The dataset is provided as `DatasetCFD.mat` (MATLAB) and/or
`DatasetCFD_converted.csv`.

- **`u`** — input: angle of attack $\alpha$ (degrees)
- **`y`** — output: lift coefficient $C_l$ (–)
- **`fs`** — sampling frequency (Hz)

It contains **32 008 samples = 8 swept-sine sweeps × 4 001 samples**, obtained
from CFD simulations of a pitching airfoil (NACA 0018, RE = 150 000).

The training script reads a `.mat` file and expects the keys `uTrain`,
`yTrain`, and optionally `fsTrain`/`fs` (if no sampling frequency is found it
falls back to 200 Hz). If you prefer to work from the `.csv`, adapt the loading
section accordingly.

---

## Repository structure

*(example layout — adjust to match your actual files)*

```
.
├── README.md
├── train_dynonet.py            # main training script (V2 / V4 / custom)
├── validate.py                 # loads saved weights + norm_stats, evaluates
├── data/
│   └── DatasetCFD.mat          
```

---

## How to run

1. **Edit the paths** at the top of the training script (see the warning above).
2. **Run the training script:**
   ```bash
   python train_dynonet.py
   ```
   It is interactive — you will be asked which version to train (`V2`, `V4` or
   `custom`) and whether to use the default hyper-parameters or override them
   (states per G-block, neurons, activation functions, epochs, learning rate,
   random seed).
3. **Find the outputs.** A run folder is created under
   `ExperimentValidation/V{2|4|N}_masked/training_…/` containing the artefacts
   listed below.
4. **Validate** a trained model with the validation script, which loads
   `model_weights.pt` together with `norm_stats.json` (the latter is required
   to normalise the input and de-normalise the predicted output).

---

## Outputs (per run)

| File                    | Contents                                             |
|-------------------------|------------------------------------------------------|
| `model_weights.pt`      | trained model parameters                             |
| `norm_stats.json`       | z-score means/stds (needed for validation)           |
| `time_plot.pdf`         | AoA, CFD vs. model $C_l$, and error vs. time         |
| `loss_curve.pdf`        | training loss (log scale)                            |
| `normalised_inputs.pdf` | sanity check of the normalised signals               |
| `results.csv`           | configuration + final metrics                        |
| `architecture.txt`      | summary of the model and masking      |

---

## Reproducibility notes

- A fixed random seed (`SEED = 42`) is set for both PyTorch and NumPy before the
  model is built, so a given configuration is reproducible.
- Training uses the **Adam** optimiser with the full sequence as a single batch
  (deterministic, not stochastic).
- The normalisation statistics in `norm_stats.json` are computed on the full
  training set; **validation must reuse them** rather than recomputing, or the
  reported fit will be wrong.
- Numerical results may still differ slightly across hardware/PyTorch versions.

---

## Citing this work

If you use this code, please cite the thesis and the dynoNet paper.

```bibtex
@mastersthesis{daSilveira,
  title  = {INVESTIGATION OF THE DYNONET ARCHITECTURE
An application to pitching wing data},
  author = {Luca Anani da Silveira},
  school = {Vrije Universiteit Brussel},
  year   = {2026}
}

@article{forgione2021dynonet,
  title   = {dynoNet: A neural network architecture for learning dynamical systems},
  author  = {Forgione, Marco and Piga, Dario},
  journal = {International Journal of Adaptive Control and Signal Processing},
  volume  = {35},
  number  = {4},
  pages   = {612--626},
  year    = {2021},
  doi     = {10.1002/acs.3216}
}
```

---



## Acknowledgements

dynoNet by Marco Forgione and Dario Piga (<https://github.com/forgi86/dynonet>).
promotor: prof. dr. ing. J. Decuyper |  supervisor: G. Van Essche | data source: L. Damiola
