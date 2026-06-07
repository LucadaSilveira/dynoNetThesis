"""
DynoNet -- unified validation of a trained V2 / V4 model
========================================================

Validation script for a trained DynoNet model.  Select the version (V2, V4 or a
custom number of branches), and the script rebuilds that model, loads the
weights and the z-score statistics saved during training, and validates it on
the monosine cases using the periodic-repetition method.

VERSIONS

  V2       : 2 parallel branches
  V4       : 3 parallel branches
  custom   : choose the number of branches and the G_out bonus yourself

PERIODIC-REPETITION TRANSIENT REMOVAL

  The monosine signals are perfectly periodic, but the IIR G-blocks start from
  zero initial states, so the first part of the response is corrupted.

    1. take one period of the input        (T samples)
    2. tile it N_REPEATS times             (N_REPEATS * T samples)
    3. z-score-normalise with the training statistics
    4. run the model on the tiled sequence
    5. denormalise the output back to physical Cl
    6. discard the first N_DISCARD_PERIODS complete periods
    7. extract N_EVAL_PERIODS steady-state period(s)
    8. compare against the CFD reference

  Discarding whole periods keeps the
  extracted period phase-aligned with the CFD reference.


OUTPUT FOLDER

ExperimentValidation\\V{N}_masked\\
    val_{label}_s{N}_n{N}[_nmix{N}]_{act}_trSmart_norm\\
        time_plot.pdf
        hysteresis_plot.pdf
        settling_plot.pdf
        results.csv
    overview_validation_s{N}_n{N}[_nmix{N}]_{act}_trSmart_norm.csv
    terminal_log_validate_*.txt
"""

import csv
import json
import os
import sys
import zipfile

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Location of the dynonet source.
sys.path.insert(0, r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\dynoNet\dynonet\src")

from dynonet.lti    import SisoLinearDynamicalOperator
from dynonet.static import SisoStaticNonLinearity, MimoStaticNonLinearity


# Plot style
plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 11,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'legend.fontsize'   : 10,
    'figure.dpi'        : 150,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : False,
    'lines.linewidth'   : 1.4,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
})
COLORS = {
    'cfd'     : '#C0392B',
    'model'   : '#2980B9',
    'error'   : '#27AE60',
    'aoa'     : '#2C3E50',
    'discard' : '#E67E22',
    'steady'  : '#27AE60',
}


# paths
# folder with the trained runs
BASE_DIR   = r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\Masterproef\dynoNet\ExperimentValidation"
# Monosine validation data.
VAL_ZIP    = r"C:\Users\lucad\Downloads\Monosines.zip"
VAL_FOLDER = r"C:\Users\lucad\Downloads\Monosines"

# defaults (match the values the model was trained with)
VERSION_DEFAULTS = {
    'V2': {
        'n_branches'   : 2,
        'g_out_bonus'  : 0,
        'n_states'     : 5,
        'n_neurons'    : 20,
        'n_neurons_mix': 30,
        'activation_branches': 'tanh',
        'activation_mix'     : 'tanh',
        'n_epochs'     : 40000,
        'lr'           : 1e-4,
        'seed'         : 42,
    },
    'V4': {
        'n_branches'   : 3,
        'g_out_bonus'  : 0,
        'n_states'     : 5,
        'n_neurons'    : 20,
        'n_neurons_mix': 30,
        'activation_branches': 'tanh',
        'activation_mix'     : 'tanh',
        'n_epochs'     : 40000,
        'lr'           : 1e-4,
        'seed'         : 42,
    },
}

# Sampling frequency (Hz) fallback if a validation .mat does not carry one.
FS_DEFAULT = 200.0

# Periodic-repetition settings.
N_REPEATS         = 10   # tile one period this many times
N_DISCARD_PERIODS = 1    # discard this many complete periods (settling)
N_EVAL_PERIODS    = 1    # steady-state periods used for the metrics
# Constraint: N_DISCARD_PERIODS + N_EVAL_PERIODS <= N_REPEATS


# Terminal logger
class Tee:
    """Write stdout to both terminal and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log      = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# Evaluation metrics
def fit_index(y_true, y_pred):
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true - np.mean(y_true))
    return 100.0 * (1.0 - num / den)


def rel_rmse(y_true, y_pred):
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.sqrt(np.mean((y_true - np.mean(y_true)) ** 2))
    return 100.0 * num / den


def abs_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def to_tensor(arr):
    return torch.tensor(arr).unsqueeze(0).unsqueeze(-1)


# DynoNet model
class DynoNetParallel(nn.Module):
    """
    N identical parallel branches followed by a mixing MLP and an output IIR
    filter.  Each branch is  G1 -> Fnl -> G2.

        n_branches    number of parallel branches (>= 1)
        n_states      states inside each G-block per branch
        g_out_bonus   extra states for the output filter G_out
        n_neurons     neurons in each branch Fnl (SisoStaticNonLinearity)
        n_neurons_mix neurons in Fnl_mix (MimoStaticNonLinearity, N -> 1)
    """
    def __init__(self,
                 n_branches,
                 n_states,
                 g_out_bonus,
                 n_neurons,
                 n_neurons_mix,
                 activation_branches,
                 activation_mix):
        super().__init__()

        if n_branches < 1:
            raise ValueError("n_branches must be >= 1")

        self.n_branches  = n_branches
        self.n_states    = n_states
        self.g_out_bonus = g_out_bonus
        self.n_out       = n_states + g_out_bonus

        if n_branches in (2, 3):
            self._explicit_branches = True
            for b in range(1, n_branches + 1):
                setattr(self, f'G1_b{b}',
                        SisoLinearDynamicalOperator(n_b=n_states, n_a=n_states, n_k=1))
                setattr(self, f'Fnl_b{b}',
                        SisoStaticNonLinearity(n_hidden=n_neurons,
                                               activation=activation_branches))
                setattr(self, f'G2_b{b}',
                        SisoLinearDynamicalOperator(n_b=n_states, n_a=n_states, n_k=0))
        else:
            self._explicit_branches = False
            self.G1 = nn.ModuleList([
                SisoLinearDynamicalOperator(n_b=n_states, n_a=n_states, n_k=1)
                for _ in range(n_branches)
            ])
            self.Fnl = nn.ModuleList([
                SisoStaticNonLinearity(n_hidden=n_neurons,
                                       activation=activation_branches)
                for _ in range(n_branches)
            ])
            self.G2 = nn.ModuleList([
                SisoLinearDynamicalOperator(n_b=n_states, n_a=n_states, n_k=0)
                for _ in range(n_branches)
            ])

        self.Fnl_mix = MimoStaticNonLinearity(
            in_channels  = n_branches,
            out_channels = 1,
            n_hidden     = n_neurons_mix,
            activation   = activation_mix,
        )
        self.G_out = SisoLinearDynamicalOperator(n_b=self.n_out,
                                                 n_a=self.n_out, n_k=0)

    def branches(self):
        if self._explicit_branches:
            return [(getattr(self, f'G1_b{b}'),
                     getattr(self, f'Fnl_b{b}'),
                     getattr(self, f'G2_b{b}'))
                    for b in range(1, self.n_branches + 1)]
        return list(zip(self.G1, self.Fnl, self.G2))

    def forward(self, u):
        outs = [G2(Fnl(G1(u))) for (G1, Fnl, G2) in self.branches()]
        x_mix = self.Fnl_mix(torch.cat(outs, dim=-1))
        return self.G_out(x_mix)


# Interactive prompt helpers
def _prompt(message, default, caster=str, choices=None):
    """Prompt the user with a default in brackets; press Enter to accept."""
    while True:
        raw = input(f"  {message} [{default}]: ").strip()
        if raw == '':
            return default
        try:
            value = caster(raw)
        except (ValueError, TypeError):
            print(f"    ! could not parse '{raw}' as {caster.__name__}, try again")
            continue
        if choices is not None and value not in choices:
            print(f"    ! must be one of {choices}")
            continue
        return value


def pick_version_and_config():
    """Ask which trained model to validate and with which hyper-parameters."""
    print("=" * 70)
    print("  DynoNet -- validation (z-score + monosine cases)")
    print("=" * 70)
    print()
    print("  Which trained model do you want to validate?")
    print("    V2      -- 2 parallel branches")
    print("    V4      -- 3 parallel branches")
    print("    custom  -- choose N branches and G_out bonus yourself")
    print()

    raw = input("  Version? [V2/V4/custom] (V4): ").strip()
    if raw == '':
        version_key = 'V4'
    else:
        version_key = raw.upper() if raw.lower() != 'custom' else 'custom'

    if version_key in ('V2', 'V4'):
        cfg = dict(VERSION_DEFAULTS[version_key])
        version_folder = f"{version_key}_masked"
    elif version_key == 'custom':
        cfg = dict(VERSION_DEFAULTS['V4'])
        cfg['n_branches']  = _prompt("Number of parallel branches", 3, int)
        cfg['g_out_bonus'] = _prompt("Extra G_out states (0=V4 conv, 2=V2 conv)",
                                     0, int)
        version_folder = f"V{cfg['n_branches']}_masked"
    else:
        raise SystemExit(f"Unknown version '{raw}'. Use V2, V4 or custom.")

    print()
    use_defaults = input("  Use the default hyper-parameters?  [Y/n]: ").strip().lower()
    if use_defaults in ('', 'y', 'yes'):
        print("  -> using defaults")
    else:
        print("  -> match the values the model was trained with")
        cfg['n_states']            = _prompt("n_states (per G-block)",
                                             cfg['n_states'], int)
        cfg['n_neurons']           = _prompt("n_neurons (per branch Fnl)",
                                             cfg['n_neurons'], int)
        cfg['n_neurons_mix']       = _prompt("n_neurons_mix (Fnl_mix N->1)",
                                             cfg['n_neurons_mix'], int)
        cfg['activation_branches'] = _prompt(
            "activation_branches",
            cfg['activation_branches'], str,
            choices={'relu', 'tanh', 'sigmoid', 'elu', 'leakyrelu'})
        cfg['activation_mix']      = _prompt(
            "activation_mix",
            cfg['activation_mix'], str,
            choices={'relu', 'tanh', 'sigmoid', 'elu', 'leakyrelu'})
        cfg['n_epochs']            = _prompt("n_epochs", cfg['n_epochs'], int)
        cfg['lr']                  = _prompt("learning rate", cfg['lr'], float)
        cfg['seed']                = _prompt("random seed", cfg['seed'], int)

    return version_key, version_folder, cfg


def build_run_name(cfg):
    """Run-folder name; must match the folder produced during training."""
    if cfg['n_branches'] == 2:
        return (f"training_s{cfg['n_states']}_n{cfg['n_neurons']}"
                f"_{cfg['activation_branches']}_ep{cfg['n_epochs']}"
                f"_trSmart_seed{cfg['seed']}_norm")
    return (f"training_s{cfg['n_states']}_n{cfg['n_neurons']}"
            f"_nmix{cfg['n_neurons_mix']}_{cfg['activation_branches']}"
            f"_ep{cfg['n_epochs']}_trSmart_seed{cfg['seed']}_norm")


def config_suffix(cfg):
    """Short tag describing the configuration, used for the output folders."""
    if cfg['n_branches'] == 2:
        return (f"s{cfg['n_states']}_n{cfg['n_neurons']}"
                f"_{cfg['activation_branches']}_trSmart_norm")
    return (f"s{cfg['n_states']}_n{cfg['n_neurons']}"
            f"_nmix{cfg['n_neurons_mix']}_{cfg['activation_branches']}"
            f"_trSmart_norm")


# Validation data
def load_validation_datasets():
    """Extract the monosine zip if needed and load every .mat case."""
    if not os.path.exists(VAL_FOLDER):
        print(f"\n  Extracting {VAL_ZIP} ...")
        with zipfile.ZipFile(VAL_ZIP, 'r') as z:
            z.extractall(os.path.dirname(VAL_ZIP))
        print("  Done")

    val_files = []
    for root, _dirs, files in os.walk(VAL_FOLDER):
        for f in files:
            if f.endswith('.mat'):
                val_files.append(os.path.join(root, f))
    val_files = sorted(val_files)

    if not val_files:
        raise SystemExit(f"No .mat files found in {VAL_FOLDER}")

    datasets = []
    for fpath in val_files:
        fname = os.path.basename(fpath)
        mat   = scipy.io.loadmat(fpath)
        alpha = mat['alpha'].squeeze().astype(np.float32)
        cl    = mat['cl'].squeeze().astype(np.float32)

        fs = None
        for key in ('fs', 'Fs', 'fsampling', 'FsTrain'):
            if key in mat:
                fs = float(np.array(mat[key]).flatten()[0])
                break
        if fs is None:
            fs = FS_DEFAULT

        label = fname.replace('CFD_Monosine_', '').replace('_Downsampled.mat', '')
        datasets.append({'filename': fname, 'label': label,
                         'alpha': alpha, 'cl': cl, 'fs': fs})

    return datasets


# Periodic-repetition evaluation
def run_periodic_validation(model, alpha_one_period, cl_ref, stats,
                            n_repeats=N_REPEATS):
    """
    Tile one period, normalise with the training statistics, run the model,
    denormalise, then discard the settling periods and keep the steady state.
    """
    T = len(alpha_one_period)
    discard  = N_DISCARD_PERIODS * T
    eval_len = N_EVAL_PERIODS * T

    assert N_DISCARD_PERIODS + N_EVAL_PERIODS <= n_repeats, (
        f"N_DISCARD_PERIODS ({N_DISCARD_PERIODS}) + N_EVAL_PERIODS "
        f"({N_EVAL_PERIODS}) must be <= N_REPEATS ({n_repeats})")

    alpha_tiled = np.tile(alpha_one_period, n_repeats).astype(np.float32)

    # normalise the input with the training statistics (never recomputed)
    alpha_norm = ((alpha_tiled - stats['alpha_mean']) / stats['alpha_std']).astype(np.float32)

    with torch.no_grad():
        cl_norm_pred = model(to_tensor(alpha_norm)).squeeze().numpy()

    # back to physical Cl
    cl_tiled_pred = cl_norm_pred * stats['cl_std'] + stats['cl_mean']

    alpha_steady   = alpha_tiled  [discard:discard + eval_len]
    cl_pred_steady = cl_tiled_pred[discard:discard + eval_len]
    cl_ref_tiled   = np.tile(cl_ref, N_EVAL_PERIODS)

    return {
        'T'              : T,
        'discard'        : discard,
        'alpha_steady'   : alpha_steady,
        'cl_pred_steady' : cl_pred_steady,
        'cl_ref'         : cl_ref_tiled,
        'cl_ref_one'     : cl_ref,
        'alpha_full'     : alpha_tiled,
        'cl_pred_full'   : cl_tiled_pred,
    }


# Plots
def plot_time(res, fs, version_key, out_path):
    T = res['T']
    time_axis = np.arange(N_EVAL_PERIODS * T) / fs

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time_axis, res['alpha_steady'], color=COLORS['aoa'], linewidth=1.0)
    axes[0].set_ylabel('AoA (deg)')

    axes[1].plot(time_axis, res['cl_ref'], color=COLORS['cfd'],
                 label='CFD', linewidth=1.4)
    axes[1].plot(time_axis, res['cl_pred_steady'], color=COLORS['model'],
                 label=f'DynoNet {version_key}', linewidth=1.2,
                 linestyle='--', dashes=(5, 2))
    axes[1].set_ylabel('$C_l$ (-)')
    axes[1].legend(frameon=False, loc='upper right')

    error = res['cl_ref'] - res['cl_pred_steady']
    axes[2].plot(time_axis, error, color=COLORS['error'], linewidth=1.0)
    axes[2].axhline(0, color='#555555', linewidth=0.7)
    axes[2].set_ylabel('Error (-)')
    axes[2].set_xlabel('Time (s)')

    for ax in axes:
        ax.tick_params(direction='out', length=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_hysteresis(res, version_key, out_path):
    T = res['T']
    alpha_one    = res['alpha_steady'][:T]
    cl_ref_one   = res['cl_ref_one']
    cl_pred_one  = res['cl_pred_steady'][:T]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(alpha_one, cl_ref_one, color=COLORS['cfd'],
            label='CFD', linewidth=1.4)
    ax.plot(alpha_one, cl_pred_one, color=COLORS['model'],
            label=f'DynoNet {version_key}', linewidth=1.2,
            linestyle='--', dashes=(5, 2))
    ax.set_xlabel(r'Angle of attack (deg)')
    ax.set_ylabel('$C_l$ (-)')
    ax.legend(frameon=False)
    ax.tick_params(direction='out', length=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_settling(res, fs, version_key, out_path):
    T         = res['T']
    discard   = res['discard']
    full_axis = np.arange(len(res['alpha_full'])) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(full_axis, res['alpha_full'], color=COLORS['aoa'], linewidth=0.8)
    axes[0].axvspan(0, discard / fs, color=COLORS['discard'], alpha=0.18,
                    label=f'discarded ({N_DISCARD_PERIODS} x T)')
    axes[0].axvspan(discard / fs, (discard + N_EVAL_PERIODS * T) / fs,
                    color=COLORS['steady'], alpha=0.18,
                    label=f'evaluated period (T = {T})')
    axes[0].set_ylabel('AoA (deg)')
    axes[0].legend(frameon=False, loc='upper right', fontsize=9)
    axes[0].tick_params(direction='out', length=3)

    axes[1].plot(full_axis, res['cl_pred_full'], color=COLORS['model'],
                 linewidth=0.8, label=f'DynoNet {version_key}')
    axes[1].axvspan(0, discard / fs, color=COLORS['discard'], alpha=0.18)
    axes[1].axvspan(discard / fs, (discard + N_EVAL_PERIODS * T) / fs,
                    color=COLORS['steady'], alpha=0.18)
    axes[1].set_ylabel('$C_l$ (-)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(frameon=False, loc='upper right', fontsize=9)
    axes[1].tick_params(direction='out', length=3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# Main
def main():
    version_key, version_folder, cfg = pick_version_and_config()

    n_branches    = cfg['n_branches']
    n_states      = cfg['n_states']
    g_out_bonus   = cfg['g_out_bonus']
    n_neurons     = cfg['n_neurons']
    n_neurons_mix = cfg['n_neurons_mix']
    act_branches  = cfg['activation_branches']
    act_mix       = cfg['activation_mix']

    version_dir = os.path.join(BASE_DIR, version_folder)
    run_name    = build_run_name(cfg)
    train_dir   = os.path.join(version_dir, run_name)
    suffix      = config_suffix(cfg)

    weights_path    = os.path.join(train_dir, 'model_weights.pt')
    norm_stats_path = os.path.join(train_dir, 'norm_stats.json')

    os.makedirs(version_dir, exist_ok=True)
    log_path = os.path.join(version_dir, f'terminal_log_validate_{suffix}.txt')
    logger     = Tee(log_path)
    sys.stdout = logger

    print("=" * 70)
    print(f"  DynoNet {version_key} -- VALIDATION (periodic repetition, z-score)")
    print(f"  Method: tile x{N_REPEATS}, discard {N_DISCARD_PERIODS} full period(s), "
          f"evaluate {N_EVAL_PERIODS}")
    print("=" * 70)

    if not os.path.exists(weights_path):
        print(f"\n  ERROR: model weights not found at:\n  {weights_path}")
        print("  Train this configuration first, and check that the version and")
        print("  hyper-parameters match the training run.")
        sys.stdout = logger.terminal
        logger.close()
        sys.exit(1)

    if not os.path.exists(norm_stats_path):
        print(f"\n  ERROR: norm_stats.json not found at:\n  {norm_stats_path}")
        sys.stdout = logger.terminal
        logger.close()
        sys.exit(1)

    # Normalisation statistics
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    stats = {
        'alpha_mean': float(norm_stats['alpha_mean']),
        'alpha_std' : float(norm_stats['alpha_std']),
        'cl_mean'   : float(norm_stats['cl_mean']),
        'cl_std'    : float(norm_stats['cl_std']),
    }
    print(f"\n  Loaded normalisation statistics from:\n  {norm_stats_path}")
    print(f"    alpha : mean = {stats['alpha_mean']:+.4f},  std = {stats['alpha_std']:.4f}")
    print(f"    cl    : mean = {stats['cl_mean']:+.4f},  std = {stats['cl_std']:.4f}")

    if 'n_branches' in norm_stats and int(norm_stats['n_branches']) != n_branches:
        print(f"  ! WARNING: norm_stats reports n_branches="
              f"{norm_stats['n_branches']} but you selected {n_branches}.")

    # Rebuild the model and load the weights
    model = DynoNetParallel(
        n_branches          = n_branches,
        n_states            = n_states,
        g_out_bonus         = g_out_bonus,
        n_neurons           = n_neurons,
        n_neurons_mix       = n_neurons_mix,
        activation_branches = act_branches,
        activation_mix      = act_mix,
    )
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Loaded model weights from:\n  {weights_path}")
    print(f"    Branches         : {n_branches}")
    print(f"    States/branch    : {n_states}  |  G_out states: {n_states + g_out_bonus}")
    print(f"    Neurons/branch   : {n_neurons}  |  Fnl_mix neurons: {n_neurons_mix}")
    print(f"    Activations      : branches {act_branches}, mix {act_mix}")
    print(f"    Trainable params : {n_params}")

    # Validation data
    print(f"\n  Searching for validation files in: {VAL_FOLDER}")
    datasets = load_validation_datasets()
    print(f"  Found {len(datasets)} file(s):")
    for vd in datasets:
        print(f"    {vd['filename']}")

    overview_csv = os.path.join(version_dir, f'overview_validation_{suffix}.csv')
    with open(overview_csv, 'w', newline='') as f:
        csv.writer(f).writerow([
            'val_case', 'T_samples', 'discard_samples',
            'aoa_min', 'aoa_max', 'fit_%', 'rel_rmse_%', 'abs_error'])

    # Evaluate every case
    print(f"\n{'='*70}")
    print("  RUNNING VALIDATION")
    print(f"{'='*70}")

    results = []
    for i, vd in enumerate(datasets, start=1):
        label   = vd['label']
        alpha_v = vd['alpha']
        cl_v    = vd['cl']
        fs_v    = vd['fs']
        dt_v    = 1.0 / fs_v
        T       = len(alpha_v)
        discard_n = N_DISCARD_PERIODS * T

        print(f"\n  [{i}/{len(datasets)}] {label}")
        print(f"    Sampling fs     : {fs_v:.2f} Hz  (period {T*dt_v:.3f} s)")
        print(f"    T (one period)  : {T} samples")
        print(f"    Tiled length    : {T * N_REPEATS} samples ({N_REPEATS}x)")
        print(f"    Discard         : {discard_n} samples "
              f"({N_DISCARD_PERIODS} complete period(s))")

        res  = run_periodic_validation(model, alpha_v, cl_v, stats)
        fit  = fit_index(res['cl_ref'], res['cl_pred_steady'])
        rmse = rel_rmse(res['cl_ref'],  res['cl_pred_steady'])
        mae  = abs_error(res['cl_ref'], res['cl_pred_steady'])

        print(f"    Fit  (steady)   : {fit:.2f}%")
        print(f"    Rel. RMSE       : {rmse:.2f}%")
        print(f"    Abs. error      : {mae:.4f}")

        safe_label = label.replace('.', 'p').replace(' ', '_')
        val_dir    = os.path.join(version_dir, f"val_{safe_label}_{suffix}")
        os.makedirs(val_dir, exist_ok=True)

        plot_time(res, fs_v, version_key, os.path.join(val_dir, 'time_plot.pdf'))
        plot_hysteresis(res, version_key, os.path.join(val_dir, 'hysteresis_plot.pdf'))
        plot_settling(res, fs_v, version_key, os.path.join(val_dir, 'settling_plot.pdf'))

        with open(os.path.join(val_dir, 'results.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['parameter',          'value'])
            w.writerow(['label',               label])
            w.writerow(['filename',            vd['filename']])
            w.writerow(['version',             version_key])
            w.writerow(['T_samples',           T])
            w.writerow(['n_repeats',           N_REPEATS])
            w.writerow(['tiled_samples',       T * N_REPEATS])
            w.writerow(['discard_samples',     discard_n])
            w.writerow(['n_discard_periods',   N_DISCARD_PERIODS])
            w.writerow(['n_eval_periods',      N_EVAL_PERIODS])
            w.writerow(['aoa_min',             f"{alpha_v.min():.1f}"])
            w.writerow(['aoa_max',             f"{alpha_v.max():.1f}"])
            w.writerow(['fit_%',               f"{fit:.4f}"])
            w.writerow(['rel_rmse_%',          f"{rmse:.4f}"])
            w.writerow(['abs_error',           f"{mae:.6f}"])
            w.writerow(['model_states',        n_states])
            w.writerow(['model_g_out_states',  n_states + g_out_bonus])
            w.writerow(['model_neurons',       n_neurons])
            w.writerow(['activation',          act_branches])
            w.writerow(['weights_file',        weights_path])

        with open(overview_csv, 'a', newline='') as f:
            csv.writer(f).writerow([
                label, T, discard_n,
                f"{alpha_v.min():.1f}", f"{alpha_v.max():.1f}",
                f"{fit:.2f}", f"{rmse:.2f}", f"{mae:.4f}"])

        print(f"    Saved to: {val_dir}")
        results.append({'label': label, 'T': T,
                        'fit': fit, 'rmse': rmse, 'mae': mae})

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY  (steady-state period, transient removed)")
    print(f"{'='*70}")
    print(f"  {'Case':<38} {'T':>5} {'Fit [%]':>8} {'RMSE [%]':>10} {'MAE':>8}")
    print(f"  {'-'*71}")
    for r in results:
        print(f"  {r['label']:<38} {r['T']:>5}"
              f" {r['fit']:>8.2f} {r['rmse']:>10.2f} {r['mae']:>8.4f}")
    if results:
        mean_fit = np.mean([r['fit'] for r in results])
        print(f"  {'-'*71}")
        print(f"  {'mean':<38} {'':>5} {mean_fit:>8.2f}")
    print(f"\n  Overview saved : {overview_csv}")
    print(f"  Terminal log   : {log_path}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()
