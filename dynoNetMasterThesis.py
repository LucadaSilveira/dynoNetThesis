"""


SUPPORTED VERSIONS
------------------
  V2       : 2 identical parallel branches
             G_out  : n_b = n_a = n_states   (legacy V2 convention)
             Fnl_mix: 2 -> 1
             Output : ExperimentValidation\\V2_masked\\training_...

  V4       : 3 identical parallel branches
             G_out  : n_b = n_a = n_states       (legacy V4 convention)
             Fnl_mix: 3 -> 1
             Output : ExperimentValidation\\V4_masked\\training_...

  custom   : N identical parallel branches (you choose N and the
             number of extra G_out states).  Output goes into
             ExperimentValidation\\V{N}_masked\\... so it does not
             collide with V2 / V4.


OUTPUT FOLDER
-------------
ExperimentValidation\\V{2|4|N}_masked\\
    training_s{N}_n{N}[_nmix{N}]_{act}_ep{N}_trSmart_seed{N}_norm\\
        model_weights.pt
        norm_stats.json
        time_plot.pdf
        loss_curve.pdf
        normalised_inputs.pdf
        results.csv
        architecture.txt
    terminal_log_train.txt
"""

import csv
import json
import os
import sys
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Where to find the dynonet source.
sys.path.insert(0, r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\dynoNet\dynonet\src")

from dynonet.lti    import SisoLinearDynamicalOperator
from dynonet.static import SisoStaticNonLinearity, MimoStaticNonLinearity



#  Plot style (identical across train scripts)
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
    'cfd'      : '#C0392B',
    'model'    : '#2980B9',
    'error'    : '#27AE60',
    'aoa'      : '#2C3E50',
    'loss'     : '#1A252F',
    'transient': '#F39C12',
}



#  PATHS  (only thing that may need editing on a new machine)
TRAIN_PATH = r"C:\Users\lucad\Downloads\DatasetCFD.mat"
BASE_DIR   = r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\Masterproef\dynoNet\ExperimentValidation"


#  DEFAULTS  (per-version sensible starting points)

# Each entry: hyper-parameters that the user can override at the prompt.
# n_branches and g_out_bonus define the architecture topology.
#   g_out_bonus = 2  -> G_out has n_states states  (V2 legacy convention)
#   g_out_bonus = 0  -> G_out has n_states states      (V4 legacy convention)
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

# Sampling frequency (Hz) fallback if the .mat does not carry one
FS_DEFAULT = 200.0

# Transient masking -- boundary-specific, identical to the dedicated scripts.
PER_SWEEP = 4001     # 32008 / 8
SWEEP_MASK = [
    (0,           50),   # sweep 1: start of data
    (4001  - 25,   0),   # sweep 2: amplitude change only
    (8002  - 100, 200),  # sweep 3: mean AoA jump +5 deg
    (12003 - 25,   0),   # sweep 4: amplitude change only
    (16004 - 100, 200),  # sweep 5: mean AoA jump +4 deg
    (20005 - 25,   0),   # sweep 6: amplitude change only
    (24006 - 100, 200),  # sweep 7: mean AoA jump +4 deg
    (28007 - 25,   0),   # sweep 8: amplitude change only
]
N_SWEEPS  = len(SWEEP_MASK)
TRANSIENT = max(t for _, t in SWEEP_MASK)
SWEEP_TYPES = [
    'start of data',  'amplitude only', 'mean AoA +5 deg',
    'amplitude only', 'mean AoA +4 deg', 'amplitude only',
    'mean AoA +4 deg','amplitude only',
]



#  Tee logger
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


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"



#  Masking + metrics  
def build_valid_mask(n_total, sweep_mask, per_sweep):
    """Boolean mask: True on samples that should contribute to the loss."""
    mask = np.zeros(n_total, dtype=bool)
    for sweep_start, transient in sweep_mask:
        valid_start = sweep_start + transient
        valid_end   = sweep_start + per_sweep
        mask[valid_start:valid_end] = True
    return mask


def masked_mse_loss(y_pred, y_true, mask_tensor):
    """MSE restricted to valid samples; gradient only flows through them."""
    y_pred_flat  = y_pred.squeeze()
    y_true_flat  = y_true.squeeze()
    y_pred_valid = y_pred_flat[mask_tensor]
    y_true_valid = y_true_flat[mask_tensor]
    return nn.MSELoss()(y_pred_valid, y_true_valid)


def fit_index_masked(y_true_np, y_pred_np, mask_np):
    y_t = y_true_np[mask_np]
    y_p = y_pred_np[mask_np]
    num = np.linalg.norm(y_t - y_p)
    den = np.linalg.norm(y_t - np.mean(y_t))
    return 100.0 * (1.0 - num / den)


def fit_index_full(y_true_np, y_pred_np):
    num = np.linalg.norm(y_true_np - y_pred_np)
    den = np.linalg.norm(y_true_np - np.mean(y_true_np))
    return 100.0 * (1.0 - num / den)


def rel_rmse_masked(y_true_np, y_pred_np, mask_np):
    y_t = y_true_np[mask_np]
    y_p = y_pred_np[mask_np]
    num = np.sqrt(np.mean((y_t - y_p) ** 2))
    den = np.sqrt(np.mean((y_t - np.mean(y_t)) ** 2))
    return 100.0 * num / den


def to_tensor(arr):
    return torch.tensor(arr).unsqueeze(0).unsqueeze(-1)



#  Generic parallel-branch DynoNet
class DynoNetParallel(nn.Module):
    """
    N identical parallel branches followed by a mixing MLP and an
    output IIR filter.  Each branch is  G1 -> Fnl -> G2.

    Topology controlled by:
        n_branches      number of parallel branches (>= 1)
        n_states        states inside each G-block per branch
        g_out_bonus     extra states for the output filter G_out:
                          0 -> G_out has n_states  (V4 convention)
                          2 -> G_out has n_states+2 (V2 convention)
                          any non-negative int is accepted
        n_neurons       neurons in each branch Fnl (SisoStaticNonLinearity)
        n_neurons_mix   neurons in Fnl_mix (MimoStaticNonLinearity, N -> 1)

    
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
            # Use the same attribute names as the dedicated scripts so
            # that state_dicts are cross-compatible.
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

        # Mixing static non-linearity (N -> 1)
        self.Fnl_mix = MimoStaticNonLinearity(
            in_channels  = n_branches,
            out_channels = 1,
            n_hidden     = n_neurons_mix,
            activation   = activation_mix,
        )

        # Output IIR filter
        self.G_out = SisoLinearDynamicalOperator(n_b=self.n_out,
                                                 n_a=self.n_out, n_k=0)

    # convenience: list of (G1, Fnl, G2) tuples regardless of storage
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



#  Interactive prompt helpers

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
    """Asks the user which version to train and which hyper-params to use."""
    print("=" * 70)
    print("  DynoNet -- unified training (z-score + masked transients)")
    print("=" * 70)
    print()
    print("  Available versions:")
    print("    V2      -- 2 parallel branches  (G_out states = n_states )")
    print("    V4      -- 3 parallel branches  (G_out states = n_states)")
    print("    custom  -- choose N branches and G_out bonus yourself")
    print()

    raw = input("  Which version do you want to train? [V2/V4/custom] (V4): ").strip()
    if raw == '':
        version_key = 'V4'
    else:
        version_key = raw.upper() if raw.lower() != 'custom' else 'custom'

    if version_key in ('V2', 'V4'):
        cfg = dict(VERSION_DEFAULTS[version_key])
        version_folder = f"{version_key}_masked"
    elif version_key == 'custom':
        cfg = dict(VERSION_DEFAULTS['V4'])    # use V4 as base
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
        print("  -> override each value (press Enter to keep the default)")
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
    """Run-folder name -- identical convention to the dedicated scripts."""
    if cfg['n_branches'] == 2:
        # V2 convention: no nmix in folder name
        return (f"training_s{cfg['n_states']}_n{cfg['n_neurons']}"
                f"_{cfg['activation_branches']}_ep{cfg['n_epochs']}"
                f"_trSmart_seed{cfg['seed']}_norm")
    return (f"training_s{cfg['n_states']}_n{cfg['n_neurons']}"
            f"_nmix{cfg['n_neurons_mix']}_{cfg['activation_branches']}"
            f"_ep{cfg['n_epochs']}_trSmart_seed{cfg['seed']}_norm")



#  Main
def main():
    version_key, version_folder, cfg = pick_version_and_config()

    n_branches    = cfg['n_branches']
    n_states      = cfg['n_states']
    g_out_bonus   = cfg['g_out_bonus']
    n_neurons     = cfg['n_neurons']
    n_neurons_mix = cfg['n_neurons_mix']
    act_branches  = cfg['activation_branches']
    act_mix       = cfg['activation_mix']
    n_epochs      = cfg['n_epochs']
    lr            = cfg['lr']
    seed          = cfg['seed']

    version_dir = os.path.join(BASE_DIR, version_folder)
    run_name    = build_run_name(cfg)
    train_dir   = os.path.join(version_dir, run_name)

    os.makedirs(version_dir, exist_ok=True)
    os.makedirs(train_dir,   exist_ok=True)

    log_path   = os.path.join(version_dir, 'terminal_log_train.txt')
    logger     = Tee(log_path)
    sys.stdout = logger

    print()
    print("=" * 70)
    print(f"  Training {version_key}: {n_branches} parallel branches "
          f"(masked + z-score)")
    print("=" * 70)

    # --- Build & report mask ------------------------------------------------
    n_total  = N_SWEEPS * PER_SWEEP            # 32008
    mask_np  = build_valid_mask(n_total, SWEEP_MASK, PER_SWEEP)
    n_valid  = int(mask_np.sum())
    n_masked = n_total - n_valid

    print(f"\nTransient mask")
    print(f"  Sweeps         : {N_SWEEPS}")
    print(f"  Samples/sweep  : {PER_SWEEP}")
    print(f"  Transient (max): {TRANSIENT} samples (per-sweep value varies)")
    print(f"  Valid samples  : {n_valid} / {n_total}  "
          f"({100*n_valid/n_total:.1f}% retained)")
    print(f"  Masked samples : {n_masked}  "
          f"({100*n_masked/n_total:.1f}% excluded from loss)")
    print(f"\n  Boundary-specific transient mask:")
    print(f"    {'Sweep':>5}  {'Start':>6}  {'Transient':>10}  "
          f"{'Valid from':>10}  {'Type'}")
    print(f"    {'-'*55}")
    for i, (s, tr) in enumerate(SWEEP_MASK):
        print(f"    {i+1:>5}  {s:>6}  {tr:>10}s  {s+tr:>10}  {SWEEP_TYPES[i]}")

    # --- Load training data -------------------------------------------------
    print(f"\nLoading training data from\n  {TRAIN_PATH}")
    mat   = scipy.io.loadmat(TRAIN_PATH)
    alpha = mat['uTrain'].squeeze().astype(np.float32)
    cl    = mat['yTrain'].squeeze().astype(np.float32)

    fs = None
    for key in ('fsTrain', 'fs', 'FsTrain', 'Fs'):
        if key in mat:
            fs = float(np.array(mat[key]).flatten()[0])
            break
    if fs is None:
        fs = FS_DEFAULT
    dt = 1.0 / fs

    assert len(alpha) == n_total, (
        f"Expected {n_total} samples, got {len(alpha)}. "
        "Check N_SWEEPS / PER_SWEEP.")

    print(f"  Time steps : {len(alpha)}  (= {N_SWEEPS} x {PER_SWEEP})")
    print(f"  Sampling   : fs = {fs:.2f} Hz  (dt = {dt*1000:.3f} ms)")
    print(f"  Duration   : {len(alpha)*dt:.2f} s "
          f"({len(alpha)*dt/N_SWEEPS:.2f} s/sweep)")
    print(f"  AoA range  : [{alpha.min():.1f}, {alpha.max():.1f}] deg")
    print(f"  Cl  range  : [{cl.min():.3f}, {cl.max():.3f}]")

    # --- Z-score normalisation ---------------------------------------------
    alpha_mean = float(alpha.mean())
    alpha_std  = float(alpha.std())
    cl_mean    = float(cl.mean())
    cl_std     = float(cl.std())

    assert alpha_std > 1e-8, "Input AoA has near-zero variance"
    assert cl_std    > 1e-8, "Output Cl has near-zero variance"

    alpha_norm = ((alpha - alpha_mean) / alpha_std).astype(np.float32)
    cl_norm    = ((cl    - cl_mean   ) / cl_std   ).astype(np.float32)

    print(f"\n  Z-score normalisation:")
    print(f"    alpha : mean = {alpha_mean:+.4f},  std = {alpha_std:.4f}  "
          f"-> [{alpha_norm.min():+.2f}, {alpha_norm.max():+.2f}]")
    print(f"    cl    : mean = {cl_mean:+.4f},  std = {cl_std:.4f}  "
          f"-> [{cl_norm.min():+.2f}, {cl_norm.max():+.2f}]")

    # Quick visual sanity check of the normalised inputs
    fig_norm, (ax_alpha, ax_cl) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_axis_norm = np.arange(len(alpha_norm)) / fs

    ax_alpha.plot(t_axis_norm, alpha_norm, color=COLORS['aoa'], alpha=0.85,
                  linewidth=1.2, label='alpha (norm)')
    for level in (-1.0, 0.0, 1.0):
        ax_alpha.axhline(level, color='#aaaaaa', linewidth=0.8,
                         linestyle='-' if level == 0.0 else '--')
    ax_alpha.set_ylabel('Norm. alpha')
    ax_alpha.set_title('Z-Score Normalized Training Data (mu=0, sigma=1)')
    ax_alpha.legend(frameon=False, loc='upper right')

    ax_cl.plot(t_axis_norm, cl_norm, color=COLORS['cfd'], alpha=0.85,
               linewidth=1.2, label='cl (norm)')
    for level in (-1.0, 0.0, 1.0):
        ax_cl.axhline(level, color='#aaaaaa', linewidth=0.8,
                      linestyle='-' if level == 0.0 else '--')
    ax_cl.set_ylabel('Norm. cl')
    ax_cl.set_xlabel('Time (s)')
    ax_cl.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'normalised_inputs.pdf'))
    plt.close(fig_norm)

    # --- Build tensors and model -------------------------------------------
    u_all      = to_tensor(alpha_norm)
    y_all      = to_tensor(cl_norm)
    mask_torch = torch.tensor(mask_np)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DynoNetParallel(
        n_branches          = n_branches,
        n_states            = n_states,
        g_out_bonus         = g_out_bonus,
        n_neurons           = n_neurons,
        n_neurons_mix       = n_neurons_mix,
        activation_branches = act_branches,
        activation_mix      = act_mix,
    )
    n_params       = sum(p.numel() for p in model.parameters())
    n_total_states = n_branches * 2 * n_states + (n_states + g_out_bonus)

    print(f"\n{'='*70}")
    print(f"  Architecture          : {n_branches} identical parallel branches")
    print(f"  States/branch G-block : {n_states}   "
          f"(G1 + G2 per branch = {2*n_states})")
    print(f"  G_out states          : {n_states + g_out_bonus}  "
          f"(bonus = {g_out_bonus})")
    print(f"  Total states          : {n_total_states}  "
          f"({n_branches} x 2 x {n_states} + {n_states + g_out_bonus})")
    print(f"  Neurons per branch    : {n_neurons}")
    print(f"  Neurons in Fnl_mix    : {n_neurons_mix}  "
          f"({n_branches} in -> 1 out)")
    print(f"  Activation branches   : {act_branches}")
    print(f"  Activation mix        : {act_mix}")
    print(f"  Epochs                : {n_epochs}")
    print(f"  Learning rate         : {lr:.0e}")
    print(f"  Trainable params      : {n_params}")
    print(f"  Seed                  : {seed}")
    print(f"  Loss computed on      : valid samples only (transients excluded)")
    print(f"  Input/output scale    : z-score (mean 0, std 1)")
    print(f"{'='*70}\n")

    # Per-component parameter count
    def numel(mod):
        return sum(p.numel() for p in mod.parameters())
    for b_idx, (G1, Fnl, G2) in enumerate(model.branches(), start=1):
        print(f"    G1_b{b_idx}   : {numel(G1):>5d} parameters")
        print(f"    Fnl_b{b_idx}  : {numel(Fnl):>5d} parameters")
        print(f"    G2_b{b_idx}   : {numel(G2):>5d} parameters")
    print(f"    Fnl_mix : {numel(model.Fnl_mix):>5d} parameters")
    print(f"    G_out   : {numel(model.G_out):>5d} parameters")
    print()

    # --- Training loop ------------------------------------------------------
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    t_start      = time.time()

    print("  Training started...\n")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(u_all)
        loss   = masked_mse_loss(y_pred, y_all, mask_torch)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 2500 == 0:
            elapsed   = time.time() - t_start
            remaining = (elapsed / (epoch + 1)) * (n_epochs - epoch - 1)
            print(f"  Epoch {epoch:6d}/{n_epochs}"
                  f" | Loss (masked): {loss.item():.6f}"
                  f" | Elapsed: {format_duration(elapsed)}"
                  f" | ETA: {format_duration(remaining)}")

    duration = time.time() - t_start

    # --- Evaluation: denormalise back to physical Cl ------------------------
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(u_all).squeeze().numpy()
    y_pred_np = y_pred_norm * cl_std + cl_mean

    fit_masked  = fit_index_masked(cl, y_pred_np, mask_np)
    fit_full    = fit_index_full(cl, y_pred_np)
    rmse_masked = rel_rmse_masked(cl, y_pred_np, mask_np)

    print(f"\n  Training complete")
    print(f"  Duration              : {format_duration(duration)}")
    print(f"  Fit (masked, valid)   : {fit_masked:.2f}%  "
          f"<-- comparable to literature")
    print(f"  Fit (full, all steps) : {fit_full:.2f}%   "
          f"<-- includes transients")
    print(f"  Rel. RMSE (masked)    : {rmse_masked:.2f}%")
    print(f"\n  The gap between masked and full fit index shows how much")
    print(f"  the transients were dragging down the reported accuracy.")

    # --- Save artefacts -----------------------------------------------------
    weights_path = os.path.join(train_dir, 'model_weights.pt')
    torch.save(model.state_dict(), weights_path)
    print(f"\n  Model weights saved : {weights_path}")

    norm_stats = {
        'alpha_mean'    : alpha_mean,
        'alpha_std'     : alpha_std,
        'cl_mean'       : cl_mean,
        'cl_std'        : cl_std,
        'normalization' : 'z-score',
        'computed_on'   : 'full training set (32008 samples, transients included)',
        'training_run'  : run_name,
        'version'       : version_key,
        'n_branches'    : n_branches,
        'g_out_bonus'   : g_out_bonus,
    }
    norm_stats_path = os.path.join(train_dir, 'norm_stats.json')
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Norm stats saved    : {norm_stats_path}")
    print(f"  -> validate.py MUST load these to normalise input + denormalise output.")

    # --- Time-domain plot (physical units, shaded transients) --------------
    time_axis = np.arange(n_total) / fs
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(time_axis, alpha, color=COLORS['aoa'], linewidth=1.0)
    for s, tr in SWEEP_MASK:
        axes[0].axvspan(s/fs, (s+tr)/fs, color=COLORS['transient'], alpha=0.20)
    axes[0].set_ylabel('AoA (deg)')

    axes[1].plot(time_axis, cl,        color=COLORS['cfd'],
                 label='CFD', linewidth=1.4)
    axes[1].plot(time_axis, y_pred_np, color=COLORS['model'],
                 label=f'DynoNet {version_key}',
                 linewidth=1.2, linestyle='--', dashes=(5, 2))
    for s, tr in SWEEP_MASK:
        axes[1].axvspan(s/fs, (s+tr)/fs, color=COLORS['transient'], alpha=0.20)
    axes[1].set_ylabel('$C_l$ (-)')
    axes[1].legend(frameon=False, loc='upper right')

    error = cl - y_pred_np
    axes[2].plot(time_axis, error, color=COLORS['error'], linewidth=1.0)
    axes[2].axhline(0, color='#555555', linewidth=0.7)
    for idx, (s, tr) in enumerate(SWEEP_MASK):
        axes[2].axvspan(s/fs, (s+tr)/fs, color=COLORS['transient'],
                        alpha=0.20, label='masked' if idx == 0 else '_')
    axes[2].set_ylabel('Error (-)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(frameon=False, loc='upper right')

    for ax in axes:
        ax.tick_params(direction='out', length=3)
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'time_plot.pdf'))
    plt.close(fig)

    # --- Loss curve --------------------------------------------------------
    fig2, ax1 = plt.subplots(figsize=(10, 4))
    ax1.semilogy(loss_history, color=COLORS['loss'], linewidth=1.2,
                 label='MSE loss (masked, z-score space)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE loss')
    ax1.tick_params(axis='y', direction='out', length=3)
    ax1.tick_params(axis='x', direction='out', length=3)
    ax1.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, 'loss_curve.pdf'))
    plt.close(fig2)

    # --- results.csv -------------------------------------------------------
    with open(os.path.join(train_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['parameter',           'value'])
        w.writerow(['version',              version_key])
        w.writerow(['architecture',
                    f"{n_branches} parallel identical branches + masked + z-score"])
        w.writerow(['transient_masking',   'enabled (boundary-specific)'])
        w.writerow(['n_sweeps',             N_SWEEPS])
        w.writerow(['samples_per_sweep',    PER_SWEEP])
        w.writerow(['transient_max',        TRANSIENT])
        w.writerow(['valid_samples',        n_valid])
        w.writerow(['masked_samples',       n_masked])
        w.writerow(['n_branches',           n_branches])
        w.writerow(['n_states',             n_states])
        w.writerow(['g_out_bonus',          g_out_bonus])
        w.writerow(['g_out_states',         n_states + g_out_bonus])
        w.writerow(['total_states',         n_total_states])
        w.writerow(['n_neurons',            n_neurons])
        w.writerow(['n_neurons_mix',        n_neurons_mix])
        w.writerow(['activation_branches',  act_branches])
        w.writerow(['activation_mix',       act_mix])
        w.writerow(['n_epochs',             n_epochs])
        w.writerow(['n_params',             n_params])
        w.writerow(['fit_masked_%',         f"{fit_masked:.4f}"])
        w.writerow(['fit_full_%',           f"{fit_full:.4f}"])
        w.writerow(['rmse_masked_%',        f"{rmse_masked:.4f}"])
        w.writerow(['learning_rate',        f"{lr:.2e}"])
        w.writerow(['normalization',        'z-score (input + output)'])
        w.writerow(['alpha_mean',           f"{alpha_mean:.6f}"])
        w.writerow(['alpha_std',            f"{alpha_std:.6f}"])
        w.writerow(['cl_mean',              f"{cl_mean:.6f}"])
        w.writerow(['cl_std',               f"{cl_std:.6f}"])
        w.writerow(['duration',             format_duration(duration)])
        w.writerow(['weights_file',         weights_path])
        w.writerow(['norm_stats_file',      norm_stats_path])

    # --- architecture.txt --------------------------------------------------
    with open(os.path.join(train_dir, 'architecture.txt'), 'w') as f:
        f.write(f"DynoNet {version_key} -- {n_branches} parallel branches "
                "(transient masking + z-score)\n")
        f.write("=" * 70 + "\n\n")

        f.write("TRANSIENT MASKING\n")
        f.write(f"  Sweeps           : {N_SWEEPS}\n")
        f.write(f"  Samples/sweep    : {PER_SWEEP}\n")
        f.write(f"  Transient type   : boundary-specific\n")
        for i, (s, tr) in enumerate(SWEEP_MASK):
            f.write(f"  Sweep {i+1}: start={s:5d}  transient={tr:3d} samples\n")
        f.write(f"  Valid samples    : {n_valid} / {n_total}\n")
        f.write(f"  Loss computed on : valid samples only\n")
        f.write(f"  Fit computed on  : valid samples only (full also reported)\n\n")

        for b in range(1, n_branches + 1):
            f.write(f"BRANCH {b}\n")
            f.write(f"  G1_b{b}  : n_b={n_states}, n_a={n_states}, n_k=1\n")
            f.write(f"  Fnl_b{b} : {n_neurons} neurons, {act_branches}\n")
            f.write(f"  G2_b{b}  : n_b={n_states}, n_a={n_states}, n_k=0\n\n")

        f.write("COMBINATION\n")
        f.write(f"  Fnl_mix : {n_branches} in, 1 out, "
                f"{n_neurons_mix} neurons, {act_mix}\n")
        f.write(f"  G_out   : n_b={n_states + g_out_bonus}, "
                f"n_a={n_states + g_out_bonus}, n_k=0 "
                f"(bonus = {g_out_bonus})\n\n")

        f.write("NORMALIZATION (z-score, applied to input AND output)\n")
        f.write(f"  alpha_mean : {alpha_mean:+.6f}\n")
        f.write(f"  alpha_std  : {alpha_std:.6f}\n")
        f.write(f"  cl_mean    : {cl_mean:+.6f}\n")
        f.write(f"  cl_std     : {cl_std:.6f}\n")
        f.write(f"  Stats computed on full training set "
                f"(transients included).\n")
        f.write(f"  Saved to   : norm_stats.json\n\n")

        f.write("TOTAL\n")
        f.write(f"  States     : {n_total_states} "
                f"({n_branches} x 2 x {n_states} + {n_states + g_out_bonus})\n")
        f.write(f"  Parameters : {n_params}\n\n")

        f.write("RESULTS  (reported in physical units after denormalisation)\n")
        f.write(f"  Fit (masked) : {fit_masked:.4f}%\n")
        f.write(f"  Fit (full)   : {fit_full:.4f}%\n")
        f.write(f"  RMSE (masked): {rmse_masked:.4f}%\n")
        f.write(f"  Duration     : {format_duration(duration)}\n")

    print(f"\n  All training files saved to: {train_dir}")
    print(f"  Terminal log: {log_path}")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()
