"""
DynoNet -- branch-contribution analysis


Outputs, per monosine case, under the trained run folder:
    branch_analysis/
        {label}_contrib_time.pdf       contribution vs time
        {label}_contrib_vs_aoa.pdf     contribution vs angle of attack
        {label}_responsibility.pdf     branch share of |contribution| vs AoA
    branch_analysis/branch_summary.csv
    branch_analysis/terminal_log_branch.txt
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

# location of the dynonet source
sys.path.insert(0, r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\dynoNet\dynonet\src")

from dynonet.lti    import SisoLinearDynamicalOperator
from dynonet.static import SisoStaticNonLinearity, MimoStaticNonLinearity


# plot style (presentation sizes)
plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 15,
    'axes.titlesize'    : 16,
    'axes.labelsize'    : 15,
    'xtick.labelsize'   : 13,
    'ytick.labelsize'   : 13,
    'legend.fontsize'   : 13,
    'figure.dpi'        : 150,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : False,
    'lines.linewidth'   : 2.0,
    'axes.linewidth'    : 1.3,
    'savefig.bbox'      : 'tight',
    'savefig.pad_inches': 0.05,
})
COLORS = {'cfd': '#C0392B', 'model': '#2980B9', 'aoa': '#2C3E50'}
BRANCH_COLORS = ['#2980B9', '#C0392B', '#27AE60', '#8E44AD', '#E67E22', '#16A085']


# paths
BASE_DIR   = r"C:\Users\lucad\OneDrive\Documenten\Bureaublad\Masterproef\dynoNet\ExperimentValidation"
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

FS_DEFAULT        = 200.0
N_REPEATS         = 10   # tile one period this many times
N_DISCARD_PERIODS = 1    # discard this many periods as settling
AOA_BINS          = 24   # bins for the responsibility plot


# terminal logger
class Tee:
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


def to_tensor(arr):
    return torch.tensor(arr).unsqueeze(0).unsqueeze(-1)


# DynoNet model
class DynoNetParallel(nn.Module):
    """N parallel branches (G1 -> Fnl -> G2), a mixing MLP, and an output filter."""
    def __init__(self, n_branches, n_states, g_out_bonus,
                 n_neurons, n_neurons_mix,
                 activation_branches, activation_mix):
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
                for _ in range(n_branches)])
            self.Fnl = nn.ModuleList([
                SisoStaticNonLinearity(n_hidden=n_neurons, activation=activation_branches)
                for _ in range(n_branches)])
            self.G2 = nn.ModuleList([
                SisoLinearDynamicalOperator(n_b=n_states, n_a=n_states, n_k=0)
                for _ in range(n_branches)])

        self.Fnl_mix = MimoStaticNonLinearity(
            in_channels=n_branches, out_channels=1,
            n_hidden=n_neurons_mix, activation=activation_mix)
        self.G_out = SisoLinearDynamicalOperator(n_b=self.n_out, n_a=self.n_out, n_k=0)

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

    def branch_signals(self, u):
        """Per-branch outputs (before the mixer)."""
        return [G2(Fnl(G1(u))) for (G1, Fnl, G2) in self.branches()]

    def combine(self, outs):
        """Run the mixer and output filter on a list of branch outputs."""
        return self.G_out(self.Fnl_mix(torch.cat(outs, dim=-1)))


# interactive prompt helpers
def _prompt(message, default, caster=str, choices=None):
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
    print("=" * 70)
    print("  DynoNet -- branch-contribution analysis")
    print("=" * 70)
    print()
    print("  Which trained model do you want to analyse?")
    print("    V2      -- 2 parallel branches")
    print("    V4      -- 3 parallel branches")
    print("    custom  -- choose N branches and G_out bonus yourself")
    print()

    raw = input("  Version? [V2/V4/custom] (V4): ").strip()
    version_key = 'V4' if raw == '' else (raw.upper() if raw.lower() != 'custom' else 'custom')

    if version_key in ('V2', 'V4'):
        cfg = dict(VERSION_DEFAULTS[version_key])
        version_folder = f"{version_key}_masked"
    elif version_key == 'custom':
        cfg = dict(VERSION_DEFAULTS['V4'])
        cfg['n_branches']  = _prompt("Number of parallel branches", 3, int)
        cfg['g_out_bonus'] = _prompt("Extra G_out states (0=V4 conv, 2=V2 conv)", 0, int)
        version_folder = f"V{cfg['n_branches']}_masked"
    else:
        raise SystemExit(f"Unknown version '{raw}'. Use V2, V4 or custom.")

    print()
    if input("  Use the default hyper-parameters?  [Y/n]: ").strip().lower() in ('', 'y', 'yes'):
        print("  -> using defaults")
    else:
        print("  -> match the values the model was trained with")
        cfg['n_states']            = _prompt("n_states", cfg['n_states'], int)
        cfg['n_neurons']           = _prompt("n_neurons", cfg['n_neurons'], int)
        cfg['n_neurons_mix']       = _prompt("n_neurons_mix", cfg['n_neurons_mix'], int)
        cfg['activation_branches'] = _prompt("activation_branches", cfg['activation_branches'],
                                             str, choices={'relu','tanh','sigmoid','elu','leakyrelu'})
        cfg['activation_mix']      = _prompt("activation_mix", cfg['activation_mix'],
                                             str, choices={'relu','tanh','sigmoid','elu','leakyrelu'})
        cfg['n_epochs']            = _prompt("n_epochs", cfg['n_epochs'], int)
        cfg['seed']                = _prompt("random seed", cfg['seed'], int)

    return version_key, version_folder, cfg


def build_run_name(cfg):
    if cfg['n_branches'] == 2:
        return (f"training_s{cfg['n_states']}_n{cfg['n_neurons']}"
                f"_{cfg['activation_branches']}_ep{cfg['n_epochs']}"
                f"_trSmart_seed{cfg['seed']}_norm")
    return (f"training_s{cfg['n_states']}_n{cfg['n_neurons']}"
            f"_nmix{cfg['n_neurons_mix']}_{cfg['activation_branches']}"
            f"_ep{cfg['n_epochs']}_trSmart_seed{cfg['seed']}_norm")


# validation data
def load_validation_datasets():
    if not os.path.exists(VAL_FOLDER):
        print(f"\n  Extracting {VAL_ZIP} ...")
        with zipfile.ZipFile(VAL_ZIP, 'r') as z:
            z.extractall(os.path.dirname(VAL_ZIP))

    val_files = sorted(os.path.join(r, f)
                       for r, _d, fs in os.walk(VAL_FOLDER)
                       for f in fs if f.endswith('.mat'))
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
        fs = fs if fs is not None else FS_DEFAULT
        label = fname.replace('CFD_Monosine_', '').replace('_Downsampled.mat', '')
        datasets.append({'filename': fname, 'label': label,
                         'alpha': alpha, 'cl': cl, 'fs': fs})
    return datasets


# branch contribution by ablation
def compute_contributions(model, alpha_period, stats):
    """
    Tile one period, run the model, and for each branch measure the change in
    the predicted Cl when that branch is held at its mean (ablated).  Returns
    the steady-state angle of attack, its rate, and the per-branch Cl
    contributions in physical units.
    """
    T       = len(alpha_period)
    discard = N_DISCARD_PERIODS * T

    alpha_tiled = np.tile(alpha_period, N_REPEATS).astype(np.float32)
    alpha_norm  = ((alpha_tiled - stats['alpha_mean']) / stats['alpha_std']).astype(np.float32)
    u = to_tensor(alpha_norm)

    with torch.no_grad():
        outs   = model.branch_signals(u)
        y_full = model.combine(outs).squeeze().numpy()

        deltas = []
        for b in range(len(outs)):
            ablated    = list(outs)
            ablated[b] = torch.full_like(outs[b], float(outs[b].mean()))
            y_wo       = model.combine(ablated).squeeze().numpy()
            # difference in z-score space -> physical Cl: the mean offset cancels
            deltas.append((y_full - y_wo) * stats['cl_std'])

    sl = slice(discard, discard + T)
    alpha_steady = alpha_tiled[sl]
    return {
        'T'      : T,
        'alpha'  : alpha_steady,
        'dalpha' : np.gradient(alpha_steady),
        'y_full' : y_full[sl] * stats['cl_std'] + stats['cl_mean'],
        'deltas' : [d[sl] for d in deltas],
    }


# plots
def plot_contrib_time(res, fs, label, out_path):
    t = np.arange(res['T']) / fs
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, res['alpha'], color=COLORS['aoa'])
    axes[0].set_ylabel(r'$\alpha$ (deg)')
    axes[0].set_title(f'{label}')

    for b, d in enumerate(res['deltas']):
        axes[1].plot(t, d, color=BRANCH_COLORS[b % len(BRANCH_COLORS)],
                     label=f'branch {b+1}')
    axes[1].axhline(0, color='#888888', linewidth=0.8)
    axes[1].set_ylabel(r'contribution to $C_l$ (-)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(frameon=False, loc='upper right', ncol=len(res['deltas']))

    for ax in axes:
        ax.tick_params(direction='out', length=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_contrib_vs_aoa(res, label, out_path):
    alpha = res['alpha']
    up    = res['dalpha'] >= 0
    dn    = ~up

    fig, ax = plt.subplots(figsize=(8, 6))
    for b, d in enumerate(res['deltas']):
        c = BRANCH_COLORS[b % len(BRANCH_COLORS)]
        ax.plot(alpha[up], d[up], color=c, linewidth=2.0, label=f'branch {b+1} (upstroke)')
        ax.plot(alpha[dn], d[dn], color=c, linewidth=2.0, linestyle='--',
                dashes=(4, 2), label=f'branch {b+1} (downstroke)')
    ax.axhline(0, color='#888888', linewidth=0.8)
    ax.set_xlabel(r'Angle of attack $\alpha$ (deg)')
    ax.set_ylabel(r'contribution to $C_l$ (-)')
    ax.set_title(f'{label}')
    ax.legend(frameon=False, fontsize=11, loc='best')
    ax.tick_params(direction='out', length=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_responsibility(res, label, out_path, n_bins=AOA_BINS):
    alpha   = res['alpha']
    deltas  = np.array([np.abs(d) for d in res['deltas']])   # (n_branches, T)
    edges   = np.linspace(alpha.min(), alpha.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx     = np.clip(np.digitize(alpha, edges) - 1, 0, n_bins - 1)

    n_branches = deltas.shape[0]
    share = np.zeros((n_branches, n_bins))
    for k in range(n_bins):
        m = idx == k
        if m.any():
            mean_abs = deltas[:, m].mean(axis=1)
            total    = mean_abs.sum()
            if total > 0:
                share[:, k] = mean_abs / total

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(n_bins)
    width  = (centers[1] - centers[0]) if n_bins > 1 else 1.0
    for b in range(n_branches):
        ax.bar(centers, share[b], bottom=bottom, width=width,
               color=BRANCH_COLORS[b % len(BRANCH_COLORS)],
               edgecolor='white', linewidth=0.3, label=f'branch {b+1}')
        bottom += share[b]
    ax.set_xlabel(r'Angle of attack $\alpha$ (deg)')
    ax.set_ylabel('share of total |contribution|')
    ax.set_ylim(0, 1)
    ax.set_title(f'{label}')
    ax.legend(frameon=False, loc='upper center', ncol=n_branches)
    ax.tick_params(direction='out', length=3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def summarise(res):
    """Per-branch share of |contribution|, AoA centre of mass, and low/high split."""
    alpha  = res['alpha']
    mid    = np.median(alpha)
    rows   = []
    abs_d  = [np.abs(d) for d in res['deltas']]
    grand  = sum(d.sum() for d in abs_d) + 1e-12
    for b, d in enumerate(abs_d):
        com   = float((alpha * d).sum() / (d.sum() + 1e-12))
        low   = d[alpha <  mid].sum()
        high  = d[alpha >= mid].sum()
        rows.append({
            'branch'      : b + 1,
            'share_total' : 100.0 * d.sum() / grand,
            'aoa_com'     : com,
            'low_share'   : 100.0 * low  / (low + high + 1e-12),
            'high_share'  : 100.0 * high / (low + high + 1e-12),
        })
    return mid, rows


# main
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
    train_dir   = os.path.join(version_dir, build_run_name(cfg))
    weights_path    = os.path.join(train_dir, 'model_weights.pt')
    norm_stats_path = os.path.join(train_dir, 'norm_stats.json')

    out_dir = os.path.join(train_dir, 'branch_analysis')
    os.makedirs(out_dir, exist_ok=True)
    logger     = Tee(os.path.join(out_dir, 'terminal_log_branch.txt'))
    sys.stdout = logger

    print("=" * 70)
    print(f"  DynoNet {version_key} -- branch-contribution analysis (ablation)")
    print("=" * 70)

    if not os.path.exists(weights_path):
        print(f"\n  ERROR: model weights not found at:\n  {weights_path}")
        sys.stdout = logger.terminal; logger.close(); sys.exit(1)
    if not os.path.exists(norm_stats_path):
        print(f"\n  ERROR: norm_stats.json not found at:\n  {norm_stats_path}")
        sys.stdout = logger.terminal; logger.close(); sys.exit(1)

    with open(norm_stats_path) as f:
        ns = json.load(f)
    stats = {'alpha_mean': float(ns['alpha_mean']), 'alpha_std': float(ns['alpha_std']),
             'cl_mean': float(ns['cl_mean']), 'cl_std': float(ns['cl_std'])}

    model = DynoNetParallel(
        n_branches=n_branches, n_states=n_states, g_out_bonus=g_out_bonus,
        n_neurons=n_neurons, n_neurons_mix=n_neurons_mix,
        activation_branches=act_branches, activation_mix=act_mix)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print(f"\n  Loaded {n_branches}-branch model from:\n  {weights_path}")

    datasets = load_validation_datasets()
    print(f"  Monosine cases: {len(datasets)}")

    summary_path = os.path.join(out_dir, 'branch_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        csv.writer(f).writerow(['case', 'branch', 'share_total_%',
                                'aoa_com_deg', 'low_aoa_share_%', 'high_aoa_share_%'])

    for vd in datasets:
        label = vd['label']
        res   = compute_contributions(model, vd['alpha'], stats)

        safe = label.replace('.', 'p').replace(' ', '_')
        plot_contrib_time(res, vd['fs'], label,
                          os.path.join(out_dir, f'{safe}_contrib_time.pdf'))
        plot_contrib_vs_aoa(res, label,
                            os.path.join(out_dir, f'{safe}_contrib_vs_aoa.pdf'))
        plot_responsibility(res, label,
                            os.path.join(out_dir, f'{safe}_responsibility.pdf'))

        mid, rows = summarise(res)
        print(f"\n  {label}  (AoA range [{res['alpha'].min():.1f}, "
              f"{res['alpha'].max():.1f}] deg, split at {mid:.1f} deg)")
        print(f"    {'branch':>6} {'share %':>9} {'AoA c.o.m.':>11} "
              f"{'low %':>7} {'high %':>7}")
        for r in rows:
            print(f"    {r['branch']:>6} {r['share_total']:>9.1f} "
                  f"{r['aoa_com']:>11.1f} {r['low_share']:>7.1f} {r['high_share']:>7.1f}")

        with open(summary_path, 'a', newline='') as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([label, r['branch'], f"{r['share_total']:.2f}",
                            f"{r['aoa_com']:.2f}", f"{r['low_share']:.2f}",
                            f"{r['high_share']:.2f}"])

    print(f"\n  Figures and summary saved to:\n  {out_dir}")
    print("\n  Reading the result:")
    print("    - if the branches specialise, their AoA centres of mass differ and")
    print("      the responsibility plot shows the share shifting between branches")
    print("      as the angle of attack moves from attached flow into stall;")
    print("    - if the bands stay roughly constant across AoA, the branches are")
    print("      not regime-specialised (still a valid, reportable outcome).")

    sys.stdout = logger.terminal
    logger.close()


if __name__ == '__main__':
    main()
