# -*- coding: utf-8 -*-
"""
M1.4 — Boxploty signifikantních kontinuálních proměnných (AFib− vs AFib+)
Výstup: figures/boxplots_continuous_significant.png/.pdf
Zdroj:  results/analysis_results_continuous_variables_zachyt_FiS_mapping_correlations.json
        Excel data (přes project pipeline)
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
FIGURES.mkdir(exist_ok=True)

CONT_JSON = RESULTS / 'analysis_results_continuous_variables_zachyt_FiS_mapping_correlations.json'

with open(CONT_JSON, encoding='utf-8') as f:
    cont_data = json.load(f)

sig_vars = {
    var: info for var, info in cont_data.items()
    if info['p_value'] < 0.05
}

if not sig_vars:
    print("Žádné signifikantní kontinuální proměnné (p < 0.05). Graf nevygenerován.")
    sys.exit(0)

print(f"Signifikantních proměnných: {len(sig_vars)}: {list(sig_vars.keys())}")

# --- Načíst raw data přes project pipeline ---
from utils.config_singleton import ConfigSingleton
from core import load_data, evaluate_logic, remove_outliers_iqr

ConfigSingleton.set()
cfg = ConfigSingleton.get()

all_vars  = list(sig_vars.keys())
ref_var   = cfg.model.reference_var

df = load_data(cfg.analysis.file_path, all_vars, ref_var)

from core import evaluate_logic
y = evaluate_logic(df, cfg.variables.conditions, cfg.variables.logic).astype(int)

# --- Layout ---
n_vars = len(sig_vars)
ncols  = min(3, n_vars)
nrows  = int(np.ceil(n_vars / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
axes_flat = np.array(axes).flatten() if n_vars > 1 else [axes]

COLORS = ['#3498db', '#e74c3c']   # AFib− modrá, AFib+ červená

for idx, (var, info) in enumerate(sig_vars.items()):
    ax = axes_flat[idx]

    col = df[var].apply(lambda v: np.nan if isinstance(v, str) else v).replace([np.inf, -np.inf], np.nan)
    valid_idx = col.dropna().index.intersection(y.dropna().index)
    col_v = col.loc[valid_idx]
    y_v   = y.loc[valid_idx]

    col_v_clean = remove_outliers_iqr(col_v, threshold=cfg.analysis.iqr_threshold)
    final_idx   = col_v_clean.index.intersection(y_v.index)
    col_final   = col_v_clean.loc[final_idx]
    y_final     = y_v.loc[final_idx]

    grp0 = col_final[y_final == 0].values
    grp1 = col_final[y_final == 1].values

    bp = ax.boxplot(
        [grp0, grp1],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='black', linewidth=2),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    p = info['p_value']
    beta = info['correlation']
    p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.set_title(f"{var}\n{p_str}  |  β = {beta:+.4f}", fontsize=8, pad=4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'AFib−\n(n={len(grp0)})', f'AFib+\n(n={len(grp1)})'], fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

# Skrýt prázdné subploty
for ax in axes_flat[n_vars:]:
    ax.set_visible(False)

fig.suptitle(
    'Signifikantní kontinuální proměnné: AFib− vs. AFib+\n(pouze p < 0.05, IQR filtr outlierů)',
    fontsize=11, fontweight='bold', y=1.01
)
plt.tight_layout()

for ext in ('png', 'pdf'):
    out = FIGURES / f'boxplots_continuous_significant.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
