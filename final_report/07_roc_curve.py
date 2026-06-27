# -*- coding: utf-8 -*-
"""
M2.2 — ROC křivka s 95% CI (1000× bootstrap)
Výstup: figures/roc_curve.png/.pdf
Zdroj:  results/risk_model_calibrated.joblib + risk_model_x_test.npy + risk_model_y_test.npy
"""
import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
FIGURES.mkdir(exist_ok=True)

x_test = np.load(RESULTS / 'risk_model_x_test.npy')
y_test = np.load(RESULTS / 'risk_model_y_test.npy')
model  = joblib.load(RESULTS / 'risk_model_calibrated.joblib')

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

y_prob = model.predict_proba(x_test)[:, 1]

# --- Hlavní ROC ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# --- Bootstrap CI (stratifikovaný, 1000×) ---
N_BOOT   = 1000
FPR_GRID = np.linspace(0, 1, 300)
rng      = np.random.RandomState(42)

pos_idx = np.where(y_test == 1)[0]
neg_idx = np.where(y_test == 0)[0]
tpr_boots = []

for _ in range(N_BOOT):
    b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
    idx   = np.concatenate([b_pos, b_neg])
    yb, pb = y_test[idx], y_prob[idx]
    if len(np.unique(yb)) < 2:
        continue
    fpr_b, tpr_b, _ = roc_curve(yb, pb)
    tpr_boots.append(np.interp(FPR_GRID, fpr_b, tpr_b))

tpr_boots = np.array(tpr_boots)
tpr_lo = np.percentile(tpr_boots, 2.5,  axis=0)
tpr_hi = np.percentile(tpr_boots, 97.5, axis=0)

# --- Youdenův bod ---
j_idx     = np.argmax(tpr - fpr)
best_fpr  = fpr[j_idx]
best_tpr  = tpr[j_idx]
best_thr  = thresholds[j_idx]

print(f"AUC: {roc_auc:.4f}")
print(f"95% CI: [{sc['test_performance']['auc_ci_lower']:.4f} – {sc['test_performance']['auc_ci_upper']:.4f}]")
print(f"Youdenův bod → FPR={best_fpr:.3f}, TPR={best_tpr:.3f}, práh={best_thr:.4f}")

# --- Graf ---
fig, ax = plt.subplots(figsize=(7, 7))

ax.fill_between(FPR_GRID, tpr_lo, tpr_hi,
                alpha=0.18, color='#2980b9',
                label='95% CI (1000× bootstrap)')
ax.plot(fpr, tpr, color='#2980b9', lw=2.5,
        label=f'Kalibrovaný model (3 features)')
ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.45, label='Náhodný klasifikátor (AUC = 0.5)')
ax.scatter([best_fpr], [best_tpr],
           color='#e74c3c', zorder=6, s=150, marker='X',
           label=f'Youdenův bod (práh = {best_thr:.3f})')

# AUC text box
ci_lo = sc['test_performance']['auc_ci_lower']
ci_hi = sc['test_performance']['auc_ci_upper']
ax.text(0.97, 0.10,
        f'AUC = {roc_auc:.3f}\n95% CI [{ci_lo:.3f} – {ci_hi:.3f}]\n(1000× stratif. bootstrap)',
        transform=ax.transAxes, fontsize=10,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85,
                  edgecolor='#cccccc'))

ax.set_xlabel('False Positive Rate (1 − Specificita)', fontsize=11)
ax.set_ylabel('True Positive Rate (Senzitivita)', fontsize=11)
ax.set_title(
    'ROC křivka — kalibrovaný model záchytu FiS\n'
    '(CHA₂DS₂-VASc + Teritoriální infarkt + Hyperlipidémie)',
    fontsize=11, fontweight='bold', pad=10,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3, linestyle=':')

# Testovací set info
n_test = sc['test_performance']['n_test']
n_pos  = sc['test_performance']['n_positive']
ax.text(0.02, 0.02,
        f'Testovací set: n = {n_test}  (AFib+ = {n_pos}, {n_pos/n_test*100:.1f} %)',
        transform=ax.transAxes, fontsize=8.5, color='#555555', style='italic')

plt.tight_layout()

for ext in ('png', 'pdf'):
    out = FIGURES / f'roc_curve.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
