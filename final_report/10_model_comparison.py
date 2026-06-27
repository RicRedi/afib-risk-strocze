# -*- coding: utf-8 -*-
"""
M3.1 — Srovnání ROC křivek: 1 vs. 2 vs. 3 features
Výstup: figures/model_comparison_roc.png/.pdf, tables/model_comparison.csv
Zdroj:  results/risk_model_calibrated.joblib + x_train/test.npy + y_train/test.npy
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
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, brier_score_loss

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
TABLES  = Path(__file__).resolve().parent / 'tables'
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

x_train = np.load(RESULTS / 'risk_model_x_train.npy')
y_train = np.load(RESULTS / 'risk_model_y_train.npy')
x_test  = np.load(RESULTS / 'risk_model_x_test.npy')
y_test  = np.load(RESULTS / 'risk_model_y_test.npy')
cal_model = joblib.load(RESULTS / 'risk_model_calibrated.joblib')

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

LR_KWARGS = dict(class_weight='balanced', random_state=42,
                 max_iter=1000, solver='lbfgs')

# --- Trénink baseline modelů na stejné sadě ---
m1 = LogisticRegression(**LR_KWARGS).fit(x_train[:, [0]],   y_train)   # jen CHA₂DS₂-VASc
m2 = LogisticRegression(**LR_KWARGS).fit(x_train[:, [0, 1]], y_train)  # + Teritoriální

prob1 = m1.predict_proba(x_test[:, [0]])[:, 1]
prob2 = m2.predict_proba(x_test[:, [0, 1]])[:, 1]
prob3 = cal_model.predict_proba(x_test)[:, 1]   # kalibrovaný 3-feature model

models = [
    {'name': 'CHA₂DS₂-VASc (1 feature)',          'probs': prob1, 'color': '#95a5a6', 'lw': 1.5, 'ls': '--'},
    {'name': 'CHA₂DS₂-VASc + Teritoriální (2)',   'probs': prob2, 'color': '#e67e22', 'lw': 1.8, 'ls': '-.'},
    {'name': 'Náš model — 3 features + kal.',      'probs': prob3, 'color': '#2980b9', 'lw': 2.5, 'ls': '-'},
]

# --- Bootstrap AUC CI a párové srovnání ---
N_BOOT = 1000
rng    = np.random.RandomState(42)
pos_idx = np.where(y_test == 1)[0]
neg_idx = np.where(y_test == 0)[0]

boot_aucs = [[] for _ in models]
for _ in range(N_BOOT):
    b = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                        rng.choice(neg_idx, len(neg_idx), replace=True)])
    yb = y_test[b]
    if len(np.unique(yb)) < 2:
        continue
    for j, m in enumerate(models):
        pb = m['probs'][b]
        fpr_b, tpr_b, _ = roc_curve(yb, pb)
        boot_aucs[j].append(auc(fpr_b, tpr_b))

# Metriky
tbl_rows = []
for j, m in enumerate(models):
    fpr, tpr, thr = roc_curve(y_test, m['probs'])
    roc_auc = auc(fpr, tpr)
    j_idx   = np.argmax(tpr - fpr)
    ba      = np.array(boot_aucs[j])
    # Youden threshold
    opt_thr  = thr[j_idx]
    y_pred   = (m['probs'] >= opt_thr).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    brier = brier_score_loss(y_test, m['probs'])
    tbl_rows.append({
        'Model':        m['name'],
        'AUC':          round(roc_auc, 4),
        'CI dolní (95%)': round(np.percentile(ba, 2.5), 4),
        'CI horní (95%)': round(np.percentile(ba, 97.5), 4),
        'Senzitivita (Youden)': round(sens, 4),
        'Specificita (Youden)': round(spec, 4),
        'Brier':        round(brier, 4),
    })
    m['fpr'] = fpr
    m['tpr'] = tpr
    m['auc'] = roc_auc
    m['ci']  = (np.percentile(ba, 2.5), np.percentile(ba, 97.5))

# Párový bootstrap test: model 3 vs model 1
diff31 = np.array(boot_aucs[2]) - np.array(boot_aucs[0])
diff32 = np.array(boot_aucs[2]) - np.array(boot_aucs[1])
p_val31 = min(np.mean(diff31 <= 0) * 2, 1.0)
p_val32 = min(np.mean(diff32 <= 0) * 2, 1.0)

tbl = pd.DataFrame(tbl_rows)
csv_path = TABLES / 'model_comparison.csv'
tbl.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {csv_path}")
print(tbl.to_string(index=False))
print(f"\nPárový bootstrap AUC test (1000×):")
print(f"  Model 3 vs Model 1:  ΔAUC = {diff31.mean():+.4f}, p = {p_val31:.4f}")
print(f"  Model 3 vs Model 2:  ΔAUC = {diff32.mean():+.4f}, p = {p_val32:.4f}")

# --- Graf ---
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.4, label='Náhodný klasifikátor')

for m in models:
    label = (f"{m['name']}\nAUC = {m['auc']:.3f} "
             f"[{m['ci'][0]:.3f}–{m['ci'][1]:.3f}]")
    ax.plot(m['fpr'], m['tpr'],
            color=m['color'], lw=m['lw'], ls=m['ls'], label=label)

# Zvýraznit vítěze
ax.text(0.03, 0.97,
        f'Párový bootstrap test (n = {N_BOOT}):\n'
        f'3 vs 1:  ΔAUC = {diff31.mean():+.3f},  p = {p_val31:.3f}\n'
        f'3 vs 2:  ΔAUC = {diff32.mean():+.3f},  p = {p_val32:.3f}',
        transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4f8',
                  edgecolor='#aaaaaa', alpha=0.9))

ax.set_xlabel('False Positive Rate (1 − Specificita)', fontsize=11)
ax.set_ylabel('True Positive Rate (Senzitivita)', fontsize=11)
ax.set_title(
    'Srovnání ROC křivek — přidaná hodnota nových prediktorů\n'
    '(všechny modely: LogReg, class_weight="balanced", testovací sada)',
    fontsize=11, fontweight='bold', pad=10,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3, linestyle=':')

plt.tight_layout()

for ext in ('png', 'pdf'):
    out = FIGURES / f'model_comparison_roc.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
