# -*- coding: utf-8 -*-
"""
M1.5 — RFECV křivka: AUC vs. počet features
Výstup: figures/rfecv_feature_selection.png/.pdf
Zdroj:  results/risk_scorecard.json  (rfecv_* klíče z M0.3)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
FIGURES.mkdir(exist_ok=True)

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

tdi = sc['training_data_info']

n_range    = tdi['rfecv_n_features_range']
mean_auc   = np.array(tdi['rfecv_mean_scores'])
std_auc    = np.array(tdi['rfecv_std_scores'])
n_selected = tdi['n_features_selected']
sel_names  = tdi['selected_features']
all_names  = tdi['all_features']
dropped    = [f for f in all_names if f not in sel_names]

best_idx = n_range.index(n_selected) if n_selected in n_range else int(np.argmax(mean_auc))
best_auc = mean_auc[best_idx]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(n_range, mean_auc, 'o-', color='#2980b9', linewidth=2, markersize=7,
        label='Průměrné AUC (5-fold CV)')
ax.fill_between(n_range,
                mean_auc - std_auc,
                mean_auc + std_auc,
                alpha=0.18, color='#2980b9', label='±1 SD')

ax.axvline(x=n_selected, color='#c0392b', linestyle='--', linewidth=1.5,
           label=f'Zvoleno: {n_selected} features')
ax.plot(n_selected, best_auc, 's', color='#c0392b', markersize=10, zorder=5)
ax.text(n_selected + 0.08, best_auc + 0.004,
        f'AUC = {best_auc:.4f}', color='#c0392b', fontsize=9, fontweight='bold')

ax.set_xlabel('Počet features', fontsize=11)
ax.set_ylabel('AUC-ROC (5-fold CV)', fontsize=11)
ax.set_title('RFECV — výběr optimálního počtu features\n'
             f'(min. 3 features, krok 1, 5-fold stratifikovaná CV)',
             fontsize=11, fontweight='bold', pad=10)
ax.set_xticks(n_range)
ax.set_ylim(max(0.4, mean_auc.min() - 0.05), min(1.0, mean_auc.max() + 0.05))
ax.grid(alpha=0.3, linestyle=':')
ax.legend(fontsize=9)

# Inset tabulka: vybrané vs. odstraněné
table_x, table_y = 0.62, 0.18
table_text = "Vybrané features:\n"
for f in sel_names:
    short = f.split('(')[0].strip()[:30]
    table_text += f"  ✓ {short}\n"
table_text += "\nOdstraněné:\n"
for f in dropped:
    short = f.split('(')[0].strip()[:30]
    table_text += f"  ✗ {short}\n"

ax.text(table_x, table_y, table_text.strip(),
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.85))

plt.tight_layout()

for ext in ('png', 'pdf'):
    out = FIGURES / f'rfecv_feature_selection.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
