# -*- coding: utf-8 -*-
"""
M3.2 — Confusion matrix s klinickým kontextem
Výstup: figures/confusion_matrix.png/.pdf
Zdroj:  results/risk_scorecard.json (přesné hodnoty z kalibrace)
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
FIGURES.mkdir(exist_ok=True)

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

cal = sc['calibration']
tp  = cal['calibrated_tp']
tn  = cal['calibrated_tn']
fp  = cal['calibrated_fp']
fn  = cal['calibrated_fn']
n   = tp + tn + fp + fn
thr = cal['calibrated_optimal_threshold']

sens = cal['calibrated_sensitivity']
spec = cal['calibrated_specificity']
ppv  = cal['calibrated_ppv']
npv  = cal['calibrated_npv']

print(f"Matice záměn (práh = {thr:.4f}):")
print(f"  TP={tp}  FP={fp}")
print(f"  FN={fn}  TN={tn}")
print(f"Senzitivita: {sens:.4f}  Specificita: {spec:.4f}")
print(f"PPV:         {ppv:.4f}   NPV:         {npv:.4f}")

# --- Graf ---
cm = np.array([[tp, fn],   # řádek 0: skutečnost FiS+ → TP | FN
               [fp, tn]])  # řádek 1: skutečnost FiS− → FP | TN

# Klinické popisky
cell_labels = [
    [f'TP = {tp}\n({tp/n*100:.1f} %)',   f'FN = {fn}\n({fn/n*100:.1f} %)'],
    [f'FP = {fp}\n({fp/n*100:.1f} %)',   f'TN = {tn}\n({tn/n*100:.1f} %)'],
]
clinical_notes = [
    ['FiS detekována\n→ MDT monitoring', 'FiS přehlédnuta\n→ missed case'],
    ['FiS nepřítomna\n→ zbytečný monitoring', 'FiS nepřítomna\n→ správně vyloučena'],
]

fig, ax = plt.subplots(figsize=(8, 7))

cmap = plt.cm.Blues
im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=cm.max() * 1.2)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predikce: FiS+\n(monitoring indikován)', 'Predikce: FiS−\n(monitoring neindikovn.)'],
                   fontsize=10)
ax.set_yticklabels(['Skutečnost:\nFiS+', 'Skutečnost:\nFiS−'], fontsize=10)
ax.set_xlabel('Predikovaná třída', fontsize=11, labelpad=10)
ax.set_ylabel('Skutečná třída', fontsize=11, labelpad=10)

# Buňky
CELL_COLORS = [['#1a5276', '#922b21'],
               ['#1a5276', '#1a5276']]
TEXT_WEIGHT  = [['bold', 'bold'], ['bold', 'normal']]

for i in range(2):
    for j in range(2):
        brightness = cm[i, j] / (cm.max() or 1)
        txt_color  = 'white' if brightness > 0.3 else '#2c3e50'
        ax.text(j, i - 0.1, cell_labels[i][j],
                ha='center', va='center', fontsize=13,
                fontweight=TEXT_WEIGHT[i][j],
                color='white')
        ax.text(j, i + 0.22, clinical_notes[i][j],
                ha='center', va='center', fontsize=7.5,
                color='white', style='italic', alpha=0.9)

ax.set_title(
    f'Confusion matrix — kalibrovaný model záchytu FiS\n'
    f'(testovací sada, n = {n}, práh = {thr:.3f} — Youdenovo optimum)',
    fontsize=11, fontweight='bold', pad=14,
)

# Metriky pod grafem
metrics_text = (
    f'Senzitivita = {sens*100:.1f} %    '
    f'Specificita = {spec*100:.1f} %    '
    f'PPV = {ppv*100:.1f} %    '
    f'NPV = {npv*100:.1f} %'
)
fig.text(0.5, 0.02, metrics_text,
         ha='center', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#eaf2fb',
                   edgecolor='#aaaaaa', alpha=0.9))

# Barevná legenda
tp_patch  = mpatches.Patch(color=cmap(0.85), label=f'TP = {tp} — správně detekovaná FiS')
tn_patch  = mpatches.Patch(color=cmap(0.5),  label=f'TN = {tn} — správně vyloučená FiS')
fp_patch  = mpatches.Patch(color=cmap(0.25), label=f'FP = {fp} — zbytečný monitoring')
fn_patch  = mpatches.Patch(color=cmap(0.08), label=f'FN = {fn} — přehlédnutá FiS')
ax.legend(handles=[tp_patch, tn_patch, fp_patch, fn_patch],
          fontsize=8, loc='upper right',
          bbox_to_anchor=(1.0, -0.16), ncol=2)

plt.subplots_adjust(bottom=0.22)

for ext in ('png', 'pdf'):
    out = FIGURES / f'confusion_matrix.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
