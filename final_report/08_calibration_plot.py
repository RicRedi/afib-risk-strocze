# -*- coding: utf-8 -*-
"""
M2.3 — Kalibrační reliability diagram (před/po Platt scaling)
Výstup: figures/calibration_reliability.png/.pdf
Zdroj:  results/risk_scorecard.json + risk_model*.joblib + x_test.npy
"""
import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
FIGURES.mkdir(exist_ok=True)

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

x_test = np.load(RESULTS / 'risk_model_x_test.npy')

base_model = joblib.load(RESULTS / 'risk_model.joblib')
cal_model  = joblib.load(RESULTS / 'risk_model_calibrated.joblib')

y_uncal = base_model.predict_proba(x_test)[:, 1]
y_cal   = cal_model.predict_proba(x_test)[:, 1]

cal_data = sc['calibration']
rl_uncal = cal_data['reliability_uncalibrated']
rl_cal   = cal_data['reliability_calibrated']

brier_u = cal_data['brier_uncalibrated']
brier_c = cal_data['brier_calibrated']
prevalence = sc['test_performance']['cohort_prevalence']

print(f"Brier nekalibrovaný:  {brier_u:.4f}")
print(f"Brier kalibrovaný:    {brier_c:.4f}")
print(f"Predikce nekalibrované: min={y_uncal.min():.3f}, max={y_uncal.max():.3f}, median={np.median(y_uncal):.3f}")
print(f"Predikce kalibrované:   min={y_cal.min():.3f}, max={y_cal.max():.3f}, median={np.median(y_cal):.3f}")

# --- Rozložení grafu: 2×2 (reliab. diagram nahoře, histogram dole) ---
fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1.2], hspace=0.35, wspace=0.35)

ax_u_rel = fig.add_subplot(gs[0, 0])
ax_c_rel = fig.add_subplot(gs[0, 1])
ax_u_his = fig.add_subplot(gs[1, 0])
ax_c_his = fig.add_subplot(gs[1, 1])

fig.suptitle('Kalibrace modelu před a po Platt scaling\n'
             '(testovací set, n = 377)',
             fontsize=12, fontweight='bold', y=0.98)

def plot_reliability(ax, frac_pos, mean_pred, brier, title, color):
    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, label='Ideální kalibrace')
    ax.scatter(mean_pred, frac_pos,
               color=color, s=100, zorder=5,
               label='Skutečné frakce pozitivních')
    ax.plot(mean_pred, frac_pos, color=color, lw=1.5, alpha=0.7)
    ax.set_xlim(0, max(max(mean_pred) * 1.15, 0.1))
    ax.set_ylim(0, max(max(frac_pos) * 1.3, 0.15))
    ax.set_xlabel('Průměrná predikovaná pravd.', fontsize=9)
    ax.set_ylabel('Skutečná frakce pozitivních', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.text(0.97, 0.07,
            f'Brier = {brier:.4f}',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='#aaaaaa', alpha=0.9))
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3, linestyle=':')

plot_reliability(
    ax_u_rel,
    rl_uncal['fraction_of_positives'],
    rl_uncal['mean_predicted_value'],
    brier_u,
    'Nekalibrovaný model\n(LogReg, class_weight="balanced")',
    '#e67e22',
)
plot_reliability(
    ax_c_rel,
    rl_cal['fraction_of_positives'],
    rl_cal['mean_predicted_value'],
    brier_c,
    'Kalibrovaný model\n(Platt scaling)',
    '#27ae60',
)

# Histogramy predikovaných pravděpodobností
ax_u_his.hist(y_uncal, bins=25, color='#e67e22', alpha=0.75, edgecolor='white')
ax_u_his.axvline(prevalence, color='#c0392b', lw=1.5, ls='--',
                  label=f'Prevalence ({prevalence*100:.1f} %)')
ax_u_his.set_xlabel('Predikovaná pravděpodobnost', fontsize=9)
ax_u_his.set_ylabel('Počet pacientů', fontsize=9)
ax_u_his.set_title('Distribuce predikcí (nekalibrovaný)', fontsize=9)
ax_u_his.legend(fontsize=7)
ax_u_his.grid(alpha=0.3, linestyle=':')

ax_c_his.hist(y_cal, bins=25, color='#27ae60', alpha=0.75, edgecolor='white')
ax_c_his.axvline(prevalence, color='#c0392b', lw=1.5, ls='--',
                  label=f'Prevalence ({prevalence*100:.1f} %)')
ax_c_his.set_xlabel('Predikovaná pravděpodobnost', fontsize=9)
ax_c_his.set_ylabel('Počet pacientů', fontsize=9)
ax_c_his.set_title('Distribuce predikcí (kalibrovaný)', fontsize=9)
ax_c_his.legend(fontsize=7)
ax_c_his.grid(alpha=0.3, linestyle=':')

for ext in ('png', 'pdf'):
    out = FIGURES / f'calibration_reliability.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
