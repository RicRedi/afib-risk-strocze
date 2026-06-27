# -*- coding: utf-8 -*-
"""
M4.1 — Citlivostní analýza prahů rizikových kategorií
Výstup: figures/threshold_sensitivity.png/.pdf, tables/threshold_sensitivity.csv
Zdroj:  results/risk_model_calibrated.joblib + x_test.npy + y_test.npy
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGURES = Path(__file__).resolve().parent / 'figures'
TABLES  = Path(__file__).resolve().parent / 'tables'
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

x_test = np.load(RESULTS / 'risk_model_x_test.npy')
y_test = np.load(RESULTS / 'risk_model_y_test.npy')
model  = joblib.load(RESULTS / 'risk_model_calibrated.joblib')

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

y_prob     = model.predict_proba(x_test)[:, 1]
prevalence = sc['test_performance']['cohort_prevalence']
n_test     = sc['test_performance']['n_test']

# --- Rozsah násobků prevalence pro analýzu ---
multipliers = np.linspace(0.3, 2.0, 200)   # jemná mřížka pro plynulou křivku
thresholds  = multipliers * prevalence

rows = []
for mult, thr in zip(multipliers, thresholds):
    mask  = y_prob < thr
    n_low = mask.sum()
    if n_low == 0:
        npv = np.nan
    else:
        n_neg_in_low = ((y_test == 0) & mask).sum()
        npv = n_neg_in_low / n_low
    rows.append({
        'Násobek prevalence': round(mult, 3),
        'Práh (%)':           round(thr * 100, 3),
        'n v Nízké':          int(n_low),
        '% kohorty':          round(n_low / n_test * 100, 1),
        'NPV (%)':            round(npv * 100, 2) if not np.isnan(npv) else None,
    })

df = pd.DataFrame(rows)

# Tabulka — klíčové body (nejbližší hodnota v mřížce)
key_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
key_idx   = [int(np.argmin(np.abs(df['Násobek prevalence'].values - m))) for m in key_mults]
df_key    = df.iloc[key_idx].copy()
csv_path = TABLES / 'threshold_sensitivity.csv'
df_key.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {csv_path}")
print(df_key.to_string(index=False))

# --- Aktuálně zvolený práh (1× prevalence) ---
chosen_mult = 1.0
chosen_thr  = chosen_mult * prevalence
chosen_mask = y_prob < chosen_thr
chosen_npv  = ((y_test == 0) & chosen_mask).sum() / chosen_mask.sum() * 100
chosen_n    = chosen_mask.sum()

print(f"\nAktuální práh (1× prev = {chosen_thr*100:.2f} %):")
print(f"  n v Nízké = {chosen_n}  ({chosen_n/n_test*100:.1f} %)  NPV = {chosen_npv:.1f} %")

# --- Graf ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

mults = df['Násobek prevalence'].values
npvs  = df['NPV (%)'].values
pcts  = df['% kohorty'].values

# Subplot 1 — NPV vs. práh
ax1.plot(mults, npvs, color='#2980b9', lw=2.2, zorder=3)
ax1.axhline(95, color='#8e44ad', ls=':', lw=1.5, alpha=0.8,
            label='95 % NPV — klinický práh')
ax1.axvline(chosen_mult, color='#e74c3c', ls='--', lw=1.5,
            label=f'Zvolený práh (1× prev.)')
ax1.scatter([chosen_mult], [chosen_npv],
            color='#e74c3c', s=100, zorder=5)
ax1.text(chosen_mult + 0.04, chosen_npv - 0.6,
         f'{chosen_npv:.1f} %', fontsize=9, color='#e74c3c', fontweight='bold')
ax1.set_ylabel('NPV kategorie Nízké (%)', fontsize=10)
ax1.set_title(
    'Citlivostní analýza: NPV a velikost kategorie Nízké\n'
    'při různých definicích horního prahu',
    fontsize=11, fontweight='bold', pad=8,
)
ax1.set_ylim(85, 101)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(alpha=0.3, linestyle=':')
ax1.spines[['top', 'right']].set_visible(False)

# Annotace kdy NPV klesne pod 95 %
cross_idx = np.where(npvs < 95)[0]
if len(cross_idx) > 0:
    cross_mult = mults[cross_idx[0]]
    ax1.axvline(cross_mult, color='#8e44ad', ls=':', lw=1.0, alpha=0.5)
    ax1.text(cross_mult + 0.03, 87.5,
             f'NPV < 95 %\npod {cross_mult:.2f}× prev.',
             fontsize=8, color='#8e44ad', style='italic')

# Subplot 2 — % kohorty v kategorii Nízké
ax2.fill_between(mults, pcts, alpha=0.25, color='#27ae60')
ax2.plot(mults, pcts, color='#27ae60', lw=2.2, zorder=3)
ax2.axvline(chosen_mult, color='#e74c3c', ls='--', lw=1.5,
            label=f'Zvolený práh (1× prev.)')

chosen_pct = chosen_n / n_test * 100
ax2.scatter([chosen_mult], [chosen_pct], color='#e74c3c', s=100, zorder=5)
ax2.text(chosen_mult + 0.04, chosen_pct + 0.5,
         f'{chosen_pct:.1f} %\n(n = {chosen_n})',
         fontsize=9, color='#e74c3c', fontweight='bold')

ax2.set_xlabel('Horní práh kategorie Nízké (násobek prevalence)', fontsize=10)
ax2.set_ylabel('% pacientů v kategorii Nízké', fontsize=10)
ax2.set_xlim(0.3, 2.0)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.3, linestyle=':')
ax2.spines[['top', 'right']].set_visible(False)

# Osa x: popisky v násobcích prevalence
ax2.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x:.2g}× prev.\n({x*prevalence*100:.1f} %)')
)
ax2.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# Zvýraznění zvoleného prahu v obou panelech
for ax in (ax1, ax2):
    ax.axvspan(chosen_mult - 0.01, chosen_mult + 0.01, color='#e74c3c', alpha=0.08)

fig.text(0.5, 0.01,
         f'Prevalence kohorty = {prevalence*100:.2f} %  |  testovací sada n = {n_test}  |  '
         f'Zvolený práh 1× = {chosen_thr*100:.2f} %',
         ha='center', fontsize=8.5, color='#666',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 1])

for ext in ('png', 'pdf'):
    out = FIGURES / f'threshold_sensitivity.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {out}")

plt.close()
print("Hotovo.")
