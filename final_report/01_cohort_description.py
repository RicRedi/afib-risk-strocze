# -*- coding: utf-8 -*-
"""
M1.1 — Popis kohorty
Výstupy: tables/cohort_demographics.csv, tables/cohort_completeness.csv
Zdroj:   results/risk_scorecard.json (+ volitelně Excel pro věk/pohlaví)
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
TABLES  = Path(__file__).resolve().parent / 'tables'
TABLES.mkdir(exist_ok=True)

with open(RESULTS / 'risk_scorecard.json', encoding='utf-8') as f:
    sc = json.load(f)

tdi = sc['training_data_info']
tp  = sc['test_performance']

n_complete = tdi['rows_with_complete_data']
n_total    = tdi['total_rows_in_dataset']
n_train    = tdi['train_samples']
n_test     = tp['n_test']
n_pos      = tdi['train_positive'] + tp['n_positive']

# --- Tabulka 1: demografické charakteristiky ---
demo_rows = [
    ('Celkem pacientů v datasetu',            f"n = {n_total}"),
    ('Kompletní záznamy (použité v modelu)',  f"n = {n_complete} ({n_complete/n_total*100:.1f} %)"),
    ('AFib+ záchyt (MDT celkově)',            f"n = {n_pos} ({n_pos/n_complete*100:.1f} %)"),
    ('Trénovací sada',                        f"n = {n_train} (80 %), AFib+ {tdi['train_positive']} ({tdi['train_positive']/n_train*100:.1f} %)"),
    ('Testovací sada',                        f"n = {n_test} (20 %), AFib+ {tp['n_positive']} ({tp['n_positive']/n_test*100:.1f} %)"),
    ('Věk — medián [IQR]',                   'viz Excel data (není v scorecard)'),
    ('Pohlaví — ženy',                       'viz Excel data (není v scorecard)'),
]

demo_df = pd.DataFrame(demo_rows, columns=['Charakteristika', 'Hodnota'])
demo_path = TABLES / 'cohort_demographics.csv'
demo_df.to_csv(demo_path, index=False, encoding='utf-8-sig')
print(f"Uloženo: {demo_path}")
print(demo_df.to_string(index=False))

# --- Tabulka 2: completeness per feature ---
completeness = tdi.get('feature_completeness', {})
comp_rows = []
for feat, info in completeness.items():
    comp_rows.append({
        'Proměnná':         feat,
        'Celkem záznamů':   info['original_count'],
        'Po čištění':       info['after_cleaning'],
        'Completeness (%)': info['completeness_%'],
    })

comp_df = pd.DataFrame(comp_rows).sort_values('Completeness (%)', ascending=True)
comp_path = TABLES / 'cohort_completeness.csv'
comp_df.to_csv(comp_path, index=False, encoding='utf-8-sig')
print(f"\nUloženo: {comp_path}")
print(comp_df.to_string(index=False))
