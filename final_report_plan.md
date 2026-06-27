# Final Report Plan — AFib Risk Calculator (STROCZE)
# ICRC / Brno University of Technology — Richard Redina

**Deadline: 15. 7. 2026** *(aktualizováno 27. 6. 2026)*
**Jazyk dokumentace: čeština** | **Kód: angličtina**
**Hosting: GitHub Pages (statická HTML stránka)**

---

## Přehled výstupů

| Modul | Výstup | Složka | Priorita | Odhad |
|-------|--------|--------|----------|-------|
| M0 | Technické prerekvizity (x_test export, Platt params, RFECV scores) | `risk_model.py` | **KRITICKÁ** | 3 hod |
| M1 | Vizualizace — popis dat a explorace (5 skriptů) | `final_report/` | VYSOKÁ | 4 hod |
| M2 | Vizualizace — výkon modelu a kalibrace (4 skripty) | `final_report/` | VYSOKÁ | 4 hod |
| M3 | Vizualizace — srovnání s CHA₂DS₂-VASc baseline (2 skripty) | `final_report/` | VYSOKÁ | 3 hod |
| M4 | Citlivostní analýza prahů rizikových kategorií (1 skript) | `final_report/` | STŘEDNÍ | 2 hod |
| M5 | TAČR dokumentace (5 dokumentů) | `docs/tacr/` | VYSOKÁ | 8 hod |
| M6 | HTML kalkulačka | `web/` | VYSOKÁ | 5 hod |
| M7 | GitHub repozitář (README, .gitignore, CI workflow) | root | STŘEDNÍ | 3 hod |
| M8 | Video funkčnosti (3–5 min) | — | STŘEDNÍ | 3 hod |

**Celkový odhad: ~35 hodin práce v 18 dnech → průměrně 2 hod/den, reálně soustředěné bloky.**

---

## M0 — Technické prerekvizity *(implementovat jako úplně první!)*

Tyto kroky odblokují všechny ostatní moduly. Bez M0 musí každý
vizualizační skript re-trénovat model, nebo HTML kalkulačka nemůže
správně počítat pravděpodobnosti.

---

### M0.1 — Export testovacích dat (`results/risk_model_x_test.npy` + `_y_test.npy`)

**Problém:** `x_test` a `y_test` existují pouze v paměti během `train()`.
Vizualizační skripty (ROC křivka, rizikové kategorie, confusion matrix) je potřebují
bez nutnosti re-trénovat model.

**Řešení:** Do `risk_model.py → export_scorecard()` přidat:
```python
np.save(model_path.replace('.joblib', '_x_test.npy'), self.x_test)
np.save(model_path.replace('.joblib', '_y_test.npy'), self.y_test)
```

**Bezpečnost:** Soubory obsahují pouze 3 číselné hodnoty per pacient (CHA₂DS₂-VASc,
0/1, 0/1) bez ID, jmen ani jiných identifikátorů. Bezpečné pro commit do gitu.

---

### M0.2 — Extrakce Platt scaling parametrů do `scorecard.json`

**Problém:** Sigmoid funkce Platt scalingu má 2 parametry `a` a `b` uvnitř
`calibrated_model.calibrators_[0].a_` a `.b_`. Bez nich nelze v JS
korektně převést logit → kalibrovanou pravděpodobnost.

**Řešení:** Do `_calibrate_model()` přidat do `calibration_results`:
```python
cal_obj = self.calibrated_model.calibrators_[0]
self.calibration_results['platt_a'] = float(cal_obj.a_)
self.calibration_results['platt_b'] = float(cal_obj.b_)
```

**Použití v JS:** `p = 1 / (1 + exp(-(a * raw_logit + b)))` kde
`raw_logit = model.intercept_ + Σ(β_i * x_i)` (neškálovaný, před bodovým převodem).

---

### M0.3 — Uložení RFECV průběhu skóre do `scorecard.json`

**Problém:** Graf AUC vs. počet features (M1.5) potřebuje `selector.cv_results_`.
RFECV objekt se po trénování zahodí.

**Řešení:** V `_select_features_rfecv()` přidat do `training_data_info`:
```python
self.training_data_info['rfecv_n_features_range'] = list(range(
    selector.min_features_to_select, len(all_features) + 1
))
self.training_data_info['rfecv_mean_scores'] = \
    selector.cv_results_['mean_test_score'].tolist()
self.training_data_info['rfecv_std_scores'] = \
    selector.cv_results_['std_test_score'].tolist()
```

---

### M0.4 — Vygenerovat `requirements.txt`

```
pip freeze > requirements.txt
```
Ručně prořezat na skutečně potřebné balíčky. Přibližný obsah:
`scikit-learn, pandas, numpy, matplotlib, statsmodels, joblib, openpyxl, pyyaml`.

---

## M1 — Vizualizace: Popis dat a explorace (`final_report/`)

Každý skript je samostatný (`python final_report/01_nazev.py`) a uloží výstupy
do `final_report/figures/` a `final_report/tables/`.

**Sdílené konvence pro všechny vizualizační skripty:**
- DPI: 300 (publikační kvalita)
- Výstupní formáty: `.png` i `.pdf` (PDF pro vkládání do Word/LaTeX bez ztráty)
- Popisky, legendy, títulky: čeština
- Nesignifikantní výsledky: šedá; p < 0.05: plná barva
- Každý skript na konci vytiskne `Uloženo: final_report/figures/XX.png`

---

### M1.1 — Popis kohorty (`01_cohort_description.py`)

**Výstupy:** `tables/cohort_demographics.csv`, `tables/cohort_completeness.csv`

**Tabulka 1 — demografické charakteristiky:**

| Charakteristika | Hodnota |
|---|---|
| Celkem pacientů v datasetu | n = ? |
| Kompletní záznamy (použité v modelu) | n = 1884 (??%) |
| AFib+ záchyt (MDT celkově) | n = 111 (5.9%) |
| Trénovací sada | n = 1507 (80%), AFib+ 89 (5.9%) |
| Testovací sada | n = 377 (20%), AFib+ 22 (5.8%) |
| Věk — medián [IQR] | z Excel dat |
| Pohlaví — ženy | z Excel dat |

**Tabulka 2 — completeness per feature:**
Per-feature % chybějících hodnot z `training_data_info.feature_completeness`.
Zdůvodňuje, proč se plný dataset zúžil na 1884 kompletních záznamů.

**Zdroj:** `results/risk_scorecard.json` + Excel soubor (věk, pohlaví).

---

### M1.2 — Forest plot binárních asociací (`02_exploratory_binary.py`)

**Výstup:** `figures/forest_plot_binary.png`, `tables/binary_associations.csv`

**Graf:**
- Horizontální forest plot, 12 binárních proměnných na ose y, seřazeno dle OR
- Každá proměnná: tečka (OR) + linie (95% CI, Woolfova metoda)
- Svislá přímka OR = 1 (nulová asociace)
- Barva: červená (p < 0.05), šedá (p ≥ 0.05)
- Vpravo od grafu: sloupec s hodnotami OR [CI] a p-hodnotou
- Zvýraznit 3 vybrané features modelu (tučně nebo hvězdičkou)

**Tabulka:**
| Proměnná | OR | 95% CI dolní | 95% CI horní | p-hodnota | Vybrána modelem |
|---|---|---|---|---|---|

**Zdroj:** `results/analysis_results_binary_variables_zachyt_FiS_mapping_correlations.json`
Pokud JSON neexistuje: skript spustí `VariableCorrelationAnalyzer().analyze_binary()`.

---

### M1.3 — Tabulka kontinuálních asociací (`03_exploratory_continuous.py`)

**Výstup:** `tables/continuous_associations.csv`

**Tabulka:**
| Proměnná | β (log-OR) | p-hodnota | Typ testu | % dat použito |
|---|---|---|---|---|
Seřadit dle p-hodnoty, zvýraznit p < 0.05.

Vizualizace grafem není nutná (19 proměnných, spojité hodnoty), ale tabulka
ukazuje, které proměnné byly statisticky asociovány a mohly být do modelu zahrnuty.

**Zdroj:** `results/analysis_results_continuous_variables_zachyt_FiS_mapping_correlations.json`

---

### M1.4 — Boxploty signifikantních kontinuálních proměnných (`04_significant_boxplots.py`)

**Výstup:** `figures/boxplots_continuous_significant.png`

**Graf:**
- Sada párových boxplotů (2×N mřížka) pouze pro proměnné s p < 0.05
- Každý boxplot: AFib− vlevo (šedý), AFib+ vpravo (modrý)
- Nad každým párem: p-hodnota a β koeficient
- Přidává vizuální důkaz k numerické tabulce M1.3

**Zdroj:** Excel + `results/analysis_results_continuous_*.json`

---

### M1.5 — RFECV křivka — výběr počtu features (`05_rfecv_curve.py`)

**Výstup:** `figures/rfecv_feature_selection.png`

**Graf:**
- Osa x: počet features (3 až 8)
- Osa y: průměrné AUC z 5-fold CV ± std (stínovaný pás)
- Svislá čárkovaná přímka na n = 3 s popiskem „Zvoleno: 3 features"
- Marker na hodnotě AUC pro n = 3
- Inset tabulka: vybrané vs. odstraněné features

**Zdroj:** Po M0.3 přímo ze `scorecard['training_data_info']['rfecv_mean_scores']` —
skript **nepotřebuje** re-trénovat model.

---

## M2 — Vizualizace: Výkon modelu a kalibrace (`final_report/`)

---

### M2.1 — Koeficienty modelu (`06_model_coefficients.py`)

**Výstup:** `figures/model_coefficients.png`, `tables/model_scorecard.csv`

**Graf:**
- Horizontální sloupcový graf, 3 features
- Osa x: `points_per_unit` (bodová hodnota na jednotku)
- Barva: modrá (pozitivní β), červená (záporný β)
- Popisky hodnot přímo na sloupcích
- Poznámka pod grafem: `* záporný koeficient Hyperlipidémie — treatment paradox
  (pacienti s hyperlipidémií pravděpodobně užívají statiny, které snižují riziko FiS;
  data STROCZE neobsahují informaci o medikaci — tato interpretace je hypotéza)`

**Tabulka:**
| Feature | β | OR (e^β) | Body/jednotku | Typ | Poznámka |
|---|---|---|---|---|---|
| CHA₂DS₂-VASc | + | > 1 | +11.0 | kontinuální | |
| Teritoriální infarkt | + | > 1 | +10.3 | binární | |
| Hyperlipidémie | − | < 1 | −10.0 | binární | treatment paradox* |

**Zdroj:** Přímo ze `scorecard['variables']` a `scorecard['intercept']` — nevyžaduje re-trénink.

---

### M2.2 — ROC křivka s 95% CI (`07_roc_curve.py`)

**Výstup:** `figures/roc_curve.png`

**Graf:**
- Hlavní ROC křivka (modrá linie) — fpr vs. tpr na testovacím setu
- Stínovaný 95% CI pás — interpolovaný z 1000× bootstrap distribuce
  (re-spustit stratifikovaný bootstrap nad x_test.npy / y_test.npy)
- Diagonála (šedá čárkovaná) — náhodný klasifikátor
- Youdenův optimální bod (červený × marker) s popiskem threshold
- Textový box v grafu: `AUC = 0.647 [0.543–0.740]  (95% CI, 1000 bootstrap)`
- Osa x: False Positive Rate (1 − Specificita), osa y: Sensitivita

**Zdroj:** `results/risk_model_calibrated.joblib` + `results/risk_model_x_test.npy`
+ `results/risk_model_y_test.npy` (vyžaduje M0.1).

---

### M2.3 — Reliability diagram — kalibrace (`08_calibration_plot.py`)

**Výstup:** `figures/calibration_reliability.png`

**Graf (2 subploty vedle sebe):**
- Vlevo: nekalibrovaný model
- Vpravo: kalibrovaný (Platt scaling)
- Každý subplot: body z reliability diagramu + ideální diagonála (šedá čárkovaná)
- Brier skóre jako textový label uvnitř každého subplotu
- Pod každým subplotem: histogram predikovaných pravděpodobností (distribuce pacientů)
- Nadpis: „Kalibrace modelu před a po Platt scaling"

**Tabulka Brier skóre (vložit do textu zprávy):**
| | Brier |
|---|---|
| Nekalibrovaný | 0.2394 |
| Kalibrovaný (Platt) | 0.0545 |
| No-skill baseline (~6%) | ~0.0550 |

**Zdroj:** Přímo ze `scorecard['calibration']` — **nepotřebuje** re-trénink.

---

### M2.4 — Rizikové kategorie: distribuce a NPV (`09_risk_categories.py`)

**Výstup:** `figures/risk_category_distribution.png`, `tables/risk_categories_summary.csv`

**Graf:**
- Sloupcový graf: počet pacientů v každé ze 4 kategorií
- Barvy: zelená / žlutá / oranžová / červená
- Nad každým sloupcem: absolutní počet + procento
- Druhá osa (pravá): NPV pro každou kategorii (horizontální linie nebo body)

**Tabulka:**
| Kategorie | Práh | Pravděp. rozsah | n (test) | % | NPV |
|---|---|---|---|---|---|
| Nízké | < 1× prevalence | < 5.8% | ? | ? | ~99% |
| Střední | 1–1.5× | 5.8–8.7% | ? | ? | ? |
| Vysoké | 1.5–2× | 8.7–11.6% | ? | ? | ? |
| Velmi vysoké | > 2× | > 11.6% | ? | ? | ? |

**Zdroj:** `risk_model_calibrated.joblib` + `x_test.npy` + `y_test.npy` (vyžaduje M0.1).

---

## M3 — Vizualizace: Srovnání s CHA₂DS₂-VASc baseline (`final_report/`)

Toto je **nejdůležitější analytická sekce** pro obhájení přidané hodnoty modelu.
Čtenář (TAČR hodnotitel, lékař) se vždy zeptá: „Proč nestačí jen CHA₂DS₂-VASc?"
Tato část odpovídá datově.

---

### M3.1 — Srovnání AUC: náš model vs. alternativy (`10_model_comparison.py`)

**Výstup:** `figures/model_comparison_roc.png`, `tables/model_comparison.csv`

**Graf:**
- Jeden panel se 3 ROC křivkami:
  1. CHA₂DS₂-VASc samotný (jen 1 feature, logistická regrese, bez kalibrace)
  2. CHA₂DS₂-VASc + Teritoriální infarkt (2 features)
  3. **Náš model** — 3 features + Platt kalibrace (zvýraznit tučnou čarou)
- Každá křivka s AUC hodnotou v legendě
- Diagonála (šedá čárkovaná)

**Tabulka srovnání:**
| Model | Features | AUC | Sensitivity (Youden) | Specificity (Youden) | Brier |
|---|---|---|---|---|---|
| CHA₂DS₂-VASc | 1 | ? | ? | ? | ? |
| + Teritoriální | 2 | ? | ? | ? | ? |
| **Náš model** | **3 + kal.** | **0.647** | **?** | **?** | **0.055** |

**Technický postup:**
1. Trénovat 3 modely na stejném `x_train` / `y_train` (vyžaduje re-trénink nebo uložení
   `x_train.npy`, `y_train.npy` — zvážit přidání do M0.1)
2. Evaluovat na stejném `x_test` / `y_test`
3. Plotovat ROC pro každý model
4. DeLong test nebo bootstrap pro testování statistické signifikance rozdílu AUC

**Poznámka:** Pokud rozdíl AUC mezi 1 a 3 features není statisticky signifikantní,
je to důležitá informace — přesto model přináší kalibrované pravděpodobnosti a kategorie,
které CHA₂DS₂-VASc samotný nenabízí. To je samo o sobě přidaná hodnota.

---

### M3.2 — Confusion matrix vizualizace (`11_confusion_matrix.py`)

**Výstup:** `figures/confusion_matrix.png`

**Graf:**
- Standardní confusion matrix heatmap pro náš model (kalibrovaný, Youden threshold)
- Hodnoty: absolutní počty + procenta
- Barevná škála: bílá → modrá
- Popisky: TP, TN, FP, FN s klinickým kontextem
  (TP = „FiS detekována, monitoring indikován" atd.)
- Pod maticí: Sensitivity, Specificity, PPV, NPV vypsané textově

**Zdroj:** `scorecard['calibration']` nebo re-spočítat z `x_test.npy` + `y_test.npy`.

---

## M4 — Citlivostní analýza prahů rizikových kategorií (`final_report/`)

---

### M4.1 — Robustnost NPV a distribuce kategorií (`12_threshold_sensitivity.py`)

**Výstup:** `figures/threshold_sensitivity.png`, `tables/threshold_sensitivity.csv`

**Motivace:** Prahy (< 1×, 1–1.5×, 1.5–2×, > 2× prevalence) jsou navržené, ne
statisticky optimalizované. Tato analýza ukáže, jak citlivé jsou výsledky na volbu prahů.

**Graf (2 subploty):**

*Subplot 1 — NPV kategorie Nízké při různých horních prazích:*
- Osa x: horní práh kategorie Nízké (0.5× až 1.5× prevalence)
- Osa y: NPV (kolik % pacientů v Nízké kategorii skutečně nemá FiS)
- Horizontální čára: 95% NPV (klinicky přijatelná hranice)
- Zvýraznit aktuálně zvolený práh (1× prevalence)

*Subplot 2 — % pacientů v každé kategorii při různých prazích:*
- Ukáže, že volba 1× je rozumný kompromis — nezatěžkává Velmi vysokou kategorii
  příliš mnoha pacienty ani ji nevyprazdňuje

**Tabulka:**
| Horní práh Nízké | NPV | n v Nízké | % kohorty |
|---|---|---|---|
| 0.5× (konzervativní) | ? | ? | ? |
| **1.0× (zvoleno)** | **~99%** | **?** | **?** |
| 1.5× (liberální) | ? | ? | ? |

**Zdroj:** `risk_model_calibrated.joblib` + `x_test.npy` + `y_test.npy`.

---

## M5 — TAČR Dokumentace (`docs/tacr/`)

Všechny soubory v češtině, formát Markdown. Lze převést na PDF přes pandoc:
`pandoc uzivatelska_prirucka.md -o uzivatelska_prirucka.pdf`

---

### M5.1 — Uživatelská příručka (`docs/tacr/uzivatelska_prirucka.md`)

**Obsah:**
1. Účel nástroje — stratifikace pacientů po CMP/TIA pro MDT monitoring FiS
2. Definice vstupních proměnných:
   - CHA₂DS₂-VASc: jak se počítá, rozsah 0–9, kde ho najdete v dokumentaci
   - Teritoriální infarkt: definice (velký kortikální nebo kortiko-subkortikální infarkt)
   - Hyperlipidémie: anamnesticky diagnostikovaná (bez ohledu na aktuální léčbu)
3. Jak používat HTML kalkulačku — krok za krokem se screenshoty
4. Interpretace výstupu — co znamená každá kategorie:
   - Nízké (< prevalence): „S 99% jistotou FiS přítomna není; standardní sledování."
   - Střední (1–1.5×): „Mírně zvýšené riziko; zvažte prodloužený monitoring."
   - Vysoké (1.5–2×): „Výrazně zvýšené riziko; MDT monitoring doporučen."
   - Velmi vysoké (> 2×): „Silné doporučení k MDT monitoringu FiS."
5. Jak spustit Python demo: `python predict.py`
6. Limitace a kontraindikace:
   - Pouze pro pacienty po iCMP/TIA (kohorta STROCZE)
   - Klinické rozhodnutí zůstává na lékaři
   - Nutná prospektivní validace před rutinním nasazením
   - Záporný vliv Hyperlipidémie — možný treatment paradox, interpretujte opatrně
7. Kontakt: Richard Redina, rredina@icloud.com / 195715@vut.cz, ICRC Brno, BUT

---

### M5.2 — Analýza funkčních požadavků (`docs/tacr/analyza_funkcionalnich_pozadavku.md`)

**Obsah:**
1. Kontext a motivace:
   - Přeléčení (přepráškování) antikoagulancii u pacientů s CHA₂DS₂-VASc ≥ 2
   - CHA₂DS₂-VASc nerozlišuje pacienty, kteří profitují z MDT monitoringu
   - Cíl: identifikovat podskupinu s vyšší pravděpodobností záchytu FiS na MDT

2. Funkční požadavky (FR):
   - FR1: Přijmout 3 klinické vstupy a vrátit rizikovou kategorii (4 stupně)
   - FR2: Vrátit kalibrovanou pravděpodobnost (0–1) reflektující kohortovou prevalenci
   - FR3: Kategorie definovány relativně ke kohortové prevalenci (~5.8%)
   - FR4: Brier score ≤ no-skill baseline (~0.055)
   - FR5: Dostupné bez instalace přes webový prohlížeč (GitHub Pages)
   - FR6: Kód open-source, reproducibilní, seed=42
   - FR7: EPV ≥ 10 pro statistickou robustnost

3. Nefunkční požadavky (NFR):
   - NFR1: AUC > 0.5 (lepší než náhoda), doloženo 95% CI
   - NFR2: NPV ≥ 95% pro kategorii Nízké (klinická bezpečnost)
   - NFR3: Interpretovatelný scorecard pro lékaře (bodový systém)
   - NFR4: Mobile-friendly HTML (lékaři používají tablety)

4. Co systém NEDĚLÁ (out of scope):
   - Diagnostika FiS (pouze predikce rizika záchytu při monitoringu)
   - Doporučení konkrétní antikoagulační terapie
   - Predikce mortality, recidivy CMP, jiných outcomes

---

### M5.3 — Technická dokumentace (`docs/tacr/technicka_dokumentace.md`)

**Obsah:**
1. Architektura systému (ASCII/Mermaid diagram):
   ```
   [Excel: STROCZE data]
       │
       ├──► analyze.py ──► VariableCorrelationAnalyzer
       │        │               • logistická regrese (19 kontin. proměnných)
       │        │               • Fisher exact test (12 binárních proměnných)
       │        └──► results/analysis_results_*.json
       │
       └──► risk_model.py ──► RiskScoreGenerator
                │               • RFECV (8 → 3 features)
                │               • LogisticRegression (class_weight='balanced')
                │               • Platt scaling (CalibratedClassifierCV)
                │               • Bootstrap CI (1000 iterací)
                └──► results/
                         ├── risk_scorecard.json
                         ├── risk_model.joblib
                         ├── risk_model_calibrated.joblib
                         ├── risk_model_x_test.npy
                         └── risk_model_y_test.npy

   predict.py ──────────────► Python demo (3 ukázkové pacienty)
   web/index.html ──────────► HTML kalkulačka (GitHub Pages)
   final_report/*.py ───────► Vizualizace a tabulky
   ```

2. Datové struktury:
   - Schéma `risk_scorecard.json` s popisem každého klíče a datového typu
   - Vstupní formát pro `predict_risk()`: `{feature_name: float/int}`

3. Výběr algoritmů a odůvodnění:
   - Logistická regrese: interpretovatelnost, klinická tradice (CHA₂DS₂-VASc je samo LR)
   - `class_weight='balanced'`: nutnost při 5.9% prevalenci (bez toho sens = 0)
   - RFECV: redukce overfitting rizika (cv-based výběr, ne threshold)
   - Platt scaling: korekce inflace pravděpodobností způsobené balanced weights
   - Stratifikovaný split: zachovává 5.9% ratio v obou sadách

4. Validační strategie:
   - 80/20 stratifikovaný split (random_state=42, reproducibilní)
   - 5-fold stratifikovaná CV (na plném datasetu, před splitem)
   - 1000× bootstrap CI pro AUC, sensitivitu, specificitu
   - EPV = 29.7 (doporučené minimum: 10)

5. Omezení:
   - Kalibraci nelze považovat za nezávislou validaci (fitted na x_test)
   - Chybějící data o medikaci (statiny) — treatment paradox nelze ověřit
   - Data ze single-center Czech kohorty — generalizovatelnost omezená

---

### M5.4 — Programátorská dokumentace (`docs/tacr/programatorska_dokumentace.md`)

**Obsah:**
1. Požadavky prostředí: Python 3.11+, závislosti z requirements.txt
2. Instalace: `git clone → pip install -r requirements.txt`
3. Struktura repozitáře (viz M7)
4. Hlavní třídy a metody — tabulka:

   | Třída / funkce | Soubor | Popis |
   |---|---|---|
   | `RiskScoreGenerator.train()` | risk_model.py | Kompletní trénink pipeline |
   | `RiskScoreGenerator.predict_risk(patient_dict)` | risk_model.py | Predikce pro pacienta |
   | `RiskScoreGenerator._calibrate_model()` | risk_model.py | Platt scaling |
   | `RiskScoreGenerator._select_features_rfecv()` | risk_model.py | RFECV výběr |
   | `RiskScoreGenerator._compute_bootstrap_ci()` | risk_model.py | 95% CI |
   | `RiskScoreGenerator.export_scorecard()` | risk_model.py | Export JSON + .joblib |
   | `VariableCorrelationAnalyzer.pipeline()` | analyze.py | Explorační analýza |
   | `predict_with_model(mdl, patient, features)` | predict.py | Inference z modelu |
   | `predict_from_scorecard(scorecard, patient)` | predict.py | Inference ze JSON |

5. Konfigurační soubory `config/*.yaml` — tabulka s popisem každého parametru
6. Ukázka end-to-end inference:
   ```python
   from utils.config_singleton import ConfigSingleton
   from risk_model import RiskScoreGenerator
   ConfigSingleton.set()
   gen = RiskScoreGenerator()
   gen.train()
   result = gen.predict_risk({
       'CHA₂DS₂-VASc': 5,
       'Typ akutní ischemie (choice=Teritoriální)': 1,
       'Osobní anamnéza (choice=Hyperlipidémie)': 0,
   })
   print(result)  # {'risk_probability': 0.092, 'risk_category': 'Vysoké', ...}
   ```
7. Jak přidat novou feature — step-by-step instrukce pro budoucí vývojáře:
   - Přidat do `variables.yaml` (příslušná sekce)
   - Přidat do `model.yaml → model_features`
   - Re-trénovat: RFECV rozhodne, zda je feature dostatečně informativní

---

### M5.5 — Ověření funkčnosti (`docs/tacr/overeni_funkčnosti.md`)

**Obsah:**
1. Testovací strategie — co testujeme a proč

2. Automatizované testy — souhrnná tabulka:
   | Fáze | Počet testů | Výsledek | Klíčové ověření |
   |---|---|---|---|
   | Split a indexování | 4 | PASS | Správné indexy po IQR filtraci |
   | Křížová validace | 12 | PASS | AUC > 0.5 ve všech foldech |
   | Class balancing | 12 | PASS | Sensitivity > 0 (bez balancingu: 0) |
   | Testovací metriky | 15 | PASS | AUC, Brier, NPV v očekávaném rozsahu |
   | Bootstrap CI | 15 | PASS | CI dolní mez > 0.5 (lepší než náhoda) |
   | RFECV výběr | 16 | PASS | Vybrány 3 features z 8 |
   | EPV check | 9 | PASS | EPV = 29.7 ≥ 10 |
   | Kalibrace | 18 | PASS | Brier snížen z 0.239 na 0.055 |
   | **Celkem** | **101** | **PASS** | |

3. Výkonnostní metriky na testovacím setu (n = 377, AFib+ = 22):
   | Metrika | Hodnota | 95% CI |
   |---|---|---|
   | AUC-ROC | 0.647 | [0.543 – 0.740] |
   | Brier (kalibrovaný) | 0.055 | — (in-sample) |
   | NPV (kategorie Nízké) | 99.1% | — |
   | EPV | 29.7 | — |

4. Reprodukovatelnost: spuštění `python risk_model.py` s `random_state=42`
   vždy produkuje identické výsledky

5. Ukázkové predikce (3 pacienti) — reprodukovatelný výstup z `predict.py`

6. Limitace a doporučení pro budoucí validaci:
   - Interní testovací set není nezávislou validací
   - Doporučení: prospektivní validace na nových pacientech STROCZE nebo externích datech
   - Kalibraci ověřit na nezávislém setu před klinickým nasazením

---

## M6 — HTML Kalkulačka (`web/`)

Statická stránka, hostovaná na GitHub Pages.
URL: `https://RicRedi.github.io/Chadsvasc/`

---

### M6.1 — Generátor JS souboru (`web/generate_scorecard_js.py`)

Spustit po každém re-trénování modelu:
```
python web/generate_scorecard_js.py
```

Skript načte `results/risk_scorecard.json` a `results/risk_model_calibrated.joblib`,
extrahuje:
- `intercept_` z logistické regrese (raw logit intercept)
- `coef_` pro každou feature
- Platt scaling parametry `a_` a `b_`
- Kohortovou prevalenci

Vygeneruje `web/assets/scorecard.js`:
```javascript
// AUTO-GENERATED — needitovat ručně, spusť generate_scorecard_js.py
const SCORECARD = {
  lr_intercept: -3.42,
  lr_coefs: { "CHA2DS2VASc": 0.31, "Teritorialni": 0.29, "Hyperlipidemie": -0.28 },
  platt_a: -1.87,
  platt_b: 0.95,
  prevalence: 0.058,
  feature_order: ["CHA2DS2VASc", "Teritorialni", "Hyperlipidemie"]
};
```

**Kalkulační vzorec v JS:**
```javascript
function predict(cha2ds2, terit, hlp) {
  const logit = SCORECARD.lr_intercept
    + SCORECARD.lr_coefs.CHA2DS2VASc * cha2ds2
    + SCORECARD.lr_coefs.Teritorialni * terit
    + SCORECARD.lr_coefs.Hyperlipidemie * hlp;
  const prob = 1 / (1 + Math.exp(-(SCORECARD.platt_a * logit + SCORECARD.platt_b)));
  const ratio = prob / SCORECARD.prevalence;
  return { prob, ratio, category: getCategory(ratio) };
}
```

---

### M6.2 — HTML struktura (`web/index.html`)

**Layout (single-page, mobile-friendly):**
```
┌─────────────────────────────────────────────────────┐
│  🫀 AFib Risk Kalkulačka                            │
│  STROCZE kohorta · ICRC Brno · BUT                 │
├─────────────────────────────────────────────────────┤
│  VSTUPY                                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ CHA₂DS₂-VASc skóre:                        │   │
│  │ [  0  ▲▼ ]  (zadejte celé číslo 0–9)       │   │
│  │                                              │   │
│  │ Typ ischemie:                                │   │
│  │ ● Teritoriální    ○ Jiný / neznámý          │   │
│  │                                              │   │
│  │ Hyperlipidémie v anamnéze:                   │   │
│  │ ● Ano             ○ Ne / neznámo             │   │
│  └─────────────────────────────────────────────┘   │
│               [ Vypočítat riziko FiS ]              │
├─────────────────────────────────────────────────────┤
│  VÝSLEDEK                    (barva dle kategorie)  │
│  ┌─────────────────────────────────────────────┐   │
│  │  RIZIKO FiS:  Vysoké                        │   │
│  │  Výrazně zvýšené riziko záchytu FiS          │   │
│  │  Pravděpodobnost záchytu: 9.2 %             │   │
│  │  (1.6× kohortová prevalence 5.8 %)          │   │
│  └─────────────────────────────────────────────┘   │
│  [ⓘ Poznámka k Hyperlipidémii]  (rozbalovací)      │
├─────────────────────────────────────────────────────┤
│  VÝKON MODELU  (sbalitelné)                         │
│  AUC 0.647 [0.543–0.740] · NPV 99.1% · Brier 0.055│
├─────────────────────────────────────────────────────┤
│  [GitHub] [Dokumentace] [Publikace]                 │
│  ⚠ Pouze pro výzkumné účely. Klinické rozhodnutí    │
│  zůstává v kompetenci ošetřujícího lékaře.          │
└─────────────────────────────────────────────────────┘
```

**Technické detaily:**
- Čistý HTML5 + vanilla JS (žádné npm, žádné frameworky — 0 závislostí)
- CSS: responsivní grid, funguje na telefonu i tabletu
- Barvy dle kategorie: #2ecc71 (nízké) / #f39c12 (střední) / #e67e22 (vysoké) / #e74c3c (velmi vysoké)
- Rozbalovací sekce [ⓘ] s celým textem o treatment paradoxu Hyperlipidémie
- Výsledek se zobrazí pouze po kliknutí na tlačítko (validace vstupu)
- Input validace: CHA₂DS₂-VASc musí být 0–9 (integer)
- Disclaimer fixně viditelný v patičce

---

### M6.3 — GitHub Pages nasazení

- `web/` složka jako GitHub Pages source: Settings → Pages → Deploy from branch `/web`
- `web/.nojekyll` soubor (prázdný) — zabraňuje Jekyll zpracování
- `web/index.html` je vstupní bod
- Po nasazení ověřit na mobilním prohlížeči

---

## M7 — GitHub Repozitář

### Struktura repozitáře

```
Chadsvasc/
├── README.md                          ← vstupní bod (CZ primárně + EN abstrakt)
├── requirements.txt                   ← pip závislosti
├── .gitignore                         ← viz níže
│
├── config/                            ← YAML konfigurace
│   ├── config.yaml
│   ├── variables.yaml
│   ├── model.yaml
│   ├── analysis.yaml
│   └── plotting.yaml
│
├── core.py                            ← příprava dat, IQR filter, binárization
├── analyze.py                         ← VariableCorrelationAnalyzer
├── risk_model.py                      ← RiskScoreGenerator
├── predict.py                         ← Python demo (3 ukázkové pacienty)
├── plotting.py                        ← CorrelationPlotter
├── utils/                             ← ConfigSingleton a helpers
│
├── final_report/                      ← vizualizační skripty
│   ├── 01_cohort_description.py
│   ├── 02_exploratory_binary.py
│   ├── 03_exploratory_continuous.py
│   ├── 04_significant_boxplots.py
│   ├── 05_rfecv_curve.py
│   ├── 06_model_coefficients.py
│   ├── 07_roc_curve.py
│   ├── 08_calibration_plot.py
│   ├── 09_risk_categories.py
│   ├── 10_model_comparison.py
│   ├── 11_confusion_matrix.py
│   ├── 12_threshold_sensitivity.py
│   ├── figures/                       ← PNG + PDF výstupy
│   └── tables/                        ← CSV výstupy
│
├── docs/
│   └── tacr/
│       ├── uzivatelska_prirucka.md
│       ├── analyza_funkcionalnich_pozadavku.md
│       ├── technicka_dokumentace.md
│       ├── programatorska_dokumentace.md
│       └── overeni_funkčnosti.md
│
├── web/                               ← GitHub Pages (statická HTML kalkulačka)
│   ├── index.html
│   ├── .nojekyll
│   ├── assets/
│   │   ├── scorecard.js               ← AUTO-GENERATED (generate_scorecard_js.py)
│   │   └── style.css
│   └── generate_scorecard_js.py
│
└── results/                           ← GITIGNORED (kromě scorecard.json)
    ├── risk_scorecard.json            ← ✓ DO GITU (jen čísla, žádná data)
    ├── risk_model.joblib              ← ✗ gitignored (binární)
    ├── risk_model_calibrated.joblib   ← ✗ gitignored (binární)
    ├── risk_model_x_test.npy          ← ✓ DO GITU (žádná PII)
    └── risk_model_y_test.npy          ← ✓ DO GITU (jen 0/1 labels)
```

### `.gitignore` klíčové položky
```gitignore
# Patient data — NEVER commit
src/

# Virtual environment
.env/
.venv/
venv/

# Binary model files (large, reproducible from training)
results/*.joblib

# Python cache
__pycache__/
*.pyc
*.pyo

# OS
.DS_Store
Thumbs.db
```

### `README.md` obsah
1. Badges: Python 3.11 | scikit-learn 1.6 | License: MIT
2. Název a stručný popis (čeština — 3 věty)
3. Anglický abstrakt (1 odstavec) — pro mezinárodní čtenáře
4. Quickstart:
   ```bash
   git clone https://github.com/RicRedi/Chadsvasc.git
   pip install -r requirements.txt
   python predict.py       # Python demo
   # nebo otevřít web/index.html v prohlížeči
   ```
5. Odkaz na HTML kalkulačku (GitHub Pages)
6. Odkaz na TAČR dokumentaci (`docs/tacr/`)
7. Citace projektu (STROCZE kohorta, ICRC, BUT)
8. Disclaimer: „Výzkumný nástroj. Není schválen pro klinické rozhodování."

### GitHub Actions CI workflow (`.github/workflows/tests.yml`)
Spustit automaticky při každém push:
```yaml
- python risk_model.py   # trénink (ověří, že pipeline projde)
- python predict.py      # demo (ověří, že inference funguje)
```
Jednoduchý smoke test — ne unit testy (ty vyžadují data).

---

## M8 — Video funkčnosti

**Délka:** 3–5 minut
**Formát:** MP4, 1080p, mluvený komentář v češtině
**Nástroj:** OBS Studio (zdarma) nebo Windows Game Bar (Win + G)

**Scénář:**

| Čas | Obsah | Zobrazení na obrazovce |
|-----|-------|----------------------|
| 00:00–00:30 | Úvod — kontext FiS po CMP, motivace | Titulní slide nebo prázdná plocha |
| 00:30–01:30 | `python predict.py` — spuštění dema | Terminál |
| 01:30–02:00 | Výpis features a performance summary | Terminál — zvýraznit AUC a NPV |
| 02:00–03:00 | HTML kalkulačka — 2 pacienti | Prohlížeč |
| 03:00–03:30 | GitHub repo — struktura, dokumentace | Prohlížeč / VS Code |
| 03:30–04:00 | Závěr — limitace, nutnost validace | Terminál nebo slide |

**Klíčové momenty k verbálnímu zdůraznění:**
- Proč 3 features místo 8 (RFECV)
- Co znamená NPV 99% pro kategorii Nízké
- Záporný koeficient Hyperlipidémie — zmínit treatment paradox a limitaci

---

## Harmonogram (18 dní, deadline 15. 7. 2026)

| Týden | Dny | Co dělat | Klíčové výstupy |
|-------|-----|----------|-----------------|
| **Týden 1** | 27.6 – 3.7 | M0 prerekvizity + M1 explorace + M2 výkon modelu | `x_test.npy`, grafy M1+M2, Platt params v JSON |
| **Týden 2** | 4.7 – 10.7 | M3 srovnání + M4 citlivost + M6 HTML kalkulačka | Baseline comparison, HTML online |
| **Týden 3** | 11.7 – 15.7 | M5 TAČR docs + M7 GitHub cleanup + M8 video | Kompletní dokumentace, video, finální push |

**Detailní harmonogram — Týden 1:**
| Den | Cíl |
|-----|-----|
| Pá 27.6 | M0 (x_test export, Platt params, RFECV scores, requirements.txt) |
| So 28.6 | M2.3 kalibrační plot (nepotřebuje x_test) + M2.1 koeficienty |
| Ne 29.6 | M1.2 forest plot + M1.1 kohorta |
| Po 30.6 | M1.3 kontinuální tabulka + M1.4 boxploty |
| Út 1.7  | M1.5 RFECV křivka + M2.2 ROC křivka |
| St 2.7  | M2.4 rizikové kategorie + M2.1 confusion matrix |
| Čt 3.7  | Review všech grafů, opravy, buffer |

**Detailní harmonogram — Týden 2:**
| Den | Cíl |
|-----|-----|
| Pá 4.7  | M3.1 srovnání modelů (AUC tabulka + ROC srovnání) |
| So 5.7  | M3.2 confusion matrix + M4.1 citlivostní analýza |
| Ne 6.7  | M6.1 generate_scorecard_js.py + extrakce koeficientů |
| Po 7.7  | M6.2 index.html + style.css |
| Út 8.7  | M6.3 GitHub Pages nasazení + testování na mobilu |
| St 9.7  | Buffer / opravy HTML |
| Čt 10.7 | Review HTML, cross-browser test |

**Detailní harmonogram — Týden 3:**
| Den | Cíl |
|-----|-----|
| Pá 11.7 | M5.1 uživatelská příručka + M5.2 funkční požadavky |
| So 12.7 | M5.3 technická dok. + M5.4 programátorská dok. |
| Ne 13.7 | M5.5 ověření funkčnosti + M7 README + .gitignore |
| Po 14.7 | M8 video nahrávka + M7 GitHub Actions |
| Út 15.7 | Finální review, git push, odevzdání ✓ |

---

## Otevřené otázky

1. **Export `x_train.npy` a `y_train.npy`:** M3.1 potřebuje trénovat baseline modely
   na stejné trénovací sadě. Buď přidat jejich export (jako x_test), nebo M3 skript
   vždy re-trénuje celý model (pomalejší, ale jednodušší).
   **Doporučení:** Přidat do M0.1 — malé soubory, bez PII.

2. **DeLong test pro srovnání AUC:** Je třeba externího balíčku (`scipy` nestačí).
   Možnosti: `statsmodels`, nebo vlastní bootstrap přístup (spočítat rozdíl AUC
   v každém bootstrap samplu → CI pro rozdíl). Bootstrap je robustnější volba.

3. **results/risk_scorecard.json v gitu:** Obsahuje pouze koeficienty a metriky,
   žádná individuální data. → Bezpečné pro commit.

4. **GitHub repo název:** Aktuální název `Chadsvasc` působí neoficiálně. Zvážit
   přejmenování na `afib-risk-strocze` nebo `strocze-afib-calculator` pro lepší
   dohledatelnost. (Přejmenování repo neovlivní kód, GitHub automaticky přesměruje.)

---

*Plán vytvořen: 27. 6. 2026 | Aktualizován: 27. 6. 2026*
*Deadline: 15. 7. 2026*
*Autor: Richard Redina / Claude Sonnet 4.6*
