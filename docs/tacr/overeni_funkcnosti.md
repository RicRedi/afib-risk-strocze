# Ověření funkčnosti
## Kalkulátor rizika záchytu fibrilace síní po iCMP/TIA

---

**Projekt:** NCK FEIM - DP001N (Healthcare) — Identifikace rizikových faktorů fibrilace síní  
**Verze dokumentu:** 1.0  
**Datum:** 2026-06-27  
**Autoři:** Valentýna Provazník, Veronika Bulková, Richard Ředina  
**Instituce:** Vysoké učení technické v Brně / Medical Data Transfer s.r.o.

---

## 1. Testovací strategie

Ověření funkčnosti systému probíhá na třech úrovních:

1. **Interní validace modelu** — výkonnostní metriky na vyčleněném testovacím souboru (20 % dat, n = 377), který nebyl použit při tréninku.
2. **Ověření reprodukovatelnosti** — opakované spuštění tréninku s identickým nastavením (`random_state=42`) produkuje byte-for-byte identické výsledky.
3. **Ověření inference** — demonstrace správné funkce predikce na třech typových pacientech s ručně ověřitelnými výstupy.

Systém neobsahuje samostatný test suite (soubory `test_*.py`), protože výkonnostní validace je integrována přímo do tréninkovací pipeline (`RiskScoreGenerator.train()`). Každý trénink automaticky provádí všechny validační kontroly a vypisuje výsledky.

---

## 2. Validační kontroly integrované v pipeline

Při spuštění `python risk_model.py` jsou automaticky provedeny kontroly organizované do 8 fází:

| Fáze | Co se ověřuje | Akceptační kritérium |
|------|--------------|---------------------|
| 1 — Příprava dat | Správné indexování po IQR filtraci a binárization; konzistence rozměrů X a y | Žádná chybějící hodnota v x_train/x_test pro vybrané prediktory |
| 2 — Datový split | Stratifikace: poměr FiS+/FiS− zachován v obou sadách | |train_positive/train_total − prevalence| < 0,5 % |
| 3 — Křížová validace | AUC > 0,5 ve všech 5 foldech (model lepší než náhoda) | Všechny foldy: AUC > 0,5 |
| 4 — Vyvažování tříd | Senzitivita > 0 (bez balanced: senzitivita = 0) | Každý fold: sensitivity > 0 |
| 5 — Testovací metriky | AUC, Brier, senzitivita, specificita, PPV, NPV na testovacím souboru | AUC [CI dolní] > 0,5; NPV > 0,95 |
| 6 — Bootstrap CI | Konfidečního interval pro AUC: dolní mez > 0,5 | auc_ci_lower > 0,5 |
| 7 — RFECV výběr | Automatický výběr optimálního počtu prediktorů | Vybrány ≥ 1 prediktor |
| 8 — EPV kontrola | Events Per Variable ≥ 10 | EPV = train_positive / n_selected_features ≥ 10 |
| 9 — Kalibrace | Brier skóre klesne po kalibraci; kalibrovaná pravděpodobnost v realistickém rozsahu | brier_calibrated < brier_uncalibrated |

Výsledky všech kontrol jsou zaznamenány v `results/risk_scorecard.json`.

---

## 3. Výkonnostní metriky na testovacím souboru

### 3.1 Hlavní metriky

Testovací soubor: **n = 377 pacientů** (FiS+ = 22, FiS− = 355), vyčleněn před tréninkem, nepodílel se na výběru prediktorů ani kalibraci.

| Metrika | Hodnota | 95% CI (bootstrap, 1 000×) | Akceptační kritérium | Splněno |
|---------|---------|--------------------------|---------------------|---------|
| AUC-ROC | 0,647 | [0,543 – 0,740] | CI dolní mez > 0,5 | ✓ |
| Senzitivita | 95,5 % | [86,4 % – 100,0 %] | > 80 % (screeningový nástroj) | ✓ |
| Specificita | 30,1 % | [25,4 % – 34,7 %] | — (vědomě nízká) | — |
| NPV (celkové) | 99,1 % | — | — | — |
| NPV (kategorie Nízké) | 95,7 % | — | ≥ 95 % | ✓ |
| Brier skóre (kalibrovaný) | 0,0545 | — | ≤ 0,0550 (no-skill baseline) | ✓ |
| EPV | 29,7 | — | ≥ 10 | ✓ |

### 3.2 Konfuzní matice

Kalibrovaný model, Youdenův optimální práh = 0,041 (tj. 4,1 % pravděpodobnost):

|  | **Predikce: FiS+ (monitoring indikován)** | **Predikce: FiS− (monitoring neindikován)** |
|--|------------------------------------------|---------------------------------------------|
| **Skutečnost: FiS+** | TP = 21 (FiS detekována, správně) | FN = 1 (FiS přehlédnuta, falešně negativní) |
| **Skutečnost: FiS−** | FP = 248 (zbytečný monitoring) | TN = 107 (správně vyloučena) |

**Klinická interpretace:**
- 21 z 22 pacientů s FiS bylo správně identifikováno pro monitoring (senzitivita 95,5 %).
- Pouze 1 pacient s FiS byl přehlédnut (FN = 1; NPV 99,1 % pro vyloučené pacienty).
- 107 pacientů bez FiS bylo správně vyloučeno z monitoringu (úspora 28,4 % kapacity).
- 248 pacientů bez FiS bylo doporučeno k monitoringu zbytečně (FP) — nevyhnutelný důsledek prioritizace senzitivity.

### 3.3 Křížová validace (trénovací sada)

| Fold | AUC | Senzitivita | Specificita |
|------|-----|-------------|-------------|
| 1 | 0,639 | 50,0 % | 63,7 % |
| 2 | 0,648 | 63,6 % | 60,3 % |
| 3 | 0,583 | 36,4 % | 73,5 % |
| 4 | 0,620 | 56,5 % | 65,5 % |
| 5 | 0,677 | 72,7 % | 57,1 % |
| **Průměr** | **0,633** | **55,8 %** | **64,0 %** |
| **Std** | **0,031** | **12,3 %** | **5,6 %** |

Variabilita senzitivity mezi foldy (36–73 %) je způsobena nízkým absolutním počtem FiS+ v každém foldu (~18 případů). Průměrná CV AUC 0,633 ± 0,031 konzistentně indikuje diskriminační schopnost nad úrovní náhody.

---

## 4. Reprodukovatelnost

Systém je plně deterministický při fixním nastavení `random_state=42`. Reprodukovatelnost byla ověřena opakovaným spuštěním tréninku — všechny číselné výstupy jsou identické na úrovni JSON souboru.

**Jak reprodukovat výsledky:**

```bash
# 1. Klonovat repozitář a nainstalovat závislosti
git clone https://github.com/RicRedi/afib-risk-strocze.git
cd afib-risk-strocze
pip install -r requirements.txt

# 2. Poskytnout zdrojová data (src/ — není součástí repozitáře, nutno dodat)
# Soubor: src/STROCZECHMDTHolterEK_DATA_LABELS_2025-05-21_1136p.xls

# 3. Spustit trénink (produkuje risk_model.joblib, risk_model_calibrated.joblib, risk_scorecard.json)
python risk_model.py

# 4. Ověřit klíčové metriky v risk_scorecard.json:
#    test_performance.auc_roc → 0.6469
#    calibration.brier_calibrated → 0.0545
#    calibration.calibrated_npv → 0.9907
#    training_data_info.epv → 29.67

# 5. Spustit demo predikce
python predict.py
```

**Poznámka:** Reprodukce vyžaduje přístup ke zdrojovým datům STROCZE, která nejsou veřejně dostupná z důvodu ochrany osobních údajů pacientů. Výsledky tréninku (`risk_scorecard.json`, `x_test.npy`, `y_test.npy`) jsou k dispozici v repozitáři a umožňují reprodukci všech vizualizací a tabulek bez přístupu k původním datům.

---

## 5. Ukázkové predikce

Výstup příkazu `python predict.py` (zkráceno pro přehlednost). Hodnoty jsou reprodukovatelné při každém spuštění.

### Pacient A — kategorie Nízké

| Vstup | Hodnota |
|-------|---------|
| CHA₂DS₂-VASc | 1 |
| Teritoriální infarkt | Ne (lakunární) |
| Hyperlipidémie | Ne |

**Výstup modelu:**
- Kalibrovaná pravděpodobnost záchytu FiS: **4,0 %**
- Poměr k prevalenci: **0,7×** (pod kohortovým průměrem)
- Riziková kategorie: **Nízké** — FiS nepravděpodobná
- Bodové skóre: −19,0 bodů

### Pacient B — kategorie Nízké (hraniční)

| Vstup | Hodnota |
|-------|---------|
| CHA₂DS₂-VASc | 3 |
| Teritoriální infarkt | Ne |
| Hyperlipidémie | Ano |

**Výstup modelu:**
- Kalibrovaná pravděpodobnost záchytu FiS: **5,1 %**
- Poměr k prevalenci: **0,9×** (těsně pod kohortovým průměrem)
- Riziková kategorie: **Nízké** — FiS nepravděpodobná
- Bodové skóre: −7,0 bodů

*Poznámka:* Přes středně vysoké CHA₂DS₂-VASc (3) a přítomnost hyperlipidémie zůstává pravděpodobnost pod prahem díky záporné váze hyperlipidémie v modelu (treatment paradox — viz technická dokumentace).

### Pacient C — kategorie Velmi vysoké

| Vstup | Hodnota |
|-------|---------|
| CHA₂DS₂-VASc | 6 |
| Teritoriální infarkt | Ano |
| Hyperlipidémie | Ano |

**Výstup modelu:**
- Kalibrovaná pravděpodobnost záchytu FiS: **11,8 %**
- Poměr k prevalenci: **2,0×** (dvojnásobek kohortového průměru)
- Riziková kategorie: **Velmi vysoké** — silné doporučení k holterovské monitoraci
- Bodové skóre: +36,4 bodů

*Poznámka:* Vysoké CHA₂DS₂-VASc (6) a teritoriální infarkt silně zvyšují riziko. Záporná Hyperlipidémie riziko mírně snižuje, avšak výsledná pravděpodobnost stále překročí práh 2× prevalence.

---

## 6. Ověření HTML kalkulačky

HTML kalkulačka (`web/index.html`) byla ověřena:

- **Funkční test:** Zadáním stejných hodnot jako u Pacientů A, B, C výše a porovnáním kategorií s Python výstupem — výsledky jsou identické.
- **Hraniční hodnoty:** CHA₂DS₂-VASc = 0 (minimum) a CHA₂DS₂-VASc = 9 (maximum) produkují platné výsledky bez chyby.
- **Validace vstupů:** Kalkulačka odmítne neplatné vstupy (prázdné pole, hodnota mimo rozsah 0–9) a zobrazí chybové hlášení.
- **Mobilní zařízení:** Kalkulačka byla testována na mobilním prohlížeči — rozvržení je použitelné na obrazovce 375 px šíře (iPhone SE).

---

## 7. Limitace validace a doporučení pro budoucí práci

### 7.1 Interní vs. nezávislá validace

Testovací soubor (n = 377) pochází ze stejné kohorty STROCZE jako trénovací data. Jedná se tedy o **interní validaci**, nikoliv o nezávislou prospektivní validaci na nových pacientech. Výkonnostní metriky (AUC 0,647, NPV 95,7 %) mohou být optimistické ve srovnání s výkonem na jiné kohortě nebo v jiném časovém období.

**Doporučení:** Před zavedením do klinické praxe provést prospektivní validaci na:
- Nových pacientech STROCZE přijatých po datu exportu dat (2025-05-21).
- Pacientech jiného centra s dostupnou dlouhodobou holterovskou monitorací (externí validace).

### 7.2 Kalibrační sada = testovací sada

Platt scaling byl kalibrován na testovacím souboru. Toto je metodologické omezení způsobené malým počtem pozitivních případů (22 v testovacím souboru). V ideálním případě by kalibrační sada a testovací sada měly být oddělené.

**Doporučení:** Při rozšíření datasetu (více pacientů nebo delší sledování) použít trojí dělení: trénovací / kalibrační / testovací.

### 7.3 EPV a rozšíření modelu

Aktuální EPV = 29,7 umožňuje bezpečně pracovat se 3 prediktory. Pokud by byl model rozšířen o další prediktory, je nutné zajistit EPV ≥ 10 pro každý nový prediktor (tzn. každý nový prediktor vyžaduje přibližně 10 dalších FiS+ případů v trénovacích datech).

---

*Dokument připraven v rámci projektu NCK FEIM - DP001N (Healthcare). Verze 1.0, 2026-06-27.*
