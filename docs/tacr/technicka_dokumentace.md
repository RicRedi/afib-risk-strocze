# Technická dokumentace
## Kalkulátor rizika záchytu fibrilace síní po iCMP/TIA

---

**Projekt:** NCK FEIM - DP001N (Healthcare) — Identifikace rizikových faktorů fibrilace síní  
**Verze dokumentu:** 1.0  
**Datum:** 2026-06-27  
**Autoři:** Valentýna Provazník, Veronika Bulková, Richard Ředina  
**Instituce:** Vysoké učení technické v Brně / Medical Data Transfer s.r.o.

---

## 1. Přehled systému

### 1.1 Vstupní data

Zdrojová data pocházejí z databáze **STROCZE (Czech Stroke Research database)** shromažďované centrem MDT Brno (Medical Data Transfer s.r.o.). Dataset obsahuje záznamy pacientů hospitalizovaných po ischemické CMP nebo TIA, u nichž byla provedena dlouhodobá holterovská monitorace (Holter EKG, 24–48 h).

| Parametr | Hodnota |
|----------|---------|
| Celkem záznamů v databázi | 5 083 |
| Záznamy s kompletními daty | 1 884 (37,1 %) |
| Datum exportu dat | 2025-05-21 |
| Cílová proměnná | Záchyt FiS při holterovské monitoraci (binárně: 0/1) |
| Celkem FiS+ | 111 (5,9 % prevalence) |
| Celkem FiS− | 1 773 |

Nízká kompletnost dat (37,1 %) je způsobena tím, že ne všechny sledované proměnné jsou dostupné u každého pacienta. Proměnné CHA₂DS₂-VASc, Teritoriální infarkt a Hyperlipidémie, které model používá, mají kompletnost 99,8–100 %, tedy prakticky bez chybějících hodnot.

### 1.2 Architektura systému

```
[STROCZE Excel — data pacientů, NIKDY do Gitu]
         │
         ├──► analyze.py ──► VariableCorrelationAnalyzer
         │        │             • logistická regrese (19 kontinuálních proměnných)
         │        │             • Fisherův exaktní test (12 binárních proměnných)
         │        └──► results/analysis_results_*.json  (statistické asociace)
         │
         └──► risk_model.py ──► RiskScoreGenerator
                  │               • Příprava dat (IQR filtrace, binárization)
                  │               • RFECV: 8 kandidátů → 3 vybrané prediktory
                  │               • LogisticRegression (class_weight='balanced')
                  │               • Platt scaling (CalibratedClassifierCV)
                  │               • Bootstrap CI (1 000 stratifikovaných iterací)
                  │               • Export scorecard + modely
                  └──► results/
                           ├── risk_scorecard.json         ← koeficienty, metriky
                           ├── risk_model.joblib            ← základní LR model
                           ├── risk_model_calibrated.joblib ← kalibrovaný model
                           ├── risk_model_x_test.npy        ← testovací features
                           └── risk_model_y_test.npy        ← testovací labely

predict.py ────────────────► Python demo (3 ukázkové pacienty)
final_report/*.py ─────────► Vizualizace a analytické tabulky

results/risk_scorecard.json
         │
         └──► web/generate_scorecard_js.py ──► web/assets/scorecard.js
                                                        │
                                               web/index.html ──► GitHub Pages
                   (⚠ spustit po každém re-tréninku modelu)
```

---

## 2. Výběr a popis prediktorů

### 2.1 Kandidátní proměnné

Z dostupné databáze bylo identifikováno **8 kandidátních proměnných** pro modelování — proměnné, které mají dostatečnou datovou kompletnost (≥ 37 % celkového datasetu, tj. dostupné u většiny z 1 884 kompletních záznamů) a byly statisticky asociovány s záchytem FiS nebo mají klinické opodstatnění.

| Proměnná | Typ | Kompletnost |
|----------|-----|-------------|
| CHA₂DS₂-VASc | Kontinuální | 99,8 % |
| Typ akutní ischemie (Teritoriální) | Binární | 100,0 % |
| Osobní anamnéza (Hyperlipidémie) | Binární | 100,0 % |
| Enddiastolický rozměr levé komory (LVEDD) | Kontinuální | 50,3 % |
| Interventrikulární septum (IVS) | Kontinuální | 53,0 % |
| Ejekční frakce levé komory (LVEF) | Kontinuální | 52,7 % |
| BMI | Kontinuální | 72,8 % |
| Sérový kreatinin | Kontinuální | 76,8 % |

### 2.2 Výběr prediktorů — RFECV

Výběr prediktorů byl proveden metodou **Recursive Feature Elimination with Cross-Validation (RFECV)** implementovanou v scikit-learn. RFECV systematicky odstraňuje prediktory s nejnižší váhou koeficientu a v každém kroku validuje výkon modelu prostřednictvím 5-fold stratifikované křížové validace (metrika: AUC-ROC).

Výsledek RFECV: **3 prediktory** z 8 kandidátů dosahují optimálního AUC (0,625 vs. 0,616–0,624 pro 4–8 prediktorů). Přidávání dalších prediktorů AUC nezvyšuje a přitom snižuje interpretovatelnost a zvyšuje riziko overfittingu.

### 2.3 Vybrané prediktory a jejich koeficienty

| Prediktor | Typ | log-OR (β) | OR (e^β) | Body / jednotku |
|-----------|-----|-----------|----------|----------------|
| CHA₂DS₂-VASc | Kontinuální | +1,101 | 3,01 | +11,0 |
| Teritoriální infarkt | Binární | +1,032 | 2,81 | +10,3 |
| Hyperlipidémie (anamnéza) | Binární | −1,000 | 0,37 | −10,0 |

**Komentář k záporné hodnotě Hyperlipidémie:** Záporný koeficient znamená, že přítomnost hyperlipidémie v anamnéze je v modelu spojena s nižším rizikem záchytu FiS. Pravděpodobnou příčinou je konfundující vliv statinové léčby — pacienti s diagnostikovanou hyperlipidémií jsou v naprosté většině léčeni statiny, které mají prokázaný antiarytmický efekt snižující incidenci FiS. Databáze STROCZE neobsahuje informaci o medikaci, takže tento mechanismus (tzv. treatment paradox) nelze přímo ověřit.

---

## 3. Algoritmické rozhodnutí a odůvodnění

### 3.1 Logistická regrese

**Volba:** LogisticRegression (scikit-learn), solver lbfgs, max_iter 1000.

**Zdůvodnění:**
- Interpretovatelnost: koeficienty jsou přímo čitelné jako log-OR a lze je převést na klinické bodové skóre.
- Klinická tradice: CHA₂DS₂-VASc samo o sobě je výsledkem logistické regrese; lékaři jsou na tento způsob prezentace zvyklí.
- Výkon: při nízkém počtu prediktorů (3) a binárním výsledku je logistická regrese standardní volbou s prokázanou robustností.

### 3.2 Vyvažování tříd (class_weight='balanced')

**Volba:** `class_weight='balanced'` — váhy tříd jsou automaticky nastaveny na inverse frekvence (FiS+: váha ~8,5×, FiS−: váha ~0,53×).

**Zdůvodnění:** Bez vyvažování tříd by model s prevalencí 5,9 % FiS+ predikoval téměř všechny pacienty jako FiS− (senzitivita → 0) a dosahoval by vysoké celkové přesnosti díky trivialnímu řešení. Vyvažování tříd vynucuje, aby model věnoval adekvátní pozornost oběma třídám. Důsledkem je snížení specifity (30,1 %), avšak zachování vysoké senzitivity (95,5 %) — což je pro screeningový nástroj správná priorita.

### 3.3 Platt scaling (kalibrace)

**Volba:** `CalibratedClassifierCV(method='sigmoid', cv='prefit')` aplikovaná na `FrozenEstimator` (základní LR model zmrazený po tréninku).

**Zdůvodnění:** Model s `class_weight='balanced'` produkuje kalibrované pravděpodobnosti v rozsahu 0,26–0,78, zatímco skutečná prevalence je 5,9 %. Takové pravděpodobnosti jsou klinicky nesmyslné (říkají „26–78 % šance na FiS" u každého pacienta). Platt scaling (logistická regrese na výstupech základního modelu) mapuje hodnoty na realistický rozsah 2,6–14,4 % odpovídající kohortové epidemiologii.

**Parametry Platt scalingu (z risk_scorecard.json):**
- a = −0,8019
- b = +2,7756
- Kalkulační vzorec: `p_kalibrovaná = 1 / (1 + exp(a × logit + b))`

### 3.4 Stratifikovaný train/test split

**Volba:** `train_test_split(stratify=y, test_size=0.2, random_state=42)`.

**Zdůvodnění:** Stratifikace zajišťuje, že poměr FiS+ a FiS− (5,9 %) je zachován v obou sadách. Bez stratifikace by testovací sada mohla náhodně obsahovat výrazně méně pozitivních případů, což by zkreslilo odhad výkonu.

| | Celkem | FiS+ | Prevalence |
|-|--------|------|-----------|
| Trénovací sada | 1 507 | 89 | 5,9 % |
| Testovací sada | 377 | 22 | 5,8 % |

### 3.5 Bootstrap konfidečního intervalů

**Volba:** Stratifikovaný bootstrap, 1 000 iterací, `random_state=42`.

**Zdůvodnění:** Testovací sada má pouze 22 pozitivních případů — příliš málo pro asymptotické (normální aproximace) CI. Bootstrap bez stratifikace by mohl generovat iterace s velmi nízkým nebo nulovým počtem FiS+. Stratifikovaný bootstrap zajišťuje, že každá bootstrap iterace obsahuje všech 22 pozitivních případů (s opakováním), čímž se zvyšuje stabilita odhadů.

---

## 4. Výkonnostní charakteristiky modelu

### 4.1 Křížová validace (trénovací sada)

| Metrika | Průměr | Std | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|-----|--------|--------|--------|--------|--------|
| AUC-ROC | 0,633 | 0,031 | 0,639 | 0,648 | 0,583 | 0,620 | 0,677 |
| Senzitivita | 0,558 | 0,123 | 0,500 | 0,636 | 0,364 | 0,565 | 0,727 |
| Specificita | 0,640 | 0,056 | 0,637 | 0,603 | 0,735 | 0,655 | 0,571 |

### 4.2 Testovací sada (n = 377, FiS+ = 22)

| Metrika | Hodnota | 95% CI (bootstrap, 1 000×) |
|---------|---------|--------------------------|
| AUC-ROC | 0,647 | [0,543 – 0,740] |
| Senzitivita (Youdenův práh) | 95,5 % | [86,4 % – 100,0 %] |
| Specificita (Youdenův práh) | 30,1 % | [25,4 % – 34,7 %] |
| PPV | 7,8 % | — |
| NPV (celkové) | 99,1 % | — |
| Brier skóre (kalibrovaný) | 0,0545 | — (in-sample) |
| Brier skóre (nekalibrovaný) | 0,2394 | — |
| No-skill Brier baseline | ~0,0550 | — |

**Konfuzní matice (kalibrovaný model, Youdenův práh = 0,041):**

| | Predikce FiS+ | Predikce FiS− |
|-|---------------|---------------|
| **Skutečnost FiS+** | TP = 21 | FN = 1 |
| **Skutečnost FiS−** | FP = 248 | TN = 107 |

### 4.3 Rozložení do rizikových kategorií (testovací sada)

| Kategorie | Práh | n | % kohorty | NPV |
|-----------|------|---|-----------|-----|
| Nízké | < 1× prev. (< 5,84 %) | 211 | 56,0 % | 95,7 % |
| Střední | 1–1,5× prev. | 55 | 14,6 % | — |
| Vysoké | 1,5–2× prev. | 23 | 6,1 % | — |
| Velmi vysoké | > 2× prev. | 88 | 23,3 % | — |

### 4.4 Screeningová efektivita (NNM framing)

| Scénář | Monitorováno | Zachyceno FiS | NNM |
|--------|-------------|--------------|-----|
| Bez modelu (všichni) | 377 | 22/22 | 17,1 |
| S modelem (kategorie Střední+) | 269 | 21/22 | 12,8 |
| Úspora monitoringů | 108 (28,6 %) | 1 přehlédnuta (NPV 99,1 %) | −25 % |

---

## 5. Datové struktury

### 5.1 risk_scorecard.json

Soubor obsahuje veškeré parametry potřebné pro nasazení modelu (koeficienty, kalibrační parametry, výkonnostní metriky). **Neobsahuje žádná individuální pacientská data.** Je bezpečný pro uložení do veřejného repozitáře.

Klíčová struktura:
```json
{
  "intercept": -0.7829,
  "scaling_factor": 10,
  "variables": {
    "CHA₂DS₂-VASc": { "points_per_unit": 11.014, "type": "continuous" },
    "Typ akutní ischemie (choice=Teritoriální)": { "points_per_unit": 10.317, "type": "binary" },
    "Osobní anamnéza (choice=Hyperlipidémie)": { "points_per_unit": -10.0, "type": "binary" }
  },
  "calibration": {
    "platt_a": -0.8019, "platt_b": 2.7756,
    "calibrated_sensitivity": 0.9545, "calibrated_npv": 0.9907
  },
  "test_performance": {
    "auc_roc": 0.6469, "cohort_prevalence": 0.05836, "n_test": 377
  }
}
```

### 5.2 Vstupní formát pro predikci

Python (přes `predict_with_model()`):
```python
patient = {
    "CHA₂DS₂-VASc": 5,
    "Typ akutní ischemie (choice=Teritoriální)": 1,  # 1 = teritoriální, 0 = jiný
    "Osobní anamnéza (choice=Hyperlipidémie)": 0,    # 1 = ano, 0 = ne
}
```

JavaScript (HTML kalkulačka, přes scorecard.js):
```javascript
const result = predict(cha2ds2=5, terit=1, hlp=0);
// → { prob: 0.092, ratio: 1.58, category: "Vysoké" }
```

---

## 6. Bezpečnost dat

- Zdrojová data pacientů (složka `src/`) jsou trvale vyloučena ze sledování Gitem pomocí `.gitignore`.
- Testovací sada (`risk_model_x_test.npy`, `risk_model_y_test.npy`) obsahuje výhradně 3 číselné hodnoty na pacienta (CHA₂DS₂-VASc skóre a dvě binární proměnné 0/1) bez jakýchkoli identifikátorů. Tyto soubory jsou bezpečné pro zveřejnění.
- Binární soubory modelů (`.joblib`) jsou vyloučeny z Gitu z důvodu velikosti, nikoli z důvodu bezpečnosti.
- Zpracování dat probíhá výhradně lokálně na pracovní stanici výzkumníka — žádná data nejsou odesílána na servery třetích stran.

---

## 7. Technická omezení

1. **Kalibrace na testovacím souboru:** Platt scaling byl kalibrován na testovacím souboru (nikoliv na oddělené kalibrační sadě). Metriky kalibrace jsou tedy in-sample a mohou nadhodnocovat výkon kalibrace. Toto omezení je důsledkem malého počtu pozitivních případů (22 v testovací sadě), který neumožňuje trojí dělení dat.

2. **Single-center kohorta:** Data pocházejí výhradně od jednoho poskytovatele holterovské monitorace (MDT Brno). Generalizovatelnost modelu na jiná centra s odlišnou demografií nebo protokoly monitoringu není ověřena.

3. **Chybějící informace o medikaci:** STROCZE neobsahuje záznamy o konkrétní farmakologické léčbě. Záporný koeficient Hyperlipidémie (treatment paradox) nemůže být přímo ověřen — jedná se o hypotézu.

4. **Statická kohortová prevalence:** Rizikové kategorie jsou definovány relativně k prevalenci 5,84 % měřené v testovacím souboru. Při aplikaci na jinou kohortu s odlišnou prevalencí by bylo nutné prahy přepočítat.

---

*Dokument připraven v rámci projektu NCK FEIM - DP001N (Healthcare). Verze 1.0, 2026-06-27.*
