# AFib Risk Factor Identification Project

**Czech: Projekt Identifikace Rizikových Faktorů Fibrilace Síní (AFib)**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Architecture](#project-architecture)
4. [Installation & Setup](#installation--setup)
5. [Configuration System](#configuration-system)
6. [Core Components](#core-components)
7. [Running the Analysis](#running-the-analysis)
8. [Understanding the Results](#understanding-the-results)
9. [Data Processing Pipeline](#data-processing-pipeline)
10. [Extending the Project](#extending-the-project)
11. [Troubleshooting](#troubleshooting)
12. [Code Cleanup & Maintenance](#code-cleanup--maintenance)

---

## 🎯 Project Overview

This project identifies clinical variables and risk factors that correlate with **Atrial Fibrillation (AFib)** in patients. The analysis uses both **continuous** and **binary** clinical variables from patient data to determine which factors are statistically significant predictors of AFib.

### Research Context
- **Institution**: International Clinical Research Center, Brno
- **Affiliation**: Brno University of Technology
- **Focus**: Finding correlations between clinical measurements and AFib detection

### Key Objectives
1. ✅ Load clinical data from Excel files
2. ✅ Analyze continuous variables for correlation with AFib (Pearson/Logistic regression)
3. ✅ Analyze binary variables for association with AFib (Chi-square/Fisher's exact)
4. ✅ Calculate statistical significance (p-values)
5. ✅ Generate visualizations for correlation patterns
6. ✅ Export results for interpretation and further modeling

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment activated (`.env/`)
- Required packages installed (see `requirements.txt`)

### Basic Usage

```python
from analyze import VariableCorrelationAnalyzer
from utils.config_singleton import ConfigSingleton

# Initialize configuration
ConfigSingleton.set()

# Run the analysis
analyzer = VariableCorrelationAnalyzer()
analyzer.pipeline()
```

Or simply execute:
```bash
python main.py
```

### Expected Output
- Console output showing analysis progress
- Correlation statistics for each variable
- Visualization plots in `results/plots/`
- JSON results files in `results/`

---

## 🏗️ Project Architecture

### File Structure
```
Chadsvasc/
├── main.py                       # Entry point
├── analyze.py                    # Core analysis engine
├── core.py                       # Utility functions
├── plotting.py                   # Visualization
├── config/
│   ├── config.yaml              # Main configuration
│   ├── variables.yaml           # Variable definitions
│   ├── analysis.yaml            # Analysis parameters
│   └── plotting.yaml            # Plotting settings
├── utils/
│   ├── config_singleton.py      # Configuration manager
│   └── load_config.py           # YAML loader
├── src/
│   └── [source data files]      # Clinical data (Excel)
├── results/
│   ├── plots/                   # Generated visualizations
│   └── [JSON results files]     # Analysis outputs
├── requirements.txt             # Python dependencies
└── .env/                        # Virtual environment
```

### Architecture Diagram

```
Entry Point
    ↓
main.py ← ConfigSingleton ← config/
    ↓
VariableCorrelationAnalyzer
    ├→ load_data() [core.py]
    ├→ evaluate_logic() [core.py]
    ├→ analyze_continuous()
    ├→ analyze_binary()
    ├→ CorrelationPlotter [plotting.py]
    └→ save_results()
        ↓
    Clinical Results
```

---

## 💻 Installation & Setup

### 1. Clone / Setup Repository
```bash
cd C:\Chadsvasc
```

### 2. Create Virtual Environment (if not exists)
```bash
python -m venv .env
.\.env\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Key Packages
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **statsmodels** - Statistical modeling (logistic regression, GLM)
- **scipy** - Statistical tests (pearsonr, chi2, fisher_exact)
- **matplotlib & seaborn** - Visualization
- **pyyaml** - Configuration loading

### 5. Verify Installation
```bash
python -c "from analyze import VariableCorrelationAnalyzer; print('✅ Setup successful')"
```

---

## ⚙️ Configuration System

### Hierarchical YAML Configuration

The project uses a **singleton pattern** to load configuration once and access globally:

```python
ConfigSingleton.set()           # Load configuration once
cfg = ConfigSingleton.get()     # Access anywhere
cfg.analysis.file_path          # Nested access
```

### Configuration Files

#### 1. `config/analysis.yaml` - Analysis Settings
```yaml
file_path: ./src/STROCZECHMDTHolterEK_DATA_LABELS_2025-05-21_1136p.xls
iqr_threshold: 1.5              # IQR multiplier for outlier detection
z_score_threshold: 3.0
significance_level: 0.05        # p-value threshold
print_progress: False           # Console output
save_results: True              # Save JSON results
save_path: ./results            # Output directory
```

#### 2. `config/variables.yaml` - Variable Definitions
```yaml
independent_continuous_variables:
  - LVEDD                        # Left ventricular dimension
  - eGFR                         # Kidney function
  - CHA₂DS₂-VASc                 # Stroke risk score
  - [more numeric variables]

independent_binary_variables:
  - Kuřák v době vzniku          # Smoker status
  - Pohlaví                      # Sex
  - [more categorical variables]

reference_var: Záchyt FiS MDT celkově    # Target: AFib detection

conditions:
  cond1:
    col: Záchyt FiS MDT celkově
    op: "=="
    value: 1                     # Only AFib+ patients

logic: cond1                     # Boolean logic to combine conditions
```

#### 3. `config/plotting.yaml` - Visualization Settings
Contains parameters for correlation plot generation and display

---

## 🔧 Core Components

### 1. VariableCorrelationAnalyzer (`analyze.py`)

Main analysis engine that correlates clinical variables with AFib.

#### Workflow
```
1. Load Configuration & Data
   ↓
2. Generate Outcome Variable (via logical conditions)
   ↓
3. FOR EACH VARIABLE:
   ├─ Clean data (remove NaN, outliers)
   ├─ Perform statistical test
   │  ├─ Continuous vars → Logistic/Pearson correlation
   │  └─ Binary vars → Chi-square/Fisher's exact
   ├─ Generate visualization
   └─ Store results
   ↓
4. Export Results (JSON)
```

#### Key Methods

**`analyze_continuous()`**: Correlation analysis for numeric variables
- Outcome is binary → **Logistic Regression**
- Outcome is continuous → **Pearson Correlation**

**`analyze_binary()`**: Association analysis for categorical variables
- Uses **Chi-Square Test** (adequate sample size)
- Uses **Fisher's Exact Test** (small samples)

**`_logistic_regression(x, y)`**: Weighted logistic regression
- Applies class weights = inverse prevalence (handles imbalance)
- Uses GLM with Binomial family

### 2. Core Utilities (`core.py`)

**`load_data(path, variables, reference_var)`**
- Loads specific columns from Excel file
- Memory efficient (only needed columns)

**`evaluate_logic(df, conditions, logic)`**
- Applies logical filters to DataFrame
- Syntax: `logic = "cond1 & cond2"`
- Returns: Boolean mask for filtering

**`convert_column_to_binary(series)`**
- Converts text to 0/1: 'yes'→1, 'no'→0, 'true'→1, 'false'→0
- Czech: 'ano'→1, 'ne'→0

**`remove_outliers_iqr(x, threshold=1.5)`**
- Removes values beyond: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

### 3. Visualization (`plotting.py`)

**CorrelationPlotter** class generates correlation plots
- Binary outcome → Logistic regression curve
- Continuous outcome → Linear regression line
- Saved to: `results/plots/`

---

## 📊 Running the Analysis

### Step-by-Step

#### 1. Configure Variables
Edit `config/variables.yaml` to select which variables to analyze:
```yaml
independent_continuous_variables:
  - LVEDD
  - eGFR
  - CHA₂DS₂-VASc
```

#### 2. Run Analysis
```bash
python main.py
```

#### 3. Monitor Progress
If `print_progress: True`:
```
Analyzing correlation for: LVEDD vs Záchyt FiS MDT celkově
Analyzing correlation for: eGFR vs Záchyt FiS MDT celkově
...
✅ Analysis complete
```

#### 4. Review Results

**Console Output**: Summary statistics

**JSON Results**: `results/analysis_results_*.json`
```json
{
  "LVEDD": {
    "correlation": 0.342,
    "p_value": 0.0001,
    "type": "logistic",
    "data_used_%": 92.5,
    "t_statistic": 4.234,
    "t_p_value": 0.00003
  }
}
```

**Plots**: `results/plots/*.png`

---

## 📈 Understanding Results

### Statistical Outputs

**Logistic Regression** (Binary Outcome)
- `correlation`: Regression coefficient β
  - Positive: Higher value → Higher AFib probability
  - Magnitude: Effect size
- `p_value`: Statistical significance (< 0.05 is significant)

**Pearson Correlation** (Continuous Outcome)
- `correlation`: r coefficient (-1 to +1)
  - 0.7-1.0: Strong correlation
  - 0.3-0.7: Moderate correlation
  - 0.0-0.3: Weak correlation
- `p_value`: Significance of correlation

**Data Quality**
- `data_used_%`: Percentage of data retained after cleaning
  - 95%+: Good quality
  - 70-95%: Moderate outlier removal
  - <70%: High outlier prevalence

### Example Interpretation

```
Variable: LVEDD
correlation: 0.342
p_value: 0.0001
data_used_%: 92.5

➜ Significant positive association with AFib (p < 0.001)
➜ Larger heart → Higher AFib risk
➜ 92.5% data quality
```

---

## 🔄 Data Processing Pipeline

```
1. Load Config (variables.yaml)
2. Load Clinical Data (Excel)
3. Generate Outcome (via conditions)
4. FOR EACH VARIABLE:
   · Remove NaN
   · Convert to numeric
   · Remove outliers (IQR)
   · Align with outcome
   · Statistical test (Logistic/Pearson/Chi-square)
5. Generate Visualization
6. Save Results (JSON)
```

---

## 🚀 Extending the Project

### Adding a New Variable

1. Edit `config/variables.yaml`:
```yaml
independent_continuous_variables:
  - New_Variable_Name    # ← Add here
  - LVEDD
```

2. Verify column exists in Excel file

3. Run analysis:
```bash
python main.py
```

### Changing the Outcome

```yaml
reference_var: Different_Target_Variable
conditions:
  cond1:
    col: Different_Target_Variable
    op: "=="
    value: 1
```

### Multiple Conditions

```yaml
conditions:
  cond1:
    col: AFib_Detection
    op: "=="
    value: 1
  cond2:
    col: Age
    op: ">="
    value: 50

logic: (cond1 & cond2)    # Python eval syntax
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'analyze'"
**Solution**: Working directory must be project root
```bash
cd C:\Chadsvasc
python main.py
```

### "FileNotFoundError: No such file or directory"
**Solution**: Check data file path in `config/analysis.yaml`
```bash
ls ./src/
```

### "KeyError: Variable name not in DataFrame"
**Solution**: Verify variable name spelling in Excel file
- Check for Czech special characters (š, č, ž)
- Column names are case-sensitive

### "All data converted to NaN"
**Solution**: Data type mismatch
- Check if Excel column is formatted as Number
- Or move to `independent_binary_variables`

### No plots generated
**Solution**: Check directory permissions
```bash
mkdir -p results/plots
```

---

## 🧹 Code Cleanup & Maintenance

### Project History
This codebase was cleaned to remove abandoned experiments:

**Deleted Config Files**:
- `hemorrhage.yaml` - Was for HemorrhageAnalysis configuration (deleted)
- `model.yaml` - Was for model_mlp.py configuration (deleted)

**Deleted Python Code**:
- `DecisionTreePlotter` class from `plotting.py` - Only used by deleted HemorrhageAnalysis

**Deleted Python Files**:
- `hemorrhage_analysis.py` - Alternative analysis (unused)
- `data_loader.py` - PyTorch dataset (abandoned ML experiment)
- `mlp.py` - Neural network (abandoned ML experiment)
- `model_mlp.py` - MLP wrapper (incomplete)
- `splitter.py` - Only used by deleted files

**Deleted Functions from `core.py`**:
- `cmp_tia_mapping()` - Only used by `splitter.py`
- `remove_outliers_iqr_df()` - Only used by `hemorrhage_analysis.py`
- `remove_outliers_z_score()` - Never used

**Current Active Config Files** (3 total):
- ✅ `config.yaml` - Main configuration aggregator
- ✅ `analysis.yaml` - Analysis parameters
- ✅ `variables.yaml` - Variable definitions
- ✅ `plotting.yaml` - Visualization parameters

**Result**: Clean, focused codebase (~500 lines of active code)

---

## 👥 Contributing

When making changes:
1. Test with `python main.py`
2. Check results in `results/`
3. Update README if modifying configuration
4. Add docstrings for new functions

---

## 📝 License & Attribution

**Author**: Richard Redina  
**Email**: 195715@vut.cz  
**Institution**: International Clinical Research Center, Brno  
**Affiliation**: Brno University of Technology

---

## 🎓 For Future Developers

### Understanding Code Flow
1. `main.py` - Entry point
2. `analyze.py` - Main analysis engine
3. `config/variables.yaml` - Data definitions
4. `core.py` - Utility functions
5. `plotting.py` - Visualization logic

### Key Patterns
- **Singleton**: Configuration management (ConfigSingleton)
- **Pipeline**: Clean → Analyze → Visualize → Export
- **OOP**: Classes with single responsibilities

### Extending Examples
- Add statistical test → Modify `analyze_continuous()` or `analyze_binary()`
- Change data source → Update `load_data()` in `core.py`
- New output format → Add to `save_results()`
- Custom visualization → Extend `CorrelationPlotter`

Good luck! 🚀
