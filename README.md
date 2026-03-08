# Benchmark: Fairness Drift Under Model Maintenance

This repository implements a comprehensive benchmarking study on fairness metrics under different model maintenance strategies (no-retrain vs. retrain) across temporal periods using the Folktables dataset.

## 📊 Overview

The project investigates how fairness metrics (Demographic Parity gap, Equalized Odds gap) evolve over time when models are deployed without retraining versus when they are periodically retrained. This addresses an emerging gap in fairness literature: **fairness drift as a dimension of model maintenance**.

**Key Research Question:** How do different fairness-aware training methods maintain fairness under temporal distribution shifts?

---

## 🚀 Quick Setup

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Installation (5 minutes)

```bash
# 1. Clone/navigate to repository
cd benchmark-fairness-folktables

# 2. Activate conda base environment
conda activate base

# 3. Install dependencies
pip install -r requirements.txt

# This installs:
# - numpy, pandas, scikit-learn
# - folktables (ACS data source)
# - tqdm (progress tracking)
# - pyyaml (config files)
# - matplotlib (plotting)
```

### Verify Installation

```bash
python3 -c "import numpy, pandas, sklearn, folktables, tqdm; print('✓ All dependencies installed')"
```

---

## ⚡ Quick Start (Run Your First Experiment)

### Option 1: Smoke Test (Fast - 2 minutes)

Perfect for testing that everything works:

```bash
python3 -m src.benchmark_runner configs/folktables_benchmark_temporal_smoke.yaml
```

This runs:
- Few seeds (5) for speed
- Single fairness method
- Minimal data sampling

### Option 2: Full Experiment (Income Task)

```bash
python3 -m src.benchmark_runner configs/folktables_benchmark_temporal.yaml
```

Produces comprehensive results:
- 20 random seeds
- All fairness methods
- Full dataset
- Results in `results/folktables/benchmark_temporal/`

### Inspect Results

```bash
ls results/folktables/benchmark_temporal/

# Generated files:
# - benchmark_results.csv              (raw per-seed results)
# - benchmark_summary.csv              (aggregated mean/std)
# - benchmark_summary_ci.csv           (with 95% confidence intervals)
# - benchmark_results_by_year.csv      (temporal breakdown)
# - benchmark_statistical_tests.csv    (significance tests)
```

---

## 📁 Project Structure

```
benchmark-fairness-folktables/
├── configs/                      # Experiment configurations
│   ├── folktables_benchmark_temporal.yaml
│   ├── folktables_benchmark_temporal_smoke.yaml
│   └── ...
├── src/
│   ├── benchmark/
│   │   ├── data.py              # Data loading & splitting utilities
│   │   ├── tables/              # Report table generation
│   │   │   ├── cohort_table.py  (Population characteristics)
│   │   │   └── initial_performance.py
│   │   ├── progress.py          # Progress tracking utilities
│   │   ├── metrics.py           # Fairness metrics computation
│   │   ├── methods.py           # Fairness methods (baseline, reweighing, etc)
│   │   ├── reporting.py         # Results visualization & analysis
│   │   └── runner.py            # Main benchmark orchestrator
│   ├── fairness.py              # Core fairness metric functions
│   └── benchmark_runner.py      # CLI entry point
├── results/                     # Generated results (gitignored)
├── data/                        # ACS data cache (gitignored)
├── requirements.txt
└── README.md
```

---

## 🔧 Create Custom Experiment

### Step 1: Create Config File

```bash
cp configs/folktables_benchmark_temporal_smoke.yaml configs/my_experiment.yaml
```

### Step 2: Edit Configuration

```yaml
data:
  task: "income"              # or "employment"
  states: ["CA", "TX", "NY"]
  mode: "temporal"
  train_years: [2014]
  val_years: [2015]
  test_years: [2016, 2017, 2018]
  max_samples: 20000

experiment:
  name: "my_experiment"

methods:
  - baseline
  - reweighing
  - equalized_odds
  - fairness_constraint

seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

output:
  dir: "results/my_experiment"

maintenance_strategies:
  - "no-retrain"
  - "retrain"
```

### Step 3: Run Experiment

```bash
python3 -m src.benchmark_runner configs/my_experiment.yaml
```

---

## 📊 Benchmark Components

### Fairness Methods Implemented

1. **Baseline** - No fairness intervention
2. **Reweighing** - Pre-processing (Kamiran & Calders, 2012)
3. **Equalized Odds** - Post-processing (Hardt et al., 2016)
4. **Fairness Constraint** - In-processing Lagrangian (Zafar et al., 2017)

### Fairness Metrics

- **DP Gap** (Demographic Parity Gap) - $\max_g |P(\hat{Y}=1|G=g) - P(\hat{Y}=1)|$
- **EO Gap** (Equalized Odds Gap) - Gap between true positive rates across groups
- **Accuracy** - Standard classification accuracy
- **AUC** - Area under ROC curve

### Maintenance Strategies

- **No-retrain** - Train on 2014, test on 2015-2018 (observe fairness drift)
- **Retrain** - Retrain yearly before testing (maintain fairness)

---

## 📈 Example Analysis

After running an experiment, analyze results:

```python
import pandas as pd

# Load results
results = pd.read_csv("results/my_experiment/benchmark_summary_ci.csv")

# View fairness metrics by method
print(results[['method', 'eo_gap_mean', 'eo_gap_ci_lower', 'eo_gap_ci_upper']])

# Compare no-retrain vs retrain (if available)
temporal = pd.read_csv("results/my_experiment/benchmark_results_by_year.csv")
no_retrain = temporal[temporal['maintenance'] == 'no-retrain']
retrain = temporal[temporal['maintenance'] == 'retrain']
```

---

## 🔍 Understanding the Code

### Data Pipeline (`src/benchmark/data.py`)

```python
from src.benchmark.data import (
    load_folktables,              # Load aggregated data
    load_folktables_by_period,    # Load per-year or per-quarter data
    extract_sensitive_attribute,  # Binary encoding of attributes
    stratified_split,             # Balanced train/val/test
)
```

### Runner (`src/benchmark/runner.py`)

The main orchestrator handles:
1. Config loading & validation
2. Data loading (temporal or static)
3. Model training with stratification
4. Metrics computation
5. Results aggregation & statistical tests
6. Progress tracking

---

## 🎯 Research Insights

Key findings we investigate:

- **Fairness Drift**: How much do fairness metrics degrade without retraining?
- **Method Comparison**: Which methods are robust to temporal shifts?
- **Maintenance Cost**: When is retraining worth the computational cost?
- **Task Dependency**: Do patterns differ between income and employment?

---

## 📝 Contributing

To add new fairness methods or metrics:

1. Add method to `src/benchmark/methods.py`
2. Add metric to `src/benchmark/metrics.py`
3. Update config templates
4. Run experiments with new config

---

## 📚 References

- Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). "Retiring Adult: New Datasets for Fair Machine Learning"
- Hardt, M., Price, E., & Srebro, N. (2016). "Equality of Opportunity in Supervised Learning"
- Kamiran, F., & Calders, T. (2012). "Data preprocessing techniques for classification without discrimination"
- Zafar, M. B., et al. (2017). "Fairness Constraints: Mechanisms for Fair Classification"

---

## 📞 Contact & Issues

For questions or issues, open a GitHub issue or contact the maintainers.

---

**Happy experimenting! 🚀**
