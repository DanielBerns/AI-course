# Project 1: Data Pipeline, Exploratory Analysis & Algorithmic Bias Audit

## Overview & Curriculum Mapping
This project maps directly to **Module 1: Introduction and the AI Landscape** in `ai-foundations.tex`. It bridges the theoretical paradigm shift from hand-coded rules to data-driven AI, implementing the foundational data engineering and ethical auditing pipeline that precedes all machine learning model development.

### Learning Objectives
1. **Data Cleaning & Engineering**: Ingest messy tabular data, compute imputation statistics, and handle outliers using interquartile range (IQR) clipping.
2. **Exploratory Data Analysis (EDA)**: Build publication-quality data quality visualizations (missingness profiles, KDE distributions, and correlation matrices).
3. **Ethical & Algorithmic Bias Auditing**: Audit demographic representation gaps, evaluate Disparate Impact Ratios (80% rule), and compute Statistical Parity and Equalized Odds metrics.

---

## Directory Structure
```
01_data_analysis_and_ethics/
├── src/
│   ├── cleaning.py            # Production-grade DataCleaner class
│   ├── visualization.py       # EDA plot generation utilities
│   └── bias_audit.py          # BiasAuditor for fairness metrics
├── data_pipeline_demo.py      # Runnable end-to-end pipeline script
└── README.md                  # Exercise documentation
```

---

## How to Run

Execute the main data pipeline script:
```bash
python projects/01_data_analysis_and_ethics/data_pipeline_demo.py
```

Generated plots will be saved to `projects/01_data_analysis_and_ethics/output/`.

---

## Exercise Tasks for Students
1. Extend `cleaning.py` to support dynamic one-hot encoding of categorical variables with high-cardinality handling.
2. Modify `bias_audit.py` to calculate Equal Opportunity Difference ($\text{TPR}_{\text{unpriv}} - \text{TPR}_{\text{priv}}$).
3. Run `data_pipeline_demo.py` on another real-world dataset (such as California Housing or Heart Disease) and document representation gaps in a model card.
