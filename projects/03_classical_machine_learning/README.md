# Project 3: Production-Ready Classical Machine Learning Pipeline & Diagnostics

## Overview & Curriculum Mapping
This project maps directly to **Module 3: Classical Machine Learning: The Core Engine** in `ai-foundations.tex`. It builds a modular end-to-end pipeline covering supervised learning (Logistic Regression, Random Forests, XGBoost), unsupervised learning (K-Means & Isolation Forests), and performance diagnostic tools for detecting overfitting vs. underfitting.

### Key Capabilities Implemented
1. **Pipeline Preprocessing**: Scikit-Learn `ColumnTransformer` handling numerical imputation, feature scaling, and categorical encoding.
2. **Model Evaluation & Diagnostics**: Dynamic Learning Curves plotting train vs. validation ROC-AUC scores over dataset subsets to diagnose high variance (overfitting) vs. high bias (underfitting).
3. **Unsupervised Mechanics**: K-Means clustering with quantitative Silhouette Score analysis and Isolation Forest anomaly detection.
4. **Production Serialisation & Serving**: Artifact serialization (`joblib`) and `ClassicalInferenceService` for low-latency batch and single-sample inference.

---

## Directory Structure
```
03_classical_machine_learning/
├── src/
│   ├── data_prep.py            # ColumnTransformer feature pipeline builder
│   ├── model_diagnostics.py    # Overfitting/underfitting learning curve & ROC-PR plotters
│   ├── clustering.py           # K-Means & Silhouette coefficient analysis
│   ├── anomaly_detection.py    # Isolation Forest anomaly scoring engine
│   └── inference_pipeline.py   # ClassicalInferenceService for serialized pipelines
├── train_pipeline.py           # Main training, diagnostic, and artifact serialization script
└── README.md                   # Project documentation
```

---

## How to Run

Execute the training and diagnostic pipeline:
```bash
python projects/03_classical_machine_learning/train_pipeline.py
```

Outputs will be saved under `projects/03_classical_machine_learning/output/`:
- `learning_curve.png` (Train vs. Validation ROC-AUC curve over training subset sizes)
- `roc_pr_curves.png` (ROC and Precision-Recall evaluation curves)
- `clustering_evaluation.png` (Elbow inertia & Silhouette Coefficient plots)
- `model_pipeline.joblib` (Serialized champion ML pipeline)

---

## Hands-on Exercises for Students
1. Extend `train_pipeline.py` to add `RandomizedSearchCV` hyperparameter tuning over `n_estimators`, `max_depth`, and `learning_rate` for XGBoost.
2. In `model_diagnostics.py`, add a function to plot a **Validation Curve** for varying values of the regularization strength $C$ in Logistic Regression.
3. Compare class imbalance handling techniques (SMOTE vs. `class_weight='balanced'`) on a fraud detection dataset.
