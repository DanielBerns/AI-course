"""
Project 3: Production-Ready Classical Machine Learning Pipeline.

Trains and evaluates classical ML algorithms (Logistic Regression, Random Forest, XGBoost),
diagnoses overfitting/underfitting via learning curves, performs K-Means clustering,
executes Isolation Forest anomaly detection, and serializes inference pipelines.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_prep import build_preprocessing_pipeline
from model_diagnostics import plot_learning_curve, plot_roc_pr_curves
from clustering import ClusteringAnalyzer
from anomaly_detection import AnomalyDetector
from inference_pipeline import ClassicalInferenceService


def main() -> None:
    print("=" * 70)
    print("PROJECT 3: CLASSICAL MACHINE LEARNING PIPELINE & DIAGNOSTICS")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Dataset Ingestion & Splitting
    print("\n[Step 1] Ingesting Real & Benchmark Tabular Data...")
    cancer_data = load_breast_cancer(as_frame=True)
    df = cancer_data.frame
    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_cols = list(X.columns)
    categorical_cols = []  # All features numerical for this benchmark

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print(f" - Train set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f" - Test set size:  {X_test.shape[0]} samples")

    # 2. Pipeline Preprocessing & Model Evaluation
    print("\n[Step 2] Evaluating Model Family Performance via 5-Fold Stratified Cross-Validation...")
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = ""
    best_auc_score = 0.0
    best_pipeline = None

    for name, clf in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("classifier", clf)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        mean_score, std_score = np.mean(cv_scores), np.std(cv_scores)
        print(f" - Model: {name:20s} | 5-Fold ROC-AUC: {mean_score:.4f} (+/- {std_score:.4f})")

        if mean_score > best_auc_score:
            best_auc_score = mean_score
            best_model_name = name
            best_pipeline = pipe

    print(f"\n -> Selected Champion Model: {best_model_name} (Cross-Val AUC: {best_auc_score:.4f})")

    # 3. Fit Champion Pipeline & Generate Overfitting/Underfitting Diagnostics
    print("\n[Step 3] Fitting Champion Pipeline & Generating Overfitting Diagnostics...")
    best_pipeline.fit(X_train, y_train)

    fig_lc, diag_summary = plot_learning_curve(
        best_pipeline, X_train, y_train, cv=skf, scoring="roc_auc", output_path=os.path.join(output_dir, "learning_curve.png")
    )

    print(f" - Learning Curve Diagnosis: {diag_summary['diagnosis']}")
    print(f" - Train Score: {diag_summary['final_train_score']:.4f} | Validation Score: {diag_summary['final_val_score']:.4f}")

    # Holdout Test Evaluation
    y_test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_preds = best_pipeline.predict(X_test)
    test_auc = roc_auc_score(y_test, y_test_probs)

    print(f"\n Champion Test Set Evaluation:")
    print(f" - Holdout Test ROC-AUC: {test_auc:.4f}")
    print(classification_report(y_test, y_test_preds, target_names=cancer_data.target_names))

    fig_curves = plot_roc_pr_curves(y_test, y_test_probs, output_path=os.path.join(output_dir, "roc_pr_curves.png"))

    # 4. Unsupervised Learning: Clustering & Anomaly Detection
    print("\n[Step 4] Executing Unsupervised K-Means & Isolation Forest Anomaly Detection...")
    X_scaled = preprocessor.fit_transform(X_train)

    cluster_analyzer = ClusteringAnalyzer(k_range=(2, 6))
    cluster_analyzer.fit_evaluate(X_scaled)
    opt_k = cluster_analyzer.get_optimal_k()
    print(f" - Optimal K-Means Cluster Count (Max Silhouette): K = {opt_k}")
    print(f" - Optimal K Silhouette Score: {cluster_analyzer.results_[opt_k]['silhouette_score']:.4f}")

    cluster_analyzer.plot_evaluation(output_path=os.path.join(output_dir, "clustering_evaluation.png"))

    anomaly_detector = AnomalyDetector(contamination=0.04)
    anom_labels, anom_scores = anomaly_detector.fit_predict(X_scaled)
    anom_summary = anomaly_detector.summarize_anomalies(pd.DataFrame(X_scaled), anom_labels, anom_scores)
    print(f" - Isolation Forest Anomalies Detected: {anom_summary['anomalies_detected']} ({anom_summary['anomaly_ratio']:.1%})")

    # 5. Serialization & Production Inference Test
    model_artifact_path = os.path.join(output_dir, "model_pipeline.joblib")
    print(f"\n[Step 5] Serializing Champion Pipeline to '{model_artifact_path}'...")
    joblib.dump(best_pipeline, model_artifact_path)

    print(" -> Loading model into ClassicalInferenceService for verification...")
    service = ClassicalInferenceService(model_artifact_path)
    sample_batch = X_test.iloc[:3].to_dict(orient="records")
    preds = service.predict(sample_batch)
    probs = service.predict_proba(sample_batch)

    print(f" - Batch Test Input Samples: {len(sample_batch)}")
    print(f" - Model Predictions:        {preds}")
    print(f" - Class Probabilities:      \n{probs.round(4)}")

    print("\n" + "=" * 70)
    print("PROJECT 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
