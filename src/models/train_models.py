import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    log_loss, confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Project root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

FEATURE_COLS = [
    'dist_to_arc',
    'heading_diff',
    'gps_speed',
    'arc_length',
    'projection_ratio'
]
TARGET_COL = 'is_match'
GROUP_COL = 'trip_id'

def compute_top1_accuracy(df: pd.DataFrame, preds_proba: np.ndarray) -> float:
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, pt_idx), pick the candidate with highest predicted probability.
    Check if the chosen candidate is a true match (is_match == 1).
    """
    temp = df[['trip_id', 'pt_idx', 'is_match']].copy()
    temp['score'] = preds_proba
    
    # Get index of candidate with max score for each (trip_id, pt_idx)
    idx_max = temp.groupby(['trip_id', 'pt_idx'])['score'].idxmax()
    chosen = temp.loc[idx_max]
    top1_acc = (chosen['is_match'] == 1).mean()
    return float(top1_acc)

def get_models() -> dict:
    """Return 5 distinct model pipelines spanning 5 algorithm families."""
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=6, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1
        ),
        'MLP Neural Net': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=200, activation='relu',
                random_state=42, early_stopping=True, validation_fraction=0.1
            ))
        ])
    }
    return models

def evaluate_models(
    train_parquet: str = "data/processed/train_features.parquet",
    test_parquet: str = "data/processed/test_features.parquet",
    artifacts_dir: str = "artifacts",
    figures_dir: str = "reports/figures"
):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading training data from {train_parquet}...")
    df_train = pd.read_parquet(train_parquet)
    print(f"Loading student test data from {test_parquet}...")
    df_test = pd.read_parquet(test_parquet)
    
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].values
    groups_train = df_train[GROUP_COL].values
    
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL].values
    
    models = get_models()
    gkf = GroupKFold(n_splits=5)
    
    results = {}
    test_predictions = {}
    
    print("\n" + "="*80)
    print("5-FOLD GROUP-AWARE CROSS-VALIDATION ON 80 TRAINING TRIPS")
    print("="*80)
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}...")
        cv_f1_scores = []
        cv_top1_accs = []
        cv_log_losses = []
        
        # 5-Fold GroupKFold
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            df_val = df_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            val_probs = model.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            
            fold_f1 = f1_score(y_val, val_preds, average='macro')
            fold_top1 = compute_top1_accuracy(df_val, val_probs)
            fold_loss = log_loss(y_val, np.clip(val_probs, 1e-7, 1 - 1e-7))
            
            cv_f1_scores.append(fold_f1)
            cv_top1_accs.append(fold_top1)
            cv_log_losses.append(fold_loss)
            
        mean_cv_f1 = float(np.mean(cv_f1_scores))
        std_cv_f1 = float(np.std(cv_f1_scores))
        mean_cv_top1 = float(np.mean(cv_top1_accs))
        mean_cv_loss = float(np.mean(cv_log_losses))
        
        print(f"  CV Macro F1: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})")
        print(f"  CV Observation Top-1 Acc: {mean_cv_top1:.4f}")
        print(f"  CV Log Loss: {mean_cv_loss:.4f}")
        
        # Fit on full training set and evaluate on Student Test
        t_fit_start = time.time()
        model.fit(X_train, y_train)
        train_time_sec = float(time.time() - t_fit_start)
        
        t_inf_start = time.time()
        test_probs = model.predict_proba(X_test)[:, 1]
        inference_time_sec = float(time.time() - t_inf_start)
        latency_ms_per_1k = float((inference_time_sec / len(X_test)) * 1000.0 * 1000.0)
        
        test_preds = (test_probs >= 0.5).astype(int)
        
        test_f1 = float(f1_score(y_test, test_preds, average='macro'))
        test_top1 = float(compute_top1_accuracy(df_test, test_probs))
        test_precision = float(precision_score(y_test, test_preds, zero_division=0))
        test_recall = float(recall_score(y_test, test_preds, zero_division=0))
        test_loss = float(log_loss(y_test, np.clip(test_probs, 1e-7, 1 - 1e-7)))
        
        print(f"  Student Test Macro F1: {test_f1:.4f}")
        print(f"  Student Test Top-1 Acc: {test_top1:.4f}")
        print(f"  Training Time: {train_time_sec:.2f}s | Latency per 1k: {latency_ms_per_1k:.2f}ms")
        
        results[name] = {
            'family': {
                'Logistic Regression': 'Linear / Probabilistic',
                'Decision Tree': 'Tree-based',
                'Random Forest': 'Ensemble',
                'LightGBM': 'Gradient Boosting',
                'MLP Neural Net': 'Neural Network'
            }[name],
            'cv_macro_f1_mean': mean_cv_f1,
            'cv_macro_f1_std': std_cv_f1,
            'cv_top1_acc': mean_cv_top1,
            'cv_log_loss': mean_cv_loss,
            'test_macro_f1': test_f1,
            'test_top1_acc': test_top1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_log_loss': test_loss,
            'train_time_sec': train_time_sec,
            'latency_ms_per_1k': latency_ms_per_1k
        }
        test_predictions[name] = test_probs
        
    # Save comparison table
    with open(os.path.join(artifacts_dir, "model_comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison results to {artifacts_dir}/model_comparison_results.json")
    
    # Print formatted markdown table
    print("\n" + "="*115)
    print(f"{'Model':<22} | {'Family':<18} | {'CV Macro F1':<12} | {'Test Macro F1':<13} | {'Top-1 Acc':<10} | {'Log Loss':<9} | {'Train(s)':<8} | {'Latency(ms)'}")
    print("="*115)
    for name, r in results.items():
        print(f"{name:<22} | {r['family']:<18} | {r['cv_macro_f1_mean']:.4f}±{r['cv_macro_f1_std']:.3f} | {r['test_macro_f1']:<13.4f} | {r['test_top1_acc']:<10.4f} | {r['test_log_loss']:<9.4f} | {r['train_time_sec']:<8.2f} | {r['latency_ms_per_1k']:.2f}")
    print("="*115)
    
    # Generate Figures
    generate_evaluation_figures(y_test, test_predictions, results, figures_dir)
    
    # Save winning initial model
    best_model_name = max(results.items(), key=lambda x: x[1]['test_macro_f1'])[0]
    print(f"\nTop Performing Initial Model: {best_model_name}")
    winning_model = models[best_model_name]
    joblib.dump(winning_model, os.path.join(artifacts_dir, "baseline_best_model.joblib"))
    print(f"Saved initial best model to {artifacts_dir}/baseline_best_model.joblib")
    
    return results

def generate_evaluation_figures(y_test: np.ndarray, test_predictions: dict, results: dict, figures_dir: str):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, probs in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('PathSnap: ROC Curves Across 5 Algorithm Families', fontsize=13, fontweight='bold', color='#1E3A8A')
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curves.png"), dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name, probs in test_predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR AUC = {pr_auc:.4f})")
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('PathSnap: Precision-Recall Curves (Class Imbalance Analysis)', fontsize=13, fontweight='bold', color='#1E3A8A')
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pr_curves.png"), dpi=300)
    plt.close()
    
    # 3. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    model_names = list(results.keys())
    cv_f1 = [results[m]['cv_macro_f1_mean'] for m in model_names]
    test_f1 = [results[m]['test_macro_f1'] for m in model_names]
    top1 = [results[m]['test_top1_acc'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    plt.bar(x - width, cv_f1, width, label='CV Macro F1', color='#3B82F6')
    plt.bar(x, test_f1, width, label='Test Macro F1', color='#10B981')
    plt.bar(x + width, top1, width, label='Test Top-1 Acc', color='#F59E0B')
    
    plt.xlabel('Algorithm Family / Model', fontsize=11)
    plt.ylabel('Score', fontsize=11)
    plt.title('PathSnap: Comprehensive Performance Comparison Across 5 Models', fontsize=13, fontweight='bold', color='#1E3A8A')
    plt.xticks(x, model_names, fontsize=10)
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Confusion Matrices (Subplots)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, probs) in zip(axes, test_predictions.items()):
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        ax.set_title(f"{name}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices on Student-Test Dataset (5 Models)', fontsize=14, fontweight='bold', y=1.03, color='#1E3A8A')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    print(f"Generated evaluation figures in {figures_dir}")

if __name__ == '__main__':
    evaluate_models()
