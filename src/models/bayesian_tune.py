import os
import sys
import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.train_models import FEATURE_COLS, TARGET_COL, GROUP_COL, compute_top1_accuracy

# Suppress Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(
    train_parquet: str = "data/processed/train_features.parquet",
    test_parquet: str = "data/processed/test_features.parquet",
    artifacts_dir: str = "artifacts",
    figures_dir: str = "reports/figures",
    n_trials: int = 35
):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading data for Bayesian Optimization...")
    df_train = pd.read_parquet(train_parquet)
    df_test = pd.read_parquet(test_parquet)
    
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].values
    groups_train = df_train[GROUP_COL].values
    
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL].values
    
    gkf = GroupKFold(n_splits=5)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 60, 250),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 5.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            
            clf = LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            cv_scores.append(f1_score(y_val, preds, average='macro'))
            
        return float(np.mean(cv_scores))

    print(f"\n" + "="*80)
    print(f"RUNNING BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA - {n_trials} TRIALS)")
    print("="*80)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    tune_duration = time.time() - start_time
    
    print(f"\nOptimization Finished in {tune_duration:.1f}s!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best CV Macro F1: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Evaluate Untuned Baseline vs Tuned on Student Test
    print("\n" + "="*80)
    print("EVALUATING UNTUNED VS TUNED MODEL ON STUDENT-TEST DATASET")
    print("="*80)
    
    # 1. Untuned Default LightGBM
    default_params = {'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    untuned_clf = LGBMClassifier(**default_params)
    untuned_clf.fit(X_train, y_train)
    
    t0_inf = time.time()
    untuned_test_probs = untuned_clf.predict_proba(X_test)[:, 1]
    untuned_lat = ((time.time() - t0_inf) / len(X_test)) * 1e6
    untuned_preds = (untuned_test_probs >= 0.5).astype(int)
    
    untuned_cv_f1 = np.mean([
        f1_score(y_train[val_idx], untuned_clf.fit(X_train.iloc[train_idx], y_train[train_idx]).predict(X_train.iloc[val_idx]), average='macro')
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train)
    ])
    untuned_test_f1 = f1_score(y_test, untuned_preds, average='macro')
    untuned_top1 = compute_top1_accuracy(df_test, untuned_test_probs)
    untuned_loss = log_loss(y_test, np.clip(untuned_test_probs, 1e-7, 1 - 1e-7))
    
    # 2. Tuned LightGBM
    best_params = {**study.best_params, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    tuned_clf = LGBMClassifier(**best_params)
    tuned_clf.fit(X_train, y_train)
    
    t0_inf = time.time()
    tuned_test_probs = tuned_clf.predict_proba(X_test)[:, 1]
    tuned_lat = ((time.time() - t0_inf) / len(X_test)) * 1e6
    tuned_preds = (tuned_test_probs >= 0.5).astype(int)
    
    tuned_cv_f1 = study.best_value
    tuned_test_f1 = f1_score(y_test, tuned_preds, average='macro')
    tuned_top1 = compute_top1_accuracy(df_test, tuned_test_probs)
    tuned_loss = log_loss(y_test, np.clip(tuned_test_probs, 1e-7, 1 - 1e-7))
    
    comparison = {
        'metric': ['Cross-Validation Macro F1', 'Student-Test Top-1 Accuracy', 'Student-Test Macro F1', 'Log Loss', 'Inference Latency (ms/1k)'],
        'untuned_baseline': [float(untuned_cv_f1), float(untuned_top1), float(untuned_test_f1), float(untuned_loss), float(untuned_lat)],
        'tuned_model': [float(tuned_cv_f1), float(tuned_top1), float(tuned_test_f1), float(tuned_loss), float(tuned_lat)],
        'relative_improvement_pct': [
            float((tuned_cv_f1 - untuned_cv_f1) / untuned_cv_f1 * 100),
            float((tuned_top1 - untuned_top1) / untuned_top1 * 100),
            float((tuned_test_f1 - untuned_test_f1) / untuned_test_f1 * 100),
            float((untuned_loss - tuned_loss) / untuned_loss * 100), # loss decrease is improvement
            float((untuned_lat - tuned_lat) / untuned_lat * 100)
        ]
    }
    
    df_comp = pd.DataFrame(comparison)
    print("\n" + df_comp.to_string(index=False))
    
    # Save artifacts
    joblib.dump(tuned_clf, os.path.join(artifacts_dir, "best_model.joblib"))
    joblib.dump(study, os.path.join(artifacts_dir, "optuna_study.joblib"))
    
    tuning_summary = {
        'n_trials': n_trials,
        'duration_seconds': tune_duration,
        'best_hyperparameters': study.best_params,
        'best_cv_macro_f1': float(study.best_value),
        'comparison_table': comparison
    }
    with open(os.path.join(artifacts_dir, "tuning_results.json"), "w") as f:
        json.dump(tuning_summary, f, indent=2)
    print(f"\nSaved best model to {artifacts_dir}/best_model.joblib")
    print(f"Saved tuning results to {artifacts_dir}/tuning_results.json")
    
    # Plot Optuna Optimization History
    plt.figure(figsize=(9, 5))
    trial_values = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(trial_values)
    
    plt.plot(range(1, len(trial_values) + 1), trial_values, 'o', color='#93C5FD', alpha=0.7, label='Trial Macro F1')
    plt.plot(range(1, len(best_so_far) + 1), best_so_far, '-', color='#1E40AF', lw=2.5, label='Best Objective Value')
    plt.xlabel('Trial Number', fontsize=11)
    plt.ylabel('CV Macro F1', fontsize=11)
    plt.title('PathSnap: Bayesian Hyperparameter Optimization History (Optuna)', fontsize=13, fontweight='bold', color='#1E3A8A')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "optuna_optimization_history.png"), dpi=300)
    plt.close()
    print(f"Saved optimization history plot to {figures_dir}/optuna_optimization_history.png")
    
    return tuning_summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    args = parser.parse_args()
    run_bayesian_optimization(n_trials=args.n_trials)
