from pathlib import Path
import re
import warnings

import pandas as pd
import numpy as np
from scipy.stats import linregress, ttest_rel
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from collections import defaultdict
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import antropy as ant
import nolds

# 1. Data Import
DATA_FILES = {
    "2x3": "r_data/2x3_sampling_rqa_metrics_output.csv",
    "2x5": "r_data/2x5_fixed_sampling_rqa_metrics_output.csv",
    "3x3": "r_data/3x3_sampling_rqa_metrics_output.csv",
    "4x4": "r_data/4x4_sampling_rqa_metrics_output.csv",
    "4x3": "r_data/4x3_sampling_rqa_metrics_output.csv",
    "5x2": "r_data/5x2_sampling_rqa_metrics_output.csv",
    "3x4": "r_data/3x4_sampling_rqa_metrics_output.csv",
    "3x2": "r_data/3x2_sampling_rqa_metrics_output.csv",
    "2x4": "r_data/2x4_sampling_rqa_metrics_output.csv",
}
N_SPLITS = 8 # K for GroupKFold
RANDOM_STATE = 42

# 2. DATA LOADING & FEATURE ENGINEERING
def load_and_engineer_all_features(file_dict: dict) -> pd.DataFrame | None:
    """Loads all data and engineers ALL features, including non-linear ones."""
    all_dfs = []; print("📂 Loading and combining data files...")
    for difficulty, filename in file_dict.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"Warning: File not found, skipping: '{filename}'")
            continue
        df = pd.read_csv(filepath)
        df['difficulty'] = difficulty
        all_dfs.append(df)
    if not all_dfs: return None
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Preprocess string lists
    series_columns = ['window_det_series', 'window_lam_series', 'window_entr_series']
    def parse_list(s):
        if not isinstance(s, str) or not s.startswith('['): return []
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        try: return [float(num) for num in re.findall(pattern, s)]
        except: return []
    for col in series_columns: combined_df[col] = combined_df[col].apply(parse_list)

    print("🚀 Engineering all features (Temporal, Non-Linear)...")
    for metric_name in ['det', 'lam', 'entr']:
        series_col = f'window_{metric_name}_series'
        def get_clean_series(s): return [x for x in s if x is not None and not np.isnan(x)]
        
        # Standard Temporal Features
        combined_df[f'{metric_name}_mean'] = combined_df[series_col].apply(lambda s: np.mean(get_clean_series(s)) if get_clean_series(s) else np.nan)
        combined_df[f'{metric_name}_std'] = combined_df[series_col].apply(lambda s: np.std(get_clean_series(s)) if get_clean_series(s) else np.nan)
        def get_poly_coeffs(series):
            clean = get_clean_series(series)
            if len(clean) < 3: return (np.nan, np.nan) # Need 3 points for quadratic fit
            try:
                # np.polyfit returns [a, b, c] for ax^2 + bx + c
                coeffs = np.polyfit(np.arange(len(clean)), clean, 2)
                return (coeffs[0], coeffs[1]) # Return quadratic (a) and linear (b) coeffs
            except Exception:
                return (np.nan, np.nan)
            
        def get_slope(series):
            if len(series) < 2: return 0
            clean_series = [x for x in series if x is not None and not np.isnan(x)]
            if len(clean_series) < 2: return 0
            return linregress(np.arange(len(clean_series)), clean_series).slope

        poly_coeffs = combined_df[series_col].apply(get_poly_coeffs)
        combined_df[f'{metric_name}_poly_quad'] = [c[0] for c in poly_coeffs]  # Curvature (a)
        combined_df[f'{metric_name}_poly_linear'] = [c[1] for c in poly_coeffs] # Initial Slope (b)
        combined_df[f'{metric_name}_slope'] = combined_df[series_col].apply(get_slope)
        
        # --- NEW: Non-Linear Dynamics Features ---
        def sampen_safe(series):
            clean = get_clean_series(series)
            if len(clean) < 10 or np.std(clean) == 0: return np.nan
            return ant.sample_entropy(clean, order=2)

        def permen_safe(series):
            clean = get_clean_series(series)
            if len(clean) < 3 * 1: return np.nan
            return ant.perm_entropy(clean, order=3, delay=1, normalize=True)

        def hurst_safe(series):
            clean = get_clean_series(series)
            if len(clean) < 10: return np.nan
            return nolds.hurst_rs(clean)
        
        def dfa_safe(series):
            clean = np.array(series)
            if len(clean) < 5:
                return np.nan
            return nolds.dfa(clean)

        combined_df[f'{metric_name}_sampen'] = combined_df[series_col].apply(sampen_safe)
        combined_df[f'{metric_name}_permen'] = combined_df[series_col].apply(permen_safe)
        combined_df[f'{metric_name}_hurst'] = combined_df[series_col].apply(hurst_safe)
        combined_df[f'{metric_name}_dfa'] = combined_df[series_col].apply(dfa_safe)
    # Create a list of all engineered features for easy selection later
    all_engineered_features = [c for c in combined_df.columns if any(x in c for x in ['_dfa', '_mean', '_std', '_var', '_poly_quad', '_poly_linear', '_skew', '_kurtosis', '_permen', '_hurst', '_slope'])] # '_slope', '_sampen',
    
    # Drop rows where key features could not be calculated
    final_df = combined_df.dropna(subset=['global_det', 'response_length'] + all_engineered_features)
    print(f"Feature engineering complete. Using {len(final_df)} valid samples.")
    return final_df


def plot_significance_matrix(p_values_dict: dict, n_splits: int, target_col: str):
    print("📊 Generating Per-Fold Significance Matrix Plot...")
    df = pd.DataFrame({k:v for k,v in p_values_dict.items() if not k.endswith('_better')})
    df.index = [f"Fold {i+1}" for i in range(n_splits)]
    annot_df = df.applymap(lambda p: f"{p:.3f}" if p < 1.0 else "N/A")
    cmap = ['#d3d3d3', '#90ee90', '#ffcccb']
    
    color_matrix = df.copy()
    for col in df.columns:
        is_better_list = p_values_dict[f"{col}_better"]
        for i in range(len(df)):
            p = df.iloc[i][col]
            is_better = is_better_list[i]
            if p < 0.05:
                color_matrix.iloc[i, df.columns.get_loc(col)] = 1 if is_better else 2
            else:
                color_matrix.iloc[i, df.columns.get_loc(col)] = 0
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(color_matrix, annot=annot_df, fmt='s', cmap=cmap, cbar=False, linewidths=2, linecolor='white')
    plt.title(f"Per-Fold Significance (McNemar's) for Predicting '{target_col.upper()}'", fontsize=16, pad=20)
    plt.ylabel("Cross-Validation Fold"); plt.yticks(rotation=0)
    plt.tight_layout()
    if SAVE_PLOT:
        PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"significance_matrix_{target_col}.png"
        plt.savefig(PLOT_OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
        print(f"Significance matrix saved to '{PLOT_OUTPUT_DIR / filename}'")
    plt.show()


def plot_f1_heatmap(model_predictions: dict, true_labels: list, class_labels: list, n_splits: int, target_col: str):
    """
    Generates F1-score heatmaps for both Logistic Regression and Random Forest models.
    """
    print(f"📊 Generating Per-Fold F1-Score Heatmaps for predicting '{target_col.upper()}'...")
    print(f"    Using TEST SET predictions from {n_splits}-fold cross-validation")
    
    # Plot for both LR and RF models
    models_to_plot = ["Temporal RQA (LR)", "Temporal RQA (RF)"]
    
    for model_name in models_to_plot:
        if model_name not in model_predictions:
            print(f"Warning: Model '{model_name}' not found for F1 heatmap. Skipping.")
            continue

        # Collect ALL actual classes from test data across all folds
        all_test_classes = set()
        class_counts_per_fold = []
        
        for fold in range(n_splits):
            fold_true = true_labels[fold]
            unique, counts = np.unique(fold_true, return_counts=True)
            fold_counts = dict(zip(unique, counts))
            class_counts_per_fold.append(fold_counts)
            all_test_classes.update(unique)
        
        present_classes = sorted(all_test_classes)
        
        print(f"\n--- {model_name}: Classes in Test Data: {present_classes} ---")
        
        # Compute F1 scores
        f1_scores_per_fold = []
        for fold in range(n_splits):
            fold_preds = model_predictions[model_name][fold]
            fold_true = true_labels[fold]
            
            report = classification_report(fold_true, fold_preds, output_dict=True, 
                                          labels=present_classes, zero_division=0)
            
            fold_f1s = {str(label): report[str(label)]['f1-score'] for label in present_classes}
            f1_scores_per_fold.append(fold_f1s)

        f1_df = pd.DataFrame(f1_scores_per_fold)
        f1_df.index = [f"Fold {i+1}" for i in range(n_splits)]
        
        # Print detailed class distribution analysis
        print(f"\n--- {model_name}: Per-Class Test Set Distribution & Performance ---")
        print(f"{'Class':<8} {'Test Samples':<15} {'Avg F1':<12} {'Std F1':<12} {'Min F1':<12} {'Max F1':<12}")
        print("-" * 75)
        
        total_counts = {label: 0 for label in present_classes}
        for fold_counts in class_counts_per_fold:
            for label in present_classes:
                total_counts[label] += fold_counts.get(label, 0)
        
        for label in present_classes:
            count = total_counts[label]
            avg_f1 = f1_df[str(label)].mean()
            std_f1 = f1_df[str(label)].std()
            min_f1 = f1_df[str(label)].min()
            max_f1 = f1_df[str(label)].max()
            print(f"{label:<8} {count:>6d} ({count/(n_splits):.0f}/fold)  {avg_f1:>6.3f}      {std_f1:>6.3f}      {min_f1:>6.3f}      {max_f1:>6.3f}")

        # Create annotations
        annot_matrix = []
        for fold_idx in range(n_splits):
            row = []
            for class_label in [str(c) for c in present_classes]:
                f1_score = f1_df.iloc[fold_idx][class_label]
                class_in_fold = class_label in [str(c) for c in class_counts_per_fold[fold_idx].keys()]
                if not class_in_fold:
                    row.append("—")
                elif f1_score == 0.0:
                    row.append("0.00")
                else:
                    row.append(f"{f1_score:.2f}")
            annot_matrix.append(row)
        
        mean_f1_row = [f"{f1_df[str(c)].mean():.2f}" for c in present_classes]
        annot_matrix.append(mean_f1_row)
        
        mean_f1_values = [f1_df[str(c)].mean() for c in present_classes]
        f1_df_with_mean = pd.concat([f1_df, pd.DataFrame([mean_f1_values], columns=f1_df.columns, index=['Mean'])])
        
        annot_df = pd.DataFrame(annot_matrix, columns=f1_df_with_mean.columns, index=f1_df_with_mean.index)

        fig_width = max(10, len(present_classes) * 1.5)
        plt.figure(figsize=(fig_width, 9))
        
        mask = np.zeros_like(f1_df_with_mean, dtype=bool)
        
        ax = sns.heatmap(f1_df_with_mean, annot=annot_df, fmt='s', cmap="RdYlGn", linewidths=.5, 
                         vmin=0, vmax=1, cbar_kws={'label': 'F1-Score'},
                         linecolor='white', mask=mask)
        
        ax.axhline(y=n_splits, color='black', linewidth=2)
        
        model_display = "Logistic Regression" if "LR" in model_name else "Random Forest"
        plt.title(f"Per-Fold F1-Scores for '{model_display}' Model (TEST SET)\n(Predicting {target_col.upper()})", 
                  fontsize=16, pad=20)
        plt.xlabel("Target Class (Difficulty Level)", fontsize=12)
        plt.ylabel("Cross-Validation Fold", fontsize=12)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        
        if SAVE_PLOT:
            model_suffix = "lr" if "LR" in model_name else "rf"
            filename = f"f1_heatmap_{target_col}_{model_suffix}.png"
            PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(PLOT_OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
            print(f" F1-Score heatmap saved to '{PLOT_OUTPUT_DIR / filename}'")
        
        plt.show()


def plot_aggregated_confusion_matrix(model_predictions: dict, true_labels_per_fold: list, class_labels: list, target_col: str):
    """
    Generates normalized confusion matrices for both LR and RF models.
    """
    print(f"📊 Generating Aggregated Confusion Matrices for predicting '{target_col.upper()}'...")
    
    models_to_plot = ["Baseline (Length)", "Global RQA (LR)", "Temporal RQA (RF)", "Combined (RF)"]

    for model_name in models_to_plot:
        if model_name not in model_predictions:
            print(f"Warning: Model '{model_name}' not found. Skipping.")
            continue
            
        all_predictions = np.concatenate(model_predictions[model_name])
        all_true_labels = np.concatenate(true_labels_per_fold)
        
        unique_labels = sorted(set(all_true_labels))
        present_class_labels = [label for label in class_labels if label in unique_labels]
        
        cm = confusion_matrix(all_true_labels, all_predictions, labels=present_class_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                percentage = cm_normalized[i, j]
                annot[i, j] = f'{percentage:.1%}\n({count})'
        
        fig_size = max(8, len(present_class_labels) * 1.2)
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm_normalized, annot=annot, fmt='', cmap="Blues", 
                    xticklabels=present_class_labels, yticklabels=present_class_labels,
                    cbar_kws={'label': 'Recall (True Positive Rate)'}, linewidths=0.5, linecolor='gray')
        
        model_display = "Logistic Regression" if "LR" or "Baseline (Length)" in model_name else "Random Forest"
        print(model_name)
        plt.title(f"Aggregated Confusion Matrix - {model_name} (TEST SET)\n(Predicting {target_col.upper()})\nShowing Recall % (Count)", 
                  fontsize=16, pad=20)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45, ha='right')
        
        print(f"\n--- {model_name}: Per-Class Performance ---")
        for i, label in enumerate(present_class_labels):
            true_count = cm[i, :].sum()
            correct = cm[i, i]
            recall = cm_normalized[i, i]
            precision_denom = cm[:, i].sum()
            precision = cm[i, i] / precision_denom if precision_denom > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  Class {label}: Recall={recall:.1%} ({correct}/{true_count}), Precision={precision:.1%}, F1={f1:.3f}")
        
        if SAVE_PLOT:
            PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model_suffix = "lr" if "LR" in model_name else "rf"
            filename = f"confusion_matrix_{target_col}_{model_suffix}.png"
            plt.savefig(PLOT_OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to '{PLOT_OUTPUT_DIR / filename}'")
            
        plt.show()


# 3. CORE CROSS-VALIDATION AND REPORT
def run_comparison_experiment(df: pd.DataFrame, target_col: str):
    """Runs a full CV comparison for a given target variable, comparing LR vs RF."""
    
    baseline_features = ['response_length']
    global_rqa_features = ['global_det', 'global_lam', 'global_entr']
    temporal_rqa_features = [c for c in df.columns if any(x in c for x in [ '_mean', '_std', '_slope', '_dfa'])]
    combined_features = baseline_features + temporal_rqa_features
    
    # Updated model configs to explicitly separate LR and RF
    model_configs = {
        "Baseline (Length)": (baseline_features, "lr"),
        "Baseline (Length) (RF)": (baseline_features, "rf"),
        "Global RQA (LR)": (global_rqa_features, "lr"),
        "Global RQA (RF)": (global_rqa_features, "rf"),
        "Temporal RQA (LR)": (temporal_rqa_features, "lr"),
        "Temporal RQA (RF)": (temporal_rqa_features, "rf"),
        "Combined (LR)": (combined_features, "lr"),
        "Combined (RF)": (combined_features, "rf")
    }
    
    X = df[['id', 'run_id'] + baseline_features + global_rqa_features + temporal_rqa_features]
    y = df[target_col]
    groups = df['id']
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    # Store scores and predictions for all models
    model_scores = {name: [] for name in model_configs}
    model_predictions = {name: [] for name in model_configs}
    true_labels_per_fold = []

    mcnemar_results = {}
    for name in model_configs:
        if name != "Baseline (Length)":
            mcnemar_results[f"{name} vs Baseline"] = []
            mcnemar_results[f"{name} vs Baseline_better"] = []
    
    print(f"\n--- Starting {N_SPLITS}-Fold Group CV to Predict '{target_col.upper()}' ---")
    print(f"    Comparing Logistic Regression vs Random Forest")
    


    # Containers for storing results
    perm_importances_per_model = defaultdict(list)
    gini_importances_per_model = defaultdict(list)

    # Loop over GroupKFold
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        true_labels_per_fold.append(y_test.values)

        print(f"Fold {fold}: X_train={len(X_train)}, X_test={len(X_test)}, y_train={len(y_train)}, y_test={len(y_test)}")

        preds = {}
        for name, (features, model_type) in model_configs.items():
            if model_type == "rf":
                # Train RF on training fold
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight='balanced',
                    n_jobs=-1
                ).fit(X_train[features], y_train)

                # --- Store Gini (MDI) importances per fold ---
                gini_imp = pd.Series(model.feature_importances_, index=features)
                gini_importances_per_model[name].append(gini_imp)

                # --- Store Permutation importance on test fold ---
                perm = permutation_importance(
                    model,
                    X_test[features],
                    y_test,
                    n_repeats=20,
                    random_state=RANDOM_STATE,
                    scoring='balanced_accuracy',
                    n_jobs=-1
                )
                perm_imp = pd.Series(perm.importances_mean, index=features)
                perm_importances_per_model[name].append(perm_imp)

            else:  # Logistic Regression
                model = LogisticRegression(
                    random_state=RANDOM_STATE,
                    class_weight='balanced',
                    max_iter=1000
                ).fit(X_train[features], y_train)

            # --- Predictions ---
            preds[name] = model.predict(X_test[features])
            model_scores[name].append(accuracy_score(y_test, preds[name]))
            model_predictions[name].append(preds[name])

        # --- McNemar test against baseline ---
        for name in model_configs:
            if name == "Baseline (Length)":
                continue
            table = np.zeros((2, 2))
            correct_model = (preds[name] == y_test)
            correct_baseline = (preds["Baseline (Length)"] == y_test)
            table[0,1] = np.sum(correct_baseline & ~correct_model)
            table[1,0] = np.sum(~correct_baseline & correct_model)
            if table[0, 1] + table[1, 0] > 0:
                p_val = mcnemar(table, exact=True).pvalue
                mcnemar_results[f"{name} vs Baseline"].append(p_val)
                mcnemar_results[f"{name} vs Baseline_better"].append(table[1,0] > table[0,1])
            else:
                mcnemar_results[f"{name} vs Baseline"].append(1.0)
                mcnemar_results[f"{name} vs Baseline_better"].append(False)

    # After CV: aggregate and plot importances
    for name in perm_importances_per_model:
        # Permutation importances
        perm_df = pd.concat(perm_importances_per_model[name], axis=1)
        final_perm_importance = perm_df.mean(axis=1).sort_values(ascending=False)

        top15_perm = final_perm_importance.head(15).sort_values()

        plt.figure(figsize=(8, 6))
        plt.barh(top15_perm.index, top15_perm.values)
        plt.xlabel("Mean Decrease in Balanced Accuracy")
        plt.title(f"Permutation Importance ({name}, GroupKFold Average)")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.show()

        # Gini importances
        gini_df = pd.concat(gini_importances_per_model[name], axis=1)
        final_gini_importance = gini_df.mean(axis=1).sort_values(ascending=False)

        top15_gini = final_gini_importance.head(15).sort_values()

        plt.figure(figsize=(12, 8))
        top15_gini.plot(kind='barh', color='skyblue')
        plt.title(f'Gini Importance ({name}, GroupKFold Average)', fontsize=18, pad=20)
        plt.xlabel('Importance (Gini Impurity Reduction)')
        plt.ylabel('Features')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    print("\n--- Averaged Cross-Validation Accuracies ---")
    for name, scores in model_scores.items(): 
        print(f"  - {name:<25}: {np.mean(scores):.2%} ± {np.std(scores):.2%}")

    # Compare LR vs RF directly
    print("\n--- Direct Comparison: Logistic Regression vs Random Forest ---")
    lr_scores = model_scores["Temporal RQA (LR)"]
    rf_scores = model_scores["Temporal RQA (RF)"]
    stat, p_value = ttest_rel(rf_scores, lr_scores)
    mean_diff = np.mean(rf_scores) - np.mean(lr_scores)
    if p_value < 0.05:
        winner = "Random Forest" if mean_diff > 0 else "Logistic Regression"
        print(f" {winner} is SIGNIFICANTLY BETTER (p={p_value:.4f}, diff={mean_diff:.2%})")
    else:
        print(f" No significant difference between models (p={p_value:.4f}, diff={mean_diff:.2%})")

    print("\n--- Significance Test: Paired T-test (vs. Baseline) ---")
    for name, scores in model_scores.items():
        if name == "Baseline (Length)": continue
        stat, p_value = ttest_rel(scores, model_scores["Baseline (Length)"])
        is_better = np.mean(scores) > np.mean(model_scores["Baseline (Length)"])
        if p_value < 0.05 and is_better: 
            print(f"  {name} is SIGNIFICANTLY BETTER than Baseline (p={p_value:.4f})")
        else: 
            print(f"  {name} is NOT significantly better than Baseline (p={p_value:.4f})")
        
    # Call plotting functions
    plot_significance_matrix(mcnemar_results, N_SPLITS, target_col)
    
    class_labels = y.cat.categories.tolist() if pd.api.types.is_categorical_dtype(y) else sorted(y.unique())
    
    plot_f1_heatmap(model_predictions, true_labels_per_fold, class_labels, N_SPLITS, target_col)
    plot_aggregated_confusion_matrix(model_predictions, true_labels_per_fold, class_labels, target_col)


def generate_final_report(df: pd.DataFrame):
    
    print("\n\n" + "#"*80)
    print("###                FINAL PROJECT REPORT                ###")
    print("###          Logistic Regression vs Random Forest          ###")
    print("#"*80)

    print("\n" + "*"*30 + " PART 1: PREDICTING PROBLEM DIFFICULTY " + "*"*29)
    run_comparison_experiment(df, target_col='difficulty')

    print("\n\n" + "*"*32 + " PART 2: PREDICTING ACCURACY " + "*"*31)
    run_comparison_experiment(df, target_col='accuracy')


SAVE_PLOT = False  # Set to True to save plots
PLOT_OUTPUT_DIR = Path("./plots")  # Directory to save plots

maindf = load_and_engineer_all_features(DATA_FILES)
generate_final_report(maindf)