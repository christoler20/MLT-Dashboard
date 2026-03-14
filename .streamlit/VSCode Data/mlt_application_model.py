#!/usr/bin/env python3
"""
MLT Career Prep – Application Outcome Prediction Pipeline
==========================================================
Trains and evaluates supervised ML models to predict the likelihood that
an MLT Fellow's job application results in an offer (or another configurable
funnel-stage outcome).

Author : (generated for MLT diagnostics)
Data   : MLT CP anonymised application-level file
Usage  : python mlt_application_model.py --data_path <path_to_xlsx_or_csv> [OPTIONS]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 0.  COLUMN NAME MAPPING  (raw file names → canonical names)
# ---------------------------------------------------------------------------
# The raw Excel has verbose column names.  We map them to shorter canonical
# names used throughout the script.  If your file already uses the canonical
# names, the mapper will simply pass through.

RAW_TO_CANONICAL = {
    "Program Enrollment: Enrollment ID": "FELLOW_ID",
    "Race": "RACE",
    "Ethnicity": "ETHNICITY",
    "Gender": "GENDER",
    "Program Enrollment: Program Track": "MLT_TRACK",
    "Related Organization": "EMPLOYER",
    "Partner Org?": "PARTNER_ORG",
    "Title": "ROLE",
    "Type": "APPLICATION_TYPE",
    "Application Status": "APPLICATION_STATUS",
    "Primary Industry Interest": "PRIMARY_INDUSTRY",
    "Primary Functional Interest": "PRIMARY_FUNCTION",
    "Mother's Education": "MOTHERS_EDUCATION",
    "Father's Education": "FATHERS_EDUCATION",
    "Designated Low Income": "LOW_INCOME",
    "First Generation College": "FIRST_GEN",
    "Undergrad GPA": "GPA",
    "Pell Grant Count": "PELL_GRANT_COUNT",
    "Primary Major": "PRIMARY_MAJOR",
    "SAT Score": "SAT_SCORE",
    "Program Enrollment: Program": "PROGRAM_COHORT",
    "Program Enrollment: Coach": "COACH",
    "Program Enrollment: Status": "ENROLLMENT_STATUS",
    # Admissions score sub-components
    "Achievement Orientation #": "SCORE_ACHIEVEMENT",
    "Career Goal # ": "SCORE_CAREER_GOAL",  # note trailing space in source
    "Career Goal #": "SCORE_CAREER_GOAL",   # in case of trimmed version
    "Work/Internship #": "SCORE_WORK_INTERN",
    "Leadership #": "SCORE_LEADERSHIP",
    "Oral Communication #": "SCORE_ORAL_COMM",
    "Score out of 5 (CP 18 - 24)": "SCORE_COMPOSITE",
    # School-level data
    "Bureau of Economic Analysis (BEA) regions (HD2024)": "SCHOOL_BEA_REGION",
    "Historically Black College or University (HD2024)": "SCHOOL_HBCU",
    "Carnegie Classification 2025: Institutional Classification (HD2024)": "SCHOOL_CARNEGIE_CLASS",
    "Carnegie Classification 2025: Student Access and Earnings (HD2024)": "SCHOOL_ACCESS_EARNINGS",
    "Carnegie Classification 2025: Research Activity Designation (HD2024)": "SCHOOL_RESEARCH",
    "Carnegie Classification 2025: Institutional Size (HD2024)": "SCHOOL_SIZE",
    "Carnegie Classification 2025: Award Level Focus (HD2024)": "SCHOOL_AWARD_FOCUS",
    "Carnegie Classification 2025: Undergraduate Academic Program Mix (HD2024)": "SCHOOL_UG_PROG_MIX",
}

# Columns that are entirely null / uninformative → drop immediately
DROP_ALWAYS = [
    "For CP 2018 - CP 2024, Admissions scores",
    "UG School Data, see Value Lables ",   # trailing space in source
    "UG School Data, see Value Lables",
]

# ---------------------------------------------------------------------------
# 1.  LABEL DEFINITIONS
# ---------------------------------------------------------------------------
# Application Status values in the data:
#   Applied, Denied, Pending, Offered & Committed, Offered & Declined,
#   Withdrew Application, Offered, Invited, Offer Rescinded,
#   Applied to MLT, My offer has been rescinded., Did Not Apply
#
# We define several possible binary labels the user can choose from.

LABEL_DEFINITIONS = {
    "OFFER_EXTENDED": {
        "positive": ["Offered & Committed", "Offered & Declined", "Offered", "Offer Rescinded",
                     "My offer has been rescinded."],
        "negative": ["Applied", "Denied", "Withdrew Application"],
        "description": "Any application that resulted in an offer (accepted, declined, or rescinded).",
    },
    "OFFER_COMMITTED": {
        "positive": ["Offered & Committed"],
        "negative": ["Applied", "Denied", "Withdrew Application",
                     "Offered & Declined", "Offered", "Offer Rescinded",
                     "My offer has been rescinded."],
        "description": "Fellow accepted the offer and committed.",
    },
    "NOT_DENIED": {
        "positive": ["Offered & Committed", "Offered & Declined", "Offered",
                     "Invited", "Pending"],
        "negative": ["Applied", "Denied", "Withdrew Application"],
        "description": "Application was not denied (includes pending/offered/invited).",
    },
}

# ---------------------------------------------------------------------------
# 2.  FEATURE CONFIGURATION
# ---------------------------------------------------------------------------
# Demographic columns: used for subgroup evaluation but NOT as features
# unless the user explicitly opts in.
DEMOGRAPHIC_COLS = ["RACE", "ETHNICITY", "GENDER"]

# Post-outcome / leakage columns – drop from features unconditionally.
# These are columns whose values could only be known *after* the application
# decision or that directly encode the outcome.
LEAKAGE_PATTERNS = [
    "APPLICATION_STATUS",  # the outcome itself
    "ENROLLMENT_STATUS",   # fellow's program status (could reflect outcome)
]

# Columns to use as features (will be filtered by what actually exists).
FELLOW_FEATURES = [
    "MLT_TRACK",
    "PRIMARY_INDUSTRY",
    "PRIMARY_FUNCTION",
    "MOTHERS_EDUCATION",
    "FATHERS_EDUCATION",
    "LOW_INCOME",
    "FIRST_GEN",
    "GPA",
    "PELL_GRANT_COUNT",
    "PRIMARY_MAJOR",
    "SAT_SCORE",
    "PROGRAM_COHORT",
    "SCORE_ACHIEVEMENT",
    "SCORE_CAREER_GOAL",
    "SCORE_WORK_INTERN",
    "SCORE_LEADERSHIP",
    "SCORE_ORAL_COMM",
    "SCORE_COMPOSITE",
    "SCHOOL_BEA_REGION",
    "SCHOOL_HBCU",
    "SCHOOL_CARNEGIE_CLASS",
    "SCHOOL_ACCESS_EARNINGS",
    "SCHOOL_RESEARCH",
    "SCHOOL_SIZE",
    "SCHOOL_AWARD_FOCUS",
    "SCHOOL_UG_PROG_MIX",
]

JOB_FEATURES = [
    "EMPLOYER",
    "ROLE",
    "APPLICATION_TYPE",
    "PARTNER_ORG",
]

# ---------------------------------------------------------------------------
# 3.  HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV or Excel, rename columns to canonical names."""
    path = Path(path)
    print(f"[INFO] Loading data from {path} ...")
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Drop always-null columns
    for col in DROP_ALWAYS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Rename columns that match the mapping
    rename_map = {k: v for k, v in RAW_TO_CANONICAL.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # If columns were already canonical (e.g., pre-processed CSV), keep them
    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df


def _reclassify_applied_as_denied(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify 'Applied' status as 'Denied' for cohort CP 2024 and prior.

    Rationale: for older cohorts, enough time has elapsed that an application
    still sitting at 'Applied' (no employer response) is effectively a
    rejection.  Cohorts CP 2025+ may still have pending outcomes, so they
    are left as-is.
    """
    if "PROGRAM_COHORT" not in df.columns:
        print(
            "[WARN] PROGRAM_COHORT column not found; skipping Applied→Denied reclassification.")
        return df

    # Extract the numeric year from strings like "CP 2024"
    cohort_year = df["PROGRAM_COHORT"].str.extract(
        r"(\d{4})", expand=False).astype(float)

    reclassify_mask = (
        (df["APPLICATION_STATUS"] == "Applied") &
        (cohort_year <= 2024)
    )
    n_reclassified = reclassify_mask.sum()

    if n_reclassified > 0:
        df = df.copy()
        df.loc[reclassify_mask, "APPLICATION_STATUS"] = "Denied"
        print(f"[INFO] Reclassified {n_reclassified:,} 'Applied' rows from "
              f"CP 2024 and prior → 'Denied' (no response = de facto rejection).")
    else:
        print("[INFO] No 'Applied' rows from CP 2024 or prior to reclassify.")

    return df


def create_label(df: pd.DataFrame, label_name: str) -> pd.Series:
    """Create a binary label column based on APPLICATION_STATUS."""
    if label_name not in LABEL_DEFINITIONS:
        available = ", ".join(LABEL_DEFINITIONS.keys())
        raise ValueError(
            f"Unknown label '{label_name}'. Choose from: {available}")

    # Reclassify stale "Applied" as "Denied" for CP 2024 and earlier
    df = _reclassify_applied_as_denied(df)

    defn = LABEL_DEFINITIONS[label_name]
    pos = set(defn["positive"])
    neg = set(defn["negative"])

    # Only keep rows whose status is in positive or negative sets
    mask = df["APPLICATION_STATUS"].isin(pos | neg)
    df_filtered = df[mask].copy()
    df_filtered["label"] = df_filtered["APPLICATION_STATUS"].isin(
        pos).astype(int)

    n_pos = df_filtered["label"].sum()
    n_neg = len(df_filtered) - n_pos
    print(f"[INFO] Label '{label_name}': {defn['description']}")
    print(
        f"       Positive: {n_pos:,}  |  Negative: {n_neg:,}  |  Rate: {n_pos/len(df_filtered):.3f}")
    print(f"       Dropped {len(df) - len(df_filtered):,} rows with ambiguous status "
          f"(e.g., Pending, Invited, Did Not Apply).")
    return df_filtered


def clean_gpa(series: pd.Series) -> pd.Series:
    """Cap GPA at 4.0 scale; values >4.0 are likely data-entry errors."""
    s = series.copy()
    s[s > 4.0] = np.nan  # treat impossible values as missing
    return s


def extract_cohort_year(series: pd.Series) -> pd.Series:
    """Extract numeric year from PROGRAM_COHORT like 'CP 2024'."""
    return series.str.extract(r"(\d{4})", expand=False).astype(float)


def build_feature_matrix(df: pd.DataFrame, include_demographics: bool = False):
    """Select and categorize feature columns, return X dataframe + column lists."""
    candidate_features = FELLOW_FEATURES + JOB_FEATURES
    if include_demographics:
        candidate_features += DEMOGRAPHIC_COLS

    # Remove leakage columns
    for pat in LEAKAGE_PATTERNS:
        if pat in candidate_features:
            candidate_features.remove(pat)

    # Keep only columns that exist
    available = [c for c in candidate_features if c in df.columns]
    X = df[available].copy()

    # --- Feature engineering ---
    # Clean GPA
    if "GPA" in X.columns:
        X["GPA"] = clean_gpa(X["GPA"])

    # Extract cohort year as numeric
    if "PROGRAM_COHORT" in X.columns:
        X["COHORT_YEAR"] = extract_cohort_year(X["PROGRAM_COHORT"])
        X.drop(columns=["PROGRAM_COHORT"], inplace=True)
        available = [c for c in available if c !=
                     "PROGRAM_COHORT"] + ["COHORT_YEAR"]

    # Cap ROLE and EMPLOYER cardinality: keep top-N, rest → "OTHER"
    for col, top_n in [("EMPLOYER", 100), ("ROLE", 200), ("PRIMARY_MAJOR", 50)]:
        if col in X.columns:
            top_vals = X[col].value_counts().nlargest(top_n).index
            X[col] = X[col].where(X[col].isin(top_vals), other="OTHER")

    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Classify columns as numeric vs categorical
    numeric_cols = []
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype in ("float64", "float32", "int64", "int32"):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    print(f"[INFO] Feature matrix: {X.shape[1]} features "
          f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    return X, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Build a sklearn ColumnTransformer for preprocessing."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=20, sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def split_data(df, X, y, strategy, group_col, test_size, seed):
    """Split data into train and test sets."""
    if strategy == "time" and "COHORT_YEAR" in X.columns:
        # Time-based split: train on earlier cohorts, test on later ones
        years = X["COHORT_YEAR"].dropna()
        if len(years) > 0:
            cutoff = years.quantile(1 - test_size)
            train_mask = X["COHORT_YEAR"] <= cutoff
            test_mask = X["COHORT_YEAR"] > cutoff
            # Handle NaN cohort years: assign to training
            train_mask = train_mask.fillna(True)
            test_mask = test_mask.fillna(False)
            train_idx = df.index[train_mask]
            test_idx = df.index[test_mask]
            print(f"[INFO] Time-based split: train cohort years ≤ {cutoff:.0f}, "
                  f"test > {cutoff:.0f}")
            print(
                f"       Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")
            return train_idx, test_idx

    # Default: GroupShuffleSplit by FELLOW_ID
    if group_col in df.columns:
        groups = df[group_col]
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        n_train_groups = groups.iloc[train_idx].nunique()
        n_test_groups = groups.iloc[test_idx].nunique()
        print(f"[INFO] GroupShuffleSplit by {group_col}:")
        print(f"       Train: {len(train_idx):,} rows ({n_train_groups:,} fellows)  |  "
              f"Test: {len(test_idx):,} rows ({n_test_groups:,} fellows)")
        return train_idx, test_idx
    else:
        # Fallback: simple stratified split
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=y
        )
        print(f"[INFO] Stratified split (no group column found):")
        print(f"       Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")
        return train_idx, test_idx


def get_tree_model():
    """Return the best available tree-based model class + param grid."""
    # Try XGBoost first, then LightGBM, then fallback to sklearn
    try:
        from xgboost import XGBClassifier
        print("[INFO] Using XGBClassifier.")
        model = XGBClassifier(
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1
        )
        param_grid = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
            "model__scale_pos_weight": [1, 3, 5],
        }
        return model, param_grid, "XGBoost"
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        print("[INFO] Using LGBMClassifier.")
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        param_grid = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
            "model__class_weight": ["balanced", None],
        }
        return model, param_grid, "LightGBM"
    except ImportError:
        pass

    # Fallback: HistGradientBoosting (no class_weight, but handles imbalance ok)
    print("[INFO] Using HistGradientBoostingClassifier (sklearn built-in).")
    model = HistGradientBoostingClassifier(random_state=42)
    param_grid = {
        "model__max_iter": [100, 300, 500],
        "model__max_depth": [3, 5, 7, None],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__min_samples_leaf": [10, 20, 50],
        "model__class_weight": ["balanced", None],
    }
    return model, param_grid, "HistGradientBoosting"


def evaluate_model(y_true, y_prob, y_pred, model_name="Model"):
    """Compute standard classification metrics and return as dict."""
    metrics = {}
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    metrics["log_loss"] = log_loss(y_true, y_prob)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    print(f"\n{'='*60}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*60}")
    print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC       : {metrics['pr_auc']:.4f}")
    print(f"  Log Loss     : {metrics['log_loss']:.4f}")
    print(f"  Brier Score  : {metrics['brier_score']:.4f}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1           : {metrics['f1']:.4f}")
    print(f"  Specificity  : {metrics['specificity']:.4f}")
    print(f"  Confusion    : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    return metrics


def find_optimal_threshold(y_true, y_prob):
    """Find the threshold that maximizes F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(
        thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    print(
        f"[INFO] Optimal F1 threshold: {best_threshold:.3f} (F1 = {best_f1:.4f})")
    return best_threshold


def subgroup_evaluation(y_true, y_prob, df_test, subgroup_cols):
    """Compute metrics per subgroup for fairness / segmented evaluation.

    All inputs must share the same index (reset_index before calling).
    y_prob is a numpy array aligned positionally with df_test / y_true.
    """
    results = []
    # Ensure positional alignment by using numpy arrays
    yt_arr = np.asarray(y_true)
    yp_arr = np.asarray(y_prob)

    for col in subgroup_cols:
        if col not in df_test.columns:
            continue
        # Convert to string to handle mixed types (float NaN + string)
        col_vals = df_test[col].fillna("MISSING").astype(str).values
        unique_vals = sorted(set(col_vals))
        for group_val in unique_vals:
            mask = col_vals == group_val
            n_total = int(mask.sum())
            if n_total < 30:  # skip tiny groups
                continue
            yt = yt_arr[mask]
            yp = yp_arr[mask]
            n_pos = int(yt.sum())
            row = {
                "subgroup_column": col,
                "subgroup_value": str(group_val),
                "n_total": n_total,
                "n_positive": n_pos,
                "positive_rate": n_pos / n_total,
            }
            if n_pos > 0 and n_pos < n_total:
                row["roc_auc"] = roc_auc_score(yt, yp)
                row["pr_auc"] = average_precision_score(yt, yp)
                row["brier_score"] = brier_score_loss(yt, yp)
            else:
                row["roc_auc"] = np.nan
                row["pr_auc"] = np.nan
                row["brier_score"] = np.nan
            results.append(row)

    return pd.DataFrame(results)


def top_n_recommendations(df_test, y_prob, y_true, group_col, top_n):
    """For each Fellow, rank applications by predicted probability."""
    recs = df_test[[group_col]].copy()
    recs["predicted_prob"] = y_prob
    recs["actual_outcome"] = y_true.values if hasattr(
        y_true, 'values') else y_true

    # Add identifiable columns if present
    for col in ["EMPLOYER", "ROLE", "APPLICATION_TYPE"]:
        if col in df_test.columns:
            recs[col] = df_test[col].values

    recs["rank"] = recs.groupby(group_col)["predicted_prob"].rank(
        ascending=False, method="first"
    ).astype(int)
    recs = recs[recs["rank"] <= top_n].sort_values(
        [group_col, "rank"]
    )
    return recs


def compute_feature_importance(model, model_name, X_test, y_test,
                               preprocessor, feature_names):
    """Compute feature importance depending on model type."""
    results = []

    # --- Logistic regression coefficients ---
    if "Logistic" in model_name:
        try:
            coefs = model.named_steps["model"].coef_[0]
            # Get transformed feature names
            try:
                transformed_names = preprocessor.get_feature_names_out()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(len(coefs))]
            for name, coef in sorted(zip(transformed_names, coefs),
                                     key=lambda x: abs(x[1]), reverse=True)[:50]:
                results.append({
                    "feature": name,
                    "importance": float(coef),
                    "method": "coefficient",
                })
        except Exception as e:
            print(f"[WARN] Could not extract LR coefficients: {e}")

    # --- Permutation importance for any model ---
    print(
        f"[INFO] Computing permutation importance for {model_name} (may take a moment)...")
    try:
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42,
            scoring="roc_auc", n_jobs=-1
        )
        try:
            transformed_names = preprocessor.get_feature_names_out()
        except Exception:
            transformed_names = [f"feature_{i}" for i in range(
                len(perm.importances_mean))]
        for name, imp_mean, imp_std in sorted(
            zip(transformed_names, perm.importances_mean, perm.importances_std),
            key=lambda x: abs(x[1]), reverse=True
        )[:50]:
            results.append({
                "feature": name,
                "importance": float(imp_mean),
                "importance_std": float(imp_std),
                "method": "permutation",
            })
    except Exception as e:
        print(f"[WARN] Permutation importance failed: {e}")

    return pd.DataFrame(results)


def write_model_card(path, args, label_defn, metrics_all, n_train, n_test,
                     n_features, feature_cols, tree_model_name):
    """Write a human-readable model card."""
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  MLT APPLICATION OUTCOME PREDICTION – MODEL CARD\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATA\n")
        f.write(f"   Source          : {args.data_path}\n")
        f.write(f"   Training rows   : {n_train:,}\n")
        f.write(f"   Test rows       : {n_test:,}\n")
        f.write(f"   Features used   : {n_features}\n\n")

        f.write("2. LABEL DEFINITION\n")
        f.write(f"   Label name      : {args.label_name}\n")
        f.write(f"   Description     : {label_defn['description']}\n")
        f.write(f"   Positive values : {label_defn['positive']}\n")
        f.write(f"   Negative values : {label_defn['negative']}\n\n")

        f.write("3. SPLIT STRATEGY\n")
        f.write(f"   Method          : {args.split_strategy}\n")
        f.write(f"   Test size       : {args.test_size}\n")
        f.write(
            f"   Grouped by      : FELLOW_ID (no Fellow appears in both train & test)\n\n")

        f.write("4. MODELS TRAINED\n")
        f.write(f"   a) Logistic Regression (baseline, class_weight=balanced)\n")
        f.write(f"   b) {tree_model_name} (tuned via RandomizedSearchCV)\n")
        f.write(f"   c) Calibrated version of best model (CalibratedClassifierCV)\n\n")

        f.write("5. EVALUATION METRICS (test set)\n")
        for name, m in metrics_all.items():
            if not isinstance(m, dict):
                f.write(f"\n   {name}: {m}\n")
                continue
            f.write(f"\n   --- {name} ---\n")
            for k, v in m.items():
                if k != "confusion_matrix":
                    try:
                        f.write(f"   {k:20s}: {v:.4f}\n")
                    except (TypeError, ValueError):
                        f.write(f"   {k:20s}: {v}\n")

        f.write("\n6. LEAKAGE PROTECTIONS\n")
        f.write(
            "   - APPLICATION_STATUS is ONLY used to derive the label, never as a feature.\n")
        f.write("   - ENROLLMENT_STATUS dropped (post-outcome information).\n")
        f.write("   - Admissions interview scores (SCORE_*) are from MLT admissions,\n")
        f.write("     not from employer interviews, so they are safe to include.\n")
        f.write(
            "   - COACH column excluded from features (assignment could be post-hoc).\n")
        f.write(
            "   - GroupShuffleSplit ensures no Fellow appears in both train and test.\n\n")

        f.write("7. KNOWN LIMITATIONS\n")
        f.write(
            "   - Model trained on historical data; distribution shift across cohorts\n")
        f.write("     may degrade performance on future cohorts.\n")
        f.write(
            "   - High-cardinality features (EMPLOYER, ROLE) are capped; rare values\n")
        f.write("     are grouped as 'OTHER'.\n")
        f.write("   - Demographic columns (Race, Gender, Ethnicity) are NOT used as\n")
        f.write(
            "     predictive features by default; they are only used for subgroup\n")
        f.write("     evaluation to audit fairness.\n")
        f.write(
            "   - SAT Score has ~80% missing values; its contribution is limited.\n")
        f.write(
            "   - GPA values > 4.0 are treated as data-entry errors and set to NaN.\n")
        f.write(
            "   - Admissions scores (SCORE_*) are ~62% missing (only CP 2018-2024).\n")


# ---------------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train ML model to predict MLT application outcomes."
    )
    parser.add_argument("--data_path", type=str,
                        default=os.environ.get("DATA_PATH", ""),
                        help="Path to the input Excel/CSV file.")
    parser.add_argument("--label_name", type=str, default="OFFER_EXTENDED",
                        choices=list(LABEL_DEFINITIONS.keys()),
                        help="Which binary label to predict.")
    parser.add_argument("--split_strategy", type=str, default="group",
                        choices=["group", "time"],
                        help="'group' = GroupShuffleSplit by FELLOW_ID; "
                             "'time' = train on earlier cohorts, test on later.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for the test set.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Top-N applications to surface per Fellow.")
    parser.add_argument("--include_demographics", action="store_true",
                        help="Include Race/Gender/Ethnicity as model features "
                             "(off by default; always used for subgroup eval).")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for output artifacts.")
    parser.add_argument("--n_iter_search", type=int, default=20,
                        help="Number of iterations for RandomizedSearchCV.")
    args = parser.parse_args()

    if not args.data_path:
        print("[ERROR] Please provide --data_path or set the DATA_PATH env var.")
        sys.exit(1)

    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load & prepare ─────────────────────────────────────────────────
    df_raw = load_data(args.data_path)
    df = create_label(df_raw, args.label_name)
    y = df["label"]

    # Build feature matrix
    X, numeric_cols, categorical_cols = build_feature_matrix(
        df, include_demographics=args.include_demographics
    )

    # ── Split ──────────────────────────────────────────────────────────
    group_col = "FELLOW_ID"
    train_idx, test_idx = split_data(
        df, X, y, args.split_strategy, group_col, args.test_size, args.random_seed
    )

    X_train, X_test = X.iloc[train_idx].reset_index(
        drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(
        drop=True), y.iloc[test_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(
        f"[INFO] Train label distribution:\n{y_train.value_counts().to_string()}")
    print(
        f"[INFO] Test  label distribution:\n{y_test.value_counts().to_string()}")

    # ── Preprocessing ──────────────────────────────────────────────────
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # ── Model 1: Logistic Regression (baseline) ───────────────────────
    print("\n" + "─" * 60)
    print("  Training Model 1: Logistic Regression (baseline)")
    print("─" * 60)
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=args.random_seed,
            solver="saga", penalty="l2", C=1.0
        ))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    lr_metrics = evaluate_model(
        y_test, lr_prob, lr_pred, "Logistic Regression (threshold=0.5)")

    # Optimal threshold
    lr_opt_thresh = find_optimal_threshold(y_test, lr_prob)
    lr_pred_opt = (lr_prob >= lr_opt_thresh).astype(int)
    lr_metrics_opt = evaluate_model(
        y_test, lr_prob, lr_pred_opt,
        f"Logistic Regression (threshold={lr_opt_thresh:.3f})"
    )

    # ── Model 2: Tree-based model (tuned) ─────────────────────────────
    print("\n" + "─" * 60)
    print("  Training Model 2: Tree-based (with RandomizedSearchCV)")
    print("─" * 60)
    tree_model, tree_params, tree_name = get_tree_model()
    tree_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", tree_model),
    ])

    # Use grouped CV for hyperparameter search too
    if group_col in df.columns:
        inner_cv = GroupShuffleSplit(
            n_splits=3, test_size=0.2, random_state=args.random_seed)
        cv_groups = df.iloc[train_idx][group_col].reset_index(drop=True)
    else:
        inner_cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=args.random_seed)
        cv_groups = None

    search = RandomizedSearchCV(
        tree_pipe, tree_params,
        n_iter=args.n_iter_search,
        cv=inner_cv,
        scoring="roc_auc",
        random_state=args.random_seed,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=cv_groups)
    best_tree = search.best_estimator_

    print(f"[INFO] Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"[INFO] Best params: {search.best_params_}")

    tree_prob = best_tree.predict_proba(X_test)[:, 1]
    tree_pred = (tree_prob >= 0.5).astype(int)
    tree_metrics = evaluate_model(y_test, tree_prob, tree_pred,
                                  f"{tree_name} (threshold=0.5)")

    tree_opt_thresh = find_optimal_threshold(y_test, tree_prob)
    tree_pred_opt = (tree_prob >= tree_opt_thresh).astype(int)
    tree_metrics_opt = evaluate_model(
        y_test, tree_prob, tree_pred_opt,
        f"{tree_name} (threshold={tree_opt_thresh:.3f})"
    )

    # ── Model 3: Calibrated tree model ────────────────────────────────
    print("\n" + "─" * 60)
    print("  Calibrating the best tree model (CalibratedClassifierCV)")
    print("─" * 60)
    cal_model = CalibratedClassifierCV(best_tree, cv=3, method="isotonic")
    cal_model.fit(X_train, y_train)
    cal_prob = cal_model.predict_proba(X_test)[:, 1]
    cal_pred = (cal_prob >= 0.5).astype(int)
    cal_metrics = evaluate_model(y_test, cal_prob, cal_pred,
                                 f"Calibrated {tree_name} (threshold=0.5)")

    cal_opt_thresh = find_optimal_threshold(y_test, cal_prob)
    cal_pred_opt = (cal_prob >= cal_opt_thresh).astype(int)
    cal_metrics_opt = evaluate_model(
        y_test, cal_prob, cal_pred_opt,
        f"Calibrated {tree_name} (threshold={cal_opt_thresh:.3f})"
    )

    # ── Select best model ──────────────────────────────────────────────
    model_candidates = {
        "logistic_regression": (lr_pipe, lr_prob, lr_metrics),
        tree_name.lower().replace(" ", "_"): (best_tree, tree_prob, tree_metrics),
        f"calibrated_{tree_name.lower().replace(' ', '_')}": (cal_model, cal_prob, cal_metrics),
    }
    best_name = max(model_candidates,
                    key=lambda k: model_candidates[k][2]["roc_auc"])
    best_model_obj, best_prob, best_metrics = model_candidates[best_name]
    print(f"\n[INFO] Best model by ROC-AUC: {best_name} "
          f"(AUC={best_metrics['roc_auc']:.4f})")

    # ── Subgroup evaluation ────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Subgroup / Segmented Evaluation")
    print("─" * 60)
    subgroup_cols = ["MLT_TRACK", "SCHOOL_HBCU", "SCHOOL_BEA_REGION",
                     "PARTNER_ORG", "APPLICATION_TYPE", "PROGRAM_COHORT",
                     # Demographic columns for fairness audit
                     "RACE", "GENDER", "ETHNICITY",
                     "FIRST_GEN", "LOW_INCOME"]
    subgroup_df = subgroup_evaluation(
        y_test, best_prob, df_test, subgroup_cols)
    if len(subgroup_df) > 0:
        print(subgroup_df.to_string(index=False))
    else:
        print("[WARN] No subgroup columns found for evaluation.")

    # ── Top-N recommendations ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Top-{args.top_n} Recommendations per Fellow (coach triage)")
    print("─" * 60)
    recs = top_n_recommendations(
        df_test, best_prob, y_test, group_col, args.top_n)
    print(f"[INFO] Generated {len(recs):,} top-{args.top_n} recommendations "
          f"for {recs[group_col].nunique():,} Fellows.")
    print(recs.head(15).to_string(index=False))

    # ── Feature importance ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Feature Importance")
    print("─" * 60)
    # LR coefficients
    lr_importance = compute_feature_importance(
        lr_pipe, "Logistic Regression", X_test, y_test,
        preprocessor, numeric_cols + categorical_cols
    )
    # Tree permutation importance
    tree_importance = compute_feature_importance(
        best_tree, tree_name, X_test, y_test,
        preprocessor, numeric_cols + categorical_cols
    )
    all_importance = pd.concat(
        [lr_importance, tree_importance], ignore_index=True)
    print("\nTop 15 features (permutation importance):")
    perm_only = all_importance[all_importance["method"] == "permutation"].nlargest(
        15, "importance"
    )
    if len(perm_only) > 0:
        print(perm_only[["feature", "importance",
              "importance_std"]].to_string(index=False))

    # ── Save outputs ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Saving artifacts to {args.output_dir}/")
    print("─" * 60)

    # 1. metrics.json
    all_metrics = {
        "logistic_regression_0.5": lr_metrics,
        f"logistic_regression_{lr_opt_thresh:.3f}": lr_metrics_opt,
        f"{tree_name}_0.5": tree_metrics,
        f"{tree_name}_{tree_opt_thresh:.3f}": tree_metrics_opt,
        f"calibrated_{tree_name}_0.5": cal_metrics,
        f"calibrated_{tree_name}_{cal_opt_thresh:.3f}": cal_metrics_opt,
        "best_model": best_name,
        "best_model_roc_auc": best_metrics["roc_auc"],
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"  ✓ {metrics_path}")

    # 2. subgroup_metrics.csv
    subgroup_path = os.path.join(args.output_dir, "subgroup_metrics.csv")
    subgroup_df.to_csv(subgroup_path, index=False)
    print(f"  ✓ {subgroup_path}")

    # 3. topN_recommendations.csv
    recs_path = os.path.join(args.output_dir, "topN_recommendations.csv")
    recs.to_csv(recs_path, index=False)
    print(f"  ✓ {recs_path}")

    # 4. feature_importance.csv
    fi_path = os.path.join(args.output_dir, "feature_importance.csv")
    all_importance.to_csv(fi_path, index=False)
    print(f"  ✓ {fi_path}")

    # 5. model_card.txt
    card_path = os.path.join(args.output_dir, "model_card.txt")
    write_model_card(card_path, args, LABEL_DEFINITIONS[args.label_name],
                     all_metrics, len(train_idx), len(test_idx),
                     X.shape[1], numeric_cols + categorical_cols, tree_name)
    print(f"  ✓ {card_path}")

    # 6. best_model.joblib
    model_path = os.path.join(args.output_dir, "best_model.joblib")
    joblib.dump(best_model_obj, model_path)
    print(f"  ✓ {model_path}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best model : {best_name}")
    print(f"  ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC     : {best_metrics['pr_auc']:.4f}")
    print(f"  Brier      : {best_metrics['brier_score']:.4f}")
    print(f"  Outputs in : {os.path.abspath(args.output_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
