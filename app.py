"""
MLT Career Prep - Offer Prediction Dashboard
=============================================
Executive and coach-facing interactive dashboard for ML-driven offer
prediction, likelihood triage, subgroup fairness monitoring, and model insight.

Model  : Further-Reduced LASSO Logistic Regression (L1, balanced weights)
Train  : CP 2018-2023
Validate: CP 2024
Score  : CP 2025 (current fellows)
"""

# ────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import re
import io
import os
import warnings

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────
# 2. PAGE CONFIGURATION
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLT Career Prep \u00b7 Offer Prediction Dashboard",
    page_icon="\U0001f3af",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────
# 3. CONSTANTS
# ────────────────────────────────────────────────────────────────────
POSITIVE_STATUSES = [
    "Offered",
    "Offered & Committed",
    "Offered & Declined",
    "Offer Rescinded",
    "My offer has been rescinded.",
]
NEGATIVE_STATUSES = ["Denied", "Pending"]
ALL_OUTCOME_STATUSES = POSITIVE_STATUSES + NEGATIVE_STATUSES

TRAIN_COHORTS = ["CP 2018", "CP 2020", "CP 2021", "CP 2022", "CP 2023"]
VALIDATION_COHORT = "CP 2024"
THRESHOLD = 0.50

# Approximate Fortune 500 / major-employer lookup for the Is_Fortune500 feature
FORTUNE_500 = {
    "Amazon", "Target", "Google", "Visa Inc.", "Dell Technologies Inc.",
    "Citi", "AT&T", "Morgan Stanley", "JPMorgan Chase", "Goldman Sachs",
    "Bank of America", "Wells Fargo", "Microsoft", "Apple", "Meta",
    "Meta Platforms", "IBM", "Intel", "Oracle", "Cisco",
    "Johnson & Johnson", "Procter & Gamble", "PepsiCo", "Coca-Cola",
    "Nike", "Walt Disney", "Netflix", "Salesforce", "Adobe", "PayPal",
    "American Express", "Capital One", "T-Mobile", "Verizon",
    "Home Depot", "Walmart", "Costco", "General Electric",
    "Honeywell", "Lockheed Martin", "Boeing", "General Motors", "Ford",
    "ExxonMobil", "Chevron", "Pfizer", "Merck", "Eli Lilly",
    "Accenture", "Uber", "Mastercard", "Starbucks Coffee Company",
    "LinkedIn", "Deloitte", "BlackRock", "Charles Schwab",
}

RISK_COLORS = {"Red": "#DC2626", "Yellow": "#F59E0B", "Green": "#059669"}
RISK_LABELS = {"Red": "Likely Denied", "Yellow": "Moderate", "Green": "Likely Offered"}

OUTPUT_COLS = [
    "Program Enrollment: Enrollment ID",
    "Program Enrollment: Coach",
    "Program Enrollment: Program Track",
    "Related Organization",
    "Title",
    "Primary Functional Interest",
    "Predicted_Probability",
    "Risk_Flag",
    "Predicted_Label",
    "Predicted_Outcome",
    "Actual_Label",
    "Actual_Outcome",
    "Correct",
    "Suggested_Coach_Action",
    "Coach_Notes",
    "Likely_Role_Alignment",
]

# ────────────────────────────────────────────────────────────────────
# 4. CSS STYLING
# ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Overall layout ──────────────────────────────────────── */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
section[data-testid="stSidebar"] {
    background: #F8F9FB;
    border-right: 1px solid #E5E7EB;
}

/* ── Dashboard header ────────────────────────────────────── */
.dash-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #2C5F8A 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
}
.dash-header h1 {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}
.dash-header p {
    font-size: 0.82rem;
    opacity: 0.85;
    margin: 0.3rem 0 0;
    font-weight: 400;
}

/* ── KPI metric cards ────────────────────────────────────── */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    text-align: center;
    border-top: 3px solid #2C5F8A;
    min-height: 100px;
}
.kpi-card.accent-green { border-top-color: #059669; }
.kpi-card.accent-red   { border-top-color: #DC2626; }
.kpi-card.accent-amber  { border-top-color: #F59E0B; }
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #1A1A2E;
    margin: 0.25rem 0 0.15rem;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.7rem;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}

/* ── Section card ────────────────────────────────────────── */
.section-card {
    background: white;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
    border: 1px solid #F0F0F3;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1B2A4A;
    margin-bottom: 0.6rem;
}
.section-caption {
    font-size: 0.78rem;
    color: #6B7280;
    margin-top: -0.3rem;
    margin-bottom: 0.8rem;
}

/* ── Likelihood badges ──────────────────────────────────── */
.risk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.risk-badge.red   { background: #FEE2E2; color: #DC2626; }
.risk-badge.yellow { background: #FEF3C7; color: #D97706; }
.risk-badge.green  { background: #D1FAE5; color: #059669; }

/* ── Legend ───────────────────────────────────────────────── */
.legend-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 0.8rem;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: #374151;
}
.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Tab label sizing ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] button {
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
}

/* ── Smaller default dataframe font ──────────────────────── */
.stDataFrame { font-size: 0.82rem; }

/* ── Info icon tooltip ───────────────────────────────────── */
.info-wrap {
    position: relative;
    display: inline-block;
    cursor: pointer;
    vertical-align: middle;
    margin-left: 4px;
}
.info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #E5E7EB;
    color: #6B7280;
    font-size: 11px;
    font-weight: 700;
    font-style: normal;
    line-height: 1;
    transition: background 0.15s, color 0.15s;
}
.info-wrap:hover .info-icon {
    background: #2C5F8A;
    color: #fff;
}
.info-tip {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    z-index: 9999;
    bottom: calc(100% + 10px);
    left: 50%;
    transform: translateX(-50%);
    width: 260px;
    background: #1B2A4A;
    color: #F3F4F6;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 0.74rem;
    font-weight: 400;
    line-height: 1.45;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    pointer-events: none;
    transition: opacity 0.18s, visibility 0.18s;
    text-align: left;
}
.info-tip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 6px;
    border-style: solid;
    border-color: #1B2A4A transparent transparent transparent;
}
.info-wrap:hover .info-tip {
    visibility: visible;
    opacity: 1;
}

/* ── Sidebar branding ────────────────────────────────────── */
.sidebar-brand {
    font-size: 0.78rem;
    color: #6B7280;
    border-top: 1px solid #E5E7EB;
    padding-top: 1rem;
    margin-top: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────


def safe_col(df, col, default=np.nan):
    """Return column if present, else a Series of *default*."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, name=col)


def assign_risk(prob):
    """Map predicted probability to a likelihood flag."""
    if prob < 0.35:
        return "Red"
    if prob <= 0.60:
        return "Yellow"
    return "Green"


def suggest_action(flag):
    """Return suggested coach action for a given likelihood flag."""
    return {
        "Red": "Immediate intervention: refine strategy, target fit, interview prep, and outreach",
        "Yellow": "Moderate coaching: strengthen positioning, sharpen application materials",
        "Green": "Maintain momentum: prepare for interviews and close opportunities",
    }.get(flag, "")


def role_alignment(row):
    """Heuristic: does the applicant's functional interest align with the role title?"""
    title = str(row.get("Title", "")).lower()
    func = str(row.get("Primary Functional Interest", "")).lower()
    kw_map = {
        "consulting": ["consulting", "consultant", "strategy", "advisory"],
        "software": ["software", "engineer", "developer", "swe", "programming"],
        "marketing": ["marketing", "brand", "digital", "growth", "content"],
        "finance": ["finance", "banking", "analyst", "investment", "wealth", "trading"],
        "product": ["product", "pm", "product manager"],
        "operations": ["operations", "supply chain", "logistics"],
        "human resources": ["hr", "human resources", "talent", "people"],
        "research": ["research", "analytics", "data", "scientist"],
        "sales": ["sales", "business development", "account"],
        "engineering": ["engineering", "mechanical", "electrical"],
    }
    matched_key = None
    for key, terms in kw_map.items():
        if any(t in func for t in terms):
            matched_key = key
            break
    if matched_key is None:
        return "Moderate"
    if any(t in title for t in kw_map[matched_key]):
        return "Strong"
    if title in ("intern", "internship", "") or not title:
        return "Moderate"
    return "Low"


def readable_feature(name):
    """Convert internal feature name to a human-readable label."""
    static = {
        "Undergrad_GPA": "Undergraduate GPA",
        "Pell_Grant_Count": "Pell Grant Count",
        "SAT_Score": "SAT Score",
        "SAT_Available": "SAT Score Reported",
        "Title_Word_Count": "Job Title Word Count",
        "Designated_Low_Income": "Designated Low Income",
        "First_Gen_College": "First-Generation College Student",
        "Is_Partner_Active": "Active MLT Partner Organization",
        "Is_Rising_Junior": "Rising Junior Internship",
        "Is_Fortune500": "Fortune 500 Company",
    }
    if name in static:
        return static[name]
    if name.startswith("Func_"):
        return "Interest: " + name[5:].replace("_", " ").strip()
    if name.startswith("Ind_"):
        return "Industry: " + name[4:].replace("_", " ").strip()
    if name.startswith("Track_"):
        return "Track: " + name[6:].replace("_", " ").strip()
    return name.replace("_", " ")


def compute_fairness(df, group_col, actual_col="Actual_Label", pred_col="Predicted_Label", prob_col="Predicted_Probability"):
    """Compute subgroup fairness metrics. Returns DataFrame or None."""
    if group_col not in df.columns:
        return None
    subset = df.dropna(subset=[group_col, actual_col])
    if len(subset) == 0:
        return None
    rows = []
    for grp, gdf in subset.groupby(group_col):
        n = len(gdf)
        if n < 5:
            continue
        y_true = gdf[actual_col].astype(int).values
        y_pred = gdf[pred_col].astype(int).values
        y_prob = gdf[prob_col].values
        if len(np.unique(y_true)) < 2:
            # Cannot compute meaningful metrics with single class
            rows.append({
                "Subgroup": grp, "Count": n,
                "Actual Offer Rate": round(y_true.mean(), 3),
                "Avg Predicted Prob": round(y_prob.mean(), 3),
                "Precision": None, "Recall": None,
                "FPR": None, "FNR": None,
            })
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "Subgroup": grp,
            "Count": n,
            "Actual Offer Rate": round(y_true.mean(), 3),
            "Avg Predicted Prob": round(y_prob.mean(), 3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 3),
            "FNR": round(fn / (fn + tp) if (fn + tp) > 0 else 0, 3),
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)


def info_icon(tip):
    """Return an inline HTML info-icon with hover tooltip."""
    return (
        f'<span class="info-wrap">'
        f'<span class="info-icon">i</span>'
        f'<span class="info-tip">{tip}</span>'
        f'</span>'
    )


def kpi_html(label, value, accent="", tooltip=""):
    """Return HTML for a single KPI card with optional info tooltip."""
    cls = f"kpi-card {accent}" if accent else "kpi-card"
    tip = info_icon(tooltip) if tooltip else ""
    return (
        f'<div class="{cls}">'
        f'<div class="kpi-label">{label} {tip}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'</div>'
    )


def plotly_clean(fig, height=380):
    """Apply a clean, executive-ready layout to a Plotly figure."""
    fig.update_layout(
        font=dict(family="Inter, -apple-system, system-ui, sans-serif", color="#374151", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=50, b=50),
        height=height,
        xaxis=dict(gridcolor="#F3F4F6", zerolinecolor="#E5E7EB"),
        yaxis=dict(gridcolor="#F3F4F6", zerolinecolor="#E5E7EB"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    return fig


# ────────────────────────────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────


def get_feature_config(train_df):
    """Derive encoding lists and medians from training data."""
    top_func = train_df["Primary Functional Interest"].dropna().value_counts().head(8).index.tolist()
    top_ind = train_df["Primary Industry Interest"].dropna().value_counts().head(8).index.tolist()
    medians = {
        "GPA": train_df["Undergrad GPA"].median(),
        "SAT": train_df["SAT Score"].median(),
    }
    return {"top_func": top_func, "top_ind": top_ind, "medians": medians}


def build_features(df, config):
    """Build the model feature matrix from raw data + config."""
    med = config["medians"]
    feat = pd.DataFrame(index=df.index)

    # Numeric
    feat["Undergrad_GPA"] = df["Undergrad GPA"].fillna(med["GPA"])
    feat["Pell_Grant_Count"] = df["Pell Grant Count"].fillna(0).astype(float)
    feat["SAT_Score"] = df["SAT Score"].fillna(med["SAT"])
    feat["SAT_Available"] = df["SAT Score"].notna().astype(int)
    feat["Title_Word_Count"] = (
        df["Title"].fillna("").apply(lambda x: len(str(x).split()))
    )

    # Binary
    feat["Designated_Low_Income"] = safe_col(df, "Designated Low Income", False).astype(int)
    feat["First_Gen_College"] = (safe_col(df, "First Generation College", "No") == "Yes").astype(int)
    feat["Is_Partner_Active"] = (safe_col(df, "Partner Org?", "") == "Partner - Active").astype(int)
    feat["Is_Rising_Junior"] = (safe_col(df, "Type", "") == "Internship (Rising Junior)").astype(int)
    feat["Is_Fortune500"] = (
        df["Related Organization"]
        .fillna("")
        .apply(lambda x: 1 if str(x).strip() in FORTUNE_500 else 0)
    )

    # Program track dummies
    tracks = [
        "Corporate Management",
        "Software Engineering/Technology",
        "Finance",
        "Consulting",
    ]
    for t in tracks:
        col_name = "Track_" + re.sub(r"[^a-zA-Z0-9]", "_", t)
        feat[col_name] = (safe_col(df, "Program Enrollment: Program Track", "") == t).astype(int)

    # Top functional interests
    for f in config["top_func"]:
        col_name = "Func_" + re.sub(r"[^a-zA-Z0-9]", "_", f)[:30]
        feat[col_name] = (df["Primary Functional Interest"].fillna("") == f).astype(int)

    # Top industry interests
    for ind in config["top_ind"]:
        col_name = "Ind_" + re.sub(r"[^a-zA-Z0-9]", "_", ind)[:30]
        feat[col_name] = (df["Primary Industry Interest"].fillna("") == ind).astype(int)

    return feat


# ────────────────────────────────────────────────────────────────────
# 7. PIPELINE  (cached: runs once per session)
# ────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def run_pipeline():
    """Load data -> engineer features -> train LASSO -> predict -> return results dict."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "train_set.xlsx")
    test_path = os.path.join(script_dir, "test_set.xlsx")

    for p, label in [(train_path, "train_set.xlsx"), (test_path, "test_set.xlsx")]:
        if not os.path.exists(p):
            return {"error": f"File not found: {label}. Place it in {script_dir}"}

    # ── Load ──────────────────────────────────────────────────
    train_full = pd.read_excel(train_path)
    test_raw = pd.read_excel(test_path)

    # ── Warn about missing expected columns ───────────────────
    expected_cols = [
        "Application Status", "Undergrad GPA", "SAT Score",
        "Pell Grant Count", "Designated Low Income",
        "First Generation College", "Primary Functional Interest",
        "Primary Industry Interest", "Title", "Type",
        "Related Organization", "Partner Org?",
        "Program Enrollment: Program Track",
        "Program Enrollment: Program",
        "Program Enrollment: Enrollment ID",
        "Program Enrollment: Coach",
    ]
    missing_cols = [c for c in expected_cols if c not in train_full.columns]

    # ── Filter training cohorts ───────────────────────────────
    train = train_full[train_full["Program Enrollment: Program"].isin(TRAIN_COHORTS)].copy()
    train = train[train["Application Status"].isin(ALL_OUTCOME_STATUSES)].copy()
    train["Offered"] = train["Application Status"].isin(POSITIVE_STATUSES).astype(int)

    # ── Validation cohort (CP 2024) ───────────────────────────
    val_full = train_full[train_full["Program Enrollment: Program"] == VALIDATION_COHORT].copy()
    val = val_full[val_full["Application Status"].isin(ALL_OUTCOME_STATUSES)].copy()
    val["Offered"] = val["Application Status"].isin(POSITIVE_STATUSES).astype(int)

    # ── Scoring cohort (CP 2025 - test set) ───────────────────
    score_all = test_raw.copy()
    score_all["Offered"] = np.where(
        score_all["Application Status"].isin(POSITIVE_STATUSES), 1,
        np.where(score_all["Application Status"].isin(NEGATIVE_STATUSES), 0, np.nan),
    )

    # ── Feature engineering ───────────────────────────────────
    config = get_feature_config(train)
    X_train = build_features(train, config)
    X_val = build_features(val, config)
    X_score = build_features(score_all, config)

    y_train = train["Offered"].values
    y_val = val["Offered"].values
    feature_names = X_train.columns.tolist()

    # ── Scaling ───────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train.values)
    X_val_sc = scaler.transform(X_val.values)
    X_score_sc = scaler.transform(X_score.values)

    # ── Train LASSO (L1 logistic, balanced, 5-fold CV) ────────
    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        cv=5,
        Cs=np.logspace(-3, 1, 15),
        max_iter=10000,
        random_state=42,
        scoring="roc_auc",
    )
    model.fit(X_train_sc, y_train)
    best_C = float(model.C_[0])
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])

    # ── Coefficient summary ───────────────────────────────────
    coef_df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        .assign(Abs=lambda d: d["Coefficient"].abs())
        .query("Coefficient != 0")
        .sort_values("Abs", ascending=False)
        .drop(columns="Abs")
        .reset_index(drop=True)
    )

    # ── Validation predictions & metrics ──────────────────────
    val_probs = model.predict_proba(X_val_sc)[:, 1]
    val_preds = (val_probs >= THRESHOLD).astype(int)
    val_metrics = {
        "Precision": round(precision_score(y_val, val_preds, zero_division=0), 4),
        "Recall": round(recall_score(y_val, val_preds, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_val, val_probs), 4),
        "F1": round(f1_score(y_val, val_preds, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_val, val_preds), 4),
        "Total": int(len(y_val)),
        "Predicted_Offered": int(val_preds.sum()),
        "Predicted_Denied": int((val_preds == 0).sum()),
        "Avg_Prob": round(float(val_probs.mean()), 4),
    }

    # ── Scoring predictions ───────────────────────────────────
    score_probs = model.predict_proba(X_score_sc)[:, 1]
    score_preds = (score_probs >= THRESHOLD).astype(int)

    # ── Build output DataFrames ───────────────────────────────
    def build_output(source_df, probs, preds, y_true_col="Offered"):
        out = source_df.copy()
        out["Predicted_Probability"] = np.round(probs, 4)
        out["Risk_Flag"] = out["Predicted_Probability"].apply(assign_risk)
        out["Predicted_Label"] = preds
        out["Predicted_Outcome"] = np.where(preds == 1, "Offered", "Denied")
        has_actual = out[y_true_col].notna()
        out["Actual_Label"] = out[y_true_col]
        out["Actual_Outcome"] = np.where(
            out[y_true_col] == 1, "Offered",
            np.where(out[y_true_col] == 0, "Denied", "Unknown"),
        )
        out["Correct"] = np.where(
            has_actual,
            (preds == out[y_true_col].fillna(-1).astype(int)).astype(int),
            np.nan,
        )
        out["Suggested_Coach_Action"] = out["Risk_Flag"].apply(suggest_action)
        out["Coach_Notes"] = ""
        out["Likely_Role_Alignment"] = out.apply(role_alignment, axis=1)
        return out

    val_out = build_output(val, val_probs, val_preds)
    score_out = build_output(score_all, score_probs, score_preds)

    # ── Scoring-cohort metrics (where ground truth exists) ────
    score_eval = score_out[score_out["Offered"].notna()].copy()
    score_metrics = {}
    if len(score_eval) > 0 and score_eval["Offered"].nunique() > 1:
        y_st = score_eval["Actual_Label"].astype(int).values
        y_sp = score_eval["Predicted_Label"].values
        y_sr = score_eval["Predicted_Probability"].values
        score_metrics = {
            "Precision": round(precision_score(y_st, y_sp, zero_division=0), 4),
            "Recall": round(recall_score(y_st, y_sp, zero_division=0), 4),
            "ROC_AUC": round(roc_auc_score(y_st, y_sr), 4),
            "F1": round(f1_score(y_st, y_sp, zero_division=0), 4),
            "Accuracy": round(accuracy_score(y_st, y_sp), 4),
            "Total_Eval": int(len(y_st)),
        }

    return {
        "val_out": val_out,
        "score_out": score_out,
        "val_metrics": val_metrics,
        "score_metrics": score_metrics,
        "coef_df": coef_df,
        "feature_names": feature_names,
        "coefs_full": coefs,
        "intercept": intercept,
        "best_C": best_C,
        "scaler": scaler,
        "train_n": len(train),
        "val_n": len(val),
        "score_n": len(score_all),
        "config": config,
        "X_score": X_score,
        "X_val": X_val,
        "missing_cols": missing_cols,
    }


# ────────────────────────────────────────────────────────────────────
# 8. MAIN APPLICATION
# ────────────────────────────────────────────────────────────────────

# Header
st.markdown(
    '<div class="dash-header">'
    "<h1>MLT Career Prep Offer Prediction Dashboard</h1>"
    "<p>Executive and Coach View for Offer Likelihood, Support Flags, and Fairness Monitoring</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Run pipeline
with st.spinner("Loading data and training model \u2026 this may take a moment on first run."):
    results = run_pipeline()

if isinstance(results, dict) and "error" in results:
    st.error(results["error"])
    st.stop()

# Unpack results
val_out = results["val_out"]
score_out = results["score_out"]
val_metrics = results["val_metrics"]
score_metrics = results["score_metrics"]
coef_df = results["coef_df"]
feature_names = results["feature_names"]
coefs_full = results["coefs_full"]
intercept = results["intercept"]
best_C = results["best_C"]
scaler = results["scaler"]
X_score = results["X_score"]
X_val = results["X_val"]

# Missing column warnings
if results["missing_cols"]:
    st.warning(f"Some expected columns were not found in the data: {', '.join(results['missing_cols'])}")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Dashboard Controls")
    cohort = st.radio(
        "Active Cohort",
        ["CP 2025 \u2013 Current Fellows", "CP 2024 \u2013 Validation"],
        index=0,
        help="Select which cohort to display in the dashboard tabs.",
    )
    is_val_view = cohort.startswith("CP 2024")
    active_df = val_out if is_val_view else score_out
    active_X = X_val if is_val_view else X_score
    cohort_label = "CP 2024" if is_val_view else "CP 2025"

    st.markdown("---")
    st.markdown("### Data Summary")
    st.markdown(f"**Train cohorts:** {', '.join(TRAIN_COHORTS)}")
    st.markdown(f"**Train rows:** {results['train_n']:,}")
    st.markdown(f"**Validation (CP 2024):** {results['val_n']:,}")
    st.markdown(f"**Scoring (CP 2025):** {results['score_n']:,}")
    st.markdown(f"**Threshold:** {THRESHOLD}")
    st.markdown(f"**Best C (regularization):** {best_C:.4f}")
    st.markdown(f"**Non-zero features:** {len(coef_df)}/{len(feature_names)}")

    st.markdown("---")
    st.markdown("### Downloads")

    # Predictions CSV
    pred_cols = [c for c in OUTPUT_COLS if c in active_df.columns]
    pred_csv = active_df[pred_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Predictions ({cohort_label})",
        pred_csv,
        "further_reduced_lasso_predictions.csv",
        "text/csv",
    )

    # Metrics CSV
    metrics_rows = [{"Cohort": "CP 2024 Validation", **val_metrics}]
    if score_metrics:
        metrics_rows.append({"Cohort": "CP 2025 Scoring", **score_metrics})
    metrics_csv = pd.DataFrame(metrics_rows).to_csv(index=False).encode("utf-8")
    st.download_button("Model Metrics", metrics_csv, "further_reduced_lasso_metrics.csv", "text/csv")

    # Coefficients CSV
    coef_csv = coef_df.to_csv(index=False).encode("utf-8")
    st.download_button("Coefficients", coef_csv, "further_reduced_lasso_coefficients.csv", "text/csv")

    # Fairness CSV (built later, placeholder button)
    st.markdown('<div class="sidebar-brand">MLT Career Prep Analytics<br>Further-Reduced LASSO v1.0</div>', unsafe_allow_html=True)

# ── KPI Strip ────────────────────────────────────────────────

total_scored = len(active_df)
pred_offered = int(active_df["Predicted_Label"].sum())
pred_denied = total_scored - pred_offered
avg_prob = active_df["Predicted_Probability"].mean()

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
with k1:
    st.markdown(kpi_html(
        f"{cohort_label} Scored", f"{total_scored:,}", "",
        "Total number of applications in the active cohort that were scored by the prediction model."
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_html(
        "Pred. Offered", f"{pred_offered:,}", "accent-green",
        "Number of applications the model predicts will receive an offer (predicted probability \u2265 0.50)."
    ), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_html(
        "Pred. Denied", f"{pred_denied:,}", "accent-red",
        "Number of applications the model predicts will NOT receive an offer (predicted probability &lt; 0.50)."
    ), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_html(
        "Avg Pred. Prob", f"{avg_prob:.2%}", "",
        "The average predicted offer probability across all scored applications. Higher means the cohort is generally more competitive."
    ), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_html(
        "Precision (CP24)", f"{val_metrics['Precision']:.2%}", "accent-amber",
        "Of all applications the model predicted as \u2018Offered,\u2019 this is the percentage that actually received offers. Higher precision means fewer false alarms."
    ), unsafe_allow_html=True)
with k6:
    st.markdown(kpi_html(
        "Recall (CP24)", f"{val_metrics['Recall']:.2%}", "accent-amber",
        "Of all applications that actually received offers, this is the percentage the model correctly identified. Higher recall means fewer missed offers."
    ), unsafe_allow_html=True)
with k7:
    st.markdown(kpi_html(
        "ROC-AUC (CP24)", f"{val_metrics['ROC_AUC']:.3f}", "",
        "Area Under the ROC Curve \u2014 measures the model\u2019s overall ability to distinguish between offers and denials. 1.0 is perfect; 0.5 is random chance."
    ), unsafe_allow_html=True)

st.markdown("")  # spacer

# ── Tabs ──────────────────────────────────────────────────────
tab_exec, tab_coach, tab_detail, tab_fair, tab_model = st.tabs([
    "Executive Overview",
    "Coach Action Center",
    "Application Detail",
    "Subgroup Fairness",
    "Model Insights",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 : Executive Overview
# ════════════════════════════════════════════════════════════════
with tab_exec:
    # Legend
    st.markdown(
        '<div class="legend-row">'
        '<div class="legend-item"><span class="legend-dot" style="background:#DC2626"></span> Red = Likely Denied (prob &lt; 0.35)</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#F59E0B"></span> Yellow = Moderate (0.35\u20130.60)</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#059669"></span> Green = Likely Offered (prob &gt; 0.60)</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    # ── Probability distribution ──────────────────────────────
    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Predicted Probability Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Distribution of predicted offer probabilities across all scored applications.</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=active_df["Predicted_Probability"],
            nbinsx=40,
            marker_color="#2C5F8A",
            opacity=0.85,
        ))
        # Threshold line
        fig_hist.add_vline(x=0.35, line_dash="dash", line_color="#DC2626", annotation_text="Red < 0.35", annotation_position="top left")
        fig_hist.add_vline(x=0.60, line_dash="dash", line_color="#059669", annotation_text="Green > 0.60", annotation_position="top right")
        fig_hist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Count")
        plotly_clean(fig_hist, 340)
        st.plotly_chart(fig_hist, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Support band counts ───────────────────────────────────
    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Support Band Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Count of applications by likelihood category.</div>', unsafe_allow_html=True)
        band_counts = active_df["Risk_Flag"].value_counts().reindex(["Red", "Yellow", "Green"], fill_value=0)
        fig_band = go.Figure(go.Bar(
            x=["Likely Denied (Red)", "Moderate (Yellow)", "Likely Offered (Green)"],
            y=[band_counts.get("Red", 0), band_counts.get("Yellow", 0), band_counts.get("Green", 0)],
            marker_color=[RISK_COLORS["Red"], RISK_COLORS["Yellow"], RISK_COLORS["Green"]],
            text=[band_counts.get("Red", 0), band_counts.get("Yellow", 0), band_counts.get("Green", 0)],
            textposition="outside",
        ))
        fig_band.update_layout(yaxis_title="Applications", xaxis_title="")
        plotly_clean(fig_band, 340)
        st.plotly_chart(fig_band, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ── Offer likelihood by program track ─────────────────────
    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Offer Likelihood by Program Track</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Average predicted probability by program track.</div>', unsafe_allow_html=True)
        if "Program Enrollment: Program Track" in active_df.columns:
            track_agg = (
                active_df.groupby("Program Enrollment: Program Track")["Predicted_Probability"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Avg Probability", "count": "Applications", "Program Enrollment: Program Track": "Track"})
                .sort_values("Avg Probability", ascending=True)
            )
            fig_track = px.bar(
                track_agg, x="Avg Probability", y="Track",
                orientation="h", text="Avg Probability",
                color="Avg Probability",
                color_continuous_scale=["#DC2626", "#F59E0B", "#059669"],
                range_color=[0, 1],
            )
            fig_track.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_track.update_layout(coloraxis_showscale=False, xaxis_range=[0, 1])
            plotly_clean(fig_track, 320)
            st.plotly_chart(fig_track, width="stretch")
        else:
            st.info("Program Track column not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Offer likelihood by company (top 15) ──────────────────
    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Offer Likelihood by Company (Top 15)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Average predicted probability for the 15 most-applied-to companies.</div>', unsafe_allow_html=True)
        if "Related Organization" in active_df.columns:
            org_agg = (
                active_df.groupby("Related Organization")["Predicted_Probability"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "Avg Probability", "count": "Applications", "Related Organization": "Company"})
                .nlargest(15, "Applications")
                .sort_values("Avg Probability", ascending=True)
            )
            fig_org = px.bar(
                org_agg, x="Avg Probability", y="Company",
                orientation="h", text="Avg Probability",
                color="Avg Probability",
                color_continuous_scale=["#DC2626", "#F59E0B", "#059669"],
                range_color=[0, 1],
            )
            fig_org.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_org.update_layout(coloraxis_showscale=False, xaxis_range=[0, 1])
            plotly_clean(fig_org, 420)
            st.plotly_chart(fig_org, width="stretch")
        else:
            st.info("Related Organization column not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Lowest-likelihood applications ──────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Lowest-Likelihood Applications Needing Coach Intervention</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Applications with the lowest predicted offer probabilities. Sorted by urgency.</div>', unsafe_allow_html=True)

    risk_table_cols = [
        "Program Enrollment: Enrollment ID",
        "Program Enrollment: Coach",
        "Program Enrollment: Program Track",
        "Related Organization",
        "Title",
        "Primary Functional Interest",
        "Predicted_Probability",
        "Risk_Flag",
        "Predicted_Outcome",
        "Suggested_Coach_Action",
    ]
    risk_table_cols = [c for c in risk_table_cols if c in active_df.columns]
    risk_df = active_df[risk_table_cols].sort_values("Predicted_Probability").head(30)
    st.dataframe(
        risk_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Predicted_Probability": st.column_config.ProgressColumn(
                "Pred. Prob", format="%.3f", min_value=0, max_value=1,
                help="The model\u2019s estimated probability that this application will receive an offer, from 0 (very unlikely) to 1 (very likely).",
            ),
            "Risk_Flag": st.column_config.TextColumn(
                "Likelihood",
                help="Red = probability < 0.35 (likely denied). Yellow = 0.35\u20130.60 (moderate). Green = > 0.60 (likely offered).",
            ),
            "Predicted_Outcome": st.column_config.TextColumn(
                "Pred. Outcome",
                help="The model\u2019s binary prediction: \u2018Offered\u2019 if probability \u2265 0.50, otherwise \u2018Denied.\u2019",
            ),
            "Suggested_Coach_Action": st.column_config.TextColumn(
                "Suggested Action",
                help="Recommended coaching action based on the likelihood flag level.",
            ),
        },
        height=450,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 : Coach Action Center
# ════════════════════════════════════════════════════════════════
with tab_coach:
    st.markdown(
        '<div class="legend-row">'
        '<div class="legend-item"><span class="legend-dot" style="background:#DC2626"></span> Red = Likely Denied</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#F59E0B"></span> Yellow = Moderate</div>'
        '<div class="legend-item"><span class="legend-dot" style="background:#059669"></span> Green = Likely Offered</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Filters ────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        coach_opts = sorted(active_df["Program Enrollment: Coach"].dropna().unique()) if "Program Enrollment: Coach" in active_df.columns else []
        sel_coach = st.multiselect("Coach", coach_opts, default=[], key="coach_filter")
    with f2:
        track_opts = sorted(active_df["Program Enrollment: Program Track"].dropna().unique()) if "Program Enrollment: Program Track" in active_df.columns else []
        sel_track = st.multiselect("Program Track", track_opts, default=[], key="track_filter")
    with f3:
        org_opts = sorted(active_df["Related Organization"].dropna().unique()) if "Related Organization" in active_df.columns else []
        sel_org = st.multiselect("Related Organization", org_opts, default=[], key="org_filter")

    f4, f5, f6 = st.columns(3)
    with f4:
        func_opts = sorted(active_df["Primary Functional Interest"].dropna().unique()) if "Primary Functional Interest" in active_df.columns else []
        sel_func = st.multiselect("Functional Interest", func_opts, default=[], key="func_filter")
    with f5:
        sel_risk = st.multiselect("Likelihood Flag", ["Red", "Yellow", "Green"], default=[], key="risk_filter")
    with f6:
        sel_pred = st.multiselect("Predicted Outcome", ["Offered", "Denied"], default=[], key="pred_filter")

    # Apply filters
    filtered = active_df.copy()
    if sel_coach:
        filtered = filtered[filtered["Program Enrollment: Coach"].isin(sel_coach)]
    if sel_track:
        filtered = filtered[filtered["Program Enrollment: Program Track"].isin(sel_track)]
    if sel_org:
        filtered = filtered[filtered["Related Organization"].isin(sel_org)]
    if sel_func:
        filtered = filtered[filtered["Primary Functional Interest"].isin(sel_func)]
    if sel_risk:
        filtered = filtered[filtered["Risk_Flag"].isin(sel_risk)]
    if sel_pred:
        filtered = filtered[filtered["Predicted_Outcome"].isin(sel_pred)]

    # Sort selector
    sort_col, sort_dir = st.columns([3, 1])
    with sort_col:
        sort_opt = st.radio(
            "Sort by probability",
            ["Lowest first (likely denied)", "Highest first"],
            horizontal=True,
            key="sort_prob",
        )
    ascending = sort_opt.startswith("Lowest")
    filtered = filtered.sort_values("Predicted_Probability", ascending=ascending)

    st.markdown(f"**{len(filtered):,}** applications shown")

    # ── Main table ─────────────────────────────────────────
    display_cols = [
        "Program Enrollment: Enrollment ID",
        "Program Enrollment: Coach",
        "Program Enrollment: Program Track",
        "Related Organization",
        "Title",
        "Primary Functional Interest",
        "Predicted_Probability",
        "Risk_Flag",
        "Predicted_Outcome",
        "Actual_Outcome",
        "Correct",
        "Suggested_Coach_Action",
        "Coach_Notes",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].reset_index(drop=True)

    edited = st.data_editor(
        display_df,
        width="stretch",
        hide_index=True,
        height=560,
        disabled=[c for c in display_cols if c != "Coach_Notes"],
        column_config={
            "Predicted_Probability": st.column_config.ProgressColumn(
                "Pred. Prob", format="%.3f", min_value=0, max_value=1,
                help="The model\u2019s estimated probability that this application receives an offer (0\u20131).",
            ),
            "Coach_Notes": st.column_config.TextColumn(
                "Coach Notes",
                help="Editable field for coach observations. Notes persist only within the current session.",
                width="medium",
            ),
            "Risk_Flag": st.column_config.TextColumn(
                "Likelihood",
                help="Red (< 0.35): likely denied. Yellow (0.35\u20130.60): moderate. Green (> 0.60): likely offered.",
            ),
            "Correct": st.column_config.NumberColumn(
                "Correct", format="%d",
                help="1 = model prediction matched the actual outcome. 0 = mismatch. Blank = actual outcome not yet known.",
            ),
            "Predicted_Outcome": st.column_config.TextColumn(
                "Pred. Outcome",
                help="\u2018Offered\u2019 if predicted probability \u2265 0.50, otherwise \u2018Denied.\u2019",
            ),
            "Actual_Outcome": st.column_config.TextColumn(
                "Actual Outcome",
                help="The real outcome for this application, if known. \u2018Unknown\u2019 means the result has not been determined yet.",
            ),
        },
        key="coach_table",
    )

    st.caption("Coach Notes are editable above. Note: notes persist only within the current session.")

    # ── Company Likelihood Explorer ─────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Company Likelihood Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">'
        "Select an applicant to estimate their predicted offer likelihood at different companies. "
        "The model adjusts Fortune&nbsp;500 and MLT&nbsp;Partner status based on the target company."
        "</div>",
        unsafe_allow_html=True,
    )

    if len(filtered) > 0:
        # Applicant selector
        cle_labels = (
            filtered["Program Enrollment: Enrollment ID"].astype(str)
            + " | "
            + filtered["Related Organization"].fillna("N/A").astype(str)
            + " | "
            + filtered["Predicted_Probability"].apply(lambda x: f"{x:.1%}")
        )
        cle_sel = st.selectbox(
            "Select Applicant",
            range(len(filtered)),
            format_func=lambda i: cle_labels.iloc[i],
            key="cle_app",
        )

        sel_row = filtered.iloc[cle_sel]
        sel_feat = active_X.loc[filtered.index[cle_sel]].values.copy().astype(float)

        # Build per-company profile from the full active dataset
        company_profiles = {}
        for org in active_df["Related Organization"].dropna().unique():
            is_f500 = 1 if str(org).strip() in FORTUNE_500 else 0
            org_rows = active_df[active_df["Related Organization"] == org]
            is_partner = 0
            if "Partner Org?" in org_rows.columns:
                is_partner = int((org_rows["Partner Org?"] == "Partner - Active").any())
            company_profiles[org] = {"Fortune500": is_f500, "Partner": is_partner}

        all_companies = sorted(company_profiles.keys())

        # Default to top 15 companies by application volume
        top_companies = (
            active_df["Related Organization"]
            .value_counts()
            .head(15)
            .index.tolist()
        )
        sel_companies = st.multiselect(
            "Select companies to compare (default: top 15 by volume)",
            all_companies,
            default=top_companies,
            key="cle_companies",
        )

        if sel_companies:
            f500_idx = feature_names.index("Is_Fortune500") if "Is_Fortune500" in feature_names else None
            partner_idx = feature_names.index("Is_Partner_Active") if "Is_Partner_Active" in feature_names else None

            cle_rows = []
            for company in sel_companies:
                prof = company_profiles[company]
                feat = sel_feat.copy()
                if f500_idx is not None:
                    feat[f500_idx] = prof["Fortune500"]
                if partner_idx is not None:
                    feat[partner_idx] = prof["Partner"]
                scaled = scaler.transform(feat.reshape(1, -1))[0]
                logit = intercept + float(np.dot(coefs_full, scaled))
                prob = 1.0 / (1.0 + np.exp(-logit))
                cle_rows.append({
                    "Company": company,
                    "Predicted_Probability": round(float(prob), 4),
                    "Likelihood": assign_risk(prob),
                    "Fortune 500": "Yes" if prof["Fortune500"] else "No",
                    "MLT Partner": "Yes" if prof["Partner"] else "No",
                })

            compare_df = (
                pd.DataFrame(cle_rows)
                .sort_values("Predicted_Probability", ascending=False)
                .reset_index(drop=True)
            )

            st.dataframe(
                compare_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Predicted_Probability": st.column_config.ProgressColumn(
                        "Pred. Prob", format="%.3f", min_value=0, max_value=1,
                        help="Estimated probability of receiving an offer at this company, based on the applicant\u2019s profile.",
                    ),
                    "Likelihood": st.column_config.TextColumn(
                        "Likelihood",
                        help="Red (< 0.35): likely denied. Yellow (0.35\u20130.60): moderate. Green (> 0.60): likely offered.",
                    ),
                    "Fortune 500": st.column_config.TextColumn(
                        help="Whether this company is on the approximate Fortune 500 list used by the model.",
                    ),
                    "MLT Partner": st.column_config.TextColumn(
                        help="Whether this company appears as an active MLT Partner Organization in the data.",
                    ),
                },
                height=min(500, 40 * len(compare_df) + 40),
            )

            # Horizontal bar chart
            chart_cle = compare_df.sort_values("Predicted_Probability", ascending=True)
            fig_cle = go.Figure(go.Bar(
                x=chart_cle["Predicted_Probability"],
                y=chart_cle["Company"],
                orientation="h",
                marker_color=[
                    RISK_COLORS[assign_risk(p)] for p in chart_cle["Predicted_Probability"]
                ],
                text=chart_cle["Predicted_Probability"].apply(lambda x: f"{x:.1%}"),
                textposition="outside",
            ))
            fig_cle.update_layout(
                xaxis_title="Predicted Probability",
                yaxis_title="",
                xaxis_range=[0, 1.05],
            )
            plotly_clean(fig_cle, max(300, len(chart_cle) * 28 + 80))
            st.plotly_chart(fig_cle, width="stretch")

            # Highlight current application
            current_company = sel_row.get("Related Organization", "N/A")
            current_prob = sel_row["Predicted_Probability"]
            st.info(
                f"**Current application:** {current_company} \u2014 {current_prob:.1%} "
                f"({RISK_LABELS[assign_risk(current_prob)]})"
            )
        else:
            st.info("Select one or more companies above to see predicted likelihoods.")
    else:
        st.info("No applications match the current filters.")

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 : Application Detail
# ════════════════════════════════════════════════════════════════
with tab_detail:
    # ── Narrow down selection ──────────────────────────────
    det_c1, det_c2 = st.columns(2)
    with det_c1:
        det_coach = st.selectbox(
            "Filter by Coach",
            ["All"] + sorted(active_df["Program Enrollment: Coach"].dropna().unique().tolist()),
            key="det_coach",
        )
    with det_c2:
        det_track = st.selectbox(
            "Filter by Track",
            ["All"] + sorted(active_df["Program Enrollment: Program Track"].dropna().unique().tolist()),
            key="det_track",
        )

    subset = active_df.copy()
    if det_coach != "All":
        subset = subset[subset["Program Enrollment: Coach"] == det_coach]
    if det_track != "All":
        subset = subset[subset["Program Enrollment: Program Track"] == det_track]

    if len(subset) == 0:
        st.info("No applications match the selected filters.")
    else:
        # Build labels for selector
        subset = subset.sort_values("Predicted_Probability", ascending=True).reset_index(drop=True)
        label_col = (
            subset["Program Enrollment: Enrollment ID"].astype(str)
            + " | "
            + subset["Related Organization"].fillna("N/A").astype(str)
            + " | "
            + subset["Title"].fillna("N/A").astype(str)
        )
        sel_idx = st.selectbox("Select Application", range(len(subset)), format_func=lambda i: label_col.iloc[i], key="app_sel")
        app = subset.iloc[sel_idx]

        # ── Profile card ──────────────────────────────────
        p1, p2 = st.columns([3, 2])
        with p1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Application Profile</div>', unsafe_allow_html=True)
            profile_fields = {
                "Enrollment ID": app.get("Program Enrollment: Enrollment ID", "N/A"),
                "Coach": app.get("Program Enrollment: Coach", "N/A"),
                "Program Track": app.get("Program Enrollment: Program Track", "N/A"),
                "Company": app.get("Related Organization", "N/A"),
                "Title": app.get("Title", "N/A"),
                "Functional Interest": app.get("Primary Functional Interest", "N/A"),
                "Industry Interest": app.get("Primary Industry Interest", "N/A"),
                "GPA": f"{app.get('Undergrad GPA', 'N/A')}",
                "First Gen": app.get("First Generation College", "N/A"),
                "Low Income": app.get("Designated Low Income", "N/A"),
                "Application Status": app.get("Application Status", "N/A"),
                "Role Alignment": app.get("Likely_Role_Alignment", "N/A"),
            }
            for k, v in profile_fields.items():
                st.markdown(f"**{k}:** {v}")
            st.markdown("</div>", unsafe_allow_html=True)

        with p2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
            prob_val = app["Predicted_Probability"]
            flag = app["Risk_Flag"]
            badge_cls = flag.lower()
            st.markdown(
                f'<div style="text-align:center;margin:1rem 0">'
                f'<div style="font-size:2.8rem;font-weight:700;color:{RISK_COLORS[flag]}">{prob_val:.1%}</div>'
                f'<div class="risk-badge {badge_cls}" style="margin-top:0.3rem">{RISK_LABELS[flag]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Predicted Outcome:** {app['Predicted_Outcome']}")
            st.markdown(f"**Actual Outcome:** {app['Actual_Outcome']}")
            if not np.isnan(app["Correct"]) if isinstance(app["Correct"], float) else True:
                correct_str = "Yes" if app["Correct"] == 1 else "No" if app["Correct"] == 0 else "N/A"
                st.markdown(f"**Prediction Correct:** {correct_str}")
            st.markdown(f"**Suggested Action:** {app['Suggested_Coach_Action']}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Feature contributions ─────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Contributing Model Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Approximate feature contributions based on model coefficients and scaled feature values.</div>', unsafe_allow_html=True)

        # Get the scaled feature values for this application
        raw_vals = active_X.iloc[subset.index[sel_idx]].values.reshape(1, -1)
        scaled_vals = scaler.transform(raw_vals)[0]
        contributions = coefs_full * scaled_vals

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Readable": [readable_feature(f) for f in feature_names],
            "Raw_Value": active_X.iloc[subset.index[sel_idx]].values,
            "Coefficient": coefs_full,
            "Contribution": contributions,
            "Abs_Contribution": np.abs(contributions),
        })
        contrib_df = contrib_df[contrib_df["Abs_Contribution"] > 0.005].sort_values("Abs_Contribution", ascending=False)

        if len(contrib_df) > 0:
            top_pos = contrib_df[contrib_df["Contribution"] > 0].head(5)
            top_neg = contrib_df[contrib_df["Contribution"] < 0].head(5)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**Factors Increasing Offer Likelihood**")
                if len(top_pos) > 0:
                    for _, r in top_pos.iterrows():
                        val_str = f"{r['Raw_Value']:.2f}" if abs(r["Raw_Value"]) > 1.5 else ("Yes" if r["Raw_Value"] == 1 else "No" if r["Raw_Value"] == 0 else f"{r['Raw_Value']:.2f}")
                        st.markdown(f"- **{r['Readable']}** ({val_str}) &mdash; contribution: +{r['Contribution']:.3f}")
                else:
                    st.markdown("_No strong positive contributors identified._")
            with pc2:
                st.markdown("**Factors Decreasing Offer Likelihood**")
                if len(top_neg) > 0:
                    for _, r in top_neg.iterrows():
                        val_str = f"{r['Raw_Value']:.2f}" if abs(r["Raw_Value"]) > 1.5 else ("Yes" if r["Raw_Value"] == 1 else "No" if r["Raw_Value"] == 0 else f"{r['Raw_Value']:.2f}")
                        st.markdown(f"- **{r['Readable']}** ({val_str}) &mdash; contribution: {r['Contribution']:.3f}")
                else:
                    st.markdown("_No strong negative contributors identified._")

            # Bar chart of contributions
            show_top = pd.concat([top_pos, top_neg]).sort_values("Contribution")
            fig_contrib = go.Figure(go.Bar(
                x=show_top["Contribution"],
                y=show_top["Readable"],
                orientation="h",
                marker_color=[RISK_COLORS["Green"] if c > 0 else RISK_COLORS["Red"] for c in show_top["Contribution"]],
            ))
            fig_contrib.update_layout(
                xaxis_title="Contribution to Prediction",
                yaxis_title="",
                title="Feature Contribution Breakdown",
            )
            plotly_clean(fig_contrib, max(280, len(show_top) * 35 + 80))
            st.plotly_chart(fig_contrib, width="stretch")

            # Plain-language summary
            pos_names = ", ".join(top_pos["Readable"].head(3).tolist()) if len(top_pos) > 0 else "none identified"
            neg_names = ", ".join(top_neg["Readable"].head(3).tolist()) if len(top_neg) > 0 else "none identified"
            st.info(
                f"**Interpretation:** For this application, the factors most increasing offer likelihood are "
                f"**{pos_names}**. The factors most decreasing likelihood are **{neg_names}**."
            )
        else:
            st.info("No significant feature contributions identified for this application.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Coach Notes for this application ──────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Coach Notes</div>', unsafe_allow_html=True)
        note_key = f"note_{app.get('Program Enrollment: Enrollment ID', sel_idx)}_{sel_idx}"
        coach_note = st.text_area(
            "Enter coaching notes for this application",
            value="",
            height=100,
            key=note_key,
            label_visibility="collapsed",
        )
        st.caption("Notes are session-only. For persistent notes, use the downloaded CSV.")
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 : Subgroup Fairness
# ════════════════════════════════════════════════════════════════
with tab_fair:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Subgroup Fairness Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">'
        "Fairness diagnostics across demographic and program dimensions. "
        "Metrics are computed only on applications with known outcomes."
        "</div>",
        unsafe_allow_html=True,
    )

    # Determine which DataFrame to use for fairness (need known outcomes)
    fair_df = active_df[active_df["Actual_Label"].notna()].copy()
    if len(fair_df) == 0:
        st.warning("No applications with known outcomes in the active cohort. Cannot compute fairness metrics.")
    else:
        fairness_dims = [
            ("Race", "Race"),
            ("Ethnicity", "Ethnicity"),
            ("Gender", "Gender"),
            ("First Generation College", "First Generation College"),
            ("Designated Low Income", "Designated Low Income"),
            ("Program Track", "Program Enrollment: Program Track"),
        ]

        # Overall metrics for comparison
        y_t = fair_df["Actual_Label"].astype(int).values
        y_p = fair_df["Predicted_Label"].astype(int).values
        overall_recall = recall_score(y_t, y_p, zero_division=0)
        overall_fnr = 1 - overall_recall
        overall_prec = precision_score(y_t, y_p, zero_division=0)

        selected_dim = st.selectbox(
            "Select fairness dimension",
            [label for label, _ in fairness_dims],
            key="fair_dim",
        )
        dim_col = dict(fairness_dims)[selected_dim]

        fair_result = compute_fairness(fair_df, dim_col)

        if fair_result is None:
            st.info(f"Column '{dim_col}' not available or insufficient data for fairness analysis.")
        else:
            st.markdown(
                f'**Overall recall:** {overall_recall:.3f} '
                f'{info_icon("Recall: Of all actual offers, the share the model correctly predicted. Higher is better.")}'
                f' &nbsp;|&nbsp; **Overall FNR:** {overall_fnr:.3f} '
                f'{info_icon("False Negative Rate: The share of actual offers the model incorrectly predicted as denied. Lower is better.")}'
                f' &nbsp;|&nbsp; **Overall precision:** {overall_prec:.3f} '
                f'{info_icon("Precision: Of all predicted offers, the share that actually received offers. Higher means fewer false alarms.")}',
                unsafe_allow_html=True,
            )

            # Fairness table
            st.dataframe(
                fair_result,
                width="stretch",
                hide_index=True,
                column_config={
                    "Actual Offer Rate": st.column_config.NumberColumn(
                        format="%.3f",
                        help="The actual proportion of applications in this subgroup that received an offer.",
                    ),
                    "Avg Predicted Prob": st.column_config.NumberColumn(
                        format="%.3f",
                        help="The average predicted offer probability for this subgroup from the model.",
                    ),
                    "Precision": st.column_config.NumberColumn(
                        format="%.3f",
                        help="Of all applications predicted as 'Offered' in this subgroup, the share that actually received offers.",
                    ),
                    "Recall": st.column_config.NumberColumn(
                        format="%.3f",
                        help="Of all actual offers in this subgroup, the share the model correctly predicted. Also called True Positive Rate.",
                    ),
                    "FPR": st.column_config.NumberColumn(
                        "False Pos Rate", format="%.3f",
                        help="False Positive Rate: Of all actual denials in this subgroup, the share the model incorrectly predicted as offers.",
                    ),
                    "FNR": st.column_config.NumberColumn(
                        "False Neg Rate", format="%.3f",
                        help="False Negative Rate: Of all actual offers in this subgroup, the share the model incorrectly predicted as denied.",
                    ),
                },
                height=min(400, 45 * len(fair_result) + 40),
            )

            # ── Fairness bar chart ────────────────────────────
            if "Recall" in fair_result.columns and fair_result["Recall"].notna().any():
                fig_fair = go.Figure()
                valid = fair_result.dropna(subset=["Recall", "FNR"])
                fig_fair.add_trace(go.Bar(
                    x=valid["Subgroup"], y=valid["Recall"],
                    name="Recall", marker_color="#2C5F8A",
                ))
                fig_fair.add_trace(go.Bar(
                    x=valid["Subgroup"], y=valid["FNR"],
                    name="False Negative Rate", marker_color="#DC2626",
                ))
                fig_fair.add_hline(
                    y=overall_recall, line_dash="dash", line_color="#6B7280",
                    annotation_text=f"Overall Recall ({overall_recall:.2f})",
                )
                fig_fair.update_layout(
                    barmode="group",
                    xaxis_title="",
                    yaxis_title="Rate",
                    yaxis_range=[0, 1],
                    title=f"Recall & False Negative Rate by {selected_dim}",
                )
                plotly_clean(fig_fair, 400)
                st.plotly_chart(fig_fair, width="stretch")

            # ── Warnings ───────────────────────────────────────
            flagged = []
            for _, row in fair_result.iterrows():
                if row.get("Recall") is not None and abs(row["Recall"] - overall_recall) > 0.10:
                    flagged.append(f"**{row['Subgroup']}** recall ({row['Recall']:.2f}) differs from overall ({overall_recall:.2f}) by more than 0.10")
                if row.get("FNR") is not None and abs(row["FNR"] - overall_fnr) > 0.10:
                    flagged.append(f"**{row['Subgroup']}** FNR ({row['FNR']:.2f}) differs from overall ({overall_fnr:.2f}) by more than 0.10")

            if flagged:
                st.warning("**Fairness Flags:**\n\n" + "\n\n".join(f"- {f}" for f in flagged))
            else:
                st.success("No subgroup metric differs from the overall population by more than 0.10 on recall or FNR.")

            # ── Download fairness CSV ──────────────────────────
            fair_csv = fair_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Download Fairness Summary ({selected_dim})",
                fair_csv,
                "subgroup_fairness_summary.csv",
                "text/csv",
                key="fair_dl",
            )

    st.markdown("---")
    st.caption(
        "**Disclaimer:** These fairness checks are diagnostic and should be interpreted cautiously, "
        "especially for small subgroup sample sizes. They do not constitute a full fairness audit."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 5 : Model Insights
# ════════════════════════════════════════════════════════════════
with tab_model:
    # ── Model setup summary ───────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)
    mi1, mi2, mi3 = st.columns(3)
    with mi1:
        st.markdown(
            f'**Algorithm:** L1 (LASSO) Logistic Regression {info_icon("LASSO uses L1 regularization to automatically shrink weak feature weights to zero, producing a sparse, interpretable model.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Class Weights:** Balanced {info_icon("Balanced class weights up-weight the minority class (offers) so the model does not simply predict the majority class (denials) for everything.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Cross-Validation:** 5-fold, scoring = ROC-AUC {info_icon("The training data is split into 5 folds. The model trains on 4 and validates on 1, rotating across all folds. The best regularization strength is chosen by the highest average ROC-AUC.")}',
            unsafe_allow_html=True,
        )
    with mi2:
        st.markdown(
            f'**Best Regularization (C):** {best_C:.5f} {info_icon("C controls how much the model is penalized for complexity. Smaller C = stronger regularization = fewer features kept. This value was selected by cross-validation.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Decision Threshold:** {THRESHOLD} {info_icon("If the predicted probability is at or above this threshold, the application is classified as Offered; below it, Denied.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Intercept:** {intercept:.4f} {info_icon("The model\u2019s baseline log-odds before any feature contributions. A negative intercept means the default prediction leans toward denial.")}',
            unsafe_allow_html=True,
        )
    with mi3:
        st.markdown(f"**Train Cohorts:** {', '.join(TRAIN_COHORTS)}")
        st.markdown(f"**Validation Cohort:** {VALIDATION_COHORT}")
        st.markdown(
            f'**Total Features:** {len(feature_names)} {info_icon("The total number of engineered features fed into the model, including numeric, binary, and one-hot encoded categorical variables.")}',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'**Non-Zero (Selected):** {len(coef_df)} {info_icon("After LASSO regularization, only features with non-zero coefficients remain. These are the features the model actually uses to make predictions.")}',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # ── Features increasing likelihood ────────────────���───
    with col_l:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Features Increasing Offer Likelihood</div>', unsafe_allow_html=True)
        pos_coefs = coef_df[coef_df["Coefficient"] > 0].copy()
        pos_coefs["Readable"] = pos_coefs["Feature"].apply(readable_feature)
        if len(pos_coefs) > 0:
            for _, r in pos_coefs.iterrows():
                st.markdown(f"- **{r['Readable']}**: +{r['Coefficient']:.4f}")
        else:
            st.markdown("_No positive coefficients._")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Features decreasing likelihood ────────────────────
    with col_r:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Features Decreasing Offer Likelihood</div>', unsafe_allow_html=True)
        neg_coefs = coef_df[coef_df["Coefficient"] < 0].copy()
        neg_coefs["Readable"] = neg_coefs["Feature"].apply(readable_feature)
        if len(neg_coefs) > 0:
            for _, r in neg_coefs.iterrows():
                st.markdown(f"- **{r['Readable']}**: {r['Coefficient']:.4f}")
        else:
            st.markdown("_No negative coefficients._")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Coefficient bar chart ─────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Non-Zero LASSO Coefficients</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Horizontal bar chart of all selected features, sorted by absolute magnitude.</div>', unsafe_allow_html=True)

    if len(coef_df) > 0:
        chart_df = coef_df.copy()
        chart_df["Readable"] = chart_df["Feature"].apply(readable_feature)
        chart_df = chart_df.sort_values("Coefficient", ascending=True)
        fig_coef = go.Figure(go.Bar(
            x=chart_df["Coefficient"],
            y=chart_df["Readable"],
            orientation="h",
            marker_color=[RISK_COLORS["Green"] if c > 0 else RISK_COLORS["Red"] for c in chart_df["Coefficient"]],
        ))
        fig_coef.update_layout(
            xaxis_title="Coefficient Value",
            yaxis_title="",
            title="",
        )
        plotly_clean(fig_coef, max(350, len(chart_df) * 28 + 80))
        st.plotly_chart(fig_coef, width="stretch")
    else:
        st.info("All coefficients are zero (extreme regularization). Try a wider C range.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Full coefficient table ────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Coefficient Table</div>', unsafe_allow_html=True)
    coef_display = coef_df.copy()
    coef_display["Readable"] = coef_display["Feature"].apply(readable_feature)
    coef_display = coef_display[["Readable", "Feature", "Coefficient"]].rename(
        columns={"Readable": "Feature (Readable)", "Feature": "Internal Name"}
    )
    st.dataframe(coef_display, width="stretch", hide_index=True, height=min(400, 40 * len(coef_display) + 40))
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Validation performance summary ────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Validation Performance (CP 2024)</div>', unsafe_allow_html=True)
    vm1, vm2, vm3, vm4, vm5 = st.columns(5)
    with vm1:
        st.metric("Precision", f"{val_metrics['Precision']:.3f}", help="Of all predicted offers, the share that were actual offers. Higher precision = fewer false positives.")
    with vm2:
        st.metric("Recall", f"{val_metrics['Recall']:.3f}", help="Of all actual offers, the share the model correctly identified. Higher recall = fewer missed offers.")
    with vm3:
        st.metric("ROC-AUC", f"{val_metrics['ROC_AUC']:.3f}", help="Area Under the ROC Curve. Measures the model\u2019s ability to separate offers from denials across all thresholds. 1.0 = perfect, 0.5 = random.")
    with vm4:
        st.metric("F1 Score", f"{val_metrics['F1']:.3f}", help="The harmonic mean of Precision and Recall. Balances both metrics into a single score. Higher is better.")
    with vm5:
        st.metric("Accuracy", f"{val_metrics['Accuracy']:.3f}", help="The share of all predictions (both offers and denials) that were correct. Can be misleading with imbalanced classes.")

    if score_metrics:
        st.markdown("---")
        st.markdown('<div class="section-title">Scoring Performance (CP 2025, known outcomes only)</div>', unsafe_allow_html=True)
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        with sm1:
            st.metric("Precision", f"{score_metrics.get('Precision', 0):.3f}", help="Of all predicted offers, the share that were actual offers. Higher precision = fewer false positives.")
        with sm2:
            st.metric("Recall", f"{score_metrics.get('Recall', 0):.3f}", help="Of all actual offers, the share the model correctly identified. Higher recall = fewer missed offers.")
        with sm3:
            st.metric("ROC-AUC", f"{score_metrics.get('ROC_AUC', 0):.3f}", help="Area Under the ROC Curve. Measures the model\u2019s ability to separate offers from denials. 1.0 = perfect, 0.5 = random.")
        with sm4:
            st.metric("F1 Score", f"{score_metrics.get('F1', 0):.3f}", help="The harmonic mean of Precision and Recall. Balances both into a single score. Higher is better.")
        with sm5:
            st.metric("Accuracy", f"{score_metrics.get('Accuracy', 0):.3f}", help="The share of all predictions that were correct. Can be misleading with imbalanced classes.")

    st.markdown("</div>", unsafe_allow_html=True)
