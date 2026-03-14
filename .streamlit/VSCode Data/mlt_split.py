#!/usr/bin/env python3
"""
MLT Career Prep – Grouped + Stratified Train/Test Split
========================================================
Reads the MLT application-level Excel file and produces an ML-ready
train/test split that:
  (a) Prevents leakage by keeping ALL rows for a Fellow in only one split
  (b) Approximately stratifies by outcome rate and program track

WHY GROUP-AWARE STRATIFICATION?
  Each Fellow has multiple application rows.  A naive random row-level split
  would let the same Fellow appear in both train and test, causing data leakage:
  the model would "learn" Fellow-specific patterns during training and then be
  tested on the same Fellow, inflating metrics.  By splitting at the *Fellow*
  level, we guarantee no Fellow overlap.  We then stratify at the Fellow level
  (by outcome and track) so that the train/test distributions are comparable.

Usage:
  python mlt_split.py --data_path <path> [OPTIONS]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------------
# 1.  AUTO-DETECTION HELPERS
# ---------------------------------------------------------------------------

def auto_detect_column(df, patterns, label):
    """Case-insensitive pattern matching to find a column."""
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for col in df.columns:
            if regex.search(col):
                print(f"  [AUTO-DETECT] {label} → '{col}'")
                return col
    return None


def detect_columns(df, args):
    """Detect or validate the ID, label, track, and date columns."""
    log_lines = []

    # --- Fellow ID ---
    if args.id_col and args.id_col in df.columns:
        id_col = args.id_col
        log_lines.append(f"ID column (user-specified): {id_col}")
    else:
        id_col = auto_detect_column(df, [
            r"enrollment.?id", r"fellow.?id", r"candidate.?id", r"person.?id",
            r"applicant.?id", r"student.?id",
        ], "ID column")
        if id_col is None:
            print("[ERROR] Could not auto-detect a Fellow/person ID column.")
            print("        Available columns:", list(df.columns))
            sys.exit(1)
        log_lines.append(f"ID column (auto-detected): {id_col}")

    # --- Label / outcome ---
    if args.label_col and args.label_col in df.columns:
        label_col = args.label_col
        log_lines.append(f"Label column (user-specified): {label_col}")
    else:
        label_col = auto_detect_column(df, [
            r"application.?status", r"offer.?extended", r"accepted",
            r"outcome", r"status", r"result",
        ], "Label column")
        if label_col is None:
            print("[ERROR] Could not auto-detect a label/outcome column.")
            sys.exit(1)
        log_lines.append(f"Label column (auto-detected): {label_col}")

    # --- Track ---
    if args.track_col and args.track_col in df.columns:
        track_col = args.track_col
        log_lines.append(f"Track column (user-specified): {track_col}")
    else:
        track_col = auto_detect_column(df, [
            r"program.?track", r"mlt.?track", r"career.?track", r"^track$",
        ], "Track column")
        if track_col:
            log_lines.append(f"Track column (auto-detected): {track_col}")
        else:
            log_lines.append("Track column: NOT FOUND (will stratify on outcome only)")

    # --- Date ---
    if args.date_col and args.date_col in df.columns:
        date_col = args.date_col
        log_lines.append(f"Date column (user-specified): {date_col}")
    else:
        date_col = auto_detect_column(df, [
            r"application.?date", r"submission.?time", r"created.?date",
            r"apply.?date", r"date",
        ], "Date column")
        if date_col:
            log_lines.append(f"Date column (auto-detected): {date_col}")
        else:
            log_lines.append("Date column: NOT FOUND")

    return id_col, label_col, track_col, date_col, log_lines


# ---------------------------------------------------------------------------
# 2.  LABEL CREATION
# ---------------------------------------------------------------------------

# Positive statuses (any form of offer)
POSITIVE_STATUSES = {
    "Offered & Committed", "Offered & Declined", "Offered",
    "Offer Rescinded", "My offer has been rescinded."
}
# Negative statuses (clear non-offers)
NEGATIVE_STATUSES = {
    "Applied", "Denied", "Withdrew Application",
}
# Ambiguous (excluded from the binary label)
AMBIGUOUS_STATUSES = {
    "Pending", "Invited", "Applied to MLT", "Did Not Apply",
}


def _find_cohort_column(df):
    """Auto-detect the program/cohort column (e.g., 'Program Enrollment: Program')."""
    patterns = [r"program.enrollment.*program$", r"program.?cohort",
                r"cohort", r"^program$"]
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for col in df.columns:
            if regex.search(col):
                return col
    return None


def _reclassify_applied_as_denied(df, label_col):
    """Reclassify 'Applied' status as 'Denied' for cohort CP 2025 and prior.

    Rationale: for older cohorts, enough time has elapsed that an application
    still sitting at 'Applied' (no employer response) is effectively a
    rejection.  Cohorts CP 2026+ may still have pending outcomes, so they
    are left as-is.
    """
    cohort_col = _find_cohort_column(df)
    if cohort_col is None:
        print("[WARN] No cohort column found; skipping Applied→Denied reclassification.")
        return df

    # Extract the numeric year from strings like "CP 2025"
    cohort_year = df[cohort_col].astype(str).str.extract(r"(\d{4})", expand=False).astype(float)

    reclassify_mask = (
        (df[label_col] == "Applied") &
        (cohort_year <= 2025)
    )
    n_reclassified = reclassify_mask.sum()

    if n_reclassified > 0:
        df = df.copy()
        df.loc[reclassify_mask, label_col] = "Denied"
        print(f"[INFO] Reclassified {n_reclassified:,} 'Applied' rows from "
              f"CP 2025 and prior → 'Denied' (no response = de facto rejection).")
    else:
        print("[INFO] No 'Applied' rows from CP 2025 or prior to reclassify.")

    return df


def create_binary_label(df, label_col):
    """Convert Application Status into a binary OFFER_EXTENDED label.

    If the label column is already binary (0/1), return as-is.
    Otherwise, map known status strings to 0/1 and drop ambiguous rows.
    """
    # Reclassify stale "Applied" as "Denied" for CP 2025 and earlier
    df = _reclassify_applied_as_denied(df, label_col)

    vals = df[label_col].dropna().unique()

    # Check if already binary
    if set(vals).issubset({0, 1, 0.0, 1.0, True, False, "0", "1"}):
        df["LABEL"] = df[label_col].astype(int)
        n_dropped = 0
    else:
        all_known = POSITIVE_STATUSES | NEGATIVE_STATUSES
        mask = df[label_col].isin(all_known)
        n_dropped = (~mask).sum()
        df = df[mask].copy()
        df["LABEL"] = df[label_col].isin(POSITIVE_STATUSES).astype(int)

    n_pos = df["LABEL"].sum()
    n_neg = len(df) - n_pos
    print(f"[INFO] Binary label created: Positive={n_pos:,}  Negative={n_neg:,}  "
          f"Rate={n_pos/len(df):.3f}")
    if n_dropped > 0:
        print(f"       Dropped {n_dropped:,} rows with ambiguous status.")
    return df


# ---------------------------------------------------------------------------
# 3.  FELLOW-LEVEL AGGREGATION FOR STRATIFICATION
# ---------------------------------------------------------------------------

def build_fellow_strata(df, id_col, track_col, date_col):
    """Aggregate to Fellow-level and build a stratification label.

    Strata = (modal_track, has_any_positive_outcome).
    This lets us stratify the Fellow-level split so that both train and test
    have similar track mixes AND similar proportions of Fellows who received
    at least one offer.
    """
    agg = {"LABEL": "max"}  # max → 1 if Fellow has ANY positive outcome

    # Track: pick mode (most common). If ties, pick alphabetically first.
    if track_col and track_col in df.columns:
        def modal_track(s):
            mode = s.mode()
            if len(mode) == 0:
                return "MISSING"
            return sorted(mode)[0]  # deterministic tie-break
        agg[track_col] = modal_track

    fellow_df = df.groupby(id_col).agg(agg).reset_index()
    fellow_df.rename(columns={"LABEL": "OUTCOME_ANY_POSITIVE"}, inplace=True)

    # Build stratum string
    if track_col and track_col in fellow_df.columns:
        fellow_df["TRACK_CLEAN"] = fellow_df[track_col].fillna("MISSING")
        fellow_df["STRATUM"] = (
            fellow_df["TRACK_CLEAN"] + "__" +
            fellow_df["OUTCOME_ANY_POSITIVE"].astype(str)
        )
    else:
        fellow_df["STRATUM"] = fellow_df["OUTCOME_ANY_POSITIVE"].astype(str)

    # Collapse tiny strata into "OTHER__<outcome>" so that
    # StratifiedShuffleSplit has enough members per class (need >= 2).
    # We iteratively merge until all strata have at least MIN_STRATUM_SIZE.
    MIN_STRATUM_SIZE = 10
    n_collapsed = 0
    while True:
        stratum_counts = fellow_df["STRATUM"].value_counts()
        tiny = stratum_counts[stratum_counts < MIN_STRATUM_SIZE].index
        # Don't try to merge the "OTHER__*" strata further — just merge into
        # the outcome-only stratum.
        tiny = [s for s in tiny if not s.startswith("OTHER__")]
        if len(tiny) == 0:
            break
        mask = fellow_df["STRATUM"].isin(tiny)
        fellow_df.loc[mask, "STRATUM"] = (
            "OTHER__" + fellow_df.loc[mask, "OUTCOME_ANY_POSITIVE"].astype(str)
        )
        n_collapsed += len(tiny)

    # Final check: if any OTHER__* stratum is still < 2, merge it into the
    # largest stratum with the same outcome to guarantee splittability.
    stratum_counts = fellow_df["STRATUM"].value_counts()
    for stratum in stratum_counts.index:
        if stratum_counts[stratum] < 2:
            outcome_val = fellow_df.loc[fellow_df["STRATUM"] == stratum,
                                         "OUTCOME_ANY_POSITIVE"].iloc[0]
            # Find the largest stratum with the same outcome
            same_outcome = fellow_df[
                fellow_df["OUTCOME_ANY_POSITIVE"] == outcome_val
            ]["STRATUM"].value_counts()
            candidates = same_outcome[same_outcome.index != stratum]
            if len(candidates) > 0:
                target = candidates.idxmax()
                fellow_df.loc[fellow_df["STRATUM"] == stratum, "STRATUM"] = target
                n_collapsed += 1

    if n_collapsed > 0:
        print(f"[INFO] Collapsed {n_collapsed} tiny strata (n<{MIN_STRATUM_SIZE}) "
              f"into larger groups.")

    print(f"[INFO] Fellow-level strata ({fellow_df['STRATUM'].nunique()} unique):")
    for s, cnt in fellow_df["STRATUM"].value_counts().items():
        print(f"       {s:45s}  n={cnt:,}")

    return fellow_df


# ---------------------------------------------------------------------------
# 4.  SPLIT STRATEGIES
# ---------------------------------------------------------------------------

def split_group_stratified(df, fellow_df, id_col, test_size, seed):
    """Stratified split at Fellow level, then expand to row level."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X_dummy = np.zeros(len(fellow_df))
    y_strata = fellow_df["STRATUM"]

    train_idx, test_idx = next(sss.split(X_dummy, y_strata))
    train_fellows = set(fellow_df.iloc[train_idx][id_col])
    test_fellows = set(fellow_df.iloc[test_idx][id_col])

    train_mask = df[id_col].isin(train_fellows)
    test_mask = df[id_col].isin(test_fellows)

    return df[train_mask].copy(), df[test_mask].copy(), train_fellows, test_fellows


def split_time_group(df, fellow_df, id_col, date_col, test_size, seed):
    """Time-based split: train on earlier applications, test on later.

    The time cutoff is chosen so that ~test_size of rows fall after it.
    Fellows who span both windows are assigned entirely to the window
    containing their EARLIEST application to prevent leakage.
    """
    if date_col is None or date_col not in df.columns:
        print("[WARN] No date column found; falling back to group_stratified.")
        return split_group_stratified(df, fellow_df, id_col, test_size, seed)

    # Parse dates
    df["_date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    valid_dates = df["_date_parsed"].dropna()
    if len(valid_dates) == 0:
        print("[WARN] Date column has no parseable dates; falling back to group_stratified.")
        df.drop(columns=["_date_parsed"], inplace=True)
        return split_group_stratified(df, fellow_df, id_col, test_size, seed)

    cutoff = valid_dates.quantile(1 - test_size)
    print(f"[INFO] Time cutoff: {cutoff} (approx {test_size*100:.0f}% of rows after)")

    # Assign Fellows by their earliest application date
    fellow_earliest = df.groupby(id_col)["_date_parsed"].min().reset_index()
    fellow_earliest.columns = [id_col, "earliest_date"]

    train_fellows_set = set(fellow_earliest[fellow_earliest["earliest_date"] <= cutoff][id_col])
    test_fellows_set = set(fellow_earliest[fellow_earliest["earliest_date"] > cutoff][id_col])

    # Fellows with earliest date exactly at cutoff go to train
    # Count how many Fellows span both windows but were moved
    spans_both = df.groupby(id_col)["_date_parsed"].agg(["min", "max"])
    spans_both = spans_both[(spans_both["min"] <= cutoff) & (spans_both["max"] > cutoff)]
    n_moved = len(spans_both)
    print(f"[INFO] {n_moved} Fellows span both time windows; "
          f"assigned to TRAIN (by earliest-date rule).")

    train_mask = df[id_col].isin(train_fellows_set)
    test_mask = df[id_col].isin(test_fellows_set)

    df.drop(columns=["_date_parsed"], inplace=True)

    return (df[train_mask].copy(), df[test_mask].copy(),
            train_fellows_set, test_fellows_set)


# ---------------------------------------------------------------------------
# 5.  DIAGNOSTICS
# ---------------------------------------------------------------------------

def compute_diagnostics(train_df, test_df, id_col, track_col):
    """Compute and print distribution diagnostics."""
    rows = []

    # --- Row counts ---
    rows.append({
        "metric": "Total rows",
        "train": len(train_df),
        "test": len(test_df),
        "overall": len(train_df) + len(test_df),
    })

    # --- Fellow counts ---
    train_fellows = set(train_df[id_col])
    test_fellows = set(test_df[id_col])
    overlap = train_fellows & test_fellows
    rows.append({
        "metric": "Unique Fellows",
        "train": len(train_fellows),
        "test": len(test_fellows),
        "overall": len(train_fellows | test_fellows),
    })
    rows.append({
        "metric": "Fellow overlap (MUST BE 0)",
        "train": len(overlap),
        "test": len(overlap),
        "overall": len(overlap),
    })

    # --- Row-level outcome rate ---
    train_rate = train_df["LABEL"].mean()
    test_rate = test_df["LABEL"].mean()
    all_rate = pd.concat([train_df, test_df])["LABEL"].mean()
    rows.append({
        "metric": "Outcome rate (row-level)",
        "train": f"{train_rate:.4f}",
        "test": f"{test_rate:.4f}",
        "overall": f"{all_rate:.4f}",
    })

    # --- Fellow-level outcome rate ---
    train_fellow_rate = train_df.groupby(id_col)["LABEL"].max().mean()
    test_fellow_rate = test_df.groupby(id_col)["LABEL"].max().mean()
    rows.append({
        "metric": "Outcome rate (Fellow-level)",
        "train": f"{train_fellow_rate:.4f}",
        "test": f"{test_fellow_rate:.4f}",
        "overall": "—",
    })

    # --- Track distribution ---
    if track_col and track_col in train_df.columns:
        train_tracks = train_df[track_col].fillna("MISSING").value_counts(normalize=True)
        test_tracks = test_df[track_col].fillna("MISSING").value_counts(normalize=True)
        for track in sorted(set(train_tracks.index) | set(test_tracks.index)):
            rows.append({
                "metric": f"Track: {track}",
                "train": f"{train_tracks.get(track, 0):.4f}",
                "test": f"{test_tracks.get(track, 0):.4f}",
                "overall": "—",
            })

    diag_df = pd.DataFrame(rows)
    return diag_df, len(overlap)


# ---------------------------------------------------------------------------
# 6.  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Produce a grouped + stratified train/test split for MLT data."
    )
    parser.add_argument("--data_path", type=str,
                        default=os.environ.get("DATA_PATH", ""),
                        help="Path to the input Excel or CSV file.")
    parser.add_argument("--sheet_name", type=str, default=None,
                        help="Sheet name to read (default: first sheet).")
    parser.add_argument("--id_col", type=str, default=None,
                        help="Fellow ID column (auto-detected if not given).")
    parser.add_argument("--label_col", type=str, default=None,
                        help="Label/outcome column (auto-detected if not given).")
    parser.add_argument("--track_col", type=str, default=None,
                        help="Track column (auto-detected if not given).")
    parser.add_argument("--date_col", type=str, default=None,
                        help="Date column (auto-detected if not given).")
    parser.add_argument("--split_strategy", type=str, default="group_stratified",
                        choices=["group_stratified", "time_group"],
                        help="Split strategy.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction for test set (default 0.2).")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory for output files.")
    args = parser.parse_args()

    if not args.data_path:
        print("[ERROR] Please provide --data_path or set DATA_PATH env var.")
        sys.exit(1)

    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    path = Path(args.data_path)
    print(f"[INFO] Loading {path} ...")
    sheet = args.sheet_name if args.sheet_name else 0
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # ── Detect columns ─────────────────────────────────────────────────
    print("\n[INFO] Detecting columns ...")
    id_col, label_col, track_col, date_col, log_lines = detect_columns(df, args)
    print()

    # ── Create binary label ────────────────────────────────────────────
    df = create_binary_label(df, label_col)

    # ── Build Fellow-level strata ──────────────────────────────────────
    print()
    fellow_df = build_fellow_strata(df, id_col, track_col, date_col)

    # ── Split ──────────────────────────────────────────────────────────
    print(f"\n[INFO] Split strategy: {args.split_strategy}")
    if args.split_strategy == "group_stratified":
        train_df, test_df, train_fellows, test_fellows = split_group_stratified(
            df, fellow_df, id_col, args.test_size, args.random_seed
        )
    elif args.split_strategy == "time_group":
        train_df, test_df, train_fellows, test_fellows = split_time_group(
            df, fellow_df, id_col, date_col, args.test_size, args.random_seed
        )
    else:
        raise ValueError(f"Unknown split strategy: {args.split_strategy}")

    # ── Diagnostics ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SPLIT DIAGNOSTICS")
    print("=" * 65)
    diag_df, n_overlap = compute_diagnostics(train_df, test_df, id_col, track_col)
    print(diag_df.to_string(index=False))
    print()
    if n_overlap > 0:
        print(f"[CRITICAL] Fellow overlap detected ({n_overlap} Fellows)! "
              "The split has leakage.")
    else:
        print("[OK] Zero Fellow overlap confirmed — no leakage.")

    # ── Save outputs ───────────────────────────────────────────────────
    print(f"\n[INFO] Saving outputs to {args.output_dir}/ ...")

    # 1. train.csv
    train_path = os.path.join(args.output_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"  train.csv         : {len(train_df):,} rows")

    # 2. test.csv
    test_path = os.path.join(args.output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"  test.csv          : {len(test_df):,} rows")

    # 3. split_diagnostics.csv
    diag_path = os.path.join(args.output_dir, "split_diagnostics.csv")
    diag_df.to_csv(diag_path, index=False)
    print(f"  split_diagnostics.csv")

    # 4. fellow_split_map.csv
    map_rows = (
        [(fid, "train") for fid in sorted(train_fellows)] +
        [(fid, "test") for fid in sorted(test_fellows)]
    )
    map_df = pd.DataFrame(map_rows, columns=[id_col, "split"])
    map_path = os.path.join(args.output_dir, "fellow_split_map.csv")
    map_df.to_csv(map_path, index=False)
    print(f"  fellow_split_map.csv : {len(map_df):,} Fellows")

    # 5. split_audit_log.txt
    audit_path = os.path.join(args.output_dir, "split_audit_log.txt")
    with open(audit_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("  MLT SPLIT AUDIT LOG\n")
        f.write("=" * 65 + "\n\n")

        f.write("1. INPUT\n")
        f.write(f"   File           : {args.data_path}\n")
        f.write(f"   Total rows     : {len(df):,}\n")
        f.write(f"   Total columns  : {len(df.columns)}\n\n")

        f.write("2. COLUMN DETECTION\n")
        for line in log_lines:
            f.write(f"   {line}\n")
        f.write(f"   ID column used : {id_col}\n")
        f.write(f"   Label column   : {label_col}\n")
        f.write(f"   Track column   : {track_col or 'NOT FOUND'}\n")
        f.write(f"   Date column    : {date_col or 'NOT FOUND'}\n\n")

        f.write("3. LABEL DEFINITION\n")
        f.write(f"   Positive statuses: {sorted(POSITIVE_STATUSES)}\n")
        f.write(f"   Negative statuses: {sorted(NEGATIVE_STATUSES)}\n")
        f.write(f"   Ambiguous (excluded): {sorted(AMBIGUOUS_STATUSES)}\n\n")

        f.write("4. SPLIT STRATEGY\n")
        f.write(f"   Method          : {args.split_strategy}\n")
        f.write(f"   Test size       : {args.test_size}\n")
        f.write(f"   Random seed     : {args.random_seed}\n")
        f.write(f"   Fellow overlap  : {n_overlap} (must be 0)\n\n")

        f.write("5. STRATIFICATION\n")
        f.write("   Strata are formed by combining (track, has_any_positive_outcome)\n")
        f.write("   at the Fellow level.  Fellows are split, then mapped back to rows.\n")
        f.write(f"   Unique strata   : {fellow_df['STRATUM'].nunique()}\n\n")

        f.write("6. RESULTS\n")
        f.write(diag_df.to_string(index=False))
        f.write("\n\n")

        f.write("7. LEAKAGE PROTECTIONS\n")
        f.write("   - All rows for a Fellow are in the same split (train or test).\n")
        f.write("   - GroupShuffleSplit / Fellow-level assignment prevents leakage.\n")
        f.write("   - For time_group strategy, Fellows spanning both windows are\n")
        f.write("     assigned to the window of their earliest application.\n")

    print(f"  split_audit_log.txt")

    print(f"\n[DONE] All outputs saved to {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
