"""
Data preparation for HCC Early Recurrence Prediction.

Steps:
1. Merge Ourdata clinical1 + clinical2 into a unified table
2. Generate slide tables (patient ID -> .h5 feature file mapping)
3. Filter censored patients for inspection (optional)

Run locally to check table structure; actual execution happens on server.
"""

import pandas as pd
from pathlib import Path

# ============================================================
# Paths (adjust to server paths when running on server)
# ============================================================
DATA_INFO_DIR = Path(__file__).parent.parent / "data_info"

# Server paths (used in output configs)
SERVER_OUTPUT_DIR = Path("/data3/chensx/outputs/tables")
SERVER_OURDATA_FEATURE_DIR = Path("/data3/chensx/outputs/features/ourdata_conch1_5")
SERVER_TCGA_FEATURE_DIR = Path("/data3/chensx/outputs/features/tcga_conch1_5")


# ============================================================
# Step 1: Merge Ourdata clinical tables
# ============================================================
def merge_ourdata_clinical():
    """
    Merge clinical1.csv and clinical2.csv into a unified table.

    clinical1.csv: ID column, extensive pathological features (MVI, AFP, etc.)
    clinical2.csv: 住院号 column, different feature set (CNLC, BCLC, pre/post-surgery labs)

    Both have: Time (months), Early recurrence (0/1)
    """
    df1 = pd.read_csv(DATA_INFO_DIR / "Ourdata_clinical1.csv")
    df2 = pd.read_csv(DATA_INFO_DIR / "Ourdata_clinical2.csv")

    # Standardize patient ID column name
    df1 = df1.rename(columns={"ID": "PATIENT"})
    df2 = df2.rename(columns={"住院 号": "PATIENT"})  # note: space between 院 and 号

    # Keep only columns needed for STAMP survival task
    # (add more columns if you want to use clinical features in multimodal fusion later)
    core_cols_1 = ["PATIENT", "Time", "Early recurrence"]
    core_cols_2 = ["PATIENT", "Time", "Early recurrence"]

    # Optional: add clinical features for future multimodal use
    clinical_cols_1 = [
        "AFP (ng/ml)", "微血管侵犯分级", "肿瘤大小（影像学）(mm)",
        "组织学分级（分化）", "卫星结节", "手术切缘宽度（mm）",
        "门静脉癌栓", "瘤内坏死", "瘤内出血", "性别", "年龄"
    ]
    clinical_cols_2 = [
        "AFP (ng/ml)", "CNLC", "BCLC", "分化", "性别", "年龄"
    ]

    keep_cols_1 = core_cols_1 + [c for c in clinical_cols_1 if c in df1.columns]
    keep_cols_2 = core_cols_2 + [c for c in clinical_cols_2 if c in df2.columns]

    df1_slim = df1[keep_cols_1].copy()
    df2_slim = df2[keep_cols_2].copy()

    # Add source label for traceability
    df1_slim["cohort"] = "cohort1"
    df2_slim["cohort"] = "cohort2"

    # Concatenate (outer join - NaN for columns not in both)
    merged = pd.concat([df1_slim, df2_slim], ignore_index=True, sort=False)

    # Ensure PATIENT is string and unique
    merged["PATIENT"] = merged["PATIENT"].astype(str)

    duplicates = merged.duplicated(subset=["PATIENT"])
    if duplicates.any():
        print(f"WARNING: {duplicates.sum()} duplicate PATIENT IDs found. Keeping first occurrence.")
        merged = merged.drop_duplicates(subset=["PATIENT"], keep="first")

    print(f"Merged clinical table: {len(merged)} patients")
    print(f"  Early recurrence = 1: {(merged['Early recurrence'] == 1).sum()}")
    print(f"  Early recurrence = 0: {(merged['Early recurrence'] == 0).sum()}")
    print(f"  Time range: {merged['Time'].min():.1f} - {merged['Time'].max():.1f} months")

    # Identify censored patients (status=0, follow-up < 24 months)
    censored_short = merged[
        (merged["Early recurrence"] == 0) & (merged["Time"] < 24)
    ]
    print(f"  Censored with follow-up < 24 months: {len(censored_short)} patients")

    # Flag suspicious Time=0 entries
    zero_time = merged[merged["Time"] == 0.0]
    if len(zero_time) > 0:
        print(f"  WARNING: {len(zero_time)} patient(s) with Time=0 (likely data entry error):")
        print(f"    {zero_time[['PATIENT', 'Time', 'Early recurrence', 'cohort']].to_string(index=False)}")
        print("  Consider excluding these patients or correcting the time value.")

    return merged


# ============================================================
# Step 2: Filter TCGA clinical table
# ============================================================
def prepare_tcga_clinical():
    """
    Filter TCGA-LIHC clinical table for binary classification.

    Key columns: bcr_patient_barcode, PFI (event), PFI.months,
                 Early recurrence (pre-computed at 24-month threshold)

    Censoring issue: 130 patients have PFI=No but PFI.months < 24.
    These are EXCLUDED because their label is ambiguous:
    they haven't recurred YET but may recur before 24 months.
    Only patients with definitive labels are kept:
      - PFI=Yes AND PFI.months < 24  -> Early recurrence = 1
      - PFI=No AND PFI.months >= 24  -> Early recurrence = 0
    """
    df = pd.read_csv(DATA_INFO_DIR / "TCGA_LIHC_clinical.csv")

    total = len(df)
    # Keep patients with valid PFI data
    df_valid = df[
        df["PFI"].notna() &
        df["PFI.months"].notna() &
        (df["PFI.months"] > 0)
    ].copy()

    # Identify and EXCLUDE censored patients (PFI=No but follow-up < 24 months)
    censored_mask = (df_valid["PFI"] == "No") & (df_valid["PFI.months"] < 24)
    n_censored = censored_mask.sum()
    df_valid = df_valid[~censored_mask].copy()

    print(f"TCGA: {total} total -> {n_censored} censored (PFI=No, PFI.months<24) excluded")

    # Rename for STAMP
    df_valid = df_valid.rename(columns={"bcr_patient_barcode": "PATIENT"})

    # Use pre-computed Early recurrence column (24-month threshold)
    # Fill any remaining NaN with 0 (only 1 NaN patient, likely data entry issue)
    df_valid["Early recurrence"] = df_valid["Early recurrence"].fillna(0).astype(int)

    print(f"TCGA filtered: {len(df_valid)} patients with unambiguous labels")
    print(f"  Early recurrence = 1: {(df_valid['Early recurrence'] == 1).sum()}")
    print(f"  Early recurrence = 0: {(df_valid['Early recurrence'] == 0).sum()}")

    return df_valid


# ============================================================
# Step 3: Generate slide tables
# ============================================================
def make_slide_table_ourdata(feature_dir: Path, clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate slide table mapping PATIENT -> FILENAME (.h5 path).

    The .h5 feature files are named after the original WSI filename (without extension).
    This function assumes WSI filenames match the patient IDs in some way.

    IMPORTANT: Run this on the server where feature files exist.
    The feature_dir should contain subdirectories or .h5 files.

    Expected .h5 file naming: <patient_id>.h5 or <slide_name>.h5
    You need to inspect the actual .h5 filenames after feature extraction
    and match them to patient IDs.
    """
    h5_files = list(feature_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} .h5 files in {feature_dir}")

    # Build slide table
    # The FILENAME column should be relative to feature_dir
    rows = []
    for h5_file in h5_files:
        # Extract patient ID from filename
        # Common patterns: "12345.h5", "12345_something.h5", "TCGA-XX-XXXX.h5"
        patient_id = h5_file.stem.split("_")[0]  # adjust as needed
        rows.append({
            "PATIENT": str(patient_id),
            "FILENAME": str(h5_file.relative_to(feature_dir))
        })

    slide_df = pd.DataFrame(rows)

    # Keep only patients present in clinical table
    valid_patients = set(clinical_df["PATIENT"].astype(str))
    slide_df["PATIENT"] = slide_df["PATIENT"].astype(str)
    matched = slide_df[slide_df["PATIENT"].isin(valid_patients)]

    print(f"Slide table: {len(slide_df)} slides, {len(matched)} matched to clinical table")
    unmatched = set(slide_df["PATIENT"]) - valid_patients
    if unmatched:
        print(f"  WARNING: {len(unmatched)} slides without clinical data (will be ignored by STAMP)")

    return matched


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data preparation for HCC Early Recurrence Prediction")
    parser.add_argument("--task", choices=["all", "slide_table"], default="all",
                        help="'all': merge clinical tables; 'slide_table': generate slide table from .h5 files")
    parser.add_argument("--feature_dir", type=Path,
                        help="Directory containing extracted .h5 feature files (required for slide_table task)")
    parser.add_argument("--clini_csv", type=Path,
                        help="Path to clinical CSV file for patient ID matching (required for slide_table task)")
    parser.add_argument("--output", type=Path,
                        help="Output path for slide table CSV (required for slide_table task)")
    args = parser.parse_args()

    if args.task == "slide_table":
        # Generate slide table: map PATIENT -> FILENAME (.h5)
        if not args.feature_dir or not args.clini_csv or not args.output:
            parser.error("--feature_dir, --clini_csv, and --output are required for slide_table task")

        clini_df = pd.read_csv(args.clini_csv)
        clini_df["PATIENT"] = clini_df["PATIENT"].astype(str)

        slide_df = make_slide_table_ourdata(args.feature_dir, clini_df)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        slide_df.to_csv(args.output, index=False)
        print(f"Saved slide table: {args.output}")
        print(f"  {len(slide_df)} slides matched to clinical table")
        print("\nFirst few rows:")
        print(slide_df.head().to_string(index=False))

    else:
        # Default: run all clinical table preparation steps
        print("=" * 60)
        print("Step 1: Merge Ourdata clinical tables")
        print("=" * 60)
        ourdata_merged = merge_ourdata_clinical()

        out_path = DATA_INFO_DIR / "ourdata_clinical_merged.csv"
        ourdata_merged.to_csv(out_path, index=False)
        print(f"Saved: {out_path}\n")

        print("=" * 60)
        print("Step 2: Prepare TCGA clinical table")
        print("=" * 60)
        tcga_filtered = prepare_tcga_clinical()

        out_path = DATA_INFO_DIR / "tcga_clinical_filtered.csv"
        tcga_filtered.to_csv(out_path, index=False)
        print(f"Saved: {out_path}\n")
