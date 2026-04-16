"""
generate_report.py
------------------
Generate a formatted comparison table for reporting.
Reads from experiment_results.csv and outputs a clean markdown/CSV table.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --input results/experiment_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate markdown table for paper/report."""
    lines = []
    lines.append("## Experiment Comparison Table")
    lines.append("")
    lines.append("| Exp | Method | Model | Clinical | Ourdata AUROC (5-fold CV) | TCGA AUROC (External) | ΔAUROC | Status |")
    lines.append("|-----|--------|-------|----------|--------------------------|----------------------|--------|--------|")

    for _, row in df.iterrows():
        clin = "✓" if row["Clinical"] == "Yes" else "—"
        lines.append(
            f"| {row['Exp']} | {row['Name']} | {row['Model']} | {clin} | "
            f"{row['Ourdata AUROC']} | {row['TCGA AUROC']} | {row['ΔAUROC']} | {row['Status']} |"
        )
    return "\n".join(lines)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation Study: HCC Early Recurrence Prediction}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Exp} & \textbf{Method} & \textbf{Clinical} & "
                 r"\textbf{Ourdata AUROC} & \textbf{TCGA AUROC} & \textbf{$\Delta$AUROC} \\")
    lines.append(r"\hline")

    for _, row in df.iterrows():
        if row["Status"] != "Done":
            continue
        clin = r"\checkmark" if row["Clinical"] == "Yes" else "—"
        our = row['Ourdata AUROC'].split()[0] if row['Ourdata AUROC'] != '—' else '—'
        tcga = row['TCGA AUROC'].split()[0] if row['TCGA AUROC'] != '—' else '—'
        lines.append(
            f"{row['Exp']} & {row['Name']} & {clin} & {our} & {tcga} & {row['ΔAUROC']} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def print_summary(df: pd.DataFrame):
    done = df[df["Status"] == "Done"]
    pending = df[df["Status"] == "Pending"]

    print("\n" + "=" * 60)
    print("EXPERIMENT PROGRESS")
    print("=" * 60)
    print(f"  Completed : {len(done)} / {len(df)}")
    print(f"  Pending   : {len(pending)} / {len(df)}")

    if len(done) > 0:
        print("\n" + "=" * 60)
        print("COMPLETED RESULTS")
        print("=" * 60)
        print(done[["Exp", "Name", "Ourdata AUROC", "TCGA AUROC", "ΔAUROC"]].to_string(index=False))

        # Find best
        def parse_auc(s):
            try:
                return float(str(s).split()[0])
            except Exception:
                return 0.0

        done = done.copy()
        done["_our"] = done["Ourdata AUROC"].apply(parse_auc)
        done["_tcga"] = done["TCGA AUROC"].apply(parse_auc)
        best_our = done.loc[done["_our"].idxmax()]
        best_tcga = done.loc[done["_tcga"].idxmax()]

        print(f"\n  Best Ourdata AUROC: {best_our['Exp']} — {best_our['Name']} ({best_our['Ourdata AUROC']})")
        print(f"  Best TCGA AUROC:    {best_tcga['Exp']} — {best_tcga['Name']} ({best_tcga['TCGA AUROC']})")

    if len(pending) > 0:
        print("\n" + "=" * 60)
        print("PENDING EXPERIMENTS")
        print("=" * 60)
        for _, row in pending.iterrows():
            print(f"  ⏳ {row['Exp']:6s} {row['Name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("/data3/chensx/STAMP/outputs/results/experiment_results.csv"))
    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="both")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Results file not found: {args.input}")
        print("Run: python scripts/collect_results.py first")
        exit(1)

    df = pd.read_csv(args.input)
    print_summary(df)

    out_dir = args.input.parent
    if args.format in ("markdown", "both"):
        md = generate_markdown_table(df)
        md_path = out_dir / "comparison_table.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"\nMarkdown table: {md_path}")
        print("\n" + md)

    if args.format in ("latex", "both"):
        latex = generate_latex_table(df)
        tex_path = out_dir / "comparison_table.tex"
        tex_path.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX table: {tex_path}")
