#!/usr/bin/env python3
# ============================================================
# Step14-PrintFullTextRankingDeepCutoffs.py
# ============================================================
# Purpose:
#   Print full-text ranking performance at deep cutoffs:
#   C@10, C@20, C@50, C@100, C@200, C@500, C@1000
#
# It reads:
#   Projects/<project>/Rankings/full_rank_strategy_*.csv
#
# It supports these reference-match columns:
#   is_reference
#   is_reference_match
#   reference_match
#   matched_reference
#   doi_in_references
#
# Run:
#   python Step14-PrintFullTextRankingDeepCutoffs.py
#
# Optional:
#   python Step14-PrintFullTextRankingDeepCutoffs.py --projects-dir Projects
#   python Step14-PrintFullTextRankingDeepCutoffs.py --save-csv
# ============================================================

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


TOPK_VALUES_DEFAULT = [10, 20, 50, 100, 200, 500, 1000]

STRATEGY_NAMES = {
    1: "FullConfigBM25",
    2: "FullReviewIntentSBERT",
    3: "FullTitleObjectiveSBERT",
    4: "FullConceptKeywordCoverage",
    5: "FullGeneratedQueryMatch",
    6: "FullBooleanSoftMatch",
    7: "FullFieldWeightedLexical",
    8: "FullSciBERTSemantic",
    9: "FullSPECTER2Semantic",
    10: "FullHybridFusion",
}

REFERENCE_COL_CANDIDATES = [
    "is_reference",
    "is_reference_match",
    "reference_match",
    "matched_reference",
    "matched_ref",
    "doi_in_references",
    "doi_in_refs",
    "in_references",
    "is_common",
    "common_reference",
]


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def get_strategy_number(path: Path) -> Optional[int]:
    # Important: full_rank_strategy_10.csv also contains rank_strategy_10,
    # so this regex is strict for full_rank_strategy.
    m = re.search(r"full_rank_strategy_(\d+)\.csv$", path.name)
    if not m:
        return None
    return int(m.group(1))


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)

    return s.astype(str).str.lower().str.strip().isin(
        ["true", "1", "yes", "y", "match", "matched", "reference", "ref"]
    )


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def find_reference_column(df: pd.DataFrame) -> Optional[str]:
    for col in REFERENCE_COL_CANDIDATES:
        if col in df.columns:
            return col

    # Fallback: find a likely reference flag column.
    for col in df.columns:
        c = col.lower().strip()
        if "reference" in c and any(x in c for x in ["is", "match", "in", "common"]):
            return col

    return None


def print_table(df: pd.DataFrame, title: str, max_width: int = 55) -> None:
    print("\n" + "=" * 180)
    print(title)
    print("=" * 180)

    if df.empty:
        print("No data.")
        return

    df = df.copy()

    for col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].map(lambda x: x[:max_width] if len(x) > max_width else x)

    widths = {}
    for col in df.columns:
        widths[col] = max(len(str(col)), int(df[col].map(len).max()))

    header = " | ".join(str(col).ljust(widths[col]) for col in df.columns)
    sep = "-+-".join("-" * widths[col] for col in df.columns)

    print(header)
    print(sep)

    for _, row in df.iterrows():
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in df.columns))


def discover_project_dirs(projects_dir: Path) -> List[Path]:
    if not projects_dir.exists():
        raise FileNotFoundError(f"Projects directory not found: {projects_dir.resolve()}")

    project_dirs = []
    for p in sorted(projects_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() == "rankings":
            continue
        if (p / "Rankings").exists():
            project_dirs.append(p)

    return project_dirs


def parse_topk_values(x: str) -> List[int]:
    if not x:
        return TOPK_VALUES_DEFAULT[:]

    values = []
    for part in x.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            k = int(part)
        except Exception:
            continue
        if k > 0:
            values.append(k)

    return values or TOPK_VALUES_DEFAULT[:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print deep cutoff diagnostics for Step14 full-text rankings."
    )
    parser.add_argument(
        "--projects-dir",
        type=str,
        default="Projects",
        help="Path to Projects directory.",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default=",".join(map(str, TOPK_VALUES_DEFAULT)),
        help="Comma-separated cutoffs, e.g. 10,20,50,100,200,500,1000",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save diagnostic CSV files under Projects/Rankings.",
    )
    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    topk_values = parse_topk_values(args.topk)

    print("=" * 180)
    print("STEP14 FULL-TEXT RANKING DEEP CUTOFF DIAGNOSTIC")
    print("=" * 180)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Ranking pattern    : Projects/<project>/Rankings/full_rank_strategy_*.csv")
    print(f"Top-K values       : {topk_values}")

    project_dirs = discover_project_dirs(projects_dir)
    print(f"Projects found     : {len(project_dirs)}")

    project_rows: List[Dict[str, Any]] = []
    strategy_rows: Dict[int, Dict[str, Any]] = {}

    missing_ref_col_files = []
    empty_files = []
    unreadable_files = []

    for project_dir in project_dirs:
        rankings_dir = project_dir / "Rankings"

        ranking_files = sorted(
            [
                f for f in rankings_dir.glob("full_rank_strategy_*.csv")
                if "summary" not in f.name.lower()
            ],
            key=lambda p: (get_strategy_number(p) or 10**9, p.name),
        )

        for ranking_file in ranking_files:
            strategy_num = get_strategy_number(ranking_file)
            if strategy_num is None:
                continue

            strategy_name = STRATEGY_NAMES.get(strategy_num, f"FullStrategy{strategy_num}")

            try:
                df = safe_read_csv(ranking_file)
            except Exception as e:
                unreadable_files.append((str(ranking_file), str(e)))
                continue

            if df.empty:
                empty_files.append(str(ranking_file))
                continue

            ref_col = find_reference_column(df)
            if ref_col is None:
                missing_ref_col_files.append(str(ranking_file))
                continue

            flags = to_bool_series(df[ref_col]).reset_index(drop=True)

            ranked_total = len(df)
            ceiling_hits = int(flags.sum())

            row: Dict[str, Any] = {
                "strategy": strategy_num,
                "strategy_name": strategy_name,
                "project": project_dir.name,
                "ranked": ranked_total,
                "ceiling_hits": ceiling_hits,
                "ref_col": ref_col,
                "file": str(ranking_file),
            }

            for k in topk_values:
                actual_k = min(k, ranked_total)
                c_at_k = int(flags.head(actual_k).sum())

                row[f"C@{k}"] = c_at_k

                if ceiling_hits > 0:
                    row[f"C@{k}/ceiling"] = round(c_at_k / ceiling_hits, 4)
                else:
                    row[f"C@{k}/ceiling"] = "NA"

                row[f"P@{k}"] = round(c_at_k / actual_k, 6) if actual_k else 0.0

            project_rows.append(row)

            if strategy_num not in strategy_rows:
                strategy_rows[strategy_num] = {
                    "strategy": strategy_num,
                    "strategy_name": strategy_name,
                    "projects": 0,
                    "ranked_total": 0,
                    "ceiling_hits_total": 0,
                    "with_ref_col": 0,
                }

                for k in topk_values:
                    strategy_rows[strategy_num][f"C@{k}"] = 0

            strategy_rows[strategy_num]["projects"] += 1
            strategy_rows[strategy_num]["ranked_total"] += ranked_total
            strategy_rows[strategy_num]["ceiling_hits_total"] += ceiling_hits
            strategy_rows[strategy_num]["with_ref_col"] += 1

            for k in topk_values:
                strategy_rows[strategy_num][f"C@{k}"] += row[f"C@{k}"]

    project_df = pd.DataFrame(project_rows)

    if project_df.empty:
        print("\nNo full-text ranking results found.")
        print("Check that files exist under:")
        print("  Projects/<project>/Rankings/full_rank_strategy_*.csv")
        print("and that they contain a reference flag column such as:")
        print("  is_reference")
        return

    strategy_df = pd.DataFrame(strategy_rows.values())

    # Add recall versus ceiling and global precision for each strategy.
    for k in topk_values:
        strategy_df[f"C@{k}/ceiling"] = strategy_df.apply(
            lambda r: round(r[f"C@{k}"] / r["ceiling_hits_total"], 4)
            if safe_int(r["ceiling_hits_total"]) > 0 else "NA",
            axis=1,
        )

        strategy_df[f"P@{k}"] = strategy_df.apply(
            lambda r: round(r[f"C@{k}"] / max(1, r["projects"] * k), 6),
            axis=1,
        )

    sort_cols = []
    for preferred in ["C@100", "C@200", "C@500", "C@1000"]:
        if preferred in strategy_df.columns:
            sort_cols.append(preferred)

    if not sort_cols:
        sort_cols = [f"C@{topk_values[-1]}"]

    strategy_df = strategy_df.sort_values(
        by=list(reversed(sort_cols)),
        ascending=False,
    ).reset_index(drop=True)

    # 1) Strategy-level compact table.
    compact_cols = [
        "strategy",
        "strategy_name",
        "projects",
        "ranked_total",
        "ceiling_hits_total",
    ]

    for k in topk_values:
        compact_cols.append(f"C@{k}")

    for k in topk_values:
        compact_cols.append(f"C@{k}/ceiling")

    print_table(
        strategy_df[compact_cols],
        "1) FULL-TEXT STRATEGY-LEVEL DEEP CUTOFF TABLE",
    )

    # 2) Precision table.
    precision_cols = [
        "strategy",
        "strategy_name",
        "projects",
    ]

    for k in topk_values:
        precision_cols.append(f"P@{k}")

    print_table(
        strategy_df[precision_cols],
        "2) FULL-TEXT STRATEGY-LEVEL PRECISION TABLE",
    )

    # 3) Best strategy by each cutoff.
    best_rows = []

    for k in topk_values:
        ccol = f"C@{k}"
        best = strategy_df.sort_values(ccol, ascending=False).iloc[0]

        best_rows.append({
            "cutoff": ccol,
            "best_strategy": int(best["strategy"]),
            "strategy_name": best["strategy_name"],
            "hits": int(best[ccol]),
            "ceiling_recall": best[f"C@{k}/ceiling"],
            "precision": best[f"P@{k}"],
        })

    best_df = pd.DataFrame(best_rows)

    print_table(
        best_df,
        "3) BEST FULL-TEXT STRATEGY BY CUTOFF",
    )

    # 4) Incremental gains.
    gain_rows = []

    for _, row in strategy_df.iterrows():
        g: Dict[str, Any] = {
            "strategy": row["strategy"],
            "strategy_name": row["strategy_name"],
        }

        previous_k = None
        previous_c = 0

        for k in topk_values:
            current_c = int(row[f"C@{k}"])

            if previous_k is None:
                g[f"gain_1_to_{k}"] = current_c
            else:
                g[f"gain_{previous_k}_to_{k}"] = current_c - previous_c

            previous_k = k
            previous_c = current_c

        gain_rows.append(g)

    gain_df = pd.DataFrame(gain_rows)

    print_table(
        gain_df,
        "4) FULL-TEXT INCREMENTAL GAINS BETWEEN CUTOFFS",
    )

    # 5) Project-level best strategy.
    project_best_rows = []

    main_cutoffs = [k for k in [100, 200, 500, 1000] if k in topk_values]
    if not main_cutoffs:
        main_cutoffs = [topk_values[-1]]

    for project, g in project_df.groupby("project"):
        max_ceiling = int(g["ceiling_hits"].max())

        row: Dict[str, Any] = {
            "project": project,
            "strategies_available": g["strategy"].nunique(),
            "ceiling_hits": max_ceiling,
        }

        for k in main_cutoffs:
            best = g.sort_values(f"C@{k}", ascending=False).iloc[0]
            row[f"best_C@{k}"] = int(best[f"C@{k}"])
            row[f"best_C@{k}_strategy"] = f"{int(best['strategy'])} - {best['strategy_name']}"
            row[f"best_C@{k}/ceiling"] = (
                round(int(best[f"C@{k}"]) / max_ceiling, 4)
                if max_ceiling > 0 else "NA"
            )

        project_best_rows.append(row)

    project_best_df = pd.DataFrame(project_best_rows)

    sort_by = f"best_C@{main_cutoffs[-1]}/ceiling"
    if sort_by in project_best_df.columns:
        project_best_df = project_best_df.sort_values(
            by=[sort_by, f"best_C@{main_cutoffs[-1]}"],
            ascending=[True, True],
        ).reset_index(drop=True)

    print_table(
        project_best_df,
        "5) PROJECT-LEVEL BEST FULL-TEXT STRATEGY",
    )

    # 6) Screening recommendation based on full-text rankings.
    recommendation_rows = []

    for _, row in project_best_df.iterrows():
        ceiling = safe_int(row.get("ceiling_hits", 0), 0)

        c100 = safe_int(row.get("best_C@100", 0), 0)
        c200 = safe_int(row.get("best_C@200", 0), 0)
        c500 = safe_int(row.get("best_C@500", 0), 0)
        c1000 = safe_int(row.get("best_C@1000", 0), 0)

        if ceiling == 0:
            recommendation = "NO_REFERENCE_HITS_IN_POOL"
        else:
            r100 = c100 / ceiling
            r200 = c200 / ceiling
            r500 = c500 / ceiling
            r1000 = c1000 / ceiling

            if r100 >= 0.70:
                recommendation = "SCREEN_TOP_100_ENOUGH"
            elif r200 >= 0.70:
                recommendation = "SCREEN_TOP_200_GOOD"
            elif r500 >= 0.70:
                recommendation = "SCREEN_TOP_500_RECOMMENDED"
            elif r1000 >= 0.70:
                recommendation = "SCREEN_TOP_1000_NEEDED"
            elif c1000 > c500:
                recommendation = "TOP_1000_ADDS_HITS_BUT_POOL_OR_RANKING_LIMITED"
            else:
                recommendation = "RANKING_STUCK_OR_REFERENCE_POOL_LIMITED"

        recommendation_rows.append({
            "project": row["project"],
            "ceiling_hits": ceiling,
            "best_C@100": c100,
            "best_C@200": c200,
            "best_C@500": c500,
            "best_C@1000": c1000,
            "recommendation": recommendation,
        })

    recommendation_df = pd.DataFrame(recommendation_rows)

    print_table(
        recommendation_df,
        "6) FULL-TEXT SCREENING RECOMMENDATION BY PROJECT",
    )

    # 7) Diagnostics.
    diag_rows = []
    diag_rows.append({
        "diagnostic": "ranking_rows_used",
        "count": len(project_df),
    })
    diag_rows.append({
        "diagnostic": "missing_reference_column_files",
        "count": len(missing_ref_col_files),
    })
    diag_rows.append({
        "diagnostic": "empty_files",
        "count": len(empty_files),
    })
    diag_rows.append({
        "diagnostic": "unreadable_files",
        "count": len(unreadable_files),
    })

    diag_df = pd.DataFrame(diag_rows)

    print_table(
        diag_df,
        "7) DIAGNOSTICS",
    )

    if missing_ref_col_files:
        print("\nWARNING: These files had no reference flag column:")
        for x in missing_ref_col_files[:50]:
            print(f"  {x}")
        if len(missing_ref_col_files) > 50:
            print(f"  ... plus {len(missing_ref_col_files) - 50} more")

    if unreadable_files:
        print("\nWARNING: These files could not be read:")
        for x, err in unreadable_files[:50]:
            print(f"  {x} | {err}")
        if len(unreadable_files) > 50:
            print(f"  ... plus {len(unreadable_files) - 50} more")

    if args.save_csv:
        summary_dir = projects_dir / "Rankings"
        summary_dir.mkdir(parents=True, exist_ok=True)

        out_strategy = summary_dir / "step14_fulltext_deep_cutoff_strategy_summary.csv"
        out_project = summary_dir / "step14_fulltext_deep_cutoff_project_summary.csv"
        out_best = summary_dir / "step14_fulltext_deep_cutoff_best_by_cutoff.csv"
        out_project_best = summary_dir / "step14_fulltext_deep_cutoff_project_best.csv"
        out_recommend = summary_dir / "step14_fulltext_deep_cutoff_recommendations.csv"

        strategy_df.to_csv(out_strategy, index=False)
        project_df.to_csv(out_project, index=False)
        best_df.to_csv(out_best, index=False)
        project_best_df.to_csv(out_project_best, index=False)
        recommendation_df.to_csv(out_recommend, index=False)

        print("\nSaved CSV files:")
        print(f"  {out_strategy}")
        print(f"  {out_project}")
        print(f"  {out_best}")
        print(f"  {out_project_best}")
        print(f"  {out_recommend}")

    print("\n" + "=" * 180)
    print("FINAL NOTES")
    print("=" * 180)
    print(
        "For Step14 full-text reranking, focus on C@100, C@200, and C@500 first. "
        "C@1000 is the candidate-pool ceiling after reranking because all full-text strategies "
        "rerank the same top-1000 pool selected by rank_strategy_3.csv. "
        "If C@1000 is identical across methods, that is normal; compare methods by how many "
        "reference hits they move earlier into C@100/C@200/C@500."
    )


if __name__ == "__main__":
    main()
