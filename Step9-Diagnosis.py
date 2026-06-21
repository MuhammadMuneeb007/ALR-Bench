# ============================================================
# Step9-PrintRankingDeepCutoffs.py
# ============================================================
# Purpose:
#   Print ranking performance at deep cutoffs:
#   C@10, C@20, C@50, C@100, C@500, C@1000, C@2000, C@3000
#
# It reads:
#   Projects/<project>/Rankings/rank_strategy_*.csv
#
# It requires each ranking CSV to contain:
#   is_reference_match
#
# Run:
#   python Step9-PrintRankingDeepCutoffs.py
# ============================================================

from pathlib import Path
import re
import pandas as pd


PROJECTS_DIR = Path("Projects")

TOPK_VALUES = [10, 20, 50, 100, 500, 1000, 2000, 3000]

STRATEGY_NAMES = {
    1: "ConfigWeightedBM25",
    2: "ReviewIntentSBERT",
    3: "TitleObjectiveSBERT",
    4: "ConceptKeywordCoverage",
    5: "GeneratedQueryMatch",
    6: "BooleanSoftMatch",
    7: "FieldWeightedLexical",
    8: "SciBERTSemantic",
    9: "SPECTER2Semantic",
    10: "HybridFusion",
}


def get_strategy_number(path: Path):
    m = re.search(r"rank_strategy_(\d+)", path.name)
    if not m:
        return None
    return int(m.group(1))


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)

    return s.astype(str).str.lower().str.strip().isin(
        ["true", "1", "yes", "y", "match", "matched"]
    )


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def print_table(df: pd.DataFrame, title: str):
    print("\n" + "=" * 160)
    print(title)
    print("=" * 160)

    if df.empty:
        print("No data.")
        return

    df = df.copy()
    df = df.astype(str)

    widths = {}
    for col in df.columns:
        widths[col] = max(len(str(col)), df[col].map(len).max())

    header = " | ".join(str(col).ljust(widths[col]) for col in df.columns)
    sep = "-+-".join("-" * widths[col] for col in df.columns)

    print(header)
    print(sep)

    for _, row in df.iterrows():
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in df.columns))


def discover_project_dirs():
    if not PROJECTS_DIR.exists():
        raise FileNotFoundError(f"Projects directory not found: {PROJECTS_DIR.resolve()}")

    project_dirs = []
    for p in sorted(PROJECTS_DIR.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() == "rankings":
            continue
        if (p / "Rankings").exists():
            project_dirs.append(p)

    return project_dirs


def main():
    print("=" * 160)
    print("DEEP RANKING CUTOFF DIAGNOSTIC")
    print("=" * 160)
    print(f"Projects directory : {PROJECTS_DIR.resolve()}")
    print(f"Top-K values       : {TOPK_VALUES}")

    project_dirs = discover_project_dirs()

    print(f"Projects found     : {len(project_dirs)}")

    project_rows = []
    strategy_rows = {}

    for project_dir in project_dirs:
        rankings_dir = project_dir / "Rankings"

        ranking_files = sorted(rankings_dir.glob("rank_strategy_*.csv"))

        ranking_files = [
            f for f in ranking_files
            if "summary" not in f.name.lower()
        ]

        for ranking_file in ranking_files:
            strategy_num = get_strategy_number(ranking_file)
            if strategy_num is None:
                continue

            strategy_name = STRATEGY_NAMES.get(strategy_num, f"Strategy{strategy_num}")

            try:
                df = safe_read_csv(ranking_file)
            except Exception as e:
                print(f"ERROR reading {ranking_file}: {e}")
                continue

            if df.empty:
                continue

            if "is_reference_match" not in df.columns:
                print(f"WARNING: is_reference_match column missing in {ranking_file}")
                continue

            flags = to_bool_series(df["is_reference_match"])

            ranked_total = len(df)
            ceiling_hits = int(flags.sum())

            row = {
                "strategy": strategy_num,
                "strategy_name": strategy_name,
                "project": project_dir.name,
                "ranked": ranked_total,
                "ceiling_hits": ceiling_hits,
            }

            for k in TOPK_VALUES:
                actual_k = min(k, ranked_total)
                c_at_k = int(flags.head(actual_k).sum())

                row[f"C@{k}"] = c_at_k

                if ceiling_hits > 0:
                    row[f"C@{k}/ceiling"] = round(c_at_k / ceiling_hits, 4)
                else:
                    row[f"C@{k}/ceiling"] = "NA"

            project_rows.append(row)

            if strategy_num not in strategy_rows:
                strategy_rows[strategy_num] = {
                    "strategy": strategy_num,
                    "strategy_name": strategy_name,
                    "projects": 0,
                    "ranked_total": 0,
                    "ceiling_hits_total": 0,
                }

                for k in TOPK_VALUES:
                    strategy_rows[strategy_num][f"C@{k}"] = 0

            strategy_rows[strategy_num]["projects"] += 1
            strategy_rows[strategy_num]["ranked_total"] += ranked_total
            strategy_rows[strategy_num]["ceiling_hits_total"] += ceiling_hits

            for k in TOPK_VALUES:
                strategy_rows[strategy_num][f"C@{k}"] += row[f"C@{k}"]

    project_df = pd.DataFrame(project_rows)

    if project_df.empty:
        print("\nNo ranking results found.")
        print("Check that files exist under:")
        print("  Projects/<project>/Rankings/rank_strategy_*.csv")
        print("and that they contain:")
        print("  is_reference_match")
        return

    strategy_df = pd.DataFrame(strategy_rows.values())

    # Add recall versus ceiling for each strategy
    for k in TOPK_VALUES:
        strategy_df[f"C@{k}/ceiling"] = strategy_df.apply(
            lambda r: round(r[f"C@{k}"] / r["ceiling_hits_total"], 4)
            if r["ceiling_hits_total"] > 0 else "NA",
            axis=1,
        )

        strategy_df[f"P@{k}"] = strategy_df.apply(
            lambda r: round(r[f"C@{k}"] / (r["projects"] * k), 4)
            if r["projects"] > 0 else "NA",
            axis=1,
        )

    # Sort by C@1000 first, then C@3000
    strategy_df = strategy_df.sort_values(
        by=["C@1000", "C@3000", "C@100"],
        ascending=False,
    ).reset_index(drop=True)

    # Strategy-level compact table
    compact_cols = [
        "strategy",
        "strategy_name",
        "projects",
        "ranked_total",
        "ceiling_hits_total",
    ]

    for k in TOPK_VALUES:
        compact_cols.append(f"C@{k}")

    for k in TOPK_VALUES:
        compact_cols.append(f"C@{k}/ceiling")

    print_table(
        strategy_df[compact_cols],
        "1) STRATEGY-LEVEL DEEP CUTOFF TABLE",
    )

    # Precision table
    precision_cols = [
        "strategy",
        "strategy_name",
        "projects",
    ]

    for k in TOPK_VALUES:
        precision_cols.append(f"P@{k}")

    print_table(
        strategy_df[precision_cols],
        "2) STRATEGY-LEVEL PRECISION TABLE",
    )

    # Best strategy by each cutoff
    best_rows = []

    for k in TOPK_VALUES:
        ccol = f"C@{k}"
        best = strategy_df.sort_values(ccol, ascending=False).iloc[0]

        best_rows.append({
            "cutoff": ccol,
            "best_strategy": int(best["strategy"]),
            "strategy_name": best["strategy_name"],
            "hits": int(best[ccol]),
            "ceiling_recall": best[f"C@{k}/ceiling"],
        })

    best_df = pd.DataFrame(best_rows)

    print_table(
        best_df,
        "3) BEST STRATEGY BY DEEP CUTOFF",
    )

    # Incremental gains
    gain_rows = []

    for _, row in strategy_df.iterrows():
        g = {
            "strategy": row["strategy"],
            "strategy_name": row["strategy_name"],
        }

        previous_k = None
        previous_c = 0

        for k in TOPK_VALUES:
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
        "4) INCREMENTAL GAINS BETWEEN CUTOFFS",
    )

    # Project-level best C@1000 and C@3000
    project_best_rows = []

    for project, g in project_df.groupby("project"):
        g1000 = g.sort_values("C@1000", ascending=False).iloc[0]
        g3000 = g.sort_values("C@3000", ascending=False).iloc[0]

        max_ceiling = int(g["ceiling_hits"].max())

        project_best_rows.append({
            "project": project,
            "strategies_available": g["strategy"].nunique(),
            "ceiling_hits": max_ceiling,
            "best_C@1000": int(g1000["C@1000"]),
            "best_C@1000_strategy": f"{int(g1000['strategy'])} - {g1000['strategy_name']}",
            "best_C@1000/ceiling": round(int(g1000["C@1000"]) / max_ceiling, 4) if max_ceiling > 0 else "NA",
            "best_C@3000": int(g3000["C@3000"]),
            "best_C@3000_strategy": f"{int(g3000['strategy'])} - {g3000['strategy_name']}",
            "best_C@3000/ceiling": round(int(g3000["C@3000"]) / max_ceiling, 4) if max_ceiling > 0 else "NA",
        })

    project_best_df = pd.DataFrame(project_best_rows)

    project_best_df = project_best_df.sort_values(
        by=["best_C@1000/ceiling", "best_C@1000"],
        ascending=[True, True],
    ).reset_index(drop=True)

    print_table(
        project_best_df,
        "5) PROJECT-LEVEL BEST C@1000 AND C@3000",
    )

    # Download recommendation table
    recommendation_rows = []

    for _, row in project_best_df.iterrows():
        ceiling = row["ceiling_hits"]
        c1000 = row["best_C@1000"]
        c3000 = row["best_C@3000"]

        if ceiling == 0:
            recommendation = "NO_REFERENCE_HITS_IN_POOL"
        else:
            r1000 = c1000 / ceiling
            r3000 = c3000 / ceiling

            if r1000 >= 0.70:
                recommendation = "TOP_1000_ENOUGH_FOR_FIRST_FULLTEXT"
            elif r3000 >= 0.70:
                recommendation = "TOP_3000_BETTER_THAN_TOP_1000"
            elif c3000 > c1000:
                recommendation = "DOWNLOAD_TOP_3000_OR_IMPROVE_RANKING"
            else:
                recommendation = "RETRIEVAL_POOL_LIMITED_OR_RANKING_STUCK"

        recommendation_rows.append({
            "project": row["project"],
            "ceiling_hits": ceiling,
            "best_C@1000": c1000,
            "best_C@3000": c3000,
            "recommendation": recommendation,
        })

    recommendation_df = pd.DataFrame(recommendation_rows)

    print_table(
        recommendation_df,
        "6) FULL-TEXT DOWNLOAD RECOMMENDATION BY PROJECT",
    )

    print("\n" + "=" * 160)
    print("FINAL NOTES")
    print("=" * 160)
    print(
        "Use C@1000 first. If C@1000 captures most of the ceiling, download top 1000. "
        "If C@2000 or C@3000 adds many extra reference hits, then top 1000 is too strict. "
        "If even C@3000 is low versus ceiling, then ranking needs improvement before full-text download."
    )


if __name__ == "__main__":
    main()