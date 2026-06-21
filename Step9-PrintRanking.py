# Step9-PrintEachRankingTable.py
#
# Prints one table for each ranking strategy.
#
# For example:
#   RANK STRATEGY 1 — ConfigWeightedBM25
#   project | ranked | refs_doi | C@10 | C@20 | C@50 | C@100 | file
#
#   RANK STRATEGY 2 — ReviewIntentSBERT
#   project | ranked | refs_doi | C@10 | C@20 | C@50 | C@100 | file
#
# Usage:
#   python Step9-PrintEachRankingTable.py --projects-dir Projects
#
# Optional save:
#   python Step9-PrintEachRankingTable.py --projects-dir Projects --save

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


DEFAULT_PROJECTS_DIR = Path("Projects")
RANKINGS_DIR_NAME = "Rankings"

TOP_K_VALUES = [10, 20, 50, 100]

REFERENCE_FLAG_COLUMNS = [
    "is_reference",
    "is_reference_match",
    "reference_match",
    "matched_reference",
    "matched_ref",
    "doi_in_references",
    "doi_in_refs",
    "in_references",
]

REFERENCE_JSON_FILES = [
    "references_gold.json",
    "references_resolved.json",
    "references.json",
]

REFERENCE_CSV_FILES = [
    Path("References") / "references_clean_doi_only.csv",
    Path("References") / "references_clean.csv",
    Path("References") / "references.csv",
    Path("references_clean_doi_only.csv"),
    Path("references_clean.csv"),
    Path("references.csv"),
]

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
    11: "FinalChunkedReranker",
}


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def safe_int(x: Any, default: int = 0) -> int:
    try:
        text = clean_text(x)
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def first_value(row: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = clean_text(row.get(key, ""))
        if value:
            return value
    return ""


def normalize_doi(x: Any) -> str:
    doi = clean_text(x).lower()

    for prefix in [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "doi:",
        "doi ",
    ]:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    match = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", doi, flags=re.I)
    if match:
        doi = match.group(1)

    doi = re.sub(r"\s+", "", doi).strip().strip(".,;:()[]{}<>")

    if doi in {"", "none", "null", "nan", "na", "n/a"}:
        return ""

    return doi


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}

    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def discover_project_dirs(projects_dir: Path) -> List[Path]:
    if not projects_dir.exists():
        return []

    skip = {"rankings", "search", "evaluation", "__pycache__"}

    return [
        path for path in sorted(projects_dir.iterdir())
        if path.is_dir() and path.name.lower() not in skip
    ]


def rank_file_path(project_dir: Path, strategy_id: int) -> Path:
    return project_dir / RANKINGS_DIR_NAME / f"rank_strategy_{strategy_id}.csv"


def is_reference_row(row: Dict[str, Any]) -> bool:
    for col in REFERENCE_FLAG_COLUMNS:
        value = clean_text(row.get(col, "")).lower()
        if value in {"1", "true", "yes", "y", "match", "matched", "reference", "ref"}:
            return True
    return False


def count_common_at(rows: List[Dict[str, str]], k: int) -> int:
    return sum(1 for row in rows[:k] if is_reference_row(row))


def extract_reference_dois_from_json(data: Any) -> Set[str]:
    out: Set[str] = set()
    doi_list_keys = {"dois", "doi_list", "referenced_dois", "reference_dois", "gold_dois"}
    doi_keys = {"doi", "DOI", "reference_doi", "work_doi", "article_doi", "clean_doi", "normalized_doi"}
    doi_keys_lower = {x.lower() for x in doi_keys}
    doi_pattern = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", flags=re.I)

    def add(value: Any) -> None:
        doi = normalize_doi(value)
        if doi:
            out.add(doi)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()

                if key_l in doi_keys_lower:
                    add(value)

                if key_l in doi_list_keys and isinstance(value, list):
                    for item in value:
                        add(item)

                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

        elif isinstance(obj, str):
            for match in doi_pattern.findall(obj):
                add(match)

    walk(data)
    return out


def load_reference_doi_count(project_dir: Path) -> str:
    for rel in REFERENCE_JSON_FILES:
        path = project_dir / rel
        dois = extract_reference_dois_from_json(read_json(path, default={}))

        if dois:
            return str(len(dois))

    dois: Set[str] = set()

    for rel in REFERENCE_CSV_FILES:
        path = project_dir / rel

        for row in read_csv_rows(path):
            for key, value in row.items():
                if "doi" not in str(key).lower():
                    continue

                doi = normalize_doi(value)
                if doi:
                    dois.add(doi)

    return str(len(dois)) if dois else ""


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_table(rows: List[Dict[str, Any]], fields: List[str]) -> None:
    if not rows:
        print("No rows found.")
        return

    widths: Dict[str, int] = {}

    for field in fields:
        widths[field] = max(
            len(field),
            *(len(str(row.get(field, ""))) for row in rows),
        )

    header = " | ".join(field.ljust(widths[field]) for field in fields)
    sep = "-+-".join("-" * widths[field] for field in fields)

    print(header)
    print(sep)

    for row in rows:
        print(" | ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields))


def summary_path(projects_dir: Path, strategy_id: int) -> Path:
    return projects_dir / RANKINGS_DIR_NAME / f"rank_strategy_{strategy_id}_summary.csv"


def build_strategy_table(summary_rows: List[Dict[str, str]], strategy_id: int) -> List[Dict[str, Any]]:
    table_rows: List[Dict[str, Any]] = []

    for row in summary_rows:
        out = {
            "project": row.get("project", ""),
            "status": row.get("status", ""),
            "ranked": first_value(row, ["articles_ranked", "n_articles", "ranked"]),
            "refs_total": first_value(row, ["references_total", "n_reference_total", "refs_total"]),
            "refs_doi": first_value(row, ["references_with_doi", "n_reference_dois", "refs_doi"]),
            "C@10": first_value(row, ["common_doi_at_10", "common_at_10", "C@10"]),
            "C@20": first_value(row, ["common_doi_at_20", "common_at_20", "C@20"]),
            "C@50": first_value(row, ["common_doi_at_50", "common_at_50", "C@50"]),
            "C@100": first_value(row, ["common_doi_at_100", "common_at_100", "C@100"]),
            "file": row.get("rank_file", ""),
        }

        table_rows.append(out)

    table_rows.sort(
        key=lambda r: (
            clean_text(r.get("project", "")),
            -safe_int(r.get("C@100")),
            -safe_int(r.get("C@50")),
            -safe_int(r.get("C@20")),
            -safe_int(r.get("C@10")),
        )
    )

    return table_rows


def build_rank_file_table(projects_dir: Path, strategy_id: int) -> List[Dict[str, Any]]:
    table_rows: List[Dict[str, Any]] = []

    for project_dir in discover_project_dirs(projects_dir):
        path = rank_file_path(project_dir, strategy_id)

        if not path.exists():
            continue

        rows = read_csv_rows(path)

        if not rows:
            table_rows.append({
                "project": project_dir.name,
                "status": "empty_rank_file",
                "ranked": "0",
                "refs_total": "",
                "refs_doi": load_reference_doi_count(project_dir),
                "C@10": "0",
                "C@20": "0",
                "C@50": "0",
                "C@100": "0",
                "file": str(path),
            })
            continue

        table_rows.append({
            "project": project_dir.name,
            "status": "from_rank_file",
            "ranked": str(len(rows)),
            "refs_total": "",
            "refs_doi": load_reference_doi_count(project_dir),
            "C@10": str(count_common_at(rows, 10)),
            "C@20": str(count_common_at(rows, 20)),
            "C@50": str(count_common_at(rows, 50)),
            "C@100": str(count_common_at(rows, 100)),
            "file": str(path),
        })

    return table_rows


def merge_summary_and_rank_file_rows(
    summary_table: List[Dict[str, Any]],
    rank_file_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for row in rank_file_table:
        project = clean_text(row.get("project", ""))
        if project:
            merged[project] = dict(row)

    for row in summary_table:
        project = clean_text(row.get("project", ""))
        if not project:
            continue

        existing = merged.get(project, {})
        combined = dict(existing)

        for key, value in row.items():
            value = clean_text(value)
            if value:
                combined[key] = value

        combined["project"] = project
        merged[project] = combined

    table_rows = list(merged.values())

    table_rows.sort(
        key=lambda r: (
            clean_text(r.get("project", "")),
            -safe_int(r.get("C@100")),
            -safe_int(r.get("C@50")),
            -safe_int(r.get("C@20")),
            -safe_int(r.get("C@10")),
        )
    )

    return table_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print one table per ranking strategy, containing all projects."
    )

    parser.add_argument(
        "--projects-dir",
        default=str(DEFAULT_PROJECTS_DIR),
        help="Path to Projects directory.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save one CSV table per ranking strategy.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    rankings_dir = projects_dir / RANKINGS_DIR_NAME

    fields = [
        "project",
        "status",
        "ranked",
        "refs_total",
        "refs_doi",
        "C@10",
        "C@20",
        "C@50",
        "C@100",
        "file",
    ]

    print("=" * 140)
    print("ONE TABLE PER RANKING STRATEGY")
    print("=" * 140)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Rankings directory : {rankings_dir.resolve()}")
    print(f"Save CSV outputs   : {args.save}")
    print("=" * 140)

    any_found = False

    for strategy_id in range(1, 12):
        path = summary_path(projects_dir, strategy_id)
        rows = read_csv_rows(path)

        title = STRATEGY_NAMES.get(strategy_id, f"Strategy{strategy_id}")

        print("\n" + "=" * 140)
        print(f"RANK STRATEGY {strategy_id} - {title}")
        print("=" * 140)
        print(f"Summary file: {path}")

        if not rows:
            print("Missing or empty summary file; checking per-project rank files.")

        summary_table = build_strategy_table(rows, strategy_id)
        rank_file_table = build_rank_file_table(projects_dir, strategy_id)
        table_rows = merge_summary_and_rank_file_rows(summary_table, rank_file_table)

        if not table_rows:
            print("No summary rows or per-project rank files found.")
            continue

        any_found = True
        print_table(table_rows, fields)

        total_ranked = sum(safe_int(r.get("ranked")) for r in table_rows)
        total_refs_doi = sum(safe_int(r.get("refs_doi")) for r in table_rows)
        total_c10 = sum(safe_int(r.get("C@10")) for r in table_rows)
        total_c20 = sum(safe_int(r.get("C@20")) for r in table_rows)
        total_c50 = sum(safe_int(r.get("C@50")) for r in table_rows)
        total_c100 = sum(safe_int(r.get("C@100")) for r in table_rows)

        print("-" * 140)
        print(
            f"TOTAL | projects={len(table_rows)} | "
            f"ranked={total_ranked} | refs_doi={total_refs_doi} | "
            f"C@10={total_c10} | C@20={total_c20} | "
            f"C@50={total_c50} | C@100={total_c100}"
        )

        if args.save:
            out_path = rankings_dir / f"rank_strategy_{strategy_id}_project_table.csv"
            write_csv(out_path, table_rows, fields)
            print(f"Saved: {out_path}")

    if not any_found:
        print("\nNo ranking summary files found. Run Rank1-Rank11 first.")


if __name__ == "__main__":
    main()
