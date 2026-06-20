# Step7D-CleanAndCheckCommonReferences.py
#
# Purpose:
#   Clean retrieved article metadata and then print common references.
#
# What this script does:
#   1. Reads:
#        Projects/<project>/Search/articles.csv
#
#   2. Cleans retrieved articles:
#        - removes exact duplicate rows
#        - removes rows with missing DOI
#        - removes duplicate DOI
#        - removes duplicate title
#
#   3. Saves:
#        Projects/<project>/Search/articles_clean_doi_only.csv
#
#   4. Reads reference files:
#        Projects/<project>/references.json
#        Projects/<project>/references_resolved.json
#        Projects/<project>/references_gold.json
#
#   5. Prints one clean table:
#        project
#        article_rows
#        missing_doi_removed
#        duplicate_rows_removed
#        duplicate_doi_removed
#        duplicate_title_removed
#        clean_articles
#        references_total
#        references_with_doi
#        common_doi
#        common_title
#        common_total
#        references_source
#
# What this script does NOT do:
#   - Does NOT query OpenAlex
#   - Does NOT calculate retrievable ceiling
#   - Does NOT download PDFs
#   - Does NOT download full text
#   - Does NOT rank articles
#
# Usage:
#   python Step7D-CleanAndCheckCommonReferences.py --projects-dir Projects --force
#
# One project:
#   python Step7D-CleanAndCheckCommonReferences.py --projects-dir Projects --project PROJECT_NAME --force

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")
DEFAULT_ARTICLES_REL = Path("Search") / "articles.csv"
DEFAULT_CLEAN_ARTICLES_REL = Path("Search") / "articles_clean_doi_only.csv"


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    text = str(x)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

    # Rescue DOI if embedded in messy text.
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", doi, flags=re.I)
    if m:
        doi = m.group(1)

    doi = doi.strip()
    doi = re.sub(r"\s+", "", doi)
    doi = doi.strip().strip(".,;:()[]{}<>")

    if doi in {"", "none", "null", "nan", "na", "n/a"}:
        return ""

    return doi


def normalize_title(x: Any) -> str:
    title = clean_text(x).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def normalize_row(row: Dict[str, Any]) -> str:
    parts = []
    for key in sorted(row):
        parts.append(f"{key}={clean_text(row.get(key, ''))}")
    return "||".join(parts)


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fieldnames
            })


def discover_projects(projects_dir: Path, one_project: str = "") -> List[Path]:
    if one_project:
        p = Path(one_project)
        if not p.is_absolute():
            p = projects_dir / one_project

        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Project folder not found: {p}")

        return [p]

    if not projects_dir.exists():
        raise FileNotFoundError(f"Projects directory not found: {projects_dir.resolve()}")

    skip = {"search", "evaluation", "__pycache__"}

    return [
        p for p in sorted(projects_dir.iterdir())
        if p.is_dir() and p.name.lower() not in skip
    ]


def parse_year(x: Any) -> int:
    text = clean_text(x)
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else 0


def safe_div(num: int, den: int) -> float:
    if not den:
        return 0.0
    return round(num / den, 6)


# ============================================================
# ARTICLE CLEANING
# ============================================================

def score_article_row(row: Dict[str, str]) -> int:
    score = 0
    score += len(clean_text(row.get("abstract", "")))
    score += 100 if normalize_doi(row.get("doi", "")) else 0
    score += 30 if clean_text(row.get("title", "")) else 0
    score += 25 if clean_text(row.get("url", "")) else 0
    score += 15 if clean_text(row.get("authors", "")) else 0
    score += 15 if clean_text(row.get("journal", "")) else 0
    score += 10 if parse_year(row.get("year", "")) else 0
    score += 5 if clean_text(row.get("pmid", "")) else 0
    score += 5 if clean_text(row.get("pmcid", "")) else 0
    return score


def clean_article_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Cleaning order:
      1. exact duplicate rows
      2. missing DOI
      3. duplicate DOI
      4. duplicate title

    Duplicate DOI/title keeps the row with richer metadata.
    """

    stats = {
        "article_rows": len(rows),
        "duplicate_rows_removed": 0,
        "missing_doi_removed": 0,
        "duplicate_doi_removed": 0,
        "duplicate_title_removed": 0,
        "rows_with_doi_before_dedup": 0,
        "clean_articles": 0,
    }

    # 1. Remove exact duplicate rows.
    seen_rows: Set[str] = set()
    no_duplicate_rows: List[Dict[str, str]] = []

    for row in rows:
        row_key = normalize_row(row)

        if row_key in seen_rows:
            stats["duplicate_rows_removed"] += 1
            continue

        seen_rows.add(row_key)
        no_duplicate_rows.append(row)

    # 2. Remove rows with missing DOI.
    doi_rows: List[Dict[str, str]] = []

    for row in no_duplicate_rows:
        doi = normalize_doi(row.get("doi", ""))

        if not doi:
            stats["missing_doi_removed"] += 1
            continue

        row2 = dict(row)
        row2["doi"] = doi
        doi_rows.append(row2)

    stats["rows_with_doi_before_dedup"] = len(doi_rows)

    # 3. Remove duplicate DOI.
    best_by_doi: Dict[str, Dict[str, str]] = {}

    for row in doi_rows:
        doi = normalize_doi(row.get("doi", ""))

        if doi not in best_by_doi:
            best_by_doi[doi] = row
        else:
            stats["duplicate_doi_removed"] += 1

            if score_article_row(row) > score_article_row(best_by_doi[doi]):
                best_by_doi[doi] = row

    doi_unique_rows = list(best_by_doi.values())

    # 4. Remove duplicate title.
    best_by_title: Dict[str, Dict[str, str]] = {}
    rows_without_title: List[Dict[str, str]] = []

    for row in doi_unique_rows:
        title_key = normalize_title(row.get("title", ""))

        if not title_key:
            rows_without_title.append(row)
            continue

        if title_key not in best_by_title:
            best_by_title[title_key] = row
        else:
            stats["duplicate_title_removed"] += 1

            if score_article_row(row) > score_article_row(best_by_title[title_key]):
                best_by_title[title_key] = row

    clean_rows = list(best_by_title.values()) + rows_without_title

    clean_rows = sorted(
        clean_rows,
        key=lambda r: (
            parse_year(r.get("year", "")),
            clean_text(r.get("title", "")),
            normalize_doi(r.get("doi", "")),
        ),
        reverse=True,
    )

    stats["clean_articles"] = len(clean_rows)

    return clean_rows, stats


# ============================================================
# REFERENCE EXTRACTION
# ============================================================

def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
    """
    Flexible extractor for references.json / references_resolved.json / references_gold.json.
    Extracts DOI and title when available.
    """

    records: List[Dict[str, str]] = []

    doi_keys = {
        "doi",
        "DOI",
        "digital_object_identifier",
        "reference_doi",
        "work_doi",
    }

    title_keys = {
        "title",
        "article_title",
        "display_name",
        "work_title",
        "reference_title",
    }

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            doi = ""
            title = ""

            for k, v in obj.items():
                if k in doi_keys and not doi:
                    doi = clean_text(v)

                if k in title_keys and not title:
                    title = clean_text(v)

            if doi or title:
                records.append({
                    "doi": normalize_doi(doi),
                    "title": clean_text(title),
                })

            # Handle direct DOI list fields.
            for k, v in obj.items():
                if k.lower() in {"dois", "doi_list", "referenced_dois"} and isinstance(v, list):
                    for item in v:
                        d = normalize_doi(item)
                        if d:
                            records.append({
                                "doi": d,
                                "title": "",
                            })

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    # Dedupe references by DOI first, then title.
    out: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for rec in records:
        doi = normalize_doi(rec.get("doi", ""))
        title_key = normalize_title(rec.get("title", ""))

        if doi:
            key = f"doi::{doi}"
        elif title_key:
            key = f"title::{title_key}"
        else:
            continue

        if key in seen:
            continue

        seen.add(key)

        out.append({
            "doi": doi,
            "title": clean_text(rec.get("title", "")),
        })

    return out


def load_reference_records(project_dir: Path) -> Tuple[List[Dict[str, str]], str]:
    candidates = [
        "references.json",
        "references_resolved.json",
        "references_gold.json",
    ]

    for rel in candidates:
        path = project_dir / rel

        if not path.exists():
            continue

        data = read_json(path, default={})
        records = extract_reference_records_from_json(data)

        if records:
            return records, rel

    return [], ""


def collect_reference_sets(records: List[Dict[str, str]]) -> Tuple[Set[str], Set[str], Set[str]]:
    keys: Set[str] = set()
    dois: Set[str] = set()
    titles: Set[str] = set()

    for rec in records:
        doi = normalize_doi(rec.get("doi", ""))
        title = normalize_title(rec.get("title", ""))

        if doi:
            dois.add(doi)
            keys.add(f"doi::{doi}")
        elif title:
            keys.add(f"title::{title}")

        if title:
            titles.add(title)

    return keys, dois, titles


def collect_article_sets(rows: List[Dict[str, str]]) -> Tuple[Set[str], Set[str], Set[str]]:
    keys: Set[str] = set()
    dois: Set[str] = set()
    titles: Set[str] = set()

    for row in rows:
        doi = normalize_doi(row.get("doi", ""))
        title = normalize_title(row.get("title", ""))

        if doi:
            dois.add(doi)
            keys.add(f"doi::{doi}")

        if title:
            titles.add(title)

    return keys, dois, titles


# ============================================================
# PROJECT CHECK
# ============================================================

def check_project(
    project_dir: Path,
    articles_rel: Path,
    clean_articles_rel: Path,
    force: bool,
) -> Dict[str, Any]:

    articles_path = project_dir / articles_rel
    clean_articles_path = project_dir / clean_articles_rel

    article_rows = read_csv_rows(articles_path)
    clean_rows, clean_stats = clean_article_rows(article_rows)

    # Save clean file.
    if clean_rows:
        fieldnames = list(clean_rows[0].keys())
    else:
        # Default fields if empty.
        fieldnames = [
            "title",
            "abstract",
            "year",
            "journal",
            "doi",
            "authors",
            "url",
            "article_type",
            "source_database",
            "source_query",
            "query_origin",
            "openalex_id",
            "pmid",
            "pmcid",
        ]

    if force or not clean_articles_path.exists():
        write_csv_rows(clean_articles_path, clean_rows, fieldnames)

    article_keys, article_dois, article_titles = collect_article_sets(clean_rows)

    ref_records, ref_source = load_reference_records(project_dir)
    ref_keys, ref_dois, ref_titles = collect_reference_sets(ref_records)

    common_dois = article_dois & ref_dois
    common_titles = article_titles & ref_titles

    # Avoid double-counting title matches for records already matched by DOI is hard
    # without record-level mapping, so we report both separately and combined as simple sum.
    common_total = len(common_dois) + len(common_titles)

    references_total = len(ref_records)
    references_with_doi = len(ref_dois)

    return {
        "project": project_dir.name,
        "articles_file_found": articles_path.exists(),
        "clean_articles_file": str(clean_articles_path),

        "article_rows": clean_stats["article_rows"],
        "duplicate_rows_removed": clean_stats["duplicate_rows_removed"],
        "missing_doi_removed": clean_stats["missing_doi_removed"],
        "rows_with_doi_before_dedup": clean_stats["rows_with_doi_before_dedup"],
        "duplicate_doi_removed": clean_stats["duplicate_doi_removed"],
        "duplicate_title_removed": clean_stats["duplicate_title_removed"],
        "clean_articles": clean_stats["clean_articles"],

        "references_total": references_total,
        "references_with_doi": references_with_doi,

        "common_doi": len(common_dois),
        "common_title": len(common_titles),
        "common_total": common_total,

        "raw_doi_recall_after_cleaning": safe_div(len(common_dois), references_with_doi),

        "references_source": ref_source,
    }


# ============================================================
# PRINTING
# ============================================================

COLUMNS = [
    "project",
    "article_rows",
    "missing_doi_removed",
    "duplicate_rows_removed",
    "duplicate_doi_removed",
    "duplicate_title_removed",
    "clean_articles",
    "references_total",
    "references_with_doi",
    "common_doi",
    "common_title",
    "common_total",
    "raw_doi_recall_after_cleaning",
    "references_source",
]


def print_table(rows: List[Dict[str, Any]]) -> None:
    compact_rows = []

    for row in rows:
        compact_rows.append({
            col: str(row.get(col, ""))
            for col in COLUMNS
        })

    widths: Dict[str, int] = {}

    for col in COLUMNS:
        widths[col] = max(
            len(col),
            *(len(row.get(col, "")) for row in compact_rows),
        )

    header = " | ".join(col.ljust(widths[col]) for col in COLUMNS)
    sep = "-+-".join("-" * widths[col] for col in COLUMNS)

    print(header)
    print(sep)

    for row in compact_rows:
        print(" | ".join(row.get(col, "").ljust(widths[col]) for col in COLUMNS))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean retrieved articles by DOI/title/row duplicates and print common references."
    )

    parser.add_argument(
        "--projects-dir",
        default=str(DEFAULT_PROJECTS_DIR),
        help="Path to Projects directory.",
    )

    parser.add_argument(
        "--project",
        default="",
        help="Optional single project folder name or path.",
    )

    parser.add_argument(
        "--articles-file",
        default=str(DEFAULT_ARTICLES_REL),
        help="Relative path to retrieved articles CSV inside each project.",
    )

    parser.add_argument(
        "--clean-articles-file",
        default=str(DEFAULT_CLEAN_ARTICLES_REL),
        help="Relative path for cleaned DOI-only articles CSV inside each project.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cleaned article files.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    articles_rel = Path(args.articles_file)
    clean_articles_rel = Path(args.clean_articles_file)

    projects = discover_projects(projects_dir, args.project)

    print("=" * 120)
    print("STEP 7D: CLEAN RETRIEVED ARTICLES + CHECK COMMON REFERENCES")
    print("=" * 120)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Articles file      : {articles_rel}")
    print(f"Clean output file  : {clean_articles_rel}")
    print(f"Force overwrite    : {args.force}")
    print("=" * 120)

    rows = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")
        row = check_project(
            project_dir=project_dir,
            articles_rel=articles_rel,
            clean_articles_rel=clean_articles_rel,
            force=args.force,
        )
        rows.append(row)

        print(
            f"OK | rows={row['article_rows']} | "
            f"missing_doi_removed={row['missing_doi_removed']} | "
            f"dup_rows={row['duplicate_rows_removed']} | "
            f"dup_doi={row['duplicate_doi_removed']} | "
            f"dup_title={row['duplicate_title_removed']} | "
            f"clean={row['clean_articles']} | "
            f"refs={row['references_total']} | "
            f"refs_doi={row['references_with_doi']} | "
            f"common_doi={row['common_doi']} | "
            f"common_title={row['common_title']} | "
            f"common_total={row['common_total']}"
        )

    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print_table(rows)


if __name__ == "__main__":
    main()