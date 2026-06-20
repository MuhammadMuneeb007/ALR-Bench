# Step6.2-ValidateConfigurationKeywordHits.py
#
# Purpose:
#   Validate each generated config.json by counting how many configured
#   search/screening/full-text/quality terms actually appear in the article text.
#
# Input per project:
#   Projects/<project>/config.json
#   Projects/<project>/article_text.txt
#   Optional fallbacks:
#     article_markdown.md
#     full_text.txt
#     section_files/all_sections_combined.txt
#
# Output:
#   Prints one clean table to terminal.
#   Does NOT save extra files by default.
#
# Usage:
#   python Step6.2-ValidateConfigurationKeywordHits.py
#   python Step6.2-ValidateConfigurationKeywordHits.py --projects-dir Projects
#   python Step6.2-ValidateConfigurationKeywordHits.py --project your_project_name
#   python Step6.2-ValidateConfigurationKeywordHits.py --csv results.csv

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECTS_DIR = Path("Projects")


# ============================================================
# BASIC HELPERS
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[_\-\/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_term(term: Any) -> str:
    return normalize_text(str(term or ""))


def unique_terms(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        term = str(item or "").strip()
        if not term:
            continue

        key = normalize_term(term)
        if not key:
            continue

        if key not in seen:
            out.append(term)
            seen.add(key)

    return out


def flatten_list_value(value: Any) -> List[str]:
    """
    Converts config values into a clean list of terms.

    Handles:
      list[str]
      list[dict]
      dict values
      string
    """
    terms: List[str] = []

    if value is None:
        return terms

    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    terms.append(item)
            elif isinstance(item, dict):
                # Useful for older structures such as:
                # {"database": "PubMed", "query": "..."}
                for v in item.values():
                    terms.extend(flatten_list_value(v))
            else:
                if str(item).strip():
                    terms.append(str(item))
        return unique_terms(terms)

    if isinstance(value, dict):
        for v in value.values():
            terms.extend(flatten_list_value(v))
        return unique_terms(terms)

    if str(value).strip():
        return [str(value)]

    return terms


def count_hits(article_text_norm: str, terms: List[str]) -> Tuple[int, int, List[str], List[str]]:
    """
    Returns:
      n_terms
      n_terms_found
      found_terms
      missing_terms

    Matching is normalized phrase matching.
    """
    found: List[str] = []
    missing: List[str] = []

    for term in unique_terms(terms):
        term_norm = normalize_term(term)
        if not term_norm:
            continue

        # For short acronyms like LLM, AI, use word boundary.
        # For multi-word phrases, normalized substring is fine.
        if len(term_norm.split()) == 1:
            pattern = rf"\b{re.escape(term_norm)}\b"
            is_hit = re.search(pattern, article_text_norm) is not None
        else:
            is_hit = term_norm in article_text_norm

        if is_hit:
            found.append(term)
        else:
            missing.append(term)

    return len(found) + len(missing), len(found), found, missing


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


# ============================================================
# PROJECT DISCOVERY
# ============================================================

def discover_projects(projects_dir: Path, one_project: str = "") -> List[Path]:
    if one_project:
        p = Path(one_project)
        if not p.is_absolute():
            p = projects_dir / one_project
        return [p]

    if not projects_dir.exists():
        return []

    return [p for p in sorted(projects_dir.iterdir()) if p.is_dir()]


def find_article_text_path(project_dir: Path) -> Path | None:
    candidates = [
        "article_text.txt",
        "article_markdown.md",
        "full_text.txt",
        "Article_Text.txt",
        "COMBINED.txt",
        "section_files/all_sections_combined.txt",
    ]

    for rel in candidates:
        p = project_dir / rel
        if p.exists() and p.is_file():
            return p

    return None


# ============================================================
# CONFIG TERM GROUPS
# ============================================================

TERM_GROUPS = {
    # High-level concept/search terms
    "concepts": [
        "concepts_keywords",
    ],

    # Reported PRISMA title/abstract screening terms
    "prisma_title_abstract": [
        "prisma_title_abstract_include_keywords",
        "prisma_title_abstract_exclude_keywords",
        "prisma_title_abstract_must_have_keywords",
        "prisma_title_abstract_uncertain_keywords",
    ],

    # Reported PRISMA full-text screening terms
    "prisma_full_text": [
        "prisma_full_text_include_keywords",
        "prisma_full_text_exclude_keywords",
        "prisma_full_text_must_have_keywords",
        "prisma_full_text_uncertain_keywords",
    ],

    # Reported PRISMA-S search terms
    "prisma_s_search": [
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "prisma_s_search_keywords",
        "prisma_s_controlled_vocabulary_terms",
        "prisma_s_free_text_terms",
        "prisma_s_boolean_operators_used",
    ],

    # Reported quality assessment terms/questions
    "prisma_quality": [
        "prisma_quality_assessment_keywords",
        "prisma_quality_assessment_questions",
    ],

    # Generated search strategy
    "generated_search": [
        "generated_search_queries",
        "generated_search_keywords",
    ],

    # Generated abstract screening terms
    "generated_abstract": [
        "generated_abstract_include_keywords",
        "generated_abstract_exclude_keywords",
    ],

    # Generated full-text screening terms
    "generated_full_text": [
        "generated_full_text_include_keywords",
        "generated_full_text_exclude_keywords",
    ],

    # Generated quality assessment terms/questions
    "generated_quality": [
        "generated_quality_assessment_keywords",
        "generated_quality_assessment_questions",
    ],
}


def get_terms_for_group(config: Dict[str, Any], keys: List[str]) -> List[str]:
    terms: List[str] = []
    for key in keys:
        terms.extend(flatten_list_value(config.get(key, [])))
    return unique_terms(terms)


# ============================================================
# VALIDATION
# ============================================================

def validate_project(project_dir: Path) -> Dict[str, Any]:
    config_path = project_dir / "config.json"
    article_path = find_article_text_path(project_dir)

    row: Dict[str, Any] = {
        "project": project_dir.name,
        "config_found": config_path.exists(),
        "article_text_found": article_path is not None,
        "article_words": 0,
    }

    if not config_path.exists():
        for group in TERM_GROUPS:
            row[f"{group}_found"] = 0
            row[f"{group}_total"] = 0
            row[f"{group}_pct"] = "0.0%"
        row["overall_found"] = 0
        row["overall_total"] = 0
        row["overall_pct"] = "0.0%"
        return row

    config = read_json(config_path)

    article_text = read_text(article_path) if article_path else ""
    article_norm = normalize_text(article_text)
    row["article_words"] = len(article_norm.split()) if article_norm else 0

    overall_found = 0
    overall_total = 0

    for group, keys in TERM_GROUPS.items():
        terms = get_terms_for_group(config, keys)
        n_total, n_found, found_terms, missing_terms = count_hits(article_norm, terms)

        row[f"{group}_found"] = n_found
        row[f"{group}_total"] = n_total
        row[f"{group}_pct"] = pct(n_found, n_total)

        overall_found += n_found
        overall_total += n_total

    row["overall_found"] = overall_found
    row["overall_total"] = overall_total
    row["overall_pct"] = pct(overall_found, overall_total)

    return row


# ============================================================
# TABLE PRINTING
# ============================================================

DISPLAY_COLUMNS = [
    "project",
    "article_words",
    "concepts",
    "prisma_s_search",
    "prisma_title_abstract",
    "prisma_full_text",
    "prisma_quality",
    "generated_search",
    "generated_abstract",
    "generated_full_text",
    "generated_quality",
    "overall",
]


def compact_row(row: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {
        "project": str(row.get("project", "")),
        "article_words": str(row.get("article_words", 0)),
    }

    for group in DISPLAY_COLUMNS:
        if group in {"project", "article_words"}:
            continue
        out[group] = (
            f"{row.get(group + '_found', 0)}/"
            f"{row.get(group + '_total', 0)} "
            f"({row.get(group + '_pct', '0.0%')})"
        )

    return out


def print_table(rows: List[Dict[str, Any]]) -> None:
    compact = [compact_row(r) for r in rows]

    widths = {}
    for col in DISPLAY_COLUMNS:
        widths[col] = max(
            len(col),
            *(len(str(r.get(col, ""))) for r in compact),
        )

    header = " | ".join(col.ljust(widths[col]) for col in DISPLAY_COLUMNS)
    sep = "-+-".join("-" * widths[col] for col in DISPLAY_COLUMNS)

    print(header)
    print(sep)

    for r in compact:
        print(" | ".join(str(r.get(col, "")).ljust(widths[col]) for col in DISPLAY_COLUMNS))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    compact = [compact_row(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DISPLAY_COLUMNS)
        writer.writeheader()
        writer.writerows(compact)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count how many config keywords/search/screening/full-text terms appear in each article text."
    )
    parser.add_argument(
        "--projects-dir",
        default=str(PROJECTS_DIR),
        help="Path to Projects directory.",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Optional single project folder name or path.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV path. If omitted, no file is saved.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    rows = [validate_project(p) for p in projects]
    print_table(rows)

    if args.csv:
        write_csv(Path(args.csv), rows)


if __name__ == "__main__":
    main()