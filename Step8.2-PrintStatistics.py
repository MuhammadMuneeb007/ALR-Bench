"""
Step8.2-PrintStatistics.py

Print Step7 search statistics so we can see:
  1. How many queries were formed for each project.
  2. How many article records were downloaded in total.
  3. Which databases returned the most records.
  4. Which query/database combinations worked best.
  5. Which configuration fields produced the strongest queries.

Inputs:
  Projects/<project>/Search/query_log.csv
  Projects/<project>/Search/articles.csv
  Projects/Search/step7_search_run_summary.csv       optional

Outputs:
  Terminal tables
  Projects/Search/Statistics/project_summary.csv
  Projects/Search/Statistics/project_database_summary.csv
  Projects/Search/Statistics/query_performance.csv
  Projects/Search/Statistics/database_summary.csv
  Projects/Search/Statistics/config_field_summary.csv
  Projects/Search/Statistics/project_config_field_summary.csv
  Projects/Search/Statistics/config_field_database_summary.csv

Usage:
  python Step8.2-PrintStatistics.py
  python Step8.2-PrintStatistics.py --top-queries 15
  python Step8.2-PrintStatistics.py --project deepfake_detection_a_systematic_literature_review
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PROJECTS_DIR = Path("Projects")
SUMMARY_DIR = Path("Search") / "Statistics"
SKIP_PROJECT_DIRS = {"Search", "Rankings", "Evaluation", "__pycache__"}

QUERY_ORIGIN_DESCRIPTIONS = {
    "generated_search_queries": "LLM/generated topic-level query phrases from the review configuration.",
    "search_keywords": "Individual keywords and short keyword combinations from the configuration.",
    "generated_search_keywords": "LLM/generated search keywords from the configuration.",
    "concepts_keywords": "Concept terms extracted from the review topic/configuration.",
    "prisma_s_search_keywords": "PRISMA search keywords reported or reconstructed from the review.",
    "prisma_s_full_search_strings": "Full Boolean PRISMA-style search strings.",
    "include_keywords": "Terms derived from inclusion criteria.",
    "must_keywords": "Required/must-have terms from the configuration.",
    "required_terms": "Topic required terms used as search/filter signals.",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(value: Any) -> str:
    doi = clean_text(value).lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip().strip(".")


def normalize_key(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def article_unique_key(row: Dict[str, str]) -> str:
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"

    openalex_id = clean_text(row.get("openalex_id", ""))
    if openalex_id:
        return f"openalex:{openalex_id.lower()}"

    pmid = clean_text(row.get("pmid", ""))
    if pmid:
        return f"pmid:{pmid}"

    semantic_scholar_id = clean_text(row.get("semantic_scholar_id", ""))
    if semantic_scholar_id:
        return f"semanticscholar:{semantic_scholar_id.lower()}"

    title = normalize_key(row.get("title", ""))
    year = clean_text(row.get("year", ""))
    return f"title:{title}|year:{year}"


def query_key_from_log(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        clean_text(row.get("query_origin", "")),
        clean_text(row.get("query", "")),
        clean_text(row.get("rendered_query", "")),
    )


def query_key_from_article(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        clean_text(row.get("query_origin", "")),
        clean_text(row.get("source_query", "")),
        clean_text(row.get("rendered_query", "")),
    )


def split_semicolon_values(value: Any, default: str = "unknown") -> List[str]:
    """Articles can carry merged values after dedup, e.g. 'a; b; b'."""
    text = clean_text(value)
    if not text:
        return [default]

    values: List[str] = []
    seen = set()
    for part in re.split(r"\s*;\s*", text):
        item = clean_text(part) or default
        if item not in seen:
            values.append(item)
            seen.add(item)
    return values or [default]


def split_query_origins(value: Any) -> List[str]:
    return split_semicolon_values(value, default="unknown")


def split_databases(value: Any) -> List[str]:
    return split_semicolon_values(value, default="unknown")


def origin_description(origin: str) -> str:
    return QUERY_ORIGIN_DESCRIPTIONS.get(
        clean_text(origin),
        "Configuration-derived query field; see config.json/query_log.csv for exact source.",
    )


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0"
    return f"{(100.0 * numerator / denominator):.1f}"


def config_field_recommendation(row: Dict[str, Any]) -> str:
    origin = clean_text(row.get("config_field", ""))
    productive = float(clean_text(row.get("productive_run_pct", "0")) or 0)
    zero = float(clean_text(row.get("zero_result_run_pct", "0")) or 0)
    unique_articles = to_int(row.get("unique_articles_credit", 0))

    if origin == "search_keywords":
        return "Keep as broad recall engine; prune very generic one-word terms after ranking."
    if origin == "generated_search_queries":
        return "Keep; efficient compact queries, good candidate for default first-pass search."
    if origin == "prisma_s_full_search_strings":
        return "Use as precision support; useful but narrower and more zero-result prone."
    if "prisma" in origin:
        return "Use selectively; verify syntax/database translation because many runs are empty."
    if productive >= 75 and unique_articles >= 1000:
        return "Keep; strong retrieval signal."
    if zero >= 50:
        return "Revise or downweight; too many zero-result runs."
    return "Keep for now; inspect project-level table before pruning."


def build_readable_config_field_summary(config_field_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(config_field_rows, start=1):
        rows.append({
            "rank": idx,
            "config_field": row.get("config_field", ""),
            "what_it_means": row.get("field_meaning", ""),
            "projects": row.get("projects", ""),
            "query_runs": row.get("query_runs", ""),
            "productive_%": row.get("productive_run_pct", ""),
            "zero_%": row.get("zero_result_run_pct", ""),
            "returned": row.get("total_returned", ""),
            "unique_article_credit": row.get("unique_articles_credit", ""),
            "articles_per_query": row.get("mean_unique_articles_per_query", ""),
            "best_database": row.get("best_database", ""),
            "best_query": row.get("best_query", ""),
            "recommendation": config_field_recommendation(row),
        })
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_int(value: Any) -> int:
    try:
        return int(float(clean_text(value) or "0"))
    except Exception:
        return 0


def ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0"
    return f"{numerator / denominator:.1f}"


def truncate(value: Any, width: int) -> str:
    text = clean_text(value)
    if width <= 3 or len(text) <= width:
        return text
    return text[: width - 3] + "..."


def print_table(title: str, rows: List[Dict[str, Any]], fields: List[str], max_rows: int = 25) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    if not rows:
        print("(no rows)")
        return

    display_rows = rows[:max_rows]
    widths: Dict[str, int] = {}
    for field in fields:
        max_cell = max(len(truncate(row.get(field, ""), 60)) for row in display_rows)
        widths[field] = min(max(len(field), max_cell), 60)

    header = " | ".join(field.ljust(widths[field]) for field in fields)
    sep = "-+-".join("-" * widths[field] for field in fields)
    print(header)
    print(sep)
    for row in display_rows:
        print(" | ".join(truncate(row.get(field, ""), widths[field]).ljust(widths[field]) for field in fields))

    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows not shown")


def discover_projects(projects_dir: Path, project: str = "") -> List[Path]:
    if project:
        target = projects_dir / project
        return [target] if target.exists() else []
    return sorted(
        p for p in projects_dir.iterdir()
        if p.is_dir() and p.name not in SKIP_PROJECT_DIRS and (p / "Search").exists()
    )


def collect_project(project_dir: Path) -> Dict[str, Any]:
    search_dir = project_dir / "Search"
    query_log_rows = read_csv(search_dir / "query_log.csv")
    article_rows = read_csv(search_dir / "articles.csv")

    query_keys = {query_key_from_log(row) for row in query_log_rows}
    databases = {clean_text(row.get("database", "")) for row in query_log_rows if clean_text(row.get("database", ""))}
    successful_runs = sum(1 for row in query_log_rows if clean_text(row.get("status", "")).lower() in {"searched", "success", "cached"})
    failed_runs = sum(1 for row in query_log_rows if clean_text(row.get("status", "")).lower() in {"failed", "error"} or clean_text(row.get("error", "")))
    zero_runs = sum(1 for row in query_log_rows if to_int(row.get("n_returned", 0)) == 0)
    total_returned_log = sum(to_int(row.get("n_returned", 0)) for row in query_log_rows)
    unique_articles = {article_unique_key(row) for row in article_rows if article_unique_key(row)}
    articles_with_doi = sum(1 for row in article_rows if normalize_doi(row.get("doi", "")))

    return {
        "project": project_dir.name,
        "query_log_rows": query_log_rows,
        "article_rows": article_rows,
        "summary": {
            "project": project_dir.name,
            "unique_queries_formed": len(query_keys),
            "query_database_runs": len(query_log_rows),
            "databases_used": len(databases),
            "successful_query_runs": successful_runs,
            "failed_query_runs": failed_runs,
            "zero_result_query_runs": zero_runs,
            "total_returned_before_dedup": total_returned_log,
            "article_rows_downloaded": len(article_rows),
            "unique_articles_downloaded": len(unique_articles),
            "articles_with_doi": articles_with_doi,
            "mean_returned_per_query_run": ratio(total_returned_log, len(query_log_rows)),
        },
    }


def build_project_database_summary(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    article_groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    for project in projects:
        project_name = project["project"]
        for row in project["article_rows"]:
            for database in split_databases(row.get("source_database", "unknown")):
                article_groups[(project_name, database)].append(row)

        for row in project["query_log_rows"]:
            database = clean_text(row.get("database", "unknown")) or "unknown"
            key = (project_name, database)
            group = groups.setdefault(key, {
                "project": project_name,
                "database": database,
                "query_runs": 0,
                "unique_queries": set(),
                "successful_runs": 0,
                "failed_runs": 0,
                "zero_result_runs": 0,
                "total_returned": 0,
                "max_returned_single_query": 0,
                "best_query": "",
            })

            n_returned = to_int(row.get("n_returned", 0))
            group["query_runs"] += 1
            group["unique_queries"].add(query_key_from_log(row))
            group["total_returned"] += n_returned
            if n_returned == 0:
                group["zero_result_runs"] += 1
            if clean_text(row.get("status", "")).lower() in {"failed", "error"} or clean_text(row.get("error", "")):
                group["failed_runs"] += 1
            else:
                group["successful_runs"] += 1
            if n_returned > group["max_returned_single_query"]:
                group["max_returned_single_query"] = n_returned
                group["best_query"] = clean_text(row.get("rendered_query", "")) or clean_text(row.get("query", ""))

    rows: List[Dict[str, Any]] = []
    for key, group in groups.items():
        articles = article_groups.get(key, [])
        unique_articles = {article_unique_key(row) for row in articles if article_unique_key(row)}
        rows.append({
            "project": group["project"],
            "database": group["database"],
            "query_runs": group["query_runs"],
            "unique_queries": len(group["unique_queries"]),
            "successful_runs": group["successful_runs"],
            "failed_runs": group["failed_runs"],
            "zero_result_runs": group["zero_result_runs"],
            "total_returned": group["total_returned"],
            "article_rows": len(articles),
            "unique_articles": len(unique_articles),
            "mean_returned_per_run": ratio(group["total_returned"], group["query_runs"]),
            "max_returned_single_query": group["max_returned_single_query"],
            "best_query": group["best_query"],
        })

    return sorted(rows, key=lambda r: (r["project"], -int(r["total_returned"]), r["database"]))


def build_query_performance(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    article_groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)

    for project in projects:
        project_name = project["project"]
        for row in project["article_rows"]:
            query_origin, query, rendered_query = query_key_from_article(row)
            key = (project_name, query_origin, query, rendered_query)
            article_groups[key].append(row)

        for row in project["query_log_rows"]:
            query_origin, query, rendered_query = query_key_from_log(row)
            key = (project_name, query_origin, query, rendered_query)
            group = groups.setdefault(key, {
                "project": project_name,
                "query_origin": query_origin,
                "query": query,
                "rendered_query": rendered_query,
                "databases_tried": set(),
                "successful_runs": 0,
                "failed_runs": 0,
                "zero_result_runs": 0,
                "total_returned": 0,
                "best_database": "",
                "best_database_returned": 0,
            })

            database = clean_text(row.get("database", "unknown")) or "unknown"
            n_returned = to_int(row.get("n_returned", 0))
            group["databases_tried"].add(database)
            group["total_returned"] += n_returned
            if n_returned == 0:
                group["zero_result_runs"] += 1
            if clean_text(row.get("status", "")).lower() in {"failed", "error"} or clean_text(row.get("error", "")):
                group["failed_runs"] += 1
            else:
                group["successful_runs"] += 1
            if n_returned > group["best_database_returned"]:
                group["best_database"] = database
                group["best_database_returned"] = n_returned

    rows: List[Dict[str, Any]] = []
    for key, group in groups.items():
        articles = article_groups.get(key, [])
        unique_articles = {article_unique_key(row) for row in articles if article_unique_key(row)}
        rows.append({
            "project": group["project"],
            "query_origin": group["query_origin"],
            "query": group["query"],
            "rendered_query": group["rendered_query"],
            "databases_tried": len(group["databases_tried"]),
            "successful_runs": group["successful_runs"],
            "failed_runs": group["failed_runs"],
            "zero_result_runs": group["zero_result_runs"],
            "total_returned": group["total_returned"],
            "article_rows": len(articles),
            "unique_articles": len(unique_articles),
            "best_database": group["best_database"],
            "best_database_returned": group["best_database_returned"],
        })

    return sorted(rows, key=lambda r: (r["project"], -int(r["unique_articles"]), -int(r["total_returned"])))


def build_database_summary(project_database_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}

    for row in project_database_rows:
        database = clean_text(row.get("database", "unknown")) or "unknown"
        group = groups.setdefault(database, {
            "database": database,
            "projects": 0,
            "query_runs": 0,
            "unique_queries": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "zero_result_runs": 0,
            "total_returned": 0,
            "article_rows": 0,
            "unique_articles_sum": 0,
        })
        group["projects"] += 1
        for field in [
            "query_runs",
            "unique_queries",
            "successful_runs",
            "failed_runs",
            "zero_result_runs",
            "total_returned",
            "article_rows",
            "unique_articles",
        ]:
            target_field = "unique_articles_sum" if field == "unique_articles" else field
            group[target_field] += to_int(row.get(field, 0))

    rows: List[Dict[str, Any]] = []
    for group in groups.values():
        rows.append({
            **group,
            "mean_returned_per_run": ratio(group["total_returned"], group["query_runs"]),
        })

    return sorted(rows, key=lambda r: -int(r["total_returned"]))


def build_config_field_summaries(
    projects: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    global_groups: Dict[str, Dict[str, Any]] = {}
    project_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    origin_database_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    global_article_keys: Dict[str, set] = defaultdict(set)
    project_article_keys: Dict[Tuple[str, str], set] = defaultdict(set)
    origin_database_article_keys: Dict[Tuple[str, str], set] = defaultdict(set)

    def new_group(project_name: str = "", origin: str = "", database: str = "") -> Dict[str, Any]:
        return {
            "project": project_name,
            "query_origin": origin,
            "config_field": origin,
            "field_meaning": origin_description(origin),
            "database": database,
            "projects": set(),
            "query_runs": 0,
            "unique_queries": set(),
            "successful_runs": 0,
            "failed_runs": 0,
            "productive_runs": 0,
            "zero_result_runs": 0,
            "total_returned": 0,
            "best_project": "",
            "best_database": "",
            "best_query": "",
            "best_query_returned": 0,
        }

    for project in projects:
        project_name = project["project"]

        for row in project["query_log_rows"]:
            origin = clean_text(row.get("query_origin", "")) or "unknown"
            database = clean_text(row.get("database", "")) or "unknown"
            query = clean_text(row.get("rendered_query", "")) or clean_text(row.get("query", ""))
            n_returned = to_int(row.get("n_returned", 0))
            query_key = query_key_from_log(row)
            is_failed = clean_text(row.get("status", "")).lower() in {"failed", "error"} or bool(clean_text(row.get("error", "")))

            group_specs = [
                (global_groups, origin, origin),
                (project_groups, (project_name, origin), origin),
                (origin_database_groups, (origin, database), origin),
            ]

            for groups, key, group_origin in group_specs:
                if key not in groups:
                    if groups is global_groups:
                        groups[key] = new_group(origin=group_origin)
                    elif groups is project_groups:
                        groups[key] = new_group(project_name=project_name, origin=group_origin)
                    else:
                        groups[key] = new_group(origin=group_origin, database=database)

                group = groups[key]
                group["projects"].add(project_name)
                group["query_runs"] += 1
                group["unique_queries"].add(query_key)
                group["total_returned"] += n_returned
                if is_failed:
                    group["failed_runs"] += 1
                else:
                    group["successful_runs"] += 1
                if n_returned > 0:
                    group["productive_runs"] += 1
                else:
                    group["zero_result_runs"] += 1
                if n_returned > group["best_query_returned"]:
                    group["best_query_returned"] = n_returned
                    group["best_project"] = project_name
                    group["best_database"] = database
                    group["best_query"] = query

        for row in project["article_rows"]:
            article_key = article_unique_key(row)
            if not article_key:
                continue
            databases = split_databases(row.get("source_database", ""))
            for origin in split_query_origins(row.get("query_origin", "")):
                global_article_keys[origin].add(article_key)
                project_article_keys[(project_name, origin)].add(article_key)
                for database in databases:
                    origin_database_article_keys[(origin, database)].add(article_key)

                if origin not in global_groups:
                    global_groups[origin] = new_group(origin=origin)
                if (project_name, origin) not in project_groups:
                    project_groups[(project_name, origin)] = new_group(project_name=project_name, origin=origin)
                for database in databases:
                    if (origin, database) not in origin_database_groups:
                        origin_database_groups[(origin, database)] = new_group(origin=origin, database=database)

    def finish_group(group: Dict[str, Any], unique_article_count: int) -> Dict[str, Any]:
        query_runs = to_int(group.get("query_runs", 0))
        productive_runs = to_int(group.get("productive_runs", 0))
        zero_result_runs = to_int(group.get("zero_result_runs", 0))
        failed_runs = to_int(group.get("failed_runs", 0))
        total_returned = to_int(group.get("total_returned", 0))
        projects_seen = group.get("projects", set())
        return {
            "project": group.get("project", ""),
            "query_origin": group.get("query_origin", ""),
            "config_field": group.get("config_field", ""),
            "field_meaning": group.get("field_meaning", ""),
            "database": group.get("database", ""),
            "projects": len(projects_seen) if isinstance(projects_seen, set) else projects_seen,
            "query_runs": query_runs,
            "unique_queries": len(group.get("unique_queries", set())),
            "successful_runs": to_int(group.get("successful_runs", 0)),
            "failed_runs": failed_runs,
            "productive_runs": productive_runs,
            "productive_run_pct": pct(productive_runs, query_runs),
            "zero_result_runs": zero_result_runs,
            "zero_result_run_pct": pct(zero_result_runs, query_runs),
            "total_returned": total_returned,
            "unique_articles_credit": unique_article_count,
            "mean_returned_per_run": ratio(total_returned, query_runs),
            "mean_unique_articles_per_query": ratio(unique_article_count, max(len(group.get("unique_queries", set())), 1)),
            "best_project": group.get("best_project", ""),
            "best_database": group.get("best_database", ""),
            "best_query": group.get("best_query", ""),
            "best_query_returned": group.get("best_query_returned", 0),
        }

    global_rows = [
        finish_group(group, len(global_article_keys.get(origin, set())))
        for origin, group in global_groups.items()
    ]
    project_rows = [
        finish_group(group, len(project_article_keys.get(key, set())))
        for key, group in project_groups.items()
    ]
    origin_database_rows = [
        finish_group(group, len(origin_database_article_keys.get(key, set())))
        for key, group in origin_database_groups.items()
    ]

    global_rows.sort(key=lambda r: (-to_int(r["unique_articles_credit"]), -to_int(r["total_returned"]), r["query_origin"]))
    project_rows.sort(key=lambda r: (r["project"], -to_int(r["unique_articles_credit"]), -to_int(r["total_returned"]), r["query_origin"]))
    origin_database_rows.sort(key=lambda r: (r["query_origin"], -to_int(r["unique_articles_credit"]), r["database"]))

    return global_rows, project_rows, origin_database_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Step7 search statistics by project, database, and query.")
    parser.add_argument("--projects-dir", default=str(DEFAULT_PROJECTS_DIR), help="Path to Projects directory.")
    parser.add_argument("--project", default="", help="Optional single project folder name.")
    parser.add_argument("--top-queries", type=int, default=10, help="Top query rows to print per project.")
    parser.add_argument("--max-table-rows", type=int, default=40, help="Maximum rows to print for large tables.")
    parser.add_argument("--show-all", action="store_true", help="Print all rows in every table.")
    parser.add_argument("--no-save", action="store_true", help="Print only; do not save CSV summaries.")
    args = parser.parse_args()
    if args.show_all:
        args.max_table_rows = 100000

    projects_dir = Path(args.projects_dir)
    if not projects_dir.exists():
        raise SystemExit(f"Projects directory not found: {projects_dir}")

    project_dirs = discover_projects(projects_dir, args.project)
    if not project_dirs:
        raise SystemExit("No projects with Search outputs were found.")

    projects = [collect_project(project_dir) for project_dir in project_dirs]
    project_rows = [project["summary"] for project in projects]
    project_database_rows = build_project_database_summary(projects)
    query_rows = build_query_performance(projects)
    database_rows = build_database_summary(project_database_rows)
    config_field_rows, project_config_field_rows, config_field_database_rows = build_config_field_summaries(projects)
    readable_config_field_rows = build_readable_config_field_summary(config_field_rows)

    total_query_runs = sum(to_int(row["query_database_runs"]) for row in project_rows)
    total_unique_queries = sum(to_int(row["unique_queries_formed"]) for row in project_rows)
    total_returned = sum(to_int(row["total_returned_before_dedup"]) for row in project_rows)
    total_article_rows = sum(to_int(row["article_rows_downloaded"]) for row in project_rows)
    total_unique_articles = sum(to_int(row["unique_articles_downloaded"]) for row in project_rows)

    print("=" * 120)
    print("STEP 8.2 SEARCH STATISTICS")
    print("=" * 120)
    print(f"Projects directory            : {projects_dir}")
    print(f"Projects analysed             : {len(project_rows)}")
    print(f"Unique query texts formed     : {total_unique_queries}")
    print(f"Query/database runs           : {total_query_runs}")
    print(f"Total returned before dedup   : {total_returned}")
    print(f"Article rows downloaded       : {total_article_rows}")
    print(f"Unique articles downloaded    : {total_unique_articles}")
    print("")
    print("Note: query/database runs = unique query texts multiplied across databases.")
    print("      Article rows are raw returned rows; unique articles are DOI/OpenAlex/PMID/S2/title-year deduplicated per project.")
    print("      Configuration-field tables group results by query_origin, i.e. the config field that generated the query.")
    print("      Unique article credit may overlap across fields because one deduplicated article can be found by multiple query origins.")

    readable_config_fields = [
        "rank",
        "config_field",
        "projects",
        "query_runs",
        "productive_%",
        "zero_%",
        "returned",
        "unique_article_credit",
        "articles_per_query",
        "best_database",
        "recommendation",
    ]
    print_table("READABLE SUMMARY: WHICH CONFIGURATION FIELDS WORKED?", readable_config_field_rows, readable_config_fields, max_rows=args.max_table_rows)

    readable_config_detail_fields = [
        "rank",
        "config_field",
        "what_it_means",
        "best_query",
    ]
    print_table("READABLE DETAILS: FIELD MEANING AND BEST QUERY", readable_config_field_rows, readable_config_detail_fields, max_rows=args.max_table_rows)

    project_fields = [
        "project",
        "unique_queries_formed",
        "query_database_runs",
        "databases_used",
        "successful_query_runs",
        "failed_query_runs",
        "zero_result_query_runs",
        "total_returned_before_dedup",
        "article_rows_downloaded",
        "unique_articles_downloaded",
        "articles_with_doi",
        "mean_returned_per_query_run",
    ]
    print_table("PROJECT SUMMARY", project_rows, project_fields, max_rows=args.max_table_rows)

    db_fields = [
        "database",
        "projects",
        "query_runs",
        "unique_queries",
        "successful_runs",
        "failed_runs",
        "zero_result_runs",
        "total_returned",
        "article_rows",
        "unique_articles_sum",
        "mean_returned_per_run",
    ]
    print_table("DATABASE SUMMARY ACROSS PROJECTS", database_rows, db_fields, max_rows=args.max_table_rows)

    config_field_fields = [
        "config_field",
        "projects",
        "query_runs",
        "unique_queries",
        "productive_runs",
        "productive_run_pct",
        "zero_result_runs",
        "zero_result_run_pct",
        "total_returned",
        "unique_articles_credit",
        "mean_returned_per_run",
        "mean_unique_articles_per_query",
        "best_project",
        "best_database",
        "best_query",
    ]
    print_table("CONFIGURATION FIELD PERFORMANCE ACROSS ALL PROJECTS", config_field_rows, config_field_fields, max_rows=args.max_table_rows)

    config_field_meaning_fields = [
        "config_field",
        "field_meaning",
    ]
    print_table("CONFIGURATION FIELD MEANING", config_field_rows, config_field_meaning_fields, max_rows=args.max_table_rows)

    project_config_field_fields = [
        "project",
        "config_field",
        "query_runs",
        "unique_queries",
        "productive_runs",
        "productive_run_pct",
        "zero_result_runs",
        "total_returned",
        "unique_articles_credit",
        "mean_returned_per_run",
        "best_database",
        "best_query",
    ]
    print_table("PROJECT X CONFIGURATION FIELD TABLE", project_config_field_rows, project_config_field_fields, max_rows=args.max_table_rows)

    config_field_database_fields = [
        "config_field",
        "database",
        "query_runs",
        "unique_queries",
        "productive_runs",
        "productive_run_pct",
        "total_returned",
        "unique_articles_credit",
        "mean_returned_per_run",
        "best_project",
        "best_query",
    ]
    print_table("CONFIGURATION FIELD X DATABASE TABLE", config_field_database_rows, config_field_database_fields, max_rows=args.max_table_rows)

    project_db_fields = [
        "project",
        "database",
        "query_runs",
        "unique_queries",
        "total_returned",
        "article_rows",
        "unique_articles",
        "zero_result_runs",
        "failed_runs",
        "mean_returned_per_run",
        "best_query",
    ]
    print_table("PROJECT X DATABASE TABLE", project_database_rows, project_db_fields, max_rows=args.max_table_rows)

    query_fields = [
        "project",
        "query_origin",
        "query",
        "databases_tried",
        "total_returned",
        "article_rows",
        "unique_articles",
        "best_database",
        "best_database_returned",
    ]
    top_query_rows: List[Dict[str, Any]] = []
    for project_name in sorted({row["project"] for row in query_rows}):
        project_query_rows = [row for row in query_rows if row["project"] == project_name]
        top_query_rows.extend(project_query_rows[: max(args.top_queries, 0)])
    print_table(f"TOP {args.top_queries} QUERIES PER PROJECT", top_query_rows, query_fields, max_rows=max(args.max_table_rows, len(top_query_rows)))

    if not args.no_save:
        out_dir = projects_dir / SUMMARY_DIR
        write_csv(out_dir / "project_summary.csv", project_rows, project_fields)
        write_csv(out_dir / "project_database_summary.csv", project_database_rows, project_db_fields + ["max_returned_single_query"])
        write_csv(out_dir / "query_performance.csv", query_rows, [
            "project",
            "query_origin",
            "query",
            "rendered_query",
            "databases_tried",
            "successful_runs",
            "failed_runs",
            "zero_result_runs",
            "total_returned",
            "article_rows",
            "unique_articles",
            "best_database",
            "best_database_returned",
        ])
        write_csv(out_dir / "database_summary.csv", database_rows, db_fields)
        write_csv(out_dir / "config_field_summary.csv", config_field_rows, [
            "config_field",
            "field_meaning",
            "projects",
            "query_runs",
            "unique_queries",
            "successful_runs",
            "failed_runs",
            "productive_runs",
            "productive_run_pct",
            "zero_result_runs",
            "zero_result_run_pct",
            "total_returned",
            "unique_articles_credit",
            "mean_returned_per_run",
            "mean_unique_articles_per_query",
            "best_project",
            "best_database",
            "best_query",
            "best_query_returned",
        ])
        write_csv(out_dir / "project_config_field_summary.csv", project_config_field_rows, [
            "project",
            "config_field",
            "field_meaning",
            "query_runs",
            "unique_queries",
            "successful_runs",
            "failed_runs",
            "productive_runs",
            "productive_run_pct",
            "zero_result_runs",
            "zero_result_run_pct",
            "total_returned",
            "unique_articles_credit",
            "mean_returned_per_run",
            "mean_unique_articles_per_query",
            "best_database",
            "best_query",
            "best_query_returned",
        ])
        write_csv(out_dir / "config_field_database_summary.csv", config_field_database_rows, [
            "config_field",
            "field_meaning",
            "database",
            "projects",
            "query_runs",
            "unique_queries",
            "successful_runs",
            "failed_runs",
            "productive_runs",
            "productive_run_pct",
            "zero_result_runs",
            "zero_result_run_pct",
            "total_returned",
            "unique_articles_credit",
            "mean_returned_per_run",
            "mean_unique_articles_per_query",
            "best_project",
            "best_query",
            "best_query_returned",
        ])
        write_csv(out_dir / "readable_config_field_summary.csv", readable_config_field_rows, [
            "rank",
            "config_field",
            "what_it_means",
            "projects",
            "query_runs",
            "productive_%",
            "zero_%",
            "returned",
            "unique_article_credit",
            "articles_per_query",
            "best_database",
            "best_query",
            "recommendation",
        ])
        print("\nSaved CSV summaries:")
        print(f"  {out_dir / 'project_summary.csv'}")
        print(f"  {out_dir / 'project_database_summary.csv'}")
        print(f"  {out_dir / 'query_performance.csv'}")
        print(f"  {out_dir / 'database_summary.csv'}")
        print(f"  {out_dir / 'config_field_summary.csv'}")
        print(f"  {out_dir / 'project_config_field_summary.csv'}")
        print(f"  {out_dir / 'config_field_database_summary.csv'}")
        print(f"  {out_dir / 'readable_config_field_summary.csv'}")


if __name__ == "__main__":
    main()
