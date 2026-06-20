# Step7C-ComputeRetrievableCeiling.py
#
# Purpose:
#   Compute the retrievable ceiling per review.
#
# This is ONLY the ceiling step.
#
# For each reference in each project, this script checks:
#   1. Has DOI?
#   2. Indexed in OpenAlex?
#   3. Journal-article / review-article type?
#   4. Within the search date window?
#
# Output per project:
#   Projects/<project>/Evaluation/retrievable_ceiling_references.csv
#   Projects/<project>/Evaluation/openalex_reference_cache.json
#
# Output global:
#   Projects/Evaluation/retrievable_ceiling_summary.csv
#
# What this script does NOT do:
#   - Does NOT read Search/articles.csv
#   - Does NOT clean retrieved articles
#   - Does NOT match retrieved articles
#   - Does NOT calculate recall
#   - Does NOT download PDFs or full text
#
# Usage:
#   python Step7C-ComputeRetrievableCeiling.py --projects-dir Projects --force
#
# One project:
#   python Step7C-ComputeRetrievableCeiling.py --projects-dir Projects --project PROJECT_NAME --force
#
# Requirements:
#   pip install requests tqdm

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")

DEFAULT_YEAR_START = 1990
DEFAULT_YEAR_END = 2026

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "muneebsiddique007@gmail.com").strip()

REQUEST_TIMEOUT = 45
MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 2.5
OPENALEX_SLEEP_SECONDS = 0.15
SAVE_CACHE_EVERY = 25
MAX_RETRY_AFTER_SECONDS = 30

HEADERS = {
    "User-Agent": f"ALR-RetrievableCeiling/1.0 (mailto:{CONTACT_EMAIL})",
    "Accept": "application/json",
}

# OpenAlex commonly reports type="article".
# Crossref type may be journal-article.
JOURNAL_ARTICLE_TYPES = {
    "article",
    "journal-article",
    "research-article",
    "review-article",
    "review",
}


class OpenAlexBudgetError(RuntimeError):
    """Raised when OpenAlex reports that the daily/prepaid request budget is exhausted."""


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

@dataclass
class ReferenceStatus:
    reference_index: int = 0

    doi: str = ""
    title: str = ""
    reference_year: str = ""
    reference_type: str = ""

    has_doi: bool = False

    openalex_indexed: bool = False
    openalex_id: str = ""
    openalex_year: str = ""
    openalex_type: str = ""
    openalex_type_crossref: str = ""
    openalex_title: str = ""

    within_date_window: bool = False
    journal_article_type: bool = False

    retrievable_ceiling: bool = False
    exclusion_reason: str = ""


REFERENCE_STATUS_FIELDS = [
    "reference_index",
    "doi",
    "title",
    "reference_year",
    "reference_type",
    "has_doi",
    "openalex_indexed",
    "openalex_id",
    "openalex_year",
    "openalex_type",
    "openalex_type_crossref",
    "openalex_title",
    "within_date_window",
    "journal_article_type",
    "retrievable_ceiling",
    "exclusion_reason",
]


SUMMARY_FIELDS = [
    "project",
    "references_source",
    "references_total",
    "references_with_doi",
    "references_missing_doi",
    "references_indexed_openalex",
    "references_not_indexed_openalex",
    "references_journal_article",
    "references_not_journal_article",
    "references_within_date_window",
    "references_outside_date_window",
    "retrievable_ceiling",
    "retrievable_ceiling_fraction_of_total",
    "retrievable_ceiling_fraction_of_doi_refs",
    "date_window_start",
    "date_window_end",
    "reference_status_path",
    "openalex_cache_path",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    text = html.unescape(str(x))
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

    # Rescue DOI if embedded in a messy string.
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


def parse_year(value: Any) -> int:
    text = clean_text(value)
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else 0


def safe_div(num: int, den: int) -> float:
    if not den:
        return 0.0
    return round(num / den, 6)


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def csv_bool(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y"}


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

    skip = {"search", "rankings", "evaluation", "__pycache__"}

    return [
        p for p in sorted(projects_dir.iterdir())
        if p.is_dir() and p.name.lower() not in skip
    ]


def deep_get(obj: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = obj

    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]

    return cur


# ============================================================
# DATE WINDOW FROM CONFIG
# ============================================================

def parse_date_range(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Supports both grouped config and older flat config.
    """

    y0 = 0
    y1 = 0

    # Grouped schema.
    y0 = parse_year(deep_get(config, "search_limits.generated_date_start", ""))
    y1 = parse_year(deep_get(config, "search_limits.generated_date_end", ""))

    if not y0:
        y0 = parse_year(deep_get(config, "search_dates.reported_date_coverage_start", ""))
    if not y1:
        y1 = parse_year(deep_get(config, "search_dates.reported_date_coverage_end", ""))

    if not y0:
        y0 = parse_year(deep_get(config, "search_dates.reported_date_search_conducted", ""))
    if not y1:
        y1 = parse_year(deep_get(config, "search_dates.reported_date_search_updated", ""))

    # Flat schema fallback.
    if not y0:
        y0 = parse_year(config.get("generated_date_start", ""))
    if not y1:
        y1 = parse_year(config.get("generated_date_end", ""))

    if not y0:
        y0 = parse_year(config.get("prisma_date_coverage_start", ""))
    if not y1:
        y1 = parse_year(config.get("prisma_date_coverage_end", ""))

    # Text date limits fallback.
    date_limit_texts = [
        clean_text(deep_get(config, "search_limits.reported_date_limits", "")),
        clean_text(config.get("prisma_s_date_limits", "")),
    ]

    years: List[int] = []
    for txt in date_limit_texts:
        years.extend(int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", txt))

    if years:
        if not y0:
            y0 = min(years)
        if not y1:
            y1 = max(years)

    if not y0:
        y0 = DEFAULT_YEAR_START
    if not y1:
        y1 = DEFAULT_YEAR_END

    if y0 > y1:
        y0, y1 = y1, y0

    return y0, y1


# ============================================================
# REFERENCE EXTRACTION
# ============================================================

def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
    """
    Flexible extractor for different reference JSON structures.
    Extracts DOI, title, year, and type where possible.
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

    year_keys = {
        "year",
        "publication_year",
        "published_year",
        "pub_year",
    }

    type_keys = {
        "type",
        "article_type",
        "publication_type",
        "work_type",
        "document_type",
    }

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            doi = ""
            title = ""
            year = ""
            ref_type = ""

            for k, v in obj.items():
                if k in doi_keys and not doi:
                    doi = clean_text(v)

                if k in title_keys and not title:
                    title = clean_text(v)

                if k in year_keys and not year:
                    year = clean_text(v)

                if k in type_keys and not ref_type:
                    ref_type = clean_text(v)

            if doi or title:
                records.append({
                    "doi": normalize_doi(doi),
                    "title": title,
                    "year": year,
                    "type": ref_type,
                })

            # Direct DOI-list fields.
            for k, v in obj.items():
                if k.lower() in {"dois", "doi_list", "referenced_dois"} and isinstance(v, list):
                    for item in v:
                        d = normalize_doi(item)
                        if d:
                            records.append({
                                "doi": d,
                                "title": "",
                                "year": "",
                                "type": "",
                            })

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    # Dedupe by DOI first, then title.
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
            "year": clean_text(rec.get("year", "")),
            "type": clean_text(rec.get("type", "")),
        })

    return out


def load_reference_records(project_dir: Path) -> Tuple[List[Dict[str, str]], str]:
    """
    Evaluation script: allowed to read references.
    Priority:
      1. references.json
      2. references_resolved.json
      3. references_gold.json
    """

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


# ============================================================
# OPENALEX LOOKUP
# ============================================================

def request_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_status = 0
    last_url = url
    last_body = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params=params or {},
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            last_status = r.status_code
            last_url = r.url
            last_body = clean_text(r.text[:500])

            if r.status_code == 429:
                if "insufficient budget" in last_body.lower() or "dailyremainingusd" in last_body.lower():
                    raise OpenAlexBudgetError(
                        "OpenAlex request budget/rate limit is exhausted; "
                        f"url={last_url}; body={last_body}"
                    )

                retry_after = r.headers.get("Retry-After", "")
                try:
                    sleep_for = max(float(retry_after), RETRY_SLEEP_SECONDS * attempt)
                except Exception:
                    sleep_for = RETRY_SLEEP_SECONDS * attempt
                sleep_for = min(sleep_for, MAX_RETRY_AFTER_SECONDS)
                time.sleep(sleep_for)
                continue

            if r.status_code in {500, 502, 503, 504}:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
                continue

            if r.status_code == 404:
                return {}

            r.raise_for_status()
            return r.json()

        except OpenAlexBudgetError:
            raise
        except Exception as e:
            last_error = e

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    if last_status:
        raise RuntimeError(
            "OpenAlex request failed after "
            f"{MAX_RETRIES} attempts; status={last_status}; url={last_url}; body={last_body}"
        )

    raise RuntimeError(
        "OpenAlex request failed after "
        f"{MAX_RETRIES} attempts; error={type(last_error).__name__}: {last_error}"
    ) from last_error


def openalex_cache_path(project_dir: Path) -> Path:
    return project_dir / "Evaluation" / "openalex_reference_cache.json"


def load_openalex_cache(project_dir: Path) -> Dict[str, Any]:
    return read_json(openalex_cache_path(project_dir), default={}) or {}


def save_openalex_cache(project_dir: Path, cache: Dict[str, Any]) -> None:
    write_json(openalex_cache_path(project_dir), cache)


def openalex_work_from_cached_article(record: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_text(record.get("title", ""))
    article_type = clean_text(record.get("article_type", ""))
    return {
        "id": clean_text(record.get("openalex_id", "")),
        "doi": normalize_doi(record.get("doi", "")),
        "publication_year": clean_text(record.get("year", "")),
        "type": article_type,
        "type_crossref": article_type,
        "title": title,
        "display_name": title,
        "_cache_source": "step7_search_query_cache",
    }


def seed_openalex_cache_from_search_query_cache(project_dir: Path, cache: Dict[str, Any]) -> int:
    """
    Reuse OpenAlex search results downloaded by Step7.

    This does not prove that every missing DOI is absent from OpenAlex, but every
    DOI found in an openalex_*.json cache file is definitely already indexed in
    OpenAlex and can be used for ceiling calculation without another API call.
    """
    query_cache_dir = project_dir / "Search" / "QueryCache"
    if not query_cache_dir.exists():
        return 0

    added = 0
    for path in sorted(query_cache_dir.glob("openalex_*.json")):
        data = read_json(path, default={}) or {}
        records = data.get("records", [])
        if not isinstance(records, list):
            continue

        for record in records:
            if not isinstance(record, dict):
                continue
            doi = normalize_doi(record.get("doi", ""))
            if not doi or doi in cache:
                continue

            work = openalex_work_from_cached_article(record)
            if not work.get("id"):
                continue

            cache[doi] = work
            added += 1

    return added


def lookup_openalex_by_doi(
    doi: str,
    cache: Dict[str, Any],
    allow_live_openalex: bool = True,
) -> Dict[str, Any]:
    doi = normalize_doi(doi)

    if not doi:
        return {}

    if doi in cache:
        return cache[doi]

    if not allow_live_openalex:
        return {}

    doi_url = f"https://doi.org/{doi}"

    # Primary robust lookup.
    data = request_json(
        "https://api.openalex.org/works",
        {
            "filter": f"doi:{doi_url}",
            "per-page": 1,
            "mailto": CONTACT_EMAIL,
        },
    )

    results = data.get("results", []) if isinstance(data, dict) else []
    work = results[0] if results else {}

    # Fallback direct endpoint lookup.
    if not work:
        encoded = quote(doi_url, safe="")
        work = request_json(
            f"https://api.openalex.org/works/{encoded}",
            {
                "mailto": CONTACT_EMAIL,
            },
        )

    if not isinstance(work, dict):
        work = {}

    cache[doi] = work

    time.sleep(OPENALEX_SLEEP_SECONDS)

    return work


def is_journal_article_type(
    openalex_type: str,
    openalex_type_crossref: str,
    reference_type: str,
) -> bool:
    values = {
        clean_text(openalex_type).lower(),
        clean_text(openalex_type_crossref).lower(),
        clean_text(reference_type).lower(),
    }

    for value in values:
        if not value:
            continue

        if value in JOURNAL_ARTICLE_TYPES:
            return True

        if "journal-article" in value:
            return True

        if value == "article":
            return True

    return False


def get_reference_status(
    ref: Dict[str, str],
    y0: int,
    y1: int,
    cache: Dict[str, Any],
    allow_live_openalex: bool = True,
) -> ReferenceStatus:
    doi = normalize_doi(ref.get("doi", ""))
    title = clean_text(ref.get("title", ""))
    ref_year = clean_text(ref.get("year", ""))
    ref_type = clean_text(ref.get("type", ""))

    status = ReferenceStatus(
        doi=doi,
        title=title,
        reference_year=ref_year,
        reference_type=ref_type,
        has_doi=bool(doi),
    )

    if not doi:
        status.exclusion_reason = "missing_doi"
        return status

    try:
        work = lookup_openalex_by_doi(
            doi=doi,
            cache=cache,
            allow_live_openalex=allow_live_openalex,
        )
    except OpenAlexBudgetError:
        raise
    except Exception as e:
        status.exclusion_reason = f"openalex_lookup_error: {clean_text(e)[:300]}"
        return status

    if not work:
        if allow_live_openalex:
            status.exclusion_reason = "not_indexed_openalex"
        else:
            status.exclusion_reason = "not_found_in_step7_openalex_query_cache_unverified"
        return status

    status.openalex_indexed = True
    status.openalex_id = clean_text(work.get("id", ""))
    status.openalex_year = clean_text(work.get("publication_year", ""))
    status.openalex_type = clean_text(work.get("type", ""))
    status.openalex_type_crossref = clean_text(work.get("type_crossref", ""))
    status.openalex_title = clean_text(work.get("title", "") or work.get("display_name", ""))

    work_year = parse_year(status.openalex_year) or parse_year(ref_year)
    status.within_date_window = bool(work_year and y0 <= work_year <= y1)

    status.journal_article_type = is_journal_article_type(
        status.openalex_type,
        status.openalex_type_crossref,
        ref_type,
    )

    if not status.within_date_window:
        status.exclusion_reason = "outside_date_window"
    elif not status.journal_article_type:
        status.exclusion_reason = "not_journal_article_type"
    else:
        status.retrievable_ceiling = True
        status.exclusion_reason = ""

    return status


# ============================================================
# PROJECT PROCESSING
# ============================================================

def process_project(
    project_dir: Path,
    force: bool,
    refresh_openalex_cache: bool,
    use_search_query_cache: bool,
    allow_live_openalex: bool,
) -> Dict[str, Any]:
    config_path = project_dir / "config.json"
    evaluation_dir = project_dir / "Evaluation"
    reference_status_path = evaluation_dir / "retrievable_ceiling_references.csv"
    cache_path = openalex_cache_path(project_dir)

    config = read_json(config_path, default={}) or {}
    y0, y1 = parse_date_range(config)

    if reference_status_path.exists() and not force:
        return summarize_existing_reference_status(
            project_dir=project_dir,
            reference_status_path=reference_status_path,
            cache_path=cache_path,
            y0=y0,
            y1=y1,
        )

    ref_records, ref_source = load_reference_records(project_dir)

    cache = {} if refresh_openalex_cache else load_openalex_cache(project_dir)
    seeded_from_query_cache = 0
    if use_search_query_cache:
        seeded_from_query_cache = seed_openalex_cache_from_search_query_cache(project_dir, cache)
        if seeded_from_query_cache:
            save_openalex_cache(project_dir, cache)

    statuses: List[ReferenceStatus] = []

    for idx, ref in enumerate(
        tqdm(ref_records, desc=f"Ceiling {project_dir.name}", leave=False),
        start=1,
    ):
        status = get_reference_status(
            ref=ref,
            y0=y0,
            y1=y1,
            cache=cache,
            allow_live_openalex=allow_live_openalex,
        )
        status.reference_index = idx
        statuses.append(status)

        if idx % SAVE_CACHE_EVERY == 0:
            save_openalex_cache(project_dir, cache)

    save_openalex_cache(project_dir, cache)

    write_csv(
        reference_status_path,
        [asdict(x) for x in statuses],
        REFERENCE_STATUS_FIELDS,
    )

    references_total = len(statuses)
    references_with_doi = sum(1 for x in statuses if x.has_doi)
    references_missing_doi = references_total - references_with_doi

    references_indexed_openalex = sum(1 for x in statuses if x.openalex_indexed)
    references_not_indexed_openalex = references_with_doi - references_indexed_openalex

    references_journal_article = sum(1 for x in statuses if x.journal_article_type)
    references_not_journal_article = references_with_doi - references_journal_article

    references_within_date_window = sum(1 for x in statuses if x.within_date_window)
    references_outside_date_window = references_with_doi - references_within_date_window

    retrievable_ceiling = sum(1 for x in statuses if x.retrievable_ceiling)

    return {
        "project": project_dir.name,
        "references_source": ref_source,

        "references_total": references_total,
        "references_with_doi": references_with_doi,
        "references_missing_doi": references_missing_doi,

        "references_indexed_openalex": references_indexed_openalex,
        "references_not_indexed_openalex": references_not_indexed_openalex,

        "references_journal_article": references_journal_article,
        "references_not_journal_article": references_not_journal_article,

        "references_within_date_window": references_within_date_window,
        "references_outside_date_window": references_outside_date_window,

        "retrievable_ceiling": retrievable_ceiling,
        "retrievable_ceiling_fraction_of_total": safe_div(
            retrievable_ceiling,
            references_total,
        ),
        "retrievable_ceiling_fraction_of_doi_refs": safe_div(
            retrievable_ceiling,
            references_with_doi,
        ),

        "date_window_start": y0,
        "date_window_end": y1,

        "reference_status_path": str(reference_status_path),
        "openalex_cache_path": str(cache_path),
        "seeded_from_query_cache": seeded_from_query_cache,
        "status": "success",
    }


# ============================================================
# PRINTING / SAVING
# ============================================================

def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    display_fields = [
        "project",
        "references_total",
        "references_with_doi",
        "references_indexed_openalex",
        "references_journal_article",
        "references_within_date_window",
        "retrievable_ceiling",
        "retrievable_ceiling_fraction_of_doi_refs",
    ]

    compact_rows = [
        {field: str(row.get(field, "")) for field in display_fields}
        for row in rows
    ]

    widths: Dict[str, int] = {}
    for field in display_fields:
        widths[field] = max(
            len(field),
            *(len(row.get(field, "")) for row in compact_rows),
        )

    header = " | ".join(field.ljust(widths[field]) for field in display_fields)
    sep = "-+-".join("-" * widths[field] for field in display_fields)

    print(header)
    print(sep)

    for row in compact_rows:
        print(" | ".join(row.get(field, "").ljust(widths[field]) for field in display_fields))


def save_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    write_csv(path, rows, SUMMARY_FIELDS)


def summarize_existing_reference_status(
    project_dir: Path,
    reference_status_path: Path,
    cache_path: Path,
    y0: int,
    y1: int,
) -> Dict[str, Any]:
    rows = read_csv(reference_status_path)

    references_total = len(rows)
    references_with_doi = sum(1 for x in rows if csv_bool(x.get("has_doi", "")))
    references_missing_doi = references_total - references_with_doi

    references_indexed_openalex = sum(1 for x in rows if csv_bool(x.get("openalex_indexed", "")))
    references_not_indexed_openalex = references_with_doi - references_indexed_openalex

    references_journal_article = sum(1 for x in rows if csv_bool(x.get("journal_article_type", "")))
    references_not_journal_article = references_with_doi - references_journal_article

    references_within_date_window = sum(1 for x in rows if csv_bool(x.get("within_date_window", "")))
    references_outside_date_window = references_with_doi - references_within_date_window

    retrievable_ceiling = sum(1 for x in rows if csv_bool(x.get("retrievable_ceiling", "")))

    return {
        "project": project_dir.name,
        "status": "existing_output_loaded",
        "references_source": "existing_retrievable_ceiling_references.csv",

        "references_total": references_total,
        "references_with_doi": references_with_doi,
        "references_missing_doi": references_missing_doi,

        "references_indexed_openalex": references_indexed_openalex,
        "references_not_indexed_openalex": references_not_indexed_openalex,

        "references_journal_article": references_journal_article,
        "references_not_journal_article": references_not_journal_article,

        "references_within_date_window": references_within_date_window,
        "references_outside_date_window": references_outside_date_window,

        "retrievable_ceiling": retrievable_ceiling,
        "retrievable_ceiling_fraction_of_total": safe_div(
            retrievable_ceiling,
            references_total,
        ),
        "retrievable_ceiling_fraction_of_doi_refs": safe_div(
            retrievable_ceiling,
            references_with_doi,
        ),

        "date_window_start": y0,
        "date_window_end": y1,
        "reference_status_path": str(reference_status_path),
        "openalex_cache_path": str(cache_path),
        "seeded_from_query_cache": 0,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global REQUEST_TIMEOUT, MAX_RETRIES, OPENALEX_SLEEP_SECONDS

    parser = argparse.ArgumentParser(
        description="Compute retrievable ceiling per review using reference DOIs and OpenAlex metadata."
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
        "--force",
        action="store_true",
        help="Overwrite existing retrievable ceiling outputs.",
    )

    parser.add_argument(
        "--refresh-openalex-cache",
        action="store_true",
        help="Ignore existing OpenAlex reference cache and re-query OpenAlex.",
    )

    parser.add_argument(
        "--no-search-query-cache",
        action="store_true",
        help="Do not seed OpenAlex DOI metadata from Step7 Search/QueryCache/openalex_*.json files.",
    )

    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use existing Evaluation cache and Step7 QueryCache only; do not call live OpenAlex API.",
    )

    parser.add_argument(
        "--request-timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help="Seconds to wait for each OpenAlex request before retrying.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES,
        help="Maximum retry attempts for each OpenAlex request.",
    )

    parser.add_argument(
        "--openalex-sleep",
        type=float,
        default=OPENALEX_SLEEP_SECONDS,
        help="Sleep time between successful uncached OpenAlex lookups.",
    )

    args = parser.parse_args()

    REQUEST_TIMEOUT = max(int(args.request_timeout), 1)
    MAX_RETRIES = max(int(args.max_retries), 1)
    OPENALEX_SLEEP_SECONDS = max(float(args.openalex_sleep), 0.0)

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    print("=" * 100)
    print("STEP 7C: COMPUTE RETRIEVABLE CEILING")
    print("=" * 100)
    print(f"Projects directory     : {projects_dir.resolve()}")
    print(f"Projects found         : {len(projects)}")
    print(f"Force overwrite        : {args.force}")
    print(f"Refresh OpenAlex cache : {args.refresh_openalex_cache}")
    print(f"Use Search QueryCache  : {not args.no_search_query_cache}")
    print(f"Cache-only mode        : {args.cache_only}")
    print(f"Request timeout        : {REQUEST_TIMEOUT}s")
    print(f"Max retries            : {MAX_RETRIES}")
    print(f"OpenAlex sleep         : {OPENALEX_SLEEP_SECONDS}s")
    print(f"Contact email          : {CONTACT_EMAIL}")
    print("Reads retrieved articles: NO")
    print("Matches retrieved set   : NO")
    print("=" * 100)

    rows: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        try:
            row = process_project(
                project_dir=project_dir,
                force=args.force,
                refresh_openalex_cache=args.refresh_openalex_cache,
                use_search_query_cache=not args.no_search_query_cache,
                allow_live_openalex=not args.cache_only,
            )
            rows.append(row)

            if row.get("status") == "skipped_existing_output":
                print("SKIP existing output")
            else:
                print(
                    f"OK | refs={row.get('references_total')} | "
                    f"doi={row.get('references_with_doi')} | "
                    f"openalex={row.get('references_indexed_openalex')} | "
                    f"journal={row.get('references_journal_article')} | "
                    f"date={row.get('references_within_date_window')} | "
                    f"ceiling={row.get('retrievable_ceiling')} | "
                    f"ceiling/doi={row.get('retrievable_ceiling_fraction_of_doi_refs')} | "
                    f"query_cache_seeded={row.get('seeded_from_query_cache', '')}"
                )

        except Exception as e:
            print(f"FAIL | {project_dir.name} | {type(e).__name__}: {e}")
            rows.append({
                "project": project_dir.name,
                "references_source": "",
                "references_total": "",
                "references_with_doi": "",
                "references_missing_doi": "",
                "references_indexed_openalex": "",
                "references_not_indexed_openalex": "",
                "references_journal_article": "",
                "references_not_journal_article": "",
                "references_within_date_window": "",
                "references_outside_date_window": "",
                "retrievable_ceiling": "",
                "retrievable_ceiling_fraction_of_total": "",
                "retrievable_ceiling_fraction_of_doi_refs": "",
                "date_window_start": "",
                "date_window_end": "",
                "reference_status_path": "",
                "openalex_cache_path": "",
                "status": "failed",
            })

    if args.project:
        summary_path = (
            projects_dir
            / "Evaluation"
            / f"retrievable_ceiling_summary_{Path(args.project).name}.csv"
        )
    else:
        summary_path = projects_dir / "Evaluation" / "retrievable_ceiling_summary.csv"
    save_summary_csv(summary_path, rows)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print_summary_table(rows)

    total_refs = sum(int(r.get("references_total", 0) or 0) for r in rows)
    total_doi = sum(int(r.get("references_with_doi", 0) or 0) for r in rows)
    total_openalex = sum(int(r.get("references_indexed_openalex", 0) or 0) for r in rows)
    total_journal = sum(int(r.get("references_journal_article", 0) or 0) for r in rows)
    total_date = sum(int(r.get("references_within_date_window", 0) or 0) for r in rows)
    total_ceiling = sum(int(r.get("retrievable_ceiling", 0) or 0) for r in rows)

    print("\n" + "=" * 100)
    print("TOTALS")
    print("=" * 100)
    print(f"Total references                         : {total_refs}")
    print(f"Total references with DOI                : {total_doi}")
    print(f"Total references indexed in OpenAlex     : {total_openalex}")
    print(f"Total references journal/review articles : {total_journal}")
    print(f"Total references within date window      : {total_date}")
    print(f"Total retrievable ceiling                : {total_ceiling}")
    print(f"Ceiling / all references                 : {safe_div(total_ceiling, total_refs)}")
    print(f"Ceiling / DOI references                 : {safe_div(total_ceiling, total_doi)}")
    print(f"\nSaved summary to: {summary_path}")

    print("\nPer-project outputs:")
    print("  Projects/<project>/Evaluation/retrievable_ceiling_references.csv")
    print("  Projects/<project>/Evaluation/openalex_reference_cache.json")


if __name__ == "__main__":
    main()
