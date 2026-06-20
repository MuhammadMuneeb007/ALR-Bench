# Step7-SearchLiterature.py
#
# PURE GENERIC LITERATURE SEARCH STAGE
#
# What this script does:
#   1. Reads each Projects/<project>/config.json
#   2. Builds generic search queries from grouped/flat config fields
#   3. Searches:
#        - OpenAlex
#        - Crossref
#        - Europe PMC
#        - PubMed
#        - Semantic Scholar
#   4. Saves:
#        Projects/<project>/Search/articles.csv
#        Projects/<project>/Search/query_log.csv
#
# What this script does NOT do:
#   - Does NOT read references.json
#   - Does NOT read references_gold.json
#   - Does NOT calculate overlap, recall, precision, or F1
#   - Does NOT download PDFs
#   - Does NOT rank articles
#   - Does NOT use quality-assessment keywords for search by default
#
# Usage:
#   python Step7-SearchLiterature.py --projects-dir Projects --force
#
# One project:
#   python Step7-SearchLiterature.py --projects-dir Projects --project PROJECT_NAME --force
#
# Fast test:
#   python Step7-SearchLiterature.py --projects-dir Projects --max-results-per-query 50 --max-queries 15 --force
#
# Optional:
#   set SEMANTIC_SCHOLAR_API_KEY=your_key
#
# Requirements:
#   pip install requests tqdm

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")

DEFAULT_DATABASES = [
    "openalex",
    "crossref",
    "europepmc",
    "pubmed",
    "semanticscholar",
]

DEFAULT_MAX_RESULTS_PER_QUERY = 100
DEFAULT_MAX_QUERIES = 30
DEFAULT_WORKERS = 6

DEFAULT_YEAR_START = 1990
DEFAULT_YEAR_END = 2026

REQUEST_TIMEOUT = 45
MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 3.0
SLEEP_BETWEEN_REQUESTS = 0.20
SEMANTIC_SCHOLAR_SLEEP = 1.20
CACHE_VERSION = 2

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "muneebsiddique007@gmail.com").strip()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()

USER_AGENT = f"ALR-SearchLiterature/2.0 (mailto:{CONTACT_EMAIL})"

BASE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

SEMANTIC_SCHOLAR_HEADERS = dict(BASE_HEADERS)
if SEMANTIC_SCHOLAR_API_KEY:
    SEMANTIC_SCHOLAR_HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY


# ============================================================
# OUTPUT RECORDS
# ============================================================

@dataclass
class ArticleRecord:
    title: str = ""
    abstract: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    authors: str = ""
    url: str = ""
    article_type: str = ""

    source_database: str = ""
    source_query: str = ""
    rendered_query: str = ""
    query_origin: str = ""

    openalex_id: str = ""
    pmid: str = ""
    pmcid: str = ""
    semantic_scholar_id: str = ""


ARTICLE_FIELDS = [
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
    "rendered_query",
    "query_origin",
    "openalex_id",
    "pmid",
    "pmcid",
    "semantic_scholar_id",
]


@dataclass
class QueryLogRecord:
    project: str = ""
    database: str = ""
    query_origin: str = ""
    query: str = ""
    rendered_query: str = ""
    year_start: int = 0
    year_end: int = 0
    max_results_per_query: int = 0
    status: str = ""
    n_returned: int = 0
    error: str = ""
    cache_used: bool = False
    cache_path: str = ""


QUERY_LOG_FIELDS = [
    "project",
    "database",
    "query_origin",
    "query",
    "rendered_query",
    "year_start",
    "year_end",
    "max_results_per_query",
    "status",
    "n_returned",
    "error",
    "cache_used",
    "cache_path",
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


def normalize_key(x: Any) -> str:
    text = clean_text(x).lower()
    text = re.sub(r"https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
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

    doi = doi.strip()
    doi = re.sub(r"\s+", "", doi)
    doi = doi.strip().strip(".,;:()[]{}<>")

    if doi in {"", "none", "null", "nan", "na", "n/a"}:
        return ""

    # DOI pattern rescue from messy strings.
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>]+)", doi, flags=re.I)
    if m:
        doi = m.group(1)
        doi = doi.strip().strip(".,;:()[]{}<>").lower()

    return doi


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_list(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        val = clean_text(item)
        if not val:
            continue
        key = normalize_key(val)
        if key and key not in seen:
            seen.add(key)
            out.append(val)

    return out


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def deep_get(obj: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_year(value: Any) -> int:
    text = clean_text(value)
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else 0


def stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


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

    skip = {"search", "__pycache__"}
    return [
        p for p in sorted(projects_dir.iterdir())
        if p.is_dir() and p.name.lower() not in skip
    ]


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ============================================================
# CONFIG READING: FLAT + GROUPED SUPPORT
# ============================================================

def get_list_from_path(config: Dict[str, Any], path: str) -> List[str]:
    value = deep_get(config, path, default=[])

    values: List[str] = []

    for item in as_list(value):
        if isinstance(item, str):
            if item.strip():
                values.append(item)
        elif isinstance(item, dict):
            # Handles objects like:
            # {"database": "PubMed", "query": "..."}
            # {"group_name": "...", "terms": [...]}
            for key in ["query", "search_string", "search_query", "terms", "keywords", "value"]:
                if key in item:
                    v = item[key]
                    if isinstance(v, list):
                        values.extend(str(x) for x in v if str(x).strip())
                    elif isinstance(v, str) and v.strip():
                        values.append(v)
            # Last-resort flatten
            if not any(k in item for k in ["query", "search_string", "search_query", "terms", "keywords", "value"]):
                for v in item.values():
                    if isinstance(v, str) and v.strip():
                        values.append(v)
                    elif isinstance(v, list):
                        values.extend(str(x) for x in v if str(x).strip())
        elif item is not None:
            s = str(item).strip()
            if s:
                values.append(s)

    return unique_list(values)


def get_string_from_paths(config: Dict[str, Any], paths: List[str]) -> str:
    for path in paths:
        value = deep_get(config, path, default="")
        value = clean_text(value)
        if value:
            return value
    return ""


def parse_date_range(config: Dict[str, Any]) -> Tuple[int, int]:
    y0 = 0
    y1 = 0

    # Grouped schema.
    y0 = parse_year(deep_get(config, "search_limits.generated_date_start", ""))
    y1 = parse_year(deep_get(config, "search_limits.generated_date_end", ""))

    if not y0:
        y0 = parse_year(deep_get(config, "search_dates.reported_date_coverage_start", ""))
    if not y1:
        y1 = parse_year(deep_get(config, "search_dates.reported_date_coverage_end", ""))

    # Flat schema fallback.
    if not y0:
        y0 = parse_year(config.get("generated_date_start", ""))
    if not y1:
        y1 = parse_year(config.get("generated_date_end", ""))

    if not y0:
        y0 = parse_year(config.get("prisma_date_coverage_start", ""))
    if not y1:
        y1 = parse_year(config.get("prisma_date_coverage_end", ""))

    # Text limits fallback.
    date_texts = [
        deep_get(config, "search_limits.reported_date_limits", ""),
        config.get("prisma_s_date_limits", ""),
    ]
    years: List[int] = []
    for text in date_texts:
        years.extend(int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", clean_text(text)))

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


def flatten_concept_terms(config: Dict[str, Any]) -> List[str]:
    terms: List[str] = []

    # Grouped schema.
    for path in [
        "review_scope.concepts.concept_1_terms",
        "review_scope.concepts.concept_2_terms",
        "review_scope.concepts.concept_3_terms",
        "review_scope.concepts.concept_4_terms",
    ]:
        terms.extend(get_list_from_path(config, path))

    # Flat schema.
    for key in [
        "concept_1_terms",
        "concept_2_terms",
        "concept_3_terms",
        "concept_4_terms",
        "concepts_keywords",
    ]:
        terms.extend(get_list_from_path(config, key))

    return unique_list(terms)


def get_concept_groups(config: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    groups: List[Tuple[str, List[str]]] = []

    grouped_paths = [
        ("concept_1", "review_scope.concepts.concept_1_terms"),
        ("concept_2", "review_scope.concepts.concept_2_terms"),
        ("concept_3", "review_scope.concepts.concept_3_terms"),
        ("concept_4", "review_scope.concepts.concept_4_terms"),
    ]

    for name, path in grouped_paths:
        terms = get_list_from_path(config, path)
        if terms:
            groups.append((name, terms))

    flat_keys = [
        ("concept_1", "concept_1_terms"),
        ("concept_2", "concept_2_terms"),
        ("concept_3", "concept_3_terms"),
        ("concept_4", "concept_4_terms"),
    ]

    for name, key in flat_keys:
        terms = get_list_from_path(config, key)
        if terms and not any(existing_name == name for existing_name, _ in groups):
            groups.append((name, terms))

    return groups


# ============================================================
# QUERY BUILDING
# ============================================================

def flatten_boolean_for_text_search(query: str) -> str:
    """
    OpenAlex / Crossref / Semantic Scholar search fields generally do better
    with plain keyword strings than exact nested Boolean syntax.
    """
    q = clean_text(query)
    q = q.replace("*", "")
    q = q.replace("?", "")
    q = re.sub(r"[(){}\[\]]", " ", q)
    q = re.sub(r"\b(AND|OR|NOT|NEAR/\d+|W/\d+)\b", " ", q, flags=re.I)
    q = q.replace('"', " ")
    q = q.replace("'", " ")
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def simplify_for_pubmed(query: str) -> str:
    """
    Keep PubMed query readable but remove database-specific syntax that often breaks.
    """
    q = clean_text(query)
    q = re.sub(r"\bTITLE-ABS-KEY\b", "", q, flags=re.I)
    q = re.sub(r"\bTS=\b", "", q, flags=re.I)
    q = re.sub(r"\bALL=\b", "", q, flags=re.I)
    q = q.replace("*", "")
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def quote_if_phrase(term: str) -> str:
    term = clean_text(term)
    if not term:
        return ""
    if " " in term and not (term.startswith('"') and term.endswith('"')):
        return f'"{term}"'
    return term


def make_keyword_queries(keywords: List[str], max_queries: int) -> List[str]:
    keywords = unique_list(keywords)

    if not keywords:
        return []

    queries: List[str] = []

    # Broad compact query.
    queries.append(" ".join(quote_if_phrase(k) for k in keywords[:6]))

    # Single important phrases.
    for k in keywords[:10]:
        if len(k) >= 4:
            queries.append(quote_if_phrase(k))

    # Pairwise combinations.
    core = keywords[:5]
    rest = keywords[5:15] if len(keywords) > 5 else keywords[1:10]

    for a in core:
        for b in rest:
            if normalize_key(a) == normalize_key(b):
                continue
            queries.append(f"{quote_if_phrase(a)} AND {quote_if_phrase(b)}")
            if len(queries) >= max_queries:
                return unique_list(queries)[:max_queries]

    return unique_list(queries)[:max_queries]


def make_concept_queries(groups: List[Tuple[str, List[str]]], max_queries: int) -> List[str]:
    """
    Concept queries are more precise than just dumping keywords.
    Generic logic:
      - group 1 + group 2
      - group 1 + group 3
      - group 1 + review terms
      - group 1 + group 2 + group 3
    """
    if not groups:
        return []

    limited_groups = [(name, terms[:8]) for name, terms in groups if terms]
    if not limited_groups:
        return []

    queries: List[str] = []

    # Use first group as anchor.
    anchor_name, anchor_terms = limited_groups[0]

    for anchor in anchor_terms[:5]:
        for other_name, other_terms in limited_groups[1:]:
            for other in other_terms[:5]:
                if normalize_key(anchor) == normalize_key(other):
                    continue
                queries.append(f"{quote_if_phrase(anchor)} AND {quote_if_phrase(other)}")
                if len(queries) >= max_queries:
                    return unique_list(queries)[:max_queries]

    # Three-way queries when at least three concept groups exist.
    if len(limited_groups) >= 3:
        g1 = limited_groups[0][1][:3]
        g2 = limited_groups[1][1][:3]
        g3 = limited_groups[2][1][:3]
        for a in g1:
            for b in g2:
                for c in g3:
                    if len({normalize_key(a), normalize_key(b), normalize_key(c)}) < 3:
                        continue
                    queries.append(
                        f"{quote_if_phrase(a)} AND {quote_if_phrase(b)} AND {quote_if_phrase(c)}"
                    )
                    if len(queries) >= max_queries:
                        return unique_list(queries)[:max_queries]

    return unique_list(queries)[:max_queries]


def extract_query_objects_from_grouped(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Handles grouped schema fields like:
      search_queries.reported_search_queries_by_database
    which may be a list of strings or dicts.
    """
    out: List[Tuple[str, str]] = []

    paths = [
        ("reported_full_search_strings", "search_queries.reported_full_search_strings"),
        ("reported_search_queries_by_database", "search_queries.reported_search_queries_by_database"),
        ("generated_search_queries", "search_queries.generated_search_queries"),
    ]

    for origin, path in paths:
        value = deep_get(config, path, default=[])

        for item in as_list(value):
            if isinstance(item, str):
                if item.strip():
                    out.append((origin, item))
            elif isinstance(item, dict):
                q = (
                    item.get("query")
                    or item.get("search_query")
                    or item.get("search_string")
                    or item.get("value")
                    or ""
                )
                if isinstance(q, str) and q.strip():
                    db = clean_text(item.get("database", ""))
                    origin2 = f"{origin}:{db}" if db else origin
                    out.append((origin2, q))

    return out


def build_queries(
    config: Dict[str, Any],
    max_queries: int,
    include_screening_keywords_in_search: bool = False,
    include_quality_keywords_in_search: bool = False,
) -> List[Tuple[str, str]]:
    """
    Returns:
      [(query_origin, query), ...]

    Search priority:
      1. Reported full search strings / query by database
      2. Generated search queries
      3. Search keywords
      4. Concept combinations
      5. Optional screening keywords
      6. Optional quality keywords
      7. Fallback title/objective
    """
    queries: List[Tuple[str, str]] = []

    # A) Grouped schema query fields.
    queries.extend(extract_query_objects_from_grouped(config))

    # B) Flat schema direct query fields.
    for key in [
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "generated_search_queries",
    ]:
        for q in get_list_from_path(config, key):
            queries.append((key, q))

    # C) Search keywords only.
    search_keywords: List[str] = []

    # Grouped schema.
    for path in [
        "search_keywords.reported_search_keywords",
        "search_keywords.reported_controlled_vocabulary_terms",
        "search_keywords.reported_free_text_terms",
        "search_keywords.generated_search_keywords",
        "review_scope.research_question_keywords",
    ]:
        search_keywords.extend(get_list_from_path(config, path))

    # Flat schema.
    for key in [
        "prisma_s_search_keywords",
        "prisma_s_controlled_vocabulary_terms",
        "prisma_s_free_text_terms",
        "generated_search_keywords",
        "research_question_keywords",
    ]:
        search_keywords.extend(get_list_from_path(config, key))

    for q in make_keyword_queries(search_keywords, max_queries=max_queries):
        queries.append(("search_keywords", q))

    # D) Concept combinations.
    concept_groups = get_concept_groups(config)
    for q in make_concept_queries(concept_groups, max_queries=max_queries):
        queries.append(("concept_queries", q))

    # E) Optional screening keywords.
    if include_screening_keywords_in_search:
        screening_keywords: List[str] = []

        for path in [
            "title_abstract_screening.generated_include_keywords",
            "title_abstract_screening.generated_must_have_keywords",
            "full_text_screening.generated_include_keywords",
            "full_text_screening.generated_must_have_keywords",
            "title_abstract_screening.reported_include_keywords",
            "title_abstract_screening.reported_must_have_keywords",
            "full_text_screening.reported_include_keywords",
            "full_text_screening.reported_must_have_keywords",
        ]:
            screening_keywords.extend(get_list_from_path(config, path))

        for key in [
            "generated_abstract_include_keywords",
            "generated_abstract_must_have_keywords",
            "generated_full_text_include_keywords",
            "generated_full_text_must_have_keywords",
            "prisma_title_abstract_include_keywords",
            "prisma_title_abstract_must_have_keywords",
            "prisma_full_text_include_keywords",
            "prisma_full_text_must_have_keywords",
        ]:
            screening_keywords.extend(get_list_from_path(config, key))

        for q in make_keyword_queries(screening_keywords, max_queries=max_queries):
            queries.append(("screening_keywords_optional", q))

    # F) Optional quality keywords. Default OFF.
    if include_quality_keywords_in_search:
        quality_keywords: List[str] = []

        for path in [
            "quality_assessment.generated_quality_assessment_keywords",
            "quality_assessment.generated_risk_of_bias_keywords",
            "quality_assessment.generated_methodological_quality_keywords",
            "quality_assessment.generated_reporting_quality_keywords",
        ]:
            quality_keywords.extend(get_list_from_path(config, path))

        for key in [
            "generated_quality_assessment_keywords",
            "generated_risk_of_bias_keywords",
            "generated_methodological_quality_keywords",
            "generated_reporting_quality_keywords",
        ]:
            quality_keywords.extend(get_list_from_path(config, key))

        for q in make_keyword_queries(quality_keywords, max_queries=max_queries):
            queries.append(("quality_keywords_optional", q))

    # G) Fallback.
    if not queries:
        title = get_string_from_paths(config, [
            "article_identity.title",
            "title",
        ])
        objective = get_string_from_paths(config, [
            "review_scope.objective",
            "objective",
        ])

        if title:
            queries.append(("title_fallback", title))
        if objective:
            queries.append(("objective_fallback", objective[:400]))

    # Global dedupe.
    out: List[Tuple[str, str]] = []
    seen = set()

    for origin, query in queries:
        query = clean_text(query)
        if not query:
            continue

        key = normalize_key(query)
        if not key or key in seen:
            continue

        out.append((origin, query))
        seen.add(key)

        if len(out) >= max_queries:
            break

    return out


def render_query_for_database(query: str, db: str) -> str:
    if db in {"openalex", "crossref", "semanticscholar"}:
        return flatten_boolean_for_text_search(query)

    if db == "pubmed":
        return simplify_for_pubmed(query)

    if db == "europepmc":
        return simplify_for_pubmed(query)

    return clean_text(query)


# ============================================================
# CACHE
# ============================================================

def query_cache_path(
    project_dir: Path,
    db: str,
    origin: str,
    raw_query: str,
    rendered_query: str,
    y0: int,
    y1: int,
    limit: int,
) -> Path:
    key = stable_hash({
        "cache_version": CACHE_VERSION,
        "database": db,
        "origin": origin,
        "raw_query": raw_query,
        "rendered_query": rendered_query,
        "year_start": y0,
        "year_end": y1,
        "limit": limit,
    })
    return project_dir / "Search" / "QueryCache" / f"{db}_{key}.json"


def load_cached_records(path: Path) -> List[ArticleRecord]:
    data = read_json(path, default={}) or {}
    records: List[ArticleRecord] = []

    for row in data.get("records", []) or []:
        if not isinstance(row, dict):
            continue
        records.append(
            ArticleRecord(**{field: row.get(field, "") for field in ARTICLE_FIELDS})
        )

    return records


def save_cached_records(
    path: Path,
    metadata: Dict[str, Any],
    records: List[ArticleRecord],
) -> None:
    ensure_dir(path.parent)
    payload = {
        **metadata,
        "cache_version": CACHE_VERSION,
        "n_records": len(records),
        "records": [asdict(record) for record in records],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


# ============================================================
# HTTP REQUESTS
# ============================================================

def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    db_name: str = "",
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params=params or {},
                headers=headers or BASE_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            if r.status_code == 429:
                sleep_time = RETRY_SLEEP_SECONDS * attempt
                if db_name == "semanticscholar":
                    sleep_time = max(sleep_time, SEMANTIC_SCHOLAR_SLEEP * attempt * 2)
                time.sleep(sleep_time)
                continue

            if r.status_code in {500, 502, 503, 504}:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS * attempt)

    raise last_error or RuntimeError("Unknown request error")


# ============================================================
# DATABASE SEARCHERS
# ============================================================

def reconstruct_openalex_abstract(inv: Any) -> str:
    if not isinstance(inv, dict):
        return ""

    positions: List[Tuple[int, str]] = []

    for word, indexes in inv.items():
        if not isinstance(indexes, list):
            continue
        for idx in indexes:
            try:
                positions.append((int(idx), str(word)))
            except Exception:
                pass

    positions.sort(key=lambda x: x[0])
    return clean_text(" ".join(word for _, word in positions))


def search_openalex(query: str, y0: int, y1: int, limit: int) -> List[ArticleRecord]:
    data = request_json(
        "https://api.openalex.org/works",
        {
            "search": query,
            "per-page": min(limit, 200),
            "filter": f"from_publication_date:{y0}-01-01,to_publication_date:{y1}-12-31",
            "mailto": CONTACT_EMAIL,
        },
        db_name="openalex",
    )

    records: List[ArticleRecord] = []

    for w in data.get("results", []) or []:
        loc = w.get("primary_location") or {}
        source = loc.get("source") or {}

        authors = []
        for a in w.get("authorships", []) or []:
            name = (a.get("author") or {}).get("display_name", "")
            if name:
                authors.append(name)

        records.append(
            ArticleRecord(
                title=clean_text(w.get("title") or w.get("display_name") or ""),
                abstract=reconstruct_openalex_abstract(w.get("abstract_inverted_index")),
                year=str(w.get("publication_year") or ""),
                journal=clean_text(source.get("display_name", "")),
                doi=normalize_doi(w.get("doi", "")),
                authors="; ".join(authors[:20]),
                url=clean_text(loc.get("landing_page_url", "") or w.get("id", "")),
                article_type=clean_text(w.get("type", "")),
                source_database="openalex",
                openalex_id=clean_text(w.get("id", "")),
            )
        )

    return records


def crossref_year(item: Dict[str, Any]) -> str:
    for key in ["published-print", "published-online", "published", "issued"]:
        parts = ((item.get(key) or {}).get("date-parts") or [])
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def search_crossref(query: str, y0: int, y1: int, limit: int) -> List[ArticleRecord]:
    data = request_json(
        "https://api.crossref.org/works",
        {
            "query": query,
            "rows": min(limit, 100),
            "filter": f"from-pub-date:{y0}-01-01,until-pub-date:{y1}-12-31",
            "mailto": CONTACT_EMAIL,
        },
        db_name="crossref",
    )

    records: List[ArticleRecord] = []

    for item in ((data.get("message") or {}).get("items") or []):
        title = clean_text(" ".join(item.get("title", []) or []))

        authors = []
        for a in item.get("author", []) or []:
            name = " ".join(
                x for x in [
                    clean_text(a.get("given", "")),
                    clean_text(a.get("family", "")),
                ]
                if x
            )
            if name:
                authors.append(name)

        records.append(
            ArticleRecord(
                title=title,
                abstract=clean_text(item.get("abstract", "")),
                year=crossref_year(item),
                journal=clean_text(" ".join(item.get("container-title", []) or [])),
                doi=normalize_doi(item.get("DOI", "")),
                authors="; ".join(authors[:20]),
                url=clean_text(item.get("URL", "")),
                article_type=clean_text(item.get("type", "")),
                source_database="crossref",
            )
        )

    return records


def search_europepmc(query: str, y0: int, y1: int, limit: int) -> List[ArticleRecord]:
    dated_query = f"({query}) AND FIRST_PDATE:[{y0}-01-01 TO {y1}-12-31]"

    data = request_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {
            "query": dated_query,
            "format": "json",
            "pageSize": min(limit, 1000),
            "resultType": "core",
        },
        db_name="europepmc",
    )

    records: List[ArticleRecord] = []

    for item in ((data.get("resultList") or {}).get("result") or []):
        url = ""
        ft = item.get("fullTextUrlList")
        if isinstance(ft, dict):
            urls = ft.get("fullTextUrl") or []
            if urls and isinstance(urls, list):
                url = clean_text(urls[0].get("url", ""))

        records.append(
            ArticleRecord(
                title=clean_text(item.get("title", "")),
                abstract=clean_text(item.get("abstractText", "")),
                year=clean_text(item.get("pubYear", "")),
                journal=clean_text(item.get("journalTitle", "")),
                doi=normalize_doi(item.get("doi", "")),
                authors=clean_text(item.get("authorString", "")),
                url=url,
                article_type=clean_text(item.get("pubType", "")),
                source_database="europepmc",
                pmid=clean_text(item.get("pmid", "")),
                pmcid=clean_text(item.get("pmcid", "")),
            )
        )

    return records


def search_pubmed(query: str, y0: int, y1: int, limit: int) -> List[ArticleRecord]:
    esearch = request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": min(limit, 200),
            "datetype": "pdat",
            "mindate": str(y0),
            "maxdate": str(y1),
        },
        db_name="pubmed",
    )

    ids = ((esearch.get("esearchresult") or {}).get("idlist") or [])
    if not ids:
        return []

    summary = request_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        },
        db_name="pubmed",
    )

    result = summary.get("result", {}) or {}
    records: List[ArticleRecord] = []

    for pmid in ids:
        item = result.get(str(pmid), {}) or {}

        year = ""
        m = re.search(r"\b(19|20)\d{2}\b", clean_text(item.get("pubdate", "")))
        if m:
            year = m.group(0)

        authors = []
        for a in item.get("authors", []) or []:
            name = a.get("name", "")
            if name:
                authors.append(name)

        doi = ""
        pmcid = ""

        for aid in item.get("articleids", []) or []:
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
            elif aid.get("idtype") == "pmc":
                pmcid = aid.get("value", "")

        records.append(
            ArticleRecord(
                title=clean_text(item.get("title", "")),
                abstract="",
                year=year,
                journal=clean_text(item.get("fulljournalname", "") or item.get("source", "")),
                doi=normalize_doi(doi),
                authors="; ".join(authors[:20]),
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                article_type=clean_text("; ".join(item.get("pubtype", []) or [])),
                source_database="pubmed",
                pmid=str(pmid),
                pmcid=clean_text(pmcid),
            )
        )

    return records


def search_semanticscholar(query: str, y0: int, y1: int, limit: int) -> List[ArticleRecord]:
    # Extra sleep because keyless Semantic Scholar requests are often rate-limited.
    time.sleep(SEMANTIC_SCHOLAR_SLEEP)

    data = request_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {
            "query": query,
            "limit": min(limit, 100),
            "fields": "paperId,title,abstract,year,venue,authors,url,externalIds,publicationTypes",
            "year": f"{y0}-{y1}",
        },
        headers=SEMANTIC_SCHOLAR_HEADERS,
        db_name="semanticscholar",
    )

    records: List[ArticleRecord] = []

    for item in data.get("data", []) or []:
        ext = item.get("externalIds", {}) or {}

        authors = []
        for a in item.get("authors", []) or []:
            name = a.get("name", "")
            if name:
                authors.append(name)

        records.append(
            ArticleRecord(
                title=clean_text(item.get("title", "")),
                abstract=clean_text(item.get("abstract", "")),
                year=str(item.get("year", "") or ""),
                journal=clean_text(item.get("venue", "")),
                doi=normalize_doi(ext.get("DOI", "")),
                authors="; ".join(authors[:20]),
                url=clean_text(item.get("url", "")),
                article_type=clean_text("; ".join(item.get("publicationTypes", []) or [])),
                source_database="semanticscholar",
                pmid=clean_text(ext.get("PubMed", "")),
                pmcid=clean_text(ext.get("PubMedCentral", "")),
                semantic_scholar_id=clean_text(item.get("paperId", "")),
            )
        )

    return records


SEARCHERS = {
    "openalex": search_openalex,
    "crossref": search_crossref,
    "europepmc": search_europepmc,
    "pubmed": search_pubmed,
    "semanticscholar": search_semanticscholar,
}


# ============================================================
# DEDUPLICATION
# ============================================================

def score_record(record: ArticleRecord) -> int:
    score = 0
    score += len(record.abstract or "")
    score += 100 if record.doi else 0
    score += 25 if record.url else 0
    score += 15 if record.authors else 0
    score += 15 if record.journal else 0
    score += 10 if record.year else 0
    score += 5 if record.pmid else 0
    score += 5 if record.pmcid else 0
    return score


def merge_records(a: ArticleRecord, b: ArticleRecord) -> ArticleRecord:
    best = a if score_record(a) >= score_record(b) else b
    other = b if best is a else a

    merged = ArticleRecord(**asdict(best))

    merged.source_database = "; ".join(unique_list([a.source_database, b.source_database]))
    merged.source_query = "; ".join(unique_list([a.source_query, b.source_query]))
    merged.rendered_query = "; ".join(unique_list([a.rendered_query, b.rendered_query]))
    merged.query_origin = "; ".join(unique_list([a.query_origin, b.query_origin]))

    for field in ARTICLE_FIELDS:
        if not getattr(merged, field, "") and getattr(other, field, ""):
            setattr(merged, field, getattr(other, field))

    return merged


def deduplicate(records: List[ArticleRecord]) -> List[ArticleRecord]:
    best: Dict[str, ArticleRecord] = {}

    for record in records:
        doi = normalize_doi(record.doi)
        title = normalize_key(record.title)

        if doi:
            key = f"doi::{doi}"
        elif title:
            key = f"title::{title}"
        else:
            key = f"unknown::{len(best)}"

        if key not in best:
            best[key] = record
        else:
            best[key] = merge_records(best[key], record)

    def sort_key(record: ArticleRecord):
        try:
            year = int(record.year)
        except Exception:
            year = 0
        return (year, score_record(record), record.title)

    return sorted(best.values(), key=sort_key, reverse=True)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_articles_csv(path: Path, records: List[ArticleRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARTICLE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field, "") for field in ARTICLE_FIELDS})


def save_query_log_csv(path: Path, records: List[QueryLogRecord]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUERY_LOG_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field, "") for field in QUERY_LOG_FIELDS})


# ============================================================
# SEARCH JOB
# ============================================================

def run_search_job(
    project_dir: Path,
    project_name: str,
    origin: str,
    raw_query: str,
    db: str,
    y0: int,
    y1: int,
    max_results_per_query: int,
    refresh_cache: bool,
) -> Tuple[List[ArticleRecord], QueryLogRecord]:
    rendered_query = render_query_for_database(raw_query, db)

    cache_path = query_cache_path(
        project_dir=project_dir,
        db=db,
        origin=origin,
        raw_query=raw_query,
        rendered_query=rendered_query,
        y0=y0,
        y1=y1,
        limit=max_results_per_query,
    )

    log = QueryLogRecord(
        project=project_name,
        database=db,
        query_origin=origin,
        query=raw_query,
        rendered_query=rendered_query,
        year_start=y0,
        year_end=y1,
        max_results_per_query=max_results_per_query,
        status="",
        n_returned=0,
        error="",
        cache_used=False,
        cache_path=str(cache_path),
    )

    if cache_path.exists() and not refresh_cache:
        records = load_cached_records(cache_path)
        log.status = "cached"
        log.n_returned = len(records)
        log.cache_used = True
        return records, log

    try:
        searcher = SEARCHERS[db]
        records = searcher(rendered_query, y0, y1, max_results_per_query)

        for record in records:
            record.source_database = db
            record.source_query = raw_query
            record.rendered_query = rendered_query
            record.query_origin = origin

        metadata = {
            "project": project_name,
            "database": db,
            "query_origin": origin,
            "query": raw_query,
            "rendered_query": rendered_query,
            "year_start": y0,
            "year_end": y1,
            "max_results_per_query": max_results_per_query,
        }
        save_cached_records(cache_path, metadata, records)

        log.status = "searched"
        log.n_returned = len(records)
        log.error = ""
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return records, log

    except Exception as e:
        log.status = "failed"
        log.n_returned = 0
        log.error = f"{type(e).__name__}: {e}"
        return [], log


# ============================================================
# PROCESS PROJECT
# ============================================================

def clear_previous_search_outputs(search_dir: Path) -> None:
    articles_path = search_dir / "articles.csv"
    query_log_path = search_dir / "query_log.csv"

    for path in [articles_path, query_log_path]:
        if path.exists():
            path.unlink()

    # Keep query cache unless --refresh-cache is used.
    ensure_dir(search_dir)


def process_project(
    project_dir: Path,
    databases: List[str],
    max_results_per_query: int,
    max_queries: int,
    workers: int,
    force: bool,
    refresh_cache: bool,
    include_screening_keywords_in_search: bool,
    include_quality_keywords_in_search: bool,
) -> Dict[str, Any]:
    config_path = project_dir / "config.json"
    search_dir = project_dir / "Search"
    articles_path = search_dir / "articles.csv"
    query_log_path = search_dir / "query_log.csv"

    if not config_path.exists():
        return {
            "project": project_dir.name,
            "status": "failed",
            "error": "config.json not found",
            "articles": 0,
            "unique_articles": 0,
            "queries": 0,
        }

    if articles_path.exists() and query_log_path.exists() and not force:
        return {
            "project": project_dir.name,
            "status": "skipped_existing_search",
            "error": "",
            "articles": 0,
            "unique_articles": 0,
            "queries": 0,
        }

    ensure_dir(search_dir)

    if force:
        clear_previous_search_outputs(search_dir)

    if refresh_cache:
        cache_dir = search_dir / "QueryCache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        ensure_dir(cache_dir)

    config = read_json(config_path, default={}) or {}
    y0, y1 = parse_date_range(config)

    query_pairs = build_queries(
        config=config,
        max_queries=max_queries,
        include_screening_keywords_in_search=include_screening_keywords_in_search,
        include_quality_keywords_in_search=include_quality_keywords_in_search,
    )

    if not query_pairs:
        save_articles_csv(articles_path, [])
        save_query_log_csv(query_log_path, [])
        return {
            "project": project_dir.name,
            "status": "failed",
            "error": "no queries generated from config",
            "articles": 0,
            "unique_articles": 0,
            "queries": 0,
        }

    jobs = []
    for origin, query in query_pairs:
        for db in databases:
            if db not in SEARCHERS:
                continue
            jobs.append((origin, query, db))

    all_records: List[ArticleRecord] = []
    query_logs: List[QueryLogRecord] = []

    # To reduce rate-limit collapse, run Semantic Scholar jobs with fewer parallel hits.
    regular_jobs = [job for job in jobs if job[2] != "semanticscholar"]
    semantic_jobs = [job for job in jobs if job[2] == "semanticscholar"]

    def submit_jobs(job_list: List[Tuple[str, str, str]], job_workers: int) -> None:
        nonlocal all_records, query_logs

        if not job_list:
            return

        with ThreadPoolExecutor(max_workers=max(1, job_workers)) as ex:
            future_map = {
                ex.submit(
                    run_search_job,
                    project_dir,
                    project_dir.name,
                    origin,
                    query,
                    db,
                    y0,
                    y1,
                    max_results_per_query,
                    refresh_cache,
                ): (origin, query, db)
                for origin, query, db in job_list
            }

            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc=f"Searching {project_dir.name}",
                leave=False,
            ):
                records, log = future.result()
                all_records.extend(records)
                query_logs.append(log)

    submit_jobs(regular_jobs, workers)
    submit_jobs(semantic_jobs, 1)

    unique_records = deduplicate(all_records)

    save_articles_csv(articles_path, unique_records)
    save_query_log_csv(query_log_path, query_logs)

    n_failed = sum(1 for q in query_logs if q.status == "failed")
    n_zero = sum(1 for q in query_logs if q.n_returned == 0)
    n_total_returned = sum(q.n_returned for q in query_logs)

    return {
        "project": project_dir.name,
        "status": "success",
        "error": "",
        "articles": len(all_records),
        "unique_articles": len(unique_records),
        "queries": len(query_logs),
        "failed_queries": n_failed,
        "zero_result_queries": n_zero,
        "total_returned_before_dedup": n_total_returned,
        "articles_path": str(articles_path),
        "query_log_path": str(query_log_path),
    }


# ============================================================
# SUMMARY PRINTING
# ============================================================

def print_project_result(result: Dict[str, Any]) -> None:
    status = result.get("status", "")
    project = result.get("project", "")
    error = result.get("error", "")

    if status == "success":
        print(
            f"OK | {project} | "
            f"queries={result.get('queries', 0)} | "
            f"returned={result.get('total_returned_before_dedup', 0)} | "
            f"unique={result.get('unique_articles', 0)} | "
            f"failed_queries={result.get('failed_queries', 0)} | "
            f"zero_queries={result.get('zero_result_queries', 0)}"
        )
    elif status == "skipped_existing_search":
        print(f"SKIP | {project} | existing Search/articles.csv and Search/query_log.csv")
    else:
        print(f"FAIL | {project} | {error}")


def save_run_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "project",
        "status",
        "queries",
        "total_returned_before_dedup",
        "articles",
        "unique_articles",
        "failed_queries",
        "zero_result_queries",
        "articles_path",
        "query_log_path",
        "error",
    ]

    ensure_dir(path.parent)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic literature search from ALR config.json files."
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
        "--databases",
        default=",".join(DEFAULT_DATABASES),
        help="Comma-separated databases: openalex,crossref,europepmc,pubmed,semanticscholar",
    )

    parser.add_argument(
        "--max-results-per-query",
        type=int,
        default=DEFAULT_MAX_RESULTS_PER_QUERY,
        help="Maximum records requested per query per database.",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=DEFAULT_MAX_QUERIES,
        help="Maximum query strings generated per project before database expansion.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel workers for non-Semantic Scholar databases.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Search/articles.csv and Search/query_log.csv.",
    )

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Delete and refresh cached API query results.",
    )

    parser.add_argument(
        "--include-screening-keywords-in-search",
        action="store_true",
        help="Also use screening include/must-have keywords as search queries. Default OFF.",
    )

    parser.add_argument(
        "--include-quality-keywords-in-search",
        action="store_true",
        help="Also use quality-assessment keywords as search queries. Default OFF.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    databases = [
        x.strip().lower()
        for x in args.databases.split(",")
        if x.strip().lower()
    ]

    invalid = [db for db in databases if db not in SEARCHERS]
    if invalid:
        raise ValueError(f"Invalid databases: {invalid}. Valid: {sorted(SEARCHERS)}")

    print("=" * 90)
    print("STEP 7: GENERIC LITERATURE SEARCH")
    print("=" * 90)
    print(f"Projects directory                 : {projects_dir.resolve()}")
    print(f"Projects found                     : {len(projects)}")
    print(f"Databases                          : {', '.join(databases)}")
    print(f"Max results per query              : {args.max_results_per_query}")
    print(f"Max queries per project            : {args.max_queries}")
    print(f"Workers                            : {args.workers}")
    print(f"Force overwrite                    : {args.force}")
    print(f"Refresh cache                      : {args.refresh_cache}")
    print(f"Use screening keywords in search   : {args.include_screening_keywords_in_search}")
    print(f"Use quality keywords in search     : {args.include_quality_keywords_in_search}")
    print(f"Semantic Scholar API key available : {bool(SEMANTIC_SCHOLAR_API_KEY)}")
    print("References/gold files read         : NO")
    print("=" * 90)

    results: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        result = process_project(
            project_dir=project_dir,
            databases=databases,
            max_results_per_query=args.max_results_per_query,
            max_queries=args.max_queries,
            workers=args.workers,
            force=args.force,
            refresh_cache=args.refresh_cache,
            include_screening_keywords_in_search=args.include_screening_keywords_in_search,
            include_quality_keywords_in_search=args.include_quality_keywords_in_search,
        )

        results.append(result)
        print_project_result(result)

    summary_path = projects_dir / "Search" / "step7_search_run_summary.csv"
    save_run_summary(summary_path, results)

    n_success = sum(1 for r in results if r.get("status") == "success")
    n_skip = sum(1 for r in results if r.get("status") == "skipped_existing_search")
    n_fail = sum(1 for r in results if r.get("status") == "failed")

    total_unique = sum(int(r.get("unique_articles", 0) or 0) for r in results)
    total_queries = sum(int(r.get("queries", 0) or 0) for r in results)
    total_failed_queries = sum(int(r.get("failed_queries", 0) or 0) for r in results)

    print("\n" + "=" * 90)
    print("STEP 7 COMPLETE")
    print("=" * 90)
    print(f"Success projects : {n_success}")
    print(f"Skipped projects : {n_skip}")
    print(f"Failed projects  : {n_fail}")
    print(f"Total queries    : {total_queries}")
    print(f"Failed queries   : {total_failed_queries}")
    print(f"Total unique articles saved across projects: {total_unique}")
    print(f"Run summary saved to: {summary_path}")
    print("\nPer project outputs:")
    print("  Projects/<project>/Search/articles.csv")
    print("  Projects/<project>/Search/query_log.csv")


if __name__ == "__main__":
    main()




