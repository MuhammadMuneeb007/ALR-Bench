# Step9-Rank5-GeneratedQueryMatch.py
#
# Rank Strategy 5:
#   Generated/reported query match ranking using local metadata.
#
# What this script does:
#   1. Reads:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Extracts search query text from config:
#        - generated_search_queries
#        - search_queries.generated_search_queries
#        - search_queries.reported_full_search_strings
#        - search_queries.reported_search_queries_by_database
#        - prisma_s_full_search_strings
#        - prisma_s_search_queries_by_database
#
#   3. Scores articles by:
#        - exact query phrase match
#        - query token coverage
#        - best query match
#        - average query match
#        - title query match boost
#        - multi-query coverage
#
#   4. Saves:
#        Projects/<project>/Rankings/rank_strategy_5.csv
#        Projects/Rankings/rank_strategy_5_summary.csv
#
#   5. Prints:
#        Common@10, Common@20, Common@50, Common@100
#
# Notes:
#   - This is a ranking/evaluation step, so reading references is allowed.
#   - No API calls.
#   - No PDF download.
#   - No full-text download.
#   - No LLM call.
#
# Usage:
#   python Step9-Rank5-GeneratedQueryMatch.py --projects-dir Projects --force
#
# One project:
#   python Step9-Rank5-GeneratedQueryMatch.py --projects-dir Projects --project PROJECT_NAME --force

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")

DEFAULT_ARTICLES_CLEAN_REL = Path("Search") / "articles_clean_doi_only.csv"
DEFAULT_ARTICLES_RAW_REL = Path("Search") / "articles.csv"

DEFAULT_RANKINGS_DIR = Path("Rankings")
DEFAULT_RANK_FILE = "rank_strategy_5.csv"

TOP_K_VALUES = [10, 20, 50, 100]

MIN_TOKEN_LEN = 2

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "had", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "these", "this", "those", "to", "was", "were", "which", "with",
    "using", "use", "used", "based", "between", "among", "study",
    "studies", "paper", "article", "articles", "method", "methods",
    "approach", "approaches", "result", "results", "data", "dataset",
    "review", "survey", "systematic", "literature", "meta", "analysis",
}


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


def normalize_phrase(x: Any) -> str:
    text = clean_text(x).lower()
    text = text.replace("*", "")
    text = re.sub(r"\b(title-abs-key|tiab|mesh|all|ts)\b", " ", text, flags=re.I)
    text = re.sub(r"\b(and|or|not|near/\d+|w/\d+)\b", " ", text, flags=re.I)
    text = re.sub(r"[\(\)\[\]\{\}\"']", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: Any) -> List[str]:
    text = normalize_phrase(text)
    toks = re.findall(r"[a-z0-9]+", text)

    out: List[str] = []
    for tok in toks:
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)

    return out


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


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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

    skip = {"search", "evaluation", "rankings", "__pycache__"}

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


def flatten_values(value: Any) -> List[str]:
    out: List[str] = []

    if value is None:
        return out

    if isinstance(value, str):
        if value.strip():
            out.append(value)
        return out

    if isinstance(value, list):
        for item in value:
            out.extend(flatten_values(item))
        return out

    if isinstance(value, dict):
        preferred = [
            "query",
            "search_query",
            "search_string",
            "value",
            "text",
            "terms",
            "keywords",
            "database",
        ]

        used = False

        for key in preferred:
            if key in value:
                out.extend(flatten_values(value.get(key)))
                used = True

        if not used:
            for v in value.values():
                out.extend(flatten_values(v))

        return out

    s = str(value).strip()
    if s:
        out.append(s)

    return out


def unique_list(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()

    for item in items:
        text = clean_text(item)
        if not text:
            continue

        key = normalize_phrase(text)
        if not key or key in seen:
            continue

        seen.add(key)
        out.append(text)

    return out


def get_list_from_path(config: Dict[str, Any], path: str) -> List[str]:
    return unique_list(flatten_values(deep_get(config, path, default=[])))


def safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return num / den


def normalize_scores(raw_scores: List[float]) -> List[float]:
    if not raw_scores:
        return []

    min_s = min(raw_scores)
    max_s = max(raw_scores)

    if max_s <= min_s:
        return [0.0 for _ in raw_scores]

    return [(x - min_s) / (max_s - min_s) for x in raw_scores]


# ============================================================
# QUERY PROFILE
# ============================================================

def extract_query_objects_from_grouped(config: Dict[str, Any]) -> List[str]:
    queries: List[str] = []

    paths = [
        "search_queries.reported_full_search_strings",
        "search_queries.reported_search_queries_by_database",
        "search_queries.generated_search_queries",
    ]

    for path in paths:
        value = deep_get(config, path, default=[])

        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    queries.append(item)
                elif isinstance(item, dict):
                    q = (
                        item.get("query")
                        or item.get("search_query")
                        or item.get("search_string")
                        or item.get("value")
                        or ""
                    )
                    if q:
                        queries.append(str(q))

        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, str):
                    queries.append(item)
                elif isinstance(item, list):
                    for x in item:
                        if isinstance(x, str):
                            queries.append(x)
                        elif isinstance(x, dict):
                            q = (
                                x.get("query")
                                or x.get("search_query")
                                or x.get("search_string")
                                or x.get("value")
                                or ""
                            )
                            if q:
                                queries.append(str(q))

        elif isinstance(value, str):
            queries.append(value)

    return unique_list(queries)


def build_query_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    query_terms: List[str] = []

    query_terms.extend(extract_query_objects_from_grouped(config))

    for path in [
        "generated_search_queries",
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
    ]:
        query_terms.extend(get_list_from_path(config, path))

    # Fallback: if no query strings exist, use generated keywords and concepts.
    fallback_used = False

    if not unique_list(query_terms):
        fallback_used = True
        for path in [
            "generated_search_keywords",
            "concepts_keywords",
            "search_keywords.generated_search_keywords",
            "review_scope.concepts.concept_1_terms",
            "review_scope.concepts.concept_2_terms",
            "review_scope.concepts.concept_3_terms",
            "review_scope.concepts.concept_4_terms",
        ]:
            query_terms.extend(get_list_from_path(config, path))

    query_terms = unique_list(query_terms)

    query_phrases: List[str] = []
    query_tokens: List[str] = []

    for query in query_terms:
        norm = normalize_phrase(query)
        if norm:
            query_phrases.append(norm)
        query_tokens.extend(tokenize(query))

    query_tokens = sorted(set(query_tokens))

    return {
        "query_terms": query_terms,
        "query_phrases": unique_list(query_phrases),
        "query_tokens": query_tokens,
        "fallback_used": fallback_used,
    }


# ============================================================
# MATCHING
# ============================================================

def phrase_present(phrase: str, text_norm: str) -> bool:
    p = normalize_phrase(phrase)
    if not p:
        return False

    if " " not in p:
        return bool(re.search(rf"\b{re.escape(p)}\b", text_norm))

    return p in text_norm


def query_match_score(
    query: str,
    title_norm: str,
    abstract_norm: str,
    body_norm: str,
    article_tokens: Set[str],
) -> Dict[str, Any]:

    query_norm = normalize_phrase(query)
    query_tokens = sorted(set(tokenize(query)))

    if not query_norm and not query_tokens:
        return {
            "score": 0.0,
            "title_phrase_hit": False,
            "abstract_phrase_hit": False,
            "body_phrase_hit": False,
            "token_coverage": 0.0,
            "matched_tokens": [],
        }

    title_phrase_hit = bool(query_norm and phrase_present(query_norm, title_norm))
    abstract_phrase_hit = bool(query_norm and phrase_present(query_norm, abstract_norm))
    body_phrase_hit = bool(query_norm and phrase_present(query_norm, body_norm))

    matched_tokens = [tok for tok in query_tokens if tok in article_tokens]
    token_coverage = safe_div(len(matched_tokens), len(query_tokens))

    phrase_component = 0.0

    if title_phrase_hit:
        phrase_component = 1.0
    elif abstract_phrase_hit:
        phrase_component = 0.75
    elif body_phrase_hit:
        phrase_component = 0.50

    score = 0.60 * token_coverage + 0.40 * phrase_component

    return {
        "score": score,
        "title_phrase_hit": title_phrase_hit,
        "abstract_phrase_hit": abstract_phrase_hit,
        "body_phrase_hit": body_phrase_hit,
        "token_coverage": token_coverage,
        "matched_tokens": matched_tokens,
    }


def all_query_scores(
    queries: List[str],
    title_norm: str,
    abstract_norm: str,
    body_norm: str,
    article_tokens: Set[str],
) -> Dict[str, Any]:

    if not queries:
        return {
            "best_query_score": 0.0,
            "avg_query_score": 0.0,
            "multi_query_coverage": 0.0,
            "title_query_coverage": 0.0,
            "matched_queries": [],
            "matched_title_queries": [],
            "matched_query_tokens": [],
        }

    scores: List[float] = []
    matched_queries: List[str] = []
    matched_title_queries: List[str] = []
    matched_query_tokens: List[str] = []

    for query in queries:
        res = query_match_score(
            query=query,
            title_norm=title_norm,
            abstract_norm=abstract_norm,
            body_norm=body_norm,
            article_tokens=article_tokens,
        )

        score = float(res["score"])
        scores.append(score)

        if score > 0:
            matched_queries.append(query)
            matched_query_tokens.extend(res["matched_tokens"])

        if res["title_phrase_hit"]:
            matched_title_queries.append(query)

    best_score = max(scores) if scores else 0.0
    avg_score = safe_div(sum(scores), len(scores))
    multi_query_coverage = safe_div(len(matched_queries), len(queries))
    title_query_coverage = safe_div(len(matched_title_queries), len(queries))

    return {
        "best_query_score": best_score,
        "avg_query_score": avg_score,
        "multi_query_coverage": multi_query_coverage,
        "title_query_coverage": title_query_coverage,
        "matched_queries": unique_list(matched_queries),
        "matched_title_queries": unique_list(matched_title_queries),
        "matched_query_tokens": sorted(set(matched_query_tokens)),
    }


def global_query_token_coverage(
    query_tokens: List[str],
    article_tokens: Set[str],
) -> Tuple[float, List[str]]:

    if not query_tokens:
        return 0.0, []

    matched = [tok for tok in query_tokens if tok in article_tokens]
    return safe_div(len(matched), len(query_tokens)), matched


# ============================================================
# REFERENCE LOADING
# ============================================================

def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
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


def load_reference_dois(project_dir: Path) -> Tuple[Set[str], str, int]:
    candidates = [
        "references_gold.json",
        "references_resolved.json",
        "references.json",
    ]

    for rel in candidates:
        path = project_dir / rel

        if not path.exists():
            continue

        data = read_json(path, default={})
        records = extract_reference_records_from_json(data)

        if records:
            dois = {
                normalize_doi(x.get("doi", ""))
                for x in records
                if normalize_doi(x.get("doi", ""))
            }
            return dois, rel, len(records)

    return set(), "", 0


def load_retrievable_ceiling_dois(project_dir: Path) -> Set[str]:
    path = project_dir / "Evaluation" / "retrievable_ceiling_references.csv"
    rows = read_csv_rows(path)

    out: Set[str] = set()

    for row in rows:
        doi = normalize_doi(row.get("doi", ""))
        flag = clean_text(row.get("retrievable_ceiling", "")).lower()

        if doi and flag in {"true", "1", "yes", "y"}:
            out.add(doi)

    return out


# ============================================================
# ARTICLE LOADING
# ============================================================

def load_article_rows(project_dir: Path) -> Tuple[List[Dict[str, str]], str]:
    clean_path = project_dir / DEFAULT_ARTICLES_CLEAN_REL
    raw_path = project_dir / DEFAULT_ARTICLES_RAW_REL

    if clean_path.exists():
        rows = read_csv_rows(clean_path)
        return rows, str(clean_path)

    rows = read_csv_rows(raw_path)
    return rows, str(raw_path)


# ============================================================
# RANKING
# ============================================================

def rank_articles(
    rows: List[Dict[str, str]],
    profile: Dict[str, Any],
    ref_dois: Set[str],
    ceiling_dois: Set[str],
) -> List[Dict[str, Any]]:

    if not rows:
        return []

    raw_scores: List[float] = []
    feature_rows: List[Dict[str, Any]] = []

    queries = profile["query_terms"]
    query_tokens = profile["query_tokens"]

    for row in rows:
        title = clean_text(row.get("title", ""))
        abstract = clean_text(row.get("abstract", ""))
        journal = clean_text(row.get("journal", ""))
        article_type = clean_text(row.get("article_type", ""))
        source_query = clean_text(row.get("source_query", ""))
        rendered_query = clean_text(row.get("rendered_query", ""))

        title_norm = normalize_phrase(title)
        abstract_norm = normalize_phrase(abstract)
        body_norm = normalize_phrase(" ".join([
            title,
            abstract,
            journal,
            article_type,
            source_query,
            rendered_query,
        ]))

        article_tokens = set(tokenize(" ".join([
            title,
            abstract,
            journal,
            article_type,
            source_query,
            rendered_query,
        ])))

        query_scores = all_query_scores(
            queries=queries,
            title_norm=title_norm,
            abstract_norm=abstract_norm,
            body_norm=body_norm,
            article_tokens=article_tokens,
        )

        global_token_score, global_matched_tokens = global_query_token_coverage(
            query_tokens=query_tokens,
            article_tokens=article_tokens,
        )

        source_query_bonus = 0.0
        matched_source_query_terms: List[str] = []

        for q in queries:
            q_norm = normalize_phrase(q)
            if q_norm and q_norm in normalize_phrase(source_query + " " + rendered_query):
                source_query_bonus = max(source_query_bonus, 1.0)
                matched_source_query_terms.append(q)

        missing_abstract_penalty = 0.05 if not abstract else 0.0

        raw_score = (
            0.35 * query_scores["best_query_score"]
            + 0.20 * query_scores["avg_query_score"]
            + 0.15 * query_scores["multi_query_coverage"]
            + 0.10 * query_scores["title_query_coverage"]
            + 0.15 * global_token_score
            + 0.05 * source_query_bonus
            - missing_abstract_penalty
        )

        raw_scores.append(raw_score)

        feature_rows.append({
            "score_best_query_match": query_scores["best_query_score"],
            "score_average_query_match": query_scores["avg_query_score"],
            "score_multi_query_coverage": query_scores["multi_query_coverage"],
            "score_title_query_coverage": query_scores["title_query_coverage"],
            "score_global_query_token_coverage": global_token_score,
            "score_source_query_bonus": source_query_bonus,
            "missing_abstract_penalty": missing_abstract_penalty,
            "matched_queries": "; ".join(query_scores["matched_queries"]),
            "matched_title_queries": "; ".join(query_scores["matched_title_queries"]),
            "matched_query_tokens": "; ".join(sorted(set(query_scores["matched_query_tokens"]))[:150]),
            "global_matched_query_tokens": "; ".join(global_matched_tokens[:150]),
            "matched_source_query_terms": "; ".join(unique_list(matched_source_query_terms)),
        })

    norm_scores = normalize_scores(raw_scores)

    ranked_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        doi = normalize_doi(row.get("doi", ""))

        out = dict(row)
        out["doi"] = doi
        out["rank_strategy_5_score"] = round(norm_scores[i], 8)
        out["rank_strategy_5_raw_score"] = round(raw_scores[i], 8)

        for key, value in feature_rows[i].items():
            if isinstance(value, float):
                out[key] = round(value, 8)
            else:
                out[key] = value

        out["is_reference_match"] = bool(doi and doi in ref_dois)
        out["is_retrievable_reference_match"] = bool(doi and doi in ceiling_dois)

        ranked_rows.append(out)

    ranked_rows.sort(
        key=lambda r: (
            float(r.get("rank_strategy_5_score", 0.0) or 0.0),
            clean_text(r.get("title", "")),
        ),
        reverse=True,
    )

    for idx, row in enumerate(ranked_rows, start=1):
        row["rank"] = idx

    return ranked_rows


def count_common_at_k(
    ranked_rows: List[Dict[str, Any]],
    ref_dois: Set[str],
    ceiling_dois: Set[str],
    k: int,
) -> Dict[str, Any]:

    top_rows = ranked_rows[:k]

    top_dois = {
        normalize_doi(row.get("doi", ""))
        for row in top_rows
        if normalize_doi(row.get("doi", ""))
    }

    common_raw = top_dois & ref_dois
    common_ceiling = top_dois & ceiling_dois if ceiling_dois else set()

    return {
        f"common_doi_at_{k}": len(common_raw),
        f"common_retrievable_at_{k}": len(common_ceiling) if ceiling_dois else "",
    }


# ============================================================
# PROJECT PROCESSING
# ============================================================

def process_project(project_dir: Path, force: bool) -> Dict[str, Any]:
    config_path = project_dir / "config.json"

    if not config_path.exists():
        return {
            "project": project_dir.name,
            "status": "failed_no_config",
        }

    ranking_dir = project_dir / DEFAULT_RANKINGS_DIR
    rank_path = ranking_dir / DEFAULT_RANK_FILE

    if rank_path.exists() and not force:
        return {
            "project": project_dir.name,
            "status": "skipped_existing_ranking",
            "rank_file": str(rank_path),
        }

    config = read_json(config_path, default={}) or {}

    rows, articles_source = load_article_rows(project_dir)

    rows = [
        dict(row, doi=normalize_doi(row.get("doi", "")))
        for row in rows
        if normalize_doi(row.get("doi", ""))
    ]

    ref_dois, ref_source, references_total = load_reference_dois(project_dir)
    ceiling_dois = load_retrievable_ceiling_dois(project_dir)

    profile = build_query_profile(config)

    ranked_rows = rank_articles(
        rows=rows,
        profile=profile,
        ref_dois=ref_dois,
        ceiling_dois=ceiling_dois,
    )

    base_fields = list(ranked_rows[0].keys()) if ranked_rows else [
        "rank",
        "rank_strategy_5_score",
        "rank_strategy_5_raw_score",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
    ]

    preferred_first = [
        "rank",
        "rank_strategy_5_score",
        "rank_strategy_5_raw_score",
        "score_best_query_match",
        "score_average_query_match",
        "score_multi_query_coverage",
        "score_title_query_coverage",
        "score_global_query_token_coverage",
        "score_source_query_bonus",
        "missing_abstract_penalty",
        "matched_queries",
        "matched_title_queries",
        "matched_query_tokens",
        "global_matched_query_tokens",
        "matched_source_query_terms",
        "is_reference_match",
        "is_retrievable_reference_match",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
        "source_query",
        "rendered_query",
        "query_origin",
        "url",
    ]

    fields: List[str] = []

    for f in preferred_first:
        if f in base_fields and f not in fields:
            fields.append(f)

    for f in base_fields:
        if f not in fields:
            fields.append(f)

    write_csv(rank_path, ranked_rows, fields)

    summary: Dict[str, Any] = {
        "project": project_dir.name,
        "status": "success",
        "articles_source": articles_source,
        "rank_file": str(rank_path),
        "articles_ranked": len(ranked_rows),
        "references_source": ref_source,
        "references_total": references_total,
        "references_with_doi": len(ref_dois),
        "retrievable_ceiling": len(ceiling_dois) if ceiling_dois else "",
        "n_query_terms": len(profile["query_terms"]),
        "n_query_tokens": len(profile["query_tokens"]),
        "fallback_used": profile["fallback_used"],
    }

    for k in TOP_K_VALUES:
        summary.update(count_common_at_k(
            ranked_rows=ranked_rows,
            ref_dois=ref_dois,
            ceiling_dois=ceiling_dois,
            k=k,
        ))

    return summary


# ============================================================
# PRINTING
# ============================================================

SUMMARY_FIELDS = [
    "project",
    "status",
    "articles_ranked",
    "references_total",
    "references_with_doi",
    "retrievable_ceiling",
    "common_doi_at_10",
    "common_doi_at_20",
    "common_doi_at_50",
    "common_doi_at_100",
    "common_retrievable_at_10",
    "common_retrievable_at_20",
    "common_retrievable_at_50",
    "common_retrievable_at_100",
    "n_query_terms",
    "n_query_tokens",
    "fallback_used",
    "articles_source",
    "rank_file",
    "references_source",
]


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    compact_rows = []

    for row in rows:
        compact_rows.append({
            col: str(row.get(col, ""))
            for col in SUMMARY_FIELDS
        })

    widths: Dict[str, int] = {}

    for col in SUMMARY_FIELDS:
        widths[col] = max(
            len(col),
            *(len(row.get(col, "")) for row in compact_rows),
        )

    header = " | ".join(col.ljust(widths[col]) for col in SUMMARY_FIELDS)
    sep = "-+-".join("-" * widths[col] for col in SUMMARY_FIELDS)

    print(header)
    print(sep)

    for row in compact_rows:
        print(" | ".join(row.get(col, "").ljust(widths[col]) for col in SUMMARY_FIELDS))


def save_global_summary(projects_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    out_path = projects_dir / "Rankings" / "rank_strategy_5_summary.csv"
    write_csv(out_path, rows, SUMMARY_FIELDS)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank retrieved articles using generated/reported query matching."
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
        help="Overwrite existing ranking outputs.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    print("=" * 140)
    print("STEP 9: RANK STRATEGY 5 — GENERATED QUERY MATCH RANKING")
    print("=" * 140)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Force overwrite    : {args.force}")
    print(f"Top-K values       : {TOP_K_VALUES}")
    print("Uses APIs          : NO")
    print("Uses LLM           : NO")
    print("Uses PDFs/fulltext : NO")
    print("=" * 140)

    summaries: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(project_dir, force=args.force)
            summaries.append(summary)

            if summary.get("status") == "success":
                print(
                    f"OK | ranked={summary.get('articles_ranked')} | "
                    f"refs_doi={summary.get('references_with_doi')} | "
                    f"C@10={summary.get('common_doi_at_10')} | "
                    f"C@20={summary.get('common_doi_at_20')} | "
                    f"C@50={summary.get('common_doi_at_50')} | "
                    f"C@100={summary.get('common_doi_at_100')} | "
                    f"file={summary.get('rank_file')}"
                )
            else:
                print(f"{summary.get('status')} | {summary.get('rank_file', '')}")

        except Exception as e:
            print(f"FAIL | {project_dir.name} | {type(e).__name__}: {e}")
            summaries.append({
                "project": project_dir.name,
                "status": f"failed: {type(e).__name__}: {e}",
            })

    summary_path = save_global_summary(projects_dir, summaries)

    print("\n" + "=" * 140)
    print("SUMMARY TABLE")
    print("=" * 140)
    print_summary_table(summaries)

    total_ranked = sum(int(x.get("articles_ranked", 0) or 0) for x in summaries)
    total_refs_doi = sum(int(x.get("references_with_doi", 0) or 0) for x in summaries)

    print("\n" + "=" * 140)
    print("TOTALS")
    print("=" * 140)
    print(f"Total ranked articles : {total_ranked}")
    print(f"Total reference DOIs  : {total_refs_doi}")

    for k in TOP_K_VALUES:
        total_common = sum(int(x.get(f"common_doi_at_{k}", 0) or 0) for x in summaries)
        print(f"Total Common@{k:<3}   : {total_common}")

    print(f"\nSaved global summary to: {summary_path}")
    print("\nPer-project ranking files:")
    print("  Projects/<project>/Rankings/rank_strategy_5.csv")


if __name__ == "__main__":
    main()