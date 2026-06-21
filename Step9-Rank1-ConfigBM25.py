# Step9-Rank1.py
#
# Rank Strategy 1:
#   Config-guided abstract/title ranking using only local metadata.
#
# What this script does:
#   1. Reads:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Builds a generic ranking profile from config:
#        - objective
#        - research questions
#        - concepts
#        - search keywords
#        - search queries
#        - inclusion keywords
#        - title/abstract screening keywords
#        - full-text screening keywords
#        - exclusion keywords as penalty
#
#   3. Ranks articles using:
#        - BM25-like abstract/title score
#        - exact keyword/phrase score
#        - concept coverage score
#        - inclusion/screening keyword score
#        - title relevance score
#        - exclusion penalty
#
#   4. Saves:
#        Projects/<project>/Rankings/rank_strategy_1.csv
#        Projects/Rankings/rank_strategy_1_summary.csv
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
#   - Fully generic; no project-specific hardcoding.
#
# Usage:
#   python Step9-Rank1.py --projects-dir Projects --force
#
# One project:
#   python Step9-Rank1.py --projects-dir Projects --project PROJECT_NAME --force

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")

DEFAULT_ARTICLES_CLEAN_REL = Path("Search") / "articles_clean_doi_only.csv"
DEFAULT_ARTICLES_RAW_REL = Path("Search") / "articles.csv"

DEFAULT_RANKINGS_DIR = Path("Rankings")
DEFAULT_RANK_FILE = "rank_strategy_1.csv"

TOP_K_VALUES = [10, 20, 50, 100]

MIN_TOKEN_LEN = 2

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "had", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "these", "this", "those", "to", "was", "were", "which", "with",
    "using", "use", "used", "based", "between", "among", "study",
    "studies", "review", "systematic", "literature", "scoping",
    "analysis", "research", "paper", "article", "articles", "model",
    "models", "method", "methods", "approach", "approaches"
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
    text = re.sub(r"\b(and|or|not)\b", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: Any) -> List[str]:
    text = normalize_phrase(text)
    tokens = re.findall(r"[a-z0-9]+", text)

    out = []
    for tok in tokens:
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


def flatten_values(value: Any) -> List[str]:
    """
    Generic flattener for strings/lists/dicts from config fields.
    """
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
            "query", "search_query", "search_string", "value", "text",
            "term", "terms", "keyword", "keywords", "question",
            "questions", "objective", "title"
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


def get_list_from_path(config: Dict[str, Any], path: str) -> List[str]:
    return unique_list(flatten_values(deep_get(config, path, default=[])))


# ============================================================
# CONFIG PROFILE
# ============================================================

def get_concept_groups(config: Dict[str, Any]) -> List[List[str]]:
    groups: List[List[str]] = []

    grouped_paths = [
        "review_scope.concepts.concept_1_terms",
        "review_scope.concepts.concept_2_terms",
        "review_scope.concepts.concept_3_terms",
        "review_scope.concepts.concept_4_terms",
    ]

    for path in grouped_paths:
        terms = get_list_from_path(config, path)
        if terms:
            groups.append(terms)

    flat_paths = [
        "concept_1_terms",
        "concept_2_terms",
        "concept_3_terms",
        "concept_4_terms",
    ]

    for path in flat_paths:
        terms = get_list_from_path(config, path)
        if terms:
            # Avoid duplicate concept groups.
            key = set(normalize_phrase(x) for x in terms)
            exists = any(key == set(normalize_phrase(y) for y in g) for g in groups)
            if not exists:
                groups.append(terms)

    # Fallback: single group from concepts_keywords.
    if not groups:
        terms = get_list_from_path(config, "concepts_keywords")
        if terms:
            groups.append(terms)

    return groups


def build_ranking_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build generic config-derived ranking profile.
    """

    objective_terms: List[str] = []
    rq_terms: List[str] = []
    search_terms: List[str] = []
    query_terms_text: List[str] = []
    include_terms: List[str] = []
    must_terms: List[str] = []
    exclude_terms: List[str] = []

    # Objective / RQs.
    objective_terms.extend(get_list_from_path(config, "review_scope.objective"))
    objective_terms.extend(get_list_from_path(config, "objective"))

    rq_terms.extend(get_list_from_path(config, "review_scope.research_questions_reported"))
    rq_terms.extend(get_list_from_path(config, "review_scope.research_questions_generated"))
    rq_terms.extend(get_list_from_path(config, "research_questions"))
    rq_terms.extend(get_list_from_path(config, "research_questions_reported"))
    rq_terms.extend(get_list_from_path(config, "research_questions_generated"))
    rq_terms.extend(get_list_from_path(config, "review_scope.research_question_keywords"))
    rq_terms.extend(get_list_from_path(config, "research_question_keywords"))

    # Search keywords.
    for path in [
        "search_keywords.reported_search_keywords",
        "search_keywords.reported_controlled_vocabulary_terms",
        "search_keywords.reported_free_text_terms",
        "search_keywords.generated_search_keywords",
        "prisma_s_search_keywords",
        "prisma_s_controlled_vocabulary_terms",
        "prisma_s_free_text_terms",
        "generated_search_keywords",
    ]:
        search_terms.extend(get_list_from_path(config, path))

    # Search queries.
    for path in [
        "search_queries.reported_full_search_strings",
        "search_queries.reported_search_queries_by_database",
        "search_queries.generated_search_queries",
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "generated_search_queries",
    ]:
        query_terms_text.extend(get_list_from_path(config, path))

    # Eligibility inclusion.
    for path in [
        "eligibility_criteria.reported_inclusion",
        "eligibility_criteria.generated_inclusion",
        "reported_inclusion",
        "generated_inclusion",
        "prisma_inclusion",
    ]:
        include_terms.extend(get_list_from_path(config, path))

    # Screening include / must.
    for path in [
        "title_abstract_screening.reported_include_keywords",
        "title_abstract_screening.generated_include_keywords",
        "full_text_screening.reported_include_keywords",
        "full_text_screening.generated_include_keywords",
        "prisma_title_abstract_include_keywords",
        "generated_abstract_include_keywords",
        "prisma_full_text_include_keywords",
        "generated_full_text_include_keywords",
    ]:
        include_terms.extend(get_list_from_path(config, path))

    for path in [
        "title_abstract_screening.reported_must_have_keywords",
        "title_abstract_screening.generated_must_have_keywords",
        "full_text_screening.reported_must_have_keywords",
        "full_text_screening.generated_must_have_keywords",
        "generated_abstract_must_have_keywords",
        "generated_full_text_must_have_keywords",
    ]:
        must_terms.extend(get_list_from_path(config, path))

    # Exclusion terms.
    for path in [
        "eligibility_criteria.reported_exclusion",
        "eligibility_criteria.generated_exclusion",
        "reported_exclusion",
        "generated_exclusion",
        "prisma_exclusion",
        "title_abstract_screening.reported_exclude_keywords",
        "title_abstract_screening.generated_exclude_keywords",
        "full_text_screening.reported_exclude_keywords",
        "full_text_screening.generated_exclude_keywords",
        "generated_abstract_exclude_keywords",
        "generated_full_text_exclude_keywords",
    ]:
        exclude_terms.extend(get_list_from_path(config, path))

    concept_groups = get_concept_groups(config)

    concept_terms: List[str] = []
    for group in concept_groups:
        concept_terms.extend(group)

    objective_terms = unique_list(objective_terms)
    rq_terms = unique_list(rq_terms)
    search_terms = unique_list(search_terms)
    query_terms_text = unique_list(query_terms_text)
    include_terms = unique_list(include_terms)
    must_terms = unique_list(must_terms)
    exclude_terms = unique_list(exclude_terms)
    concept_terms = unique_list(concept_terms)

    # Weighted query tokens for BM25.
    weighted_query_tokens: Dict[str, float] = defaultdict(float)

    def add_weighted_terms(items: List[str], weight: float) -> None:
        for item in items:
            for tok in tokenize(item):
                weighted_query_tokens[tok] += weight

    add_weighted_terms(objective_terms, 3.0)
    add_weighted_terms(rq_terms, 3.0)
    add_weighted_terms(concept_terms, 3.0)
    add_weighted_terms(search_terms, 2.5)
    add_weighted_terms(include_terms, 2.0)
    add_weighted_terms(must_terms, 2.5)
    add_weighted_terms(query_terms_text, 1.5)

    return {
        "objective_terms": objective_terms,
        "research_question_terms": rq_terms,
        "search_terms": search_terms,
        "query_terms_text": query_terms_text,
        "include_terms": include_terms,
        "must_terms": must_terms,
        "exclude_terms": exclude_terms,
        "concept_groups": concept_groups,
        "concept_terms": concept_terms,
        "weighted_query_tokens": dict(weighted_query_tokens),
    }


# ============================================================
# BM25 + EXACT MATCH SCORING
# ============================================================

def article_text(row: Dict[str, str]) -> str:
    title = clean_text(row.get("title", ""))
    abstract = clean_text(row.get("abstract", ""))
    journal = clean_text(row.get("journal", ""))
    article_type = clean_text(row.get("article_type", ""))

    # Title is repeated because title is highly informative.
    return " ".join([
        title,
        title,
        title,
        abstract,
        abstract,
        journal,
        article_type,
    ])


def title_text(row: Dict[str, str]) -> str:
    return clean_text(row.get("title", ""))


def build_bm25_index(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    doc_tokens: List[List[str]] = []
    doc_lens: List[int] = []
    df: Counter = Counter()

    for row in rows:
        toks = tokenize(article_text(row))
        doc_tokens.append(toks)
        doc_lens.append(len(toks))

        for tok in set(toks):
            df[tok] += 1

    n_docs = len(rows)
    avgdl = sum(doc_lens) / n_docs if n_docs else 0.0

    idf: Dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log(1.0 + ((n_docs - freq + 0.5) / (freq + 0.5)))

    return {
        "doc_tokens": doc_tokens,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "idf": idf,
        "n_docs": n_docs,
    }


def bm25_score_doc(
    tokens: List[str],
    doc_len: int,
    avgdl: float,
    idf: Dict[str, float],
    weighted_query_tokens: Dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not tokens or not weighted_query_tokens:
        return 0.0

    tf = Counter(tokens)
    score = 0.0

    for tok, q_weight in weighted_query_tokens.items():
        if tok not in tf:
            continue

        freq = tf[tok]
        tok_idf = idf.get(tok, 0.0)

        denom = freq + k1 * (1.0 - b + b * (doc_len / avgdl)) if avgdl else freq + k1
        term_score = tok_idf * ((freq * (k1 + 1.0)) / denom)

        score += q_weight * term_score

    return score


def phrase_present(phrase: str, text_norm: str) -> bool:
    p = normalize_phrase(phrase)
    if not p:
        return False

    # For single tokens, use token boundary.
    if " " not in p:
        return bool(re.search(rf"\b{re.escape(p)}\b", text_norm))

    return p in text_norm


def phrase_score(
    phrases: List[str],
    title_norm: str,
    body_norm: str,
    title_weight: float = 2.0,
    body_weight: float = 1.0,
) -> float:
    phrases = unique_list(phrases)
    if not phrases:
        return 0.0

    total_possible = 0.0
    total_hit = 0.0

    for phrase in phrases:
        p = normalize_phrase(phrase)
        if not p:
            continue

        total_possible += title_weight

        if phrase_present(p, title_norm):
            total_hit += title_weight
        elif phrase_present(p, body_norm):
            total_hit += body_weight

    if not total_possible:
        return 0.0

    return min(total_hit / total_possible, 1.0)


def concept_coverage_score(
    concept_groups: List[List[str]],
    title_norm: str,
    body_norm: str,
) -> float:
    if not concept_groups:
        return 0.0

    covered = 0

    for group in concept_groups:
        group_hit = False

        for term in group:
            if phrase_present(term, title_norm) or phrase_present(term, body_norm):
                group_hit = True
                break

        if group_hit:
            covered += 1

    return covered / len(concept_groups)


def exclusion_penalty(
    exclude_terms: List[str],
    title_norm: str,
    body_norm: str,
) -> float:
    exclude_terms = unique_list(exclude_terms)
    if not exclude_terms:
        return 0.0

    hits = 0

    for term in exclude_terms:
        if phrase_present(term, title_norm) or phrase_present(term, body_norm):
            hits += 1

    return min(hits / max(len(exclude_terms), 1), 1.0)


def normalize_scores(raw_scores: List[float]) -> List[float]:
    if not raw_scores:
        return []

    min_s = min(raw_scores)
    max_s = max(raw_scores)

    if max_s <= min_s:
        return [0.0 for _ in raw_scores]

    return [(x - min_s) / (max_s - min_s) for x in raw_scores]


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
                            records.append({"doi": d, "title": ""})

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

    index = build_bm25_index(rows)

    raw_bm25_scores: List[float] = []

    for i, row in enumerate(rows):
        score = bm25_score_doc(
            tokens=index["doc_tokens"][i],
            doc_len=index["doc_lens"][i],
            avgdl=index["avgdl"],
            idf=index["idf"],
            weighted_query_tokens=profile["weighted_query_tokens"],
        )
        raw_bm25_scores.append(score)

    bm25_norm_scores = normalize_scores(raw_bm25_scores)

    ranked_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        title = clean_text(row.get("title", ""))
        abstract = clean_text(row.get("abstract", ""))

        title_norm = normalize_phrase(title)
        body_norm = normalize_phrase(" ".join([
            title,
            abstract,
            clean_text(row.get("journal", "")),
            clean_text(row.get("article_type", "")),
        ]))

        doi = normalize_doi(row.get("doi", ""))

        bm25 = bm25_norm_scores[i]

        search_keyword_score = phrase_score(
            profile["search_terms"] + profile["query_terms_text"],
            title_norm,
            body_norm,
        )

        concept_score = concept_coverage_score(
            profile["concept_groups"],
            title_norm,
            body_norm,
        )

        include_score = phrase_score(
            profile["include_terms"] + profile["must_terms"],
            title_norm,
            body_norm,
        )

        title_relevance_score = phrase_score(
            profile["objective_terms"]
            + profile["research_question_terms"]
            + profile["concept_terms"]
            + profile["search_terms"],
            title_norm,
            title_norm,
            title_weight=1.0,
            body_weight=1.0,
        )

        penalty = exclusion_penalty(
            profile["exclude_terms"],
            title_norm,
            body_norm,
        )

        missing_abstract_penalty = 0.05 if not abstract else 0.0

        # Rank Strategy 1 weights.
        final_score = (
            0.50 * bm25
            + 0.20 * concept_score
            + 0.15 * search_keyword_score
            + 0.10 * include_score
            + 0.05 * title_relevance_score
            - 0.20 * penalty
            - missing_abstract_penalty
        )

        final_score = round(final_score, 8)

        out = dict(row)
        out["doi"] = doi
        out["rank_strategy_1_score"] = final_score
        out["score_bm25_abstract_title"] = round(bm25, 8)
        out["score_concept_coverage"] = round(concept_score, 8)
        out["score_search_keyword_match"] = round(search_keyword_score, 8)
        out["score_include_keyword_match"] = round(include_score, 8)
        out["score_title_relevance"] = round(title_relevance_score, 8)
        out["score_exclusion_penalty"] = round(penalty, 8)
        out["missing_abstract_penalty"] = missing_abstract_penalty

        out["is_reference_match"] = bool(doi and doi in ref_dois)
        out["is_retrievable_reference_match"] = bool(doi and doi in ceiling_dois)

        ranked_rows.append(out)

    ranked_rows.sort(
        key=lambda r: (
            float(r.get("rank_strategy_1_score", 0.0) or 0.0),
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

    # Keep only rows with DOI for fair DOI-based ranking evaluation.
    rows = [
        dict(row, doi=normalize_doi(row.get("doi", "")))
        for row in rows
        if normalize_doi(row.get("doi", ""))
    ]

    ref_dois, ref_source, references_total = load_reference_dois(project_dir)
    ceiling_dois = load_retrievable_ceiling_dois(project_dir)

    profile = build_ranking_profile(config)

    ranked_rows = rank_articles(
        rows=rows,
        profile=profile,
        ref_dois=ref_dois,
        ceiling_dois=ceiling_dois,
    )

    # Output fields.
    base_fields = list(ranked_rows[0].keys()) if ranked_rows else [
        "rank",
        "rank_strategy_1_score",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
    ]

    preferred_first = [
        "rank",
        "rank_strategy_1_score",
        "score_bm25_abstract_title",
        "score_concept_coverage",
        "score_search_keyword_match",
        "score_include_keyword_match",
        "score_title_relevance",
        "score_exclusion_penalty",
        "missing_abstract_penalty",
        "is_reference_match",
        "is_retrievable_reference_match",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
        "source_query",
        "query_origin",
        "url",
    ]

    fields = []
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
        "n_objective_terms": len(profile["objective_terms"]),
        "n_research_question_terms": len(profile["research_question_terms"]),
        "n_concept_groups": len(profile["concept_groups"]),
        "n_concept_terms": len(profile["concept_terms"]),
        "n_search_terms": len(profile["search_terms"]),
        "n_include_terms": len(profile["include_terms"]),
        "n_must_terms": len(profile["must_terms"]),
        "n_exclude_terms": len(profile["exclude_terms"]),
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
    "n_concept_groups",
    "n_concept_terms",
    "n_search_terms",
    "n_include_terms",
    "n_must_terms",
    "n_exclude_terms",
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
    out_path = projects_dir / "Rankings" / "rank_strategy_1_summary.csv"
    write_csv(out_path, rows, SUMMARY_FIELDS)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank retrieved articles using config-guided abstract/title ranking."
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
    print("STEP 9: RANK STRATEGY 1 — CONFIG-GUIDED ABSTRACT/TITLE RANKING")
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
    print("  Projects/<project>/Rankings/rank_strategy_1.csv")


if __name__ == "__main__":
    main()