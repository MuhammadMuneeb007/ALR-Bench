# Step9-Rank2-ReviewIntentSBERT.py
#
# Rank Strategy 2:
#   Review-intent BGE cross-encoder reranking using local metadata.
#
# What this script does:
#   1. Reads:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Builds one review-intent document from config:
#        - title
#        - objective
#        - research questions
#        - concepts
#        - generated search queries
#        - generated search keywords
#        - inclusion keywords
#        - title/abstract screening keywords
#        - full-text screening keywords
#
#   3. Scores each pair:
#        - review-intent text
#        - article title + abstract
#
#   4. Ranks articles by cross-encoder reranker score.
#
#   5. Saves:
#        Projects/<project>/Rankings/rank_strategy_2.csv
#        Projects/Rankings/rank_strategy_2_summary.csv
#
#   6. Prints:
#        Common@10, Common@20, Common@50, Common@100
#
# Notes:
#   - This is a ranking/evaluation step, so reading references is allowed.
#   - No API calls.
#   - No PDF download.
#   - No full-text download.
#   - No LLM call.
#
# Requirements:
#   Preferred: pip install sentence-transformers torch tqdm
#   Fallback:  pip install scikit-learn tqdm
#
# Usage:
#   python Step9-Rank2-ReviewIntentSBERT.py --projects-dir Projects --force
#
# One project:
#   python Step9-Rank2-ReviewIntentSBERT.py --projects-dir Projects --project PROJECT_NAME --force
#
# Optional model:
#   python Step9-Rank2-ReviewIntentSBERT.py --projects-dir Projects --model BAAI/bge-reranker-large --force
#
# If your torch/torchvision/SBERT environment is broken, use the fast local fallback:
#   python Step9-Rank2-ReviewIntentSBERT.py --projects-dir Projects --backend tfidf --force

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")

DEFAULT_ARTICLES_CLEAN_REL = Path("Search") / "articles_clean_doi_only.csv"
DEFAULT_ARTICLES_RAW_REL = Path("Search") / "articles.csv"

DEFAULT_RANKINGS_DIR = Path("Rankings")
DEFAULT_RANK_FILE = "rank_strategy_2.csv"

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

TOP_K_VALUES = [10, 20, 50, 100]


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
            "term",
            "terms",
            "keyword",
            "keywords",
            "question",
            "questions",
            "objective",
            "title",
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

        key = normalize_title(text)
        if not key or key in seen:
            continue

        seen.add(key)
        out.append(text)

    return out


def get_list_from_path(config: Dict[str, Any], path: str) -> List[str]:
    return unique_list(flatten_values(deep_get(config, path, default=[])))


def truncate_text(text: str, max_chars: int = 6000) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


# ============================================================
# CONFIG REVIEW-INTENT BUILDING
# ============================================================

def build_review_intent_sections(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Build config-derived text groups for semantic ranking.

    This is intentionally broad because many project configs have empty RQs
    and sparse PRISMA fields, but usually have title, objective, concepts,
    generated search queries, and generated keywords.
    """

    sections: Dict[str, List[str]] = {
        "title": [],
        "objective": [],
        "research_questions": [],
        "concepts": [],
        "search_queries": [],
        "search_keywords": [],
        "include_terms": [],
        "must_terms": [],
        "quality_terms": [],
    }

    # Title.
    for path in [
        "article_identity.title",
        "title",
    ]:
        sections["title"].extend(get_list_from_path(config, path))

    # Objective.
    for path in [
        "review_scope.objective",
        "objective",
    ]:
        sections["objective"].extend(get_list_from_path(config, path))

    # Research questions.
    for path in [
        "review_scope.research_questions_reported",
        "review_scope.research_questions_generated",
        "research_questions",
        "research_questions_reported",
        "research_questions_generated",
        "review_scope.research_question_keywords",
        "research_question_keywords",
    ]:
        sections["research_questions"].extend(get_list_from_path(config, path))

    # Concepts.
    for path in [
        "review_scope.concepts.concept_1_terms",
        "review_scope.concepts.concept_2_terms",
        "review_scope.concepts.concept_3_terms",
        "review_scope.concepts.concept_4_terms",
        "concept_1_terms",
        "concept_2_terms",
        "concept_3_terms",
        "concept_4_terms",
        "concepts_keywords",
    ]:
        sections["concepts"].extend(get_list_from_path(config, path))

    # Search queries.
    for path in [
        "search_queries.reported_full_search_strings",
        "search_queries.reported_search_queries_by_database",
        "search_queries.generated_search_queries",
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "generated_search_queries",
    ]:
        sections["search_queries"].extend(get_list_from_path(config, path))

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
        sections["search_keywords"].extend(get_list_from_path(config, path))

    # Inclusion terms.
    for path in [
        "eligibility_criteria.reported_inclusion",
        "eligibility_criteria.generated_inclusion",
        "reported_inclusion",
        "generated_inclusion",
        "prisma_inclusion",
        "prisma_eligibility_inclusion",
        "title_abstract_screening.reported_include_keywords",
        "title_abstract_screening.generated_include_keywords",
        "full_text_screening.reported_include_keywords",
        "full_text_screening.generated_include_keywords",
        "prisma_title_abstract_include_keywords",
        "generated_abstract_include_keywords",
        "prisma_full_text_include_keywords",
        "generated_full_text_include_keywords",
    ]:
        sections["include_terms"].extend(get_list_from_path(config, path))

    # Must-have terms.
    for path in [
        "title_abstract_screening.reported_must_have_keywords",
        "title_abstract_screening.generated_must_have_keywords",
        "full_text_screening.reported_must_have_keywords",
        "full_text_screening.generated_must_have_keywords",
        "prisma_title_abstract_must_have_keywords",
        "generated_abstract_must_have_keywords",
        "prisma_full_text_must_have_keywords",
        "generated_full_text_must_have_keywords",
    ]:
        sections["must_terms"].extend(get_list_from_path(config, path))

    # Quality terms are low-weight context only.
    for path in [
        "quality_assessment.generated_quality_assessment_keywords",
        "quality_assessment.generated_risk_of_bias_keywords",
        "quality_assessment.generated_methodological_quality_keywords",
        "quality_assessment.generated_reporting_quality_keywords",
        "generated_quality_assessment_keywords",
        "generated_quality_assessment_questions",
        "prisma_quality_assessment_keywords",
        "prisma_quality_assessment_questions",
    ]:
        sections["quality_terms"].extend(get_list_from_path(config, path))

    for key in list(sections):
        sections[key] = unique_list(sections[key])

    return sections


def build_review_intent_text(config: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    sections = build_review_intent_sections(config)

    # Repeat the most important fields to create light weighting in the embedding text.
    parts: List[str] = []

    def add_section(label: str, values: List[str], repeat: int = 1) -> None:
        if not values:
            return

        text = "; ".join(values)

        for _ in range(repeat):
            parts.append(f"{label}: {text}")

    add_section("Review title", sections["title"], repeat=3)
    add_section("Review objective", sections["objective"], repeat=3)
    add_section("Research questions", sections["research_questions"], repeat=3)
    add_section("Core concepts", sections["concepts"], repeat=3)
    add_section("Generated search queries", sections["search_queries"], repeat=2)
    add_section("Generated search keywords", sections["search_keywords"], repeat=2)
    add_section("Inclusion terms", sections["include_terms"], repeat=2)
    add_section("Must-have terms", sections["must_terms"], repeat=2)
    add_section("Quality context", sections["quality_terms"], repeat=1)

    intent_text = clean_text(" ".join(parts))

    if not intent_text:
        # Last fallback: dump simple top-level fields.
        fallback = []
        for key in ["title", "objective", "review_type"]:
            if clean_text(config.get(key, "")):
                fallback.append(f"{key}: {clean_text(config.get(key, ''))}")
        intent_text = clean_text(" ".join(fallback))

    stats = {
        "n_title_terms": len(sections["title"]),
        "n_objective_terms": len(sections["objective"]),
        "n_research_question_terms": len(sections["research_questions"]),
        "n_concept_terms": len(sections["concepts"]),
        "n_search_query_terms": len(sections["search_queries"]),
        "n_search_keyword_terms": len(sections["search_keywords"]),
        "n_include_terms": len(sections["include_terms"]),
        "n_must_terms": len(sections["must_terms"]),
        "n_quality_terms": len(sections["quality_terms"]),
        "review_intent_chars": len(intent_text),
    }

    return truncate_text(intent_text, max_chars=6000), stats


# ============================================================
# ARTICLE TEXT
# ============================================================

def build_article_semantic_text(row: Dict[str, str]) -> str:
    title = clean_text(row.get("title", ""))
    abstract = clean_text(row.get("abstract", ""))
    journal = clean_text(row.get("journal", ""))
    article_type = clean_text(row.get("article_type", ""))

    parts = []

    if title:
        # Repeat title because it is the strongest metadata signal.
        parts.append(f"Title: {title}")
        parts.append(f"Title: {title}")

    if abstract:
        parts.append(f"Abstract: {abstract}")

    if journal:
        parts.append(f"Journal: {journal}")

    if article_type:
        parts.append(f"Article type: {article_type}")

    return truncate_text(" ".join(parts), max_chars=5000)


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
    # Prefer gold file first for evaluation.
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
# EMBEDDING HELPERS
# ============================================================

def load_sbert_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it with:\n"
            "pip install sentence-transformers torch tqdm"
        ) from e

    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            "Could not load SentenceTransformer. This is usually an environment "
            "issue with torch/torchvision/transformers. The script can still run "
            "with --backend tfidf, or with --backend auto to fall back automatically."
        ) from e


def load_reranker_model(model_name: str) -> Dict[str, Any]:
    if "bge-reranker-v2-m3" in model_name.lower():
        try:
            from sentence_transformers import CrossEncoder

            return {
                "reranker_type": "sentence_transformers.CrossEncoder",
                "model": CrossEncoder(model_name),
                "flag_error": "using CrossEncoder for bge-reranker-v2-m3 to avoid FlagEmbedding XLMRobertaTokenizer.prepare_for_model error",
            }
        except Exception as cross_error:
            raise RuntimeError(
                "Could not load bge-reranker-v2-m3 with sentence_transformers.CrossEncoder. "
                "Try: pip install -U sentence-transformers transformers torch"
            ) from cross_error

    try:
        from FlagEmbedding import FlagReranker

        return {
            "reranker_type": "FlagEmbedding.FlagReranker",
            "model": FlagReranker(model_name, use_fp16=False),
        }
    except Exception as flag_error:
        try:
            from sentence_transformers import CrossEncoder

            return {
                "reranker_type": "sentence_transformers.CrossEncoder",
                "model": CrossEncoder(model_name),
                "flag_error": f"{type(flag_error).__name__}: {flag_error}",
            }
        except Exception as cross_error:
            raise RuntimeError(
                "Could not load reranker model. Install one of these options:\n"
                "  pip install FlagEmbedding torch transformers\n"
                "  pip install sentence-transformers torch transformers\n"
                f"FlagEmbedding error: {type(flag_error).__name__}: {flag_error}\n"
                f"CrossEncoder error: {type(cross_error).__name__}: {cross_error}"
            ) from cross_error


def load_rank_backend(model_name: str, backend: str) -> Dict[str, Any]:
    backend = clean_text(backend).lower() or "auto"
    if backend not in {"auto", "reranker", "sbert", "tfidf", "lexical"}:
        raise ValueError("backend must be one of: auto, reranker, sbert, tfidf, lexical")

    if backend == "auto":
        try_reranker = os.getenv("STEP9_DISABLE_RERANKER_AUTO", "0").strip().lower() not in {"1", "true", "yes", "y"}
        if try_reranker:
            try:
                reranker_info = load_reranker_model(model_name)
                return {
                    "backend": "reranker",
                    "model": reranker_info["model"],
                    "model_name": model_name,
                    "reranker_type": reranker_info.get("reranker_type", ""),
                    "error": reranker_info.get("flag_error", ""),
                }
            except Exception as e:
                return {
                    "backend": "tfidf",
                    "model": None,
                    "model_name": "tfidf_cosine_fallback",
                    "reranker_type": "",
                    "error": f"reranker failed; using TF-IDF fallback: {type(e).__name__}: {e}",
                }
        else:
            return {
                "backend": "tfidf",
                "model": None,
                "model_name": "tfidf_cosine_auto_default",
                "reranker_type": "",
                "error": "auto selected TF-IDF because STEP9_DISABLE_RERANKER_AUTO is set",
            }

    if backend == "reranker":
        reranker_info = load_reranker_model(model_name)
        return {
            "backend": "reranker",
            "model": reranker_info["model"],
            "model_name": model_name,
            "reranker_type": reranker_info.get("reranker_type", ""),
            "error": reranker_info.get("flag_error", ""),
        }

    if backend in {"auto", "sbert"}:
        try:
            return {
                "backend": "sbert",
                "model": load_sbert_model(model_name),
                "model_name": model_name,
                "reranker_type": "",
                "error": "",
            }
        except Exception as e:
            if backend == "sbert":
                raise
            return {
                "backend": "tfidf",
                "model": None,
                "model_name": "tfidf_cosine_fallback",
                "reranker_type": "",
                "error": f"{type(e).__name__}: {e}",
            }

    return {
        "backend": backend,
        "model": None,
        "model_name": f"{backend}_fallback",
        "reranker_type": "",
        "error": "",
    }


def cosine_scores_from_embeddings(query_embedding, doc_embeddings) -> List[float]:
    import numpy as np

    q = np.asarray(query_embedding, dtype="float32")
    docs = np.asarray(doc_embeddings, dtype="float32")

    if q.ndim == 1:
        q = q.reshape(1, -1)

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    d_norm = np.linalg.norm(docs, axis=1, keepdims=True)

    q_norm[q_norm == 0] = 1.0
    d_norm[d_norm == 0] = 1.0

    q = q / q_norm
    docs = docs / d_norm

    scores = docs @ q[0]
    return [float(x) for x in scores]


def tfidf_cosine_scores(query_text: str, doc_texts: List[str]) -> Tuple[List[float], str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=80000,
        )
        matrix = vectorizer.fit_transform([query_text] + doc_texts)
        scores = cosine_similarity(matrix[1:], matrix[:1]).ravel()
        return [float(x) for x in scores], "tfidf_cosine"
    except Exception as e:
        return lexical_jaccard_scores(query_text, doc_texts), f"lexical_jaccard_fallback:{type(e).__name__}"


def lexical_jaccard_scores(query_text: str, doc_texts: List[str]) -> List[float]:
    query_tokens = set(normalize_title(query_text).split())
    scores: List[float] = []
    for doc in doc_texts:
        doc_tokens = set(normalize_title(doc).split())
        union = query_tokens | doc_tokens
        scores.append(float(len(query_tokens & doc_tokens) / len(union)) if union else 0.0)
    return scores


def reranker_scores(query_text: str, doc_texts: List[str], backend_info: Dict[str, Any], batch_size: int) -> Tuple[List[float], str]:
    pairs = [[query_text, doc] for doc in doc_texts]
    model = backend_info["model"]
    reranker_type = clean_text(backend_info.get("reranker_type", ""))

    if "FlagReranker" in reranker_type:
        try:
            scores = model.compute_score(pairs, batch_size=batch_size, normalize=True)
        except TypeError:
            scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        return [float(x) for x in scores], "bge_reranker_pair_score"

    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return [float(x) for x in scores], "cross_encoder_pair_score"


def normalize_scores(raw_scores: List[float]) -> List[float]:
    if not raw_scores:
        return []

    min_s = min(raw_scores)
    max_s = max(raw_scores)

    if max_s <= min_s:
        return [0.0 for _ in raw_scores]

    return [(x - min_s) / (max_s - min_s) for x in raw_scores]


# ============================================================
# RANKING
# ============================================================

def rank_articles(
    rows: List[Dict[str, str]],
    review_intent_text: str,
    backend_info: Dict[str, Any],
    ref_dois: Set[str],
    ceiling_dois: Set[str],
    batch_size: int,
) -> List[Dict[str, Any]]:

    if not rows:
        return []

    article_texts = [build_article_semantic_text(row) for row in rows]
    backend = clean_text(backend_info.get("backend", "tfidf"))

    if backend == "reranker":
        raw_scores, score_backend = reranker_scores(
            review_intent_text,
            article_texts,
            backend_info,
            batch_size=batch_size,
        )
    elif backend == "sbert":
        model = backend_info["model"]
        query_embedding = model.encode(
            [review_intent_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        article_embeddings = model.encode(
            article_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        raw_scores = cosine_scores_from_embeddings(query_embedding, article_embeddings)
        score_backend = "sbert_cosine"
    elif backend == "lexical":
        raw_scores = lexical_jaccard_scores(review_intent_text, article_texts)
        score_backend = "lexical_jaccard"
    else:
        raw_scores, score_backend = tfidf_cosine_scores(review_intent_text, article_texts)

    norm_scores = normalize_scores(raw_scores)

    ranked_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        doi = normalize_doi(row.get("doi", ""))

        out = dict(row)
        out["doi"] = doi
        out["rank_strategy_2_score"] = round(norm_scores[i], 8)
        out["score_sbert_cosine_raw"] = round(raw_scores[i], 8)
        out["score_semantic_raw"] = round(raw_scores[i], 8)
        out["score_backend"] = score_backend
        out["article_semantic_text_chars"] = len(article_texts[i])
        out["is_reference_match"] = bool(doi and doi in ref_dois)
        out["is_retrievable_reference_match"] = bool(doi and doi in ceiling_dois)

        ranked_rows.append(out)

    ranked_rows.sort(
        key=lambda r: (
            float(r.get("rank_strategy_2_score", 0.0) or 0.0),
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

def process_project(
    project_dir: Path,
    force: bool,
    backend_info: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:

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

    review_intent_text, intent_stats = build_review_intent_text(config)

    ranked_rows = rank_articles(
        rows=rows,
        review_intent_text=review_intent_text,
        backend_info=backend_info,
        ref_dois=ref_dois,
        ceiling_dois=ceiling_dois,
        batch_size=batch_size,
    )

    base_fields = list(ranked_rows[0].keys()) if ranked_rows else [
        "rank",
        "rank_strategy_2_score",
        "score_sbert_cosine_raw",
        "score_semantic_raw",
        "score_backend",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
    ]

    preferred_first = [
        "rank",
        "rank_strategy_2_score",
        "score_sbert_cosine_raw",
        "score_semantic_raw",
        "score_backend",
        "article_semantic_text_chars",
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
        "model_name": backend_info.get("model_name", ""),
        "embedding_backend": backend_info.get("backend", ""),
        "reranker_type": backend_info.get("reranker_type", ""),
        "backend_error": backend_info.get("error", ""),
        "articles_ranked": len(ranked_rows),
        "references_source": ref_source,
        "references_total": references_total,
        "references_with_doi": len(ref_dois),
        "retrievable_ceiling": len(ceiling_dois) if ceiling_dois else "",
        **intent_stats,
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
    "embedding_backend",
    "reranker_type",
    "model_name",
    "backend_error",
    "review_intent_chars",
    "n_title_terms",
    "n_objective_terms",
    "n_research_question_terms",
    "n_concept_terms",
    "n_search_query_terms",
    "n_search_keyword_terms",
    "n_include_terms",
    "n_must_terms",
    "n_quality_terms",
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
    out_path = projects_dir / "Rankings" / "rank_strategy_2_summary.csv"
    write_csv(out_path, rows, SUMMARY_FIELDS)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank retrieved articles using review-intent SBERT semantic ranking."
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

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Reranker/SentenceTransformer model name or local model path.",
    )

    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "reranker", "sbert", "tfidf", "lexical"],
        help="Ranking backend. auto tries the BGE reranker then falls back to TF-IDF.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    print("=" * 140)
    print("STEP 9: RANK STRATEGY 2 — REVIEW-INTENT SBERT SEMANTIC RANKING")
    print("=" * 140)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Force overwrite    : {args.force}")
    print(f"Top-K values       : {TOP_K_VALUES}")
    print(f"Model              : {args.model}")
    print(f"Backend            : {args.backend}")
    print(f"Batch size         : {args.batch_size}")
    print("Uses APIs          : NO")
    print("Uses LLM           : NO")
    print("Uses PDFs/fulltext : NO")
    print("=" * 140)

    print("\nLoading ranking backend...")
    backend_info = load_rank_backend(args.model, args.backend)
    print(f"Backend loaded     : {backend_info.get('backend')}")
    print(f"Backend model/name : {backend_info.get('model_name')}")
    if backend_info.get("error"):
        print(f"Backend warning    : {backend_info.get('error')}")

    summaries: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(
                project_dir=project_dir,
                force=args.force,
                backend_info=backend_info,
                batch_size=args.batch_size,
            )

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
    print("  Projects/<project>/Rankings/rank_strategy_2.csv")


if __name__ == "__main__":
    main()
