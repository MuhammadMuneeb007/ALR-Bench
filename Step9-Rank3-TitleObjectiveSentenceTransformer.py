# Step9-Rank3-TitleObjectiveSentenceTransformer.py
#
# Rank Strategy 3:
#   Title + Objective BGE-large embedding ranking using local metadata.
#
# What this script does:
#   1. Reads:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Builds a semantic query only from:
#        - review title
#        - review objective
#
#   3. Embeds:
#        - title + objective query
#        - article title + abstract
#
#   4. Ranks articles by cosine similarity.
#
#   5. Saves:
#        Projects/<project>/Rankings/rank_strategy_3.csv
#        Projects/Rankings/rank_strategy_3_summary.csv
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
#
# Usage:
#   python Step9-Rank3-TitleObjectiveSentenceTransformer.py --projects-dir Projects --force
#
# One project:
#   python Step9-Rank3-TitleObjectiveSentenceTransformer.py --projects-dir Projects --project PROJECT_NAME --force

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_RANK_FILE = "rank_strategy_3.csv"

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

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
# TITLE + OBJECTIVE QUERY BUILDING
# ============================================================

def build_title_objective_query(config: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    """
    Build a clean semantic query from title + objective + concepts.

    Keep this label-free because BGE-large should spend its token budget on
    topical signal, not scaffolding like "Review title:".
    """

    title_terms: List[str] = []
    objective_terms: List[str] = []
    concept_terms: List[str] = []

    for path in [
        "article_identity.title",
        "title",
    ]:
        title_terms.extend(get_list_from_path(config, path))

    for path in [
        "review_scope.objective",
        "objective",
    ]:
        objective_terms.extend(get_list_from_path(config, path))

    for path in [
        "concepts_keywords",
        "concepts",
        "keywords",
        "main_concepts",
        "review_concepts",
    ]:
        concept_terms.extend(get_list_from_path(config, path))

    title_terms = unique_list(title_terms)
    objective_terms = unique_list(objective_terms)
    concept_terms = unique_list(concept_terms)

    parts: List[str] = []

    if title_terms:
        title_text = "; ".join(title_terms)
        # Title is concise and strongly defines the review topic.
        parts.append(title_text)
        parts.append(title_text)

    if objective_terms:
        objective_text = "; ".join(objective_terms)
        parts.append(objective_text)

    if concept_terms:
        parts.append("; ".join(concept_terms))

    query_text = clean_text(" ".join(parts))

    # Fallback if the focused fields are missing.
    if not query_text:
        fallback_fields = [
            "generated_search_queries",
            "generated_search_keywords",
            "review_type",
        ]

        fallback_parts: List[str] = []

        for path in fallback_fields:
            vals = get_list_from_path(config, path)
            if vals:
                fallback_parts.append("; ".join(vals))

        query_text = clean_text(" ".join(fallback_parts))

    stats = {
        "n_title_terms": len(title_terms),
        "n_objective_terms": len(objective_terms),
        "n_concept_terms": len(concept_terms),
        "title_objective_query_chars": len(query_text),
    }

    return truncate_text(query_text, max_chars=6000), stats


# ============================================================
# ARTICLE TEXT
# ============================================================

def build_article_semantic_text(row: Dict[str, str]) -> str:
    title = clean_text(row.get("title", ""))
    abstract = clean_text(row.get("abstract", ""))

    parts: List[str] = []

    if title:
        parts.append(f"Title: {title}")
        parts.append(f"Title: {title}")

    if abstract:
        parts.append(f"Abstract: {abstract}")

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
# RERANKER HELPERS
# ============================================================

def load_sentence_transformer_model(model_name: str) -> Dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for bge-large. Install it with:\n"
            "pip install -U sentence-transformers torch tqdm"
        ) from e

    model = SentenceTransformer(model_name)

    return {
        "reranker_type": "sentence_transformers.SentenceTransformer",
        "model": model,
        "error": "",
    }


def load_transformers_pair_reranker(model_name: str) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Could not import torch/transformers for the safe reranker path. "
            "Install/update with: pip install -U torch transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return {
        "reranker_type": "transformers.AutoModelForSequenceClassification",
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "torch": torch,
        "error": (
            "using direct transformers pair scorer; this avoids "
            "XLMRobertaTokenizer.prepare_for_model"
        ),
    }


def load_reranker_model(model_name: str) -> Dict[str, Any]:
    model_l = model_name.lower()

    if "reranker" not in model_l:
        return load_sentence_transformer_model(model_name)

    if "bge-reranker-v2-m3" in model_name.lower():
        try:
            return load_transformers_pair_reranker(model_name)
        except Exception as direct_error:
            return {
                "reranker_type": "lexical.safe_fallback",
                "model": None,
                "error": (
                    "direct transformers scorer failed; using lexical fallback so Rank 3 "
                    "does not hit XLMRobertaTokenizer.prepare_for_model. "
                    f"Direct error: {type(direct_error).__name__}: {direct_error}"
                ),
            }

    try:
        from FlagEmbedding import FlagReranker

        return {
            "reranker_type": "FlagEmbedding.FlagReranker",
            "model": FlagReranker(model_name, use_fp16=False),
            "error": "",
        }
    except Exception as flag_error:
        try:
            return load_transformers_pair_reranker(model_name)
        except Exception as direct_error:
            try:
                from sentence_transformers import CrossEncoder

                return {
                    "reranker_type": "sentence_transformers.CrossEncoder",
                    "model": CrossEncoder(model_name),
                    "error": (
                        f"FlagEmbedding unavailable: {type(flag_error).__name__}: {flag_error}; "
                        f"direct transformers unavailable: {type(direct_error).__name__}: {direct_error}"
                    ),
                }
            except Exception as cross_error:
                raise RuntimeError(
                    "Could not load reranker model. Install one of these options:\n"
                    "  pip install -U torch transformers\n"
                    "  pip install FlagEmbedding torch transformers\n"
                    "  pip install sentence-transformers torch transformers\n"
                    f"FlagEmbedding error: {type(flag_error).__name__}: {flag_error}\n"
                    f"Direct transformers error: {type(direct_error).__name__}: {direct_error}\n"
                    f"CrossEncoder error: {type(cross_error).__name__}: {cross_error}"
                ) from cross_error


def transformers_pair_scores(query_text: str, doc_texts: List[str], reranker_info: Dict[str, Any], batch_size: int) -> List[float]:
    torch = reranker_info["torch"]
    tokenizer = reranker_info["tokenizer"]
    model = reranker_info["model"]
    device = reranker_info["device"]

    scores: List[float] = []

    for start in tqdm(range(0, len(doc_texts), batch_size), desc="reranker scoring", unit="batch"):
        batch_docs = doc_texts[start:start + batch_size]
        batch_queries = [query_text] * len(batch_docs)

        encoded = tokenizer(
            batch_queries,
            batch_docs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits

            if logits.ndim == 2 and logits.shape[1] > 1:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)

        scores.extend(float(x) for x in batch_scores.detach().cpu().tolist())

    return scores


def lexical_pair_scores(query_text: str, doc_texts: List[str]) -> List[float]:
    query_tokens = set(re.findall(r"[a-z0-9]+", clean_text(query_text).lower()))
    query_tokens = {tok for tok in query_tokens if len(tok) >= 2}

    if not query_tokens:
        return [0.0 for _ in doc_texts]

    scores: List[float] = []

    for doc in doc_texts:
        doc_tokens = set(re.findall(r"[a-z0-9]+", clean_text(doc).lower()))
        doc_tokens = {tok for tok in doc_tokens if len(tok) >= 2}

        if not doc_tokens:
            scores.append(0.0)
            continue

        overlap = len(query_tokens & doc_tokens)
        score = overlap / ((len(query_tokens) * len(doc_tokens)) ** 0.5)
        scores.append(float(score))

    return scores


def sentence_transformer_scores(query_text: str, doc_texts: List[str], reranker_info: Dict[str, Any], batch_size: int) -> List[float]:
    model = reranker_info["model"]

    # BGE v1.5 benefits from this query instruction; documents stay unprefixed.
    query_for_embedding = f"Represent this sentence for searching relevant passages: {query_text}"

    query_embedding = model.encode(
        [query_for_embedding],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    doc_embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return [float(doc_embedding @ query_embedding) for doc_embedding in doc_embeddings]


def reranker_scores(query_text: str, doc_texts: List[str], reranker_info: Dict[str, Any], batch_size: int) -> Tuple[List[float], str]:
    reranker_type = clean_text(reranker_info.get("reranker_type", ""))

    if "SentenceTransformer" in reranker_type:
        scores = sentence_transformer_scores(query_text, doc_texts, reranker_info, batch_size)
        return scores, "bge_large_cosine_similarity"

    if "AutoModelForSequenceClassification" in reranker_type:
        scores = transformers_pair_scores(query_text, doc_texts, reranker_info, batch_size)
        return scores, "transformers_pair_score"

    if reranker_type == "lexical.safe_fallback":
        return lexical_pair_scores(query_text, doc_texts), "lexical_safe_fallback_score"

    pairs = [[query_text, doc] for doc in doc_texts]
    model = reranker_info["model"]

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
    query_text: str,
    reranker_info: Dict[str, Any],
    ref_dois: Set[str],
    ceiling_dois: Set[str],
    batch_size: int,
) -> List[Dict[str, Any]]:

    if not rows:
        return []

    article_texts = [build_article_semantic_text(row) for row in rows]
    raw_scores, score_backend = reranker_scores(
        query_text,
        article_texts,
        reranker_info,
        batch_size=batch_size,
    )
    norm_scores = normalize_scores(raw_scores)

    ranked_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        doi = normalize_doi(row.get("doi", ""))

        out = dict(row)
        out["doi"] = doi
        out["rank_strategy_3_score"] = round(norm_scores[i], 8)
        out["score_title_objective_sentence_transformer_raw"] = round(raw_scores[i], 8)
        out["score_title_objective_reranker_raw"] = round(raw_scores[i], 8)
        out["score_backend"] = score_backend
        out["article_semantic_text_chars"] = len(article_texts[i])
        out["is_reference_match"] = bool(doi and doi in ref_dois)
        out["is_retrievable_reference_match"] = bool(doi and doi in ceiling_dois)

        ranked_rows.append(out)

    ranked_rows.sort(
        key=lambda r: (
            float(r.get("rank_strategy_3_score", 0.0) or 0.0),
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
    reranker_info: Dict[str, Any],
    model_name: str,
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

    query_text, query_stats = build_title_objective_query(config)

    ranked_rows = rank_articles(
        rows=rows,
        query_text=query_text,
        reranker_info=reranker_info,
        ref_dois=ref_dois,
        ceiling_dois=ceiling_dois,
        batch_size=batch_size,
    )

    base_fields = list(ranked_rows[0].keys()) if ranked_rows else [
        "rank",
        "rank_strategy_3_score",
        "score_title_objective_sentence_transformer_raw",
        "score_title_objective_reranker_raw",
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
        "rank_strategy_3_score",
        "score_title_objective_sentence_transformer_raw",
        "score_title_objective_reranker_raw",
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
        "model_name": model_name,
        "reranker_type": reranker_info.get("reranker_type", ""),
        "model_warning": reranker_info.get("error", ""),
        "articles_ranked": len(ranked_rows),
        "references_source": ref_source,
        "references_total": references_total,
        "references_with_doi": len(ref_dois),
        "retrievable_ceiling": len(ceiling_dois) if ceiling_dois else "",
        **query_stats,
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
    "model_name",
    "reranker_type",
    "model_warning",
    "title_objective_query_chars",
    "n_title_terms",
    "n_objective_terms",
    "n_concept_terms",
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
    out_path = projects_dir / "Rankings" / "rank_strategy_3_summary.csv"
    write_csv(out_path, rows, SUMMARY_FIELDS)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank retrieved articles using title + objective BGE-large embedding scoring."
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
        help="Embedding or reranker model name/local path. Default: BAAI/bge-large-en-v1.5",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Model batch size.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    print("=" * 140)
    print("STEP 9: RANK STRATEGY 3 - TITLE + OBJECTIVE BGE-LARGE SEMANTIC RANKING")
    print("=" * 140)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Force overwrite    : {args.force}")
    print(f"Top-K values       : {TOP_K_VALUES}")
    print(f"Model              : {args.model}")
    print(f"Batch size         : {args.batch_size}")
    print("Uses APIs          : NO")
    print("Uses LLM           : NO")
    print("Uses PDFs/fulltext : NO")
    print("=" * 140)

    print("\nLoading ranking model...")
    reranker_info = load_reranker_model(args.model)
    print(f"Model backend      : {reranker_info.get('reranker_type')}")
    if reranker_info.get("error"):
        print(f"Model warning      : {reranker_info.get('error')}")

    summaries: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(
                project_dir=project_dir,
                force=args.force,
                reranker_info=reranker_info,
                model_name=args.model,
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
    print("  Projects/<project>/Rankings/rank_strategy_3.csv")


if __name__ == "__main__":
    main()
