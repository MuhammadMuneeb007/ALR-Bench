# Step9-Rank9-SPECTER2Semantic.py
#
# Rank Strategy 9:
#   SPECTER/SPECTER2 paper-level semantic ranking using local metadata.
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
#        - inclusion terms
#        - must-have terms
#
#   3. Embeds:
#        - review-intent text
#        - each article title + abstract
#
#   4. Ranks articles by cosine similarity.
#
#   5. Saves:
#        Projects/<project>/Rankings/rank_strategy_9.csv
#        Projects/Rankings/rank_strategy_9_summary.csv
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
#   pip install torch transformers tqdm numpy
#
# Usage:
#   python Step9-Rank9-SPECTER2Semantic.py --projects-dir Projects --force
#
# One project:
#   python Step9-Rank9-SPECTER2Semantic.py --projects-dir Projects --project PROJECT_NAME --force
#
# Optional model:
#   python Step9-Rank9-SPECTER2Semantic.py --projects-dir Projects --model allenai/specter2_base --force
#
# Safer fallback model:
#   python Step9-Rank9-SPECTER2Semantic.py --projects-dir Projects --model allenai/specter --force

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
DEFAULT_RANK_FILE = "rank_strategy_9.csv"

# SPECTER2 base is the intended model. If it causes environment/model-loading issues,
# run with: --model allenai/specter
DEFAULT_MODEL_NAME = "allenai/specter2_base"

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


def truncate_text(text: str, max_chars: int = 5000) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def normalize_scores(raw_scores: List[float]) -> List[float]:
    if not raw_scores:
        return []

    min_s = min(raw_scores)
    max_s = max(raw_scores)

    if max_s <= min_s:
        return [0.0 for _ in raw_scores]

    return [(x - min_s) / (max_s - min_s) for x in raw_scores]


# ============================================================
# REVIEW INTENT BUILDING
# ============================================================

def build_specter_intent_sections(config: Dict[str, Any]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {
        "title": [],
        "objective": [],
        "research_questions": [],
        "concepts": [],
        "search_queries": [],
        "search_keywords": [],
        "include_terms": [],
        "must_terms": [],
    }

    for path in [
        "article_identity.title",
        "title",
    ]:
        sections["title"].extend(get_list_from_path(config, path))

    for path in [
        "review_scope.objective",
        "objective",
    ]:
        sections["objective"].extend(get_list_from_path(config, path))

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

    for path in [
        "search_queries.reported_full_search_strings",
        "search_queries.reported_search_queries_by_database",
        "search_queries.generated_search_queries",
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "generated_search_queries",
    ]:
        sections["search_queries"].extend(get_list_from_path(config, path))

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

    for key in list(sections):
        sections[key] = unique_list(sections[key])

    return sections


def build_review_intent_text(config: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    sections = build_specter_intent_sections(config)

    parts: List[str] = []

    def add_section(label: str, values: List[str], repeat: int = 1) -> None:
        if not values:
            return

        text = "; ".join(values)

        for _ in range(repeat):
            parts.append(f"{label}: {text}")

    add_section("Title", sections["title"], repeat=3)
    add_section("Abstract", sections["objective"], repeat=3)
    add_section("Research questions", sections["research_questions"], repeat=2)
    add_section("Core concepts", sections["concepts"], repeat=3)
    add_section("Search queries", sections["search_queries"], repeat=2)
    add_section("Search keywords", sections["search_keywords"], repeat=2)
    add_section("Inclusion terms", sections["include_terms"], repeat=2)
    add_section("Must-have terms", sections["must_terms"], repeat=2)

    intent_text = clean_text(" ".join(parts))

    if not intent_text:
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
        "review_intent_chars": len(intent_text),
    }

    return truncate_text(intent_text, max_chars=5000), stats


# ============================================================
# ARTICLE TEXT
# ============================================================

def build_specter_paper_text(row: Dict[str, str]) -> str:
    """
    SPECTER-style input should mainly be title + abstract.
    Metadata like journal/database is intentionally not central here.
    """

    title = clean_text(row.get("title", ""))
    abstract = clean_text(row.get("abstract", ""))

    if title and abstract:
        text = f"{title} [SEP] {abstract}"
    elif title:
        text = title
    elif abstract:
        text = abstract
    else:
        text = clean_text(row.get("journal", ""))

    return truncate_text(text, max_chars=5000)


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
# SPECTER EMBEDDING HELPERS
# ============================================================

class SpecterEmbedder:
    def __init__(self, model_name: str, device: str = "", pooling: str = "cls") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "Required packages are not installed. Install them with:\n"
                "pip install torch transformers tqdm numpy"
            ) from e

        self.torch = torch
        self.pooling = pooling.lower().strip()

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def mean_pool(self, model_output: Any, attention_mask: Any) -> Any:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        summed = (token_embeddings * input_mask_expanded).sum(1)
        counts = input_mask_expanded.sum(1).clamp(min=1e-9)

        return summed / counts

    def cls_pool(self, model_output: Any) -> Any:
        return model_output.last_hidden_state[:, 0, :]

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 512,
        show_progress: bool = True,
    ):
        import numpy as np

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        with self.torch.no_grad():
            for start in iterator:
                batch_texts = texts[start:start + batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                encoded = {
                    k: v.to(self.device)
                    for k, v in encoded.items()
                }

                output = self.model(**encoded)

                if self.pooling == "mean":
                    pooled = self.mean_pool(output, encoded["attention_mask"])
                else:
                    pooled = self.cls_pool(output)

                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.cpu().numpy())

        if not all_embeddings:
            return np.zeros((0, 1), dtype="float32")

        return np.vstack(all_embeddings)


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


# ============================================================
# RANKING
# ============================================================

def rank_articles(
    rows: List[Dict[str, str]],
    review_intent_text: str,
    embedder: SpecterEmbedder,
    ref_dois: Set[str],
    ceiling_dois: Set[str],
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:

    if not rows:
        return []

    paper_texts = [build_specter_paper_text(row) for row in rows]

    query_embedding = embedder.encode(
        [review_intent_text],
        batch_size=1,
        max_length=max_length,
        show_progress=False,
    )[0]

    paper_embeddings = embedder.encode(
        paper_texts,
        batch_size=batch_size,
        max_length=max_length,
        show_progress=True,
    )

    raw_scores = cosine_scores_from_embeddings(query_embedding, paper_embeddings)
    norm_scores = normalize_scores(raw_scores)

    ranked_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        doi = normalize_doi(row.get("doi", ""))

        out = dict(row)
        out["doi"] = doi
        out["rank_strategy_9_score"] = round(norm_scores[i], 8)
        out["score_specter_cosine_raw"] = round(raw_scores[i], 8)
        out["specter_paper_text_chars"] = len(paper_texts[i])
        out["is_reference_match"] = bool(doi and doi in ref_dois)
        out["is_retrievable_reference_match"] = bool(doi and doi in ceiling_dois)

        ranked_rows.append(out)

    ranked_rows.sort(
        key=lambda r: (
            float(r.get("rank_strategy_9_score", 0.0) or 0.0),
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
    embedder: SpecterEmbedder,
    model_name: str,
    batch_size: int,
    max_length: int,
    pooling: str,
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
        embedder=embedder,
        ref_dois=ref_dois,
        ceiling_dois=ceiling_dois,
        batch_size=batch_size,
        max_length=max_length,
    )

    base_fields = list(ranked_rows[0].keys()) if ranked_rows else [
        "rank",
        "rank_strategy_9_score",
        "score_specter_cosine_raw",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
    ]

    preferred_first = [
        "rank",
        "rank_strategy_9_score",
        "score_specter_cosine_raw",
        "specter_paper_text_chars",
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
        "model_name": model_name,
        "pooling": pooling,
        "max_length": max_length,
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
    "model_name",
    "pooling",
    "max_length",
    "review_intent_chars",
    "n_title_terms",
    "n_objective_terms",
    "n_research_question_terms",
    "n_concept_terms",
    "n_search_query_terms",
    "n_search_keyword_terms",
    "n_include_terms",
    "n_must_terms",
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
    out_path = projects_dir / "Rankings" / "rank_strategy_9_summary.csv"
    write_csv(out_path, rows, SUMMARY_FIELDS)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank retrieved articles using SPECTER/SPECTER2 paper-level semantic ranking."
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
        help="HuggingFace transformer model name or local model path.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum transformer token length.",
    )

    parser.add_argument(
        "--device",
        default="",
        help="Optional device override: cuda, cpu, cuda:0, etc.",
    )

    parser.add_argument(
        "--pooling",
        default="cls",
        choices=["cls", "mean"],
        help="Pooling strategy. SPECTER usually uses CLS; mean is available as fallback.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    print("=" * 140)
    print("STEP 9: RANK STRATEGY 9 — SPECTER/SPECTER2 PAPER-LEVEL SEMANTIC RANKING")
    print("=" * 140)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Force overwrite    : {args.force}")
    print(f"Top-K values       : {TOP_K_VALUES}")
    print(f"Model              : {args.model}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Max length         : {args.max_length}")
    print(f"Pooling            : {args.pooling}")
    print(f"Device override    : {args.device or 'auto'}")
    print("Uses APIs          : NO")
    print("Uses LLM           : NO")
    print("Uses PDFs/fulltext : NO")
    print("=" * 140)

    print("\nLoading SPECTER/SPECTER2 model...")
    embedder = SpecterEmbedder(
        model_name=args.model,
        device=args.device,
        pooling=args.pooling,
    )
    print(f"Model loaded on device: {embedder.device}")

    summaries: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(
                project_dir=project_dir,
                force=args.force,
                embedder=embedder,
                model_name=args.model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                pooling=args.pooling,
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
    print("  Projects/<project>/Rankings/rank_strategy_9.csv")


if __name__ == "__main__":
    main()