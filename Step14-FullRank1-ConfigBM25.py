#!/usr/bin/env python3
# Step14-FullRank1-ConfigBM25.py
#
# Full-text-aware Rank Strategy 1:
#   Project-specific Config BM25 ranking.
#
# What this script does:
#   1. Reads each project's:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Builds a project-specific ranking query from config:
#        - title
#        - objective
#        - research questions
#        - concepts_keywords
#        - generated_search_keywords
#        - generated_search_queries
#        - generated_abstract_include_keywords
#        - generated_full_text_include_keywords
#        - prisma_s_search_keywords
#        - prisma_s_full_search_strings
#        - prisma_full_text_include_keywords
#        - prisma_full_text_must_have_keywords
#        - prisma_title_abstract_include_keywords
#
#   3. Builds soft exclusion penalty terms from config:
#        - generated_abstract_exclude_keywords
#        - generated_full_text_exclude_keywords
#        - prisma_title_abstract_exclude_keywords
#        - prisma_full_text_exclude_keywords
#        - prisma_eligibility_exclusion
#
#   4. For every article:
#        - uses title + abstract + metadata
#        - adds extracted full text if available
#        - if full text missing, still ranks article using title/abstract
#        - applies small missing-full-text penalty
#
#   5. Scores using:
#        - BM25-style token score
#        - exact phrase/config keyword boosts
#        - field boosts for title and abstract
#        - full-text availability boost
#        - exclusion penalty
#
#   6. Saves:
#        Projects/<project>/Rankings/full_rank_strategy_1.csv
#        Projects/Rankings/full_rank_strategy_1_summary.csv
#
#   7. Prints:
#        C@10, C@20, C@50, C@100, C@200, C@500, C@1000
#
# Notes:
#   - This script does NOT download PDFs.
#   - This script does NOT call APIs.
#   - This script does NOT use LLMs.
#   - Full text is optional. Missing full text does not remove the article.
#   - References are used only for evaluation, not ranking.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")
DEFAULT_DATABASE_DIR = Path("DataBase_Strategy3_Top1000")

DEFAULT_ARTICLES_CLEAN_REL = Path("Search") / "articles_clean_doi_only.csv"
DEFAULT_ARTICLES_RAW_REL = Path("Search") / "articles.csv"
DEFAULT_INPUT_RANK_REL = Path("Rankings") / "rank_strategy_3.csv"
DEFAULT_TOP_N_FROM_INPUT_RANK = 1000

DEFAULT_RANKINGS_DIR = Path("Rankings")
DEFAULT_RANK_FILE = "full_rank_strategy_1.csv"
DEFAULT_SUMMARY_FILE = "full_rank_strategy_1_summary.csv"

TOP_K_VALUES = [10, 20, 50, 100, 200, 500, 1000]

MIN_TOKEN_LEN = 2
MAX_FULLTEXT_CHARS = 250_000

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Field weights
TITLE_WEIGHT = 4.0
ABSTRACT_WEIGHT = 2.5
FULLTEXT_WEIGHT = 1.0
METADATA_WEIGHT = 0.5

# Soft penalties/boosts
MISSING_FULLTEXT_PENALTY = 0.05
HAS_FULLTEXT_BOOST = 0.05
EXCLUSION_TITLE_PENALTY = 3.0
EXCLUSION_ABSTRACT_PENALTY = 2.0
EXCLUSION_FULLTEXT_PENALTY = 0.75

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "had", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "these", "this", "those", "to", "was", "were", "which", "with",
    "using", "use", "used", "based", "between", "among", "study",
    "studies", "review", "systematic", "literature", "scoping",
    "analysis", "research", "paper", "article", "articles", "model",
    "models", "method", "methods", "approach", "approaches", "result",
    "results", "data", "dataset", "datasets", "new", "novel", "also",
    "can", "may", "will", "one", "two", "three", "via", "within",
    "without", "across", "including", "include", "includes", "included",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    text = str(x)
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
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
    text = text.replace("*", " ")

    # Remove database search syntax but keep useful words.
    text = re.sub(
        r"\b(title-abs-key|tiab|mesh|all|ts|ab|ti|kw|topic)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(and|or|not|near/\d+|w/\d+|adj\d+)\b",
        " ",
        text,
        flags=re.I,
    )

    text = re.sub(r"[\(\)\[\]\{\}\"']", " ", text)
    text = re.sub(r"[^a-z0-9+/.-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: Any) -> List[str]:
    text = normalize_phrase(text)
    raw_tokens = re.findall(r"[a-z0-9]+", text)

    out = []
    for tok in raw_tokens:
        if len(tok) < MIN_TOKEN_LEN:
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)

    return out


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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

    with path.open("w", encoding="utf-8", newline="", errors="ignore") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in tqdm(rows, desc=f"Writing {path.name}", unit="row", leave=True, dynamic_ncols=True):
            writer.writerow({field: row.get(field, "") for field in fields})


def flatten_values(value: Any) -> List[str]:
    """
    Recursively flatten strings/lists/dicts from config fields.
    """
    out: List[str] = []

    if value is None:
        return out

    if isinstance(value, str):
        text = clean_text(value)
        if text:
            out.append(text)
        return out

    if isinstance(value, (int, float, bool)):
        text = clean_text(value)
        if text:
            out.append(text)
        return out

    if isinstance(value, list):
        for item in value:
            out.extend(flatten_values(item))
        return out

    if isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_values(item))
        return out

    text = clean_text(value)
    if text:
        out.append(text)

    return out


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []

    for item in items:
        item = clean_text(item)
        if not item:
            continue

        key = item.lower().strip()
        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


# ============================================================
# PROJECT DISCOVERY
# ============================================================

def discover_projects(projects_dir: Path, one_project: str = "") -> List[Path]:
    if one_project:
        p = projects_dir / one_project
        return [p] if p.exists() else []

    projects = []
    if not projects_dir.exists():
        return projects

    for p in sorted(projects_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "config.json").exists():
            projects.append(p)

    return projects


# ============================================================
# CONFIG PROFILE
# ============================================================

def build_config_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build project-specific positive and negative ranking profiles.
    """

    positive_fields = [
        "title",
        "objective",
        "research_questions",
        "concepts_keywords",

        "prisma_s_search_keywords",
        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "prisma_s_controlled_vocabulary_terms",
        "prisma_s_free_text_terms",

        "generated_search_queries",
        "generated_search_keywords",
        "generated_abstract_include_keywords",
        "generated_full_text_include_keywords",
        "generated_quality_assessment_keywords",

        "prisma_title_abstract_include_keywords",
        "prisma_title_abstract_must_have_keywords",
        "prisma_title_abstract_uncertain_keywords",

        "prisma_full_text_include_keywords",
        "prisma_full_text_must_have_keywords",
        "prisma_full_text_uncertain_keywords",

        "prisma_eligibility_inclusion",
        "prisma_quality_assessment_keywords",
        "prisma_quality_assessment_questions",
    ]

    negative_fields = [
        "generated_abstract_exclude_keywords",
        "generated_full_text_exclude_keywords",
        "prisma_title_abstract_exclude_keywords",
        "prisma_full_text_exclude_keywords",
        "prisma_eligibility_exclusion",
    ]

    query_text_parts: List[str] = []
    positive_phrases: List[str] = []
    negative_phrases: List[str] = []

    for field in positive_fields:
        vals = flatten_values(config.get(field, []))
        for v in vals:
            query_text_parts.append(v)
            positive_phrases.append(v)

    for field in negative_fields:
        vals = flatten_values(config.get(field, []))
        for v in vals:
            negative_phrases.append(v)

    positive_phrases = unique_keep_order(positive_phrases)
    negative_phrases = unique_keep_order(negative_phrases)

    query_text = " ".join(query_text_parts)
    query_tokens = tokenize(query_text)

    # Token weights based on frequency in config.
    token_counts = Counter(query_tokens)

    # Remove extremely generic tokens if they appear from noisy configs.
    query_tokens_unique = []
    seen = set()
    for tok in query_tokens:
        if tok not in seen:
            seen.add(tok)
            query_tokens_unique.append(tok)

    positive_norm_phrases = []
    for p in positive_phrases:
        np = normalize_phrase(p)
        if len(np) >= 3:
            positive_norm_phrases.append(np)

    negative_norm_phrases = []
    for p in negative_phrases:
        np = normalize_phrase(p)
        if len(np) >= 3:
            negative_norm_phrases.append(np)

    return {
        "query_text": query_text,
        "query_tokens": query_tokens_unique,
        "query_token_counts": token_counts,
        "positive_phrases": positive_norm_phrases,
        "negative_phrases": negative_norm_phrases,
        "n_query_tokens": len(query_tokens_unique),
        "n_positive_phrases": len(positive_norm_phrases),
        "n_negative_phrases": len(negative_norm_phrases),
    }


# ============================================================
# ARTICLE / REFERENCE LOADING
# ============================================================

def get_first_existing(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in row and clean_text(row.get(k, "")):
            return clean_text(row.get(k, ""))
    return ""


def sort_ranked_input_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def rank_key(row: Dict[str, str]) -> Tuple[int, float]:
        rank_value = get_first_existing(row, [
            "rank_strategy_3",
            "rank",
            "Rank",
            "input_rank",
        ])
        try:
            rank_num = int(float(rank_value))
        except Exception:
            rank_num = 10**9

        score_value = get_first_existing(row, [
            "rank_strategy_3_score",
            "score_title_objective_sentence_transformer_raw",
            "score",
        ])
        try:
            score_num = float(score_value)
        except Exception:
            score_num = 0.0

        return rank_num, -score_num

    return sorted(rows, key=rank_key)


def load_articles(
    project_dir: Path,
    input_rank_rel: Path,
    top_n: int,
) -> Tuple[List[Dict[str, str]], str, int]:
    rank_path = project_dir / input_rank_rel
    rows = read_csv_rows(rank_path)
    source = str(rank_path)

    if not rows:
        return [], source, 0

    rows = sort_ranked_input_rows(rows)
    if top_n > 0:
        rows = rows[:top_n]

    # Deduplicate primarily by DOI, then normalized title.
    deduped = []
    seen = set()

    for row in rows:
        doi = normalize_doi(get_first_existing(row, ["doi", "DOI", "article_doi"]))
        title = get_first_existing(row, ["title", "Title", "article_title", "display_name"])
        title_norm = normalize_title(title)

        if doi:
            key = ("doi", doi)
        elif title_norm:
            key = ("title", title_norm)
        else:
            continue

        if key in seen:
            continue

        seen.add(key)

        row["_doi_norm"] = doi
        row["_title_norm"] = title_norm
        row["_input_rank_source"] = source
        row["_input_rank"] = get_first_existing(row, ["rank_strategy_3", "rank", "Rank"])
        row["_input_rank_score"] = get_first_existing(row, ["rank_strategy_3_score", "score"])
        deduped.append(row)

    return deduped, source, len(rows)


def possible_reference_files(project_dir: Path) -> List[Path]:
    """
    Try common reference file names from previous steps.
    Adjust/add names here if your project uses another reference file.
    """
    candidates = [
        project_dir / "References" / "references_clean.csv",
        project_dir / "References" / "references_clean_doi_only.csv",
        project_dir / "References" / "references.csv",
        project_dir / "references_clean.csv",
        project_dir / "references.csv",
        project_dir / "Search" / "references_clean.csv",
        project_dir / "Search" / "references.csv",
    ]
    return candidates


def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
    """
    Extract reference DOI/title records from nested JSON reference files.

    This matches the working Step9 logic and supports:
        Projects/<project>/references.json
        Projects/<project>/references_resolved.json
        Projects/<project>/references_gold.json

    It recursively searches dictionaries/lists for DOI-like and title-like keys.
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


def load_reference_dois(project_dir: Path) -> Set[str]:
    """
    Load reference DOIs for evaluation.

    Priority 1: JSON reference files used by the working Step9 script:
        Projects/<project>/references.json
        Projects/<project>/references_resolved.json
        Projects/<project>/references_gold.json

    Priority 2: CSV reference files from older project structures.
    """

    json_candidates = [
        "references.json",
        "references_resolved.json",
        "references_gold.json",
    ]

    for rel in json_candidates:
        path = project_dir / rel
        if not path.exists():
            continue

        data = read_json(path, default={})
        records = extract_reference_records_from_json(data)

        ref_dois = {
            normalize_doi(x.get("doi", ""))
            for x in records
            if normalize_doi(x.get("doi", ""))
        }

        if ref_dois:
            return ref_dois

    # Fallback: old CSV-style reference files.
    ref_dois: Set[str] = set()

    for path in possible_reference_files(project_dir):
        if not path.exists():
            continue

        rows = read_csv_rows(path)
        for row in rows:
            doi = normalize_doi(get_first_existing(row, ["doi", "DOI", "article_doi", "reference_doi", "work_doi"]))
            if doi:
                ref_dois.add(doi)

        if ref_dois:
            return ref_dois

    return ref_dois


# ============================================================
# FULL TEXT DISCOVERY
# ============================================================

def safe_read_text(path: Path, max_chars: int = MAX_FULLTEXT_CHARS) -> str:
    if not path.exists() or not path.is_file():
        return ""

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(text)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception:
        return ""


def find_matching_database_project_folder(database_dir: Path, project_name: str) -> Optional[Path]:
    """
    Robustly match a Projects/<project_name> folder to its database folder.

    This fixes the common case where the database folder was saved with a
    truncated project name plus an 8-character hash suffix, e.g.

        Projects:
            health_outcomes_associated_with_reallocations_of_time_between_sleep_sedentary_be

        DataBase_Strategy3_Top1000:
            health_outcomes_associated_with_reallocations_of_time_betwee_0cd0b897

    The old code only checked p.name.startswith(f"{project_name}_"), which fails
    when the database folder is truncated. This function uses exact, prefix, and
    token-overlap matching.
    """

    if not database_dir.exists():
        return None

    candidates = [p for p in database_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not candidates:
        return None

    # 1) Exact match.
    direct = database_dir / project_name
    if direct.exists() and direct.is_dir():
        return direct

    project_norm = normalize_title(project_name).replace(" ", "_")
    project_tokens = set(tokenize(project_name))

    # 2) Full prefix match either way.
    for p in candidates:
        db_name = p.name
        db_base = re.sub(r"_[a-f0-9]{8}$", "", db_name.lower())
        db_norm = normalize_title(db_base).replace(" ", "_")

        if db_name == project_name:
            return p
        if db_name.startswith(f"{project_name}_"):
            return p
        if project_norm.startswith(db_norm) and len(db_norm) >= 25:
            return p
        if db_norm.startswith(project_norm) and len(project_norm) >= 25:
            return p

    # 3) Token-overlap fallback.
    best_path: Optional[Path] = None
    best_score = 0.0

    for p in candidates:
        db_name = p.name
        db_base = re.sub(r"_[a-f0-9]{8}$", "", db_name.lower())
        db_tokens = set(tokenize(db_base))

        if not project_tokens or not db_tokens:
            continue

        # Use both directions so truncated names still score highly.
        overlap_project = len(project_tokens & db_tokens) / max(1, len(project_tokens))
        overlap_db = len(project_tokens & db_tokens) / max(1, len(db_tokens))
        score = max(overlap_project, overlap_db)

        if score > best_score:
            best_score = score
            best_path = p

    # 0.50 is intentionally permissive because long names are truncated.
    if best_path is not None and best_score >= 0.50:
        return best_path

    return None


def _temp_index_from_article_dir(article_dir: Path) -> str:
    """Return Temp index as string if folder is Temp_X, otherwise empty string."""
    m = re.match(r"^Temp_(\d+)$", article_dir.name)
    return m.group(1) if m else ""


def build_fulltext_index(database_dir: Path, project_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Build index of extracted full texts for one project.

    FIXED:
      - correctly matches truncated/hash database project folders
      - scans Temp_* article folders under the matched project folder
      - indexes text by DOI, title, folder path, and Temp index
      - merges nested base_metadata from extraction_metadata.json
    """

    index: Dict[str, Dict[str, Any]] = {}

    if not database_dir.exists():
        print(f"     database missing: {database_dir}")
        return index

    text_file_names = [
        "Article_Text.txt",
        "COMBINED.txt",
        "full_text.txt",
        "COMBINED.md",
        "fulltext.txt",
        "text.txt",
    ]

    metadata_names = [
        "extraction_metadata.json",
        "metadata.json",
        "article_metadata.json",
        "download_metadata.json",
    ]

    matched_project_dir = find_matching_database_project_folder(database_dir, project_name)

    if matched_project_dir is None:
        print(f"     fulltext roots: [NO MATCH FOUND FOR PROJECT: {project_name}]")
        return index

    print(f"     fulltext roots: {matched_project_dir}")

    # In your database structure, each article lives in Temp_0, Temp_1, ...
    temp_dirs = [
        p for p in matched_project_dir.iterdir()
        if p.is_dir() and re.match(r"^Temp_\d+$", p.name)
    ]

    temp_dirs = sorted(
        temp_dirs,
        key=lambda p: int(p.name.replace("Temp_", ""))
        if p.name.replace("Temp_", "").isdigit()
        else 10**9,
    )

    # Fallback for any older non-Temp layout.
    if not temp_dirs:
        temp_dirs = [p for p in matched_project_dir.iterdir() if p.is_dir()]
        if not temp_dirs:
            temp_dirs = [matched_project_dir]

    unique_files: List[Path] = []

    for article_dir in tqdm(
        temp_dirs,
        desc=f"{project_name} | finding combined text",
        unit="folder",
        leave=True,
        dynamic_ncols=True,
    ):
        chosen: Optional[Path] = None

        for name in text_file_names:
            candidate = article_dir / name
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                chosen = candidate
                break

        if chosen:
            unique_files.append(chosen)

    print(f"     fulltext files selected: {len(unique_files)}")

    for text_path in tqdm(
        unique_files,
        desc=f"{project_name} | indexing full text",
        unit="file",
        leave=True,
        dynamic_ncols=True,
    ):
        article_dir = text_path.parent

        text = safe_read_text(text_path)
        if not text:
            continue

        # Merge all metadata files. Later files can fill missing values.
        metadata: Dict[str, Any] = {}

        for meta_name in metadata_names:
            mp = article_dir / meta_name
            if mp.exists():
                try:
                    this_meta = read_json(mp, {})
                    if isinstance(this_meta, dict):
                        # extraction_metadata.json may contain base_metadata.
                        base_metadata = this_meta.get("base_metadata", {})
                        if isinstance(base_metadata, dict):
                            metadata.update(base_metadata)
                        metadata.update(this_meta)
                except Exception:
                    pass

        doi = normalize_doi(
            metadata.get("doi")
            or metadata.get("DOI")
            or metadata.get("article_doi")
            or metadata.get("reference_doi")
            or metadata.get("work_doi")
            or ""
        )

        title = clean_text(
            metadata.get("title")
            or metadata.get("article_title")
            or metadata.get("display_name")
            or metadata.get("work_title")
            or ""
        )

        # Sometimes DOI/title are in nested metadata dicts.
        if not doi or not title:
            for v in metadata.values():
                if isinstance(v, dict):
                    if not doi:
                        doi = normalize_doi(
                            v.get("doi")
                            or v.get("DOI")
                            or v.get("article_doi")
                            or v.get("work_doi")
                            or ""
                        )
                    if not title:
                        title = clean_text(
                            v.get("title")
                            or v.get("article_title")
                            or v.get("display_name")
                            or v.get("work_title")
                            or ""
                        )

        # Try DOI from folder path as a final fallback.
        if not doi:
            folder_text = str(article_dir)
            m = re.search(r"(10\.\d{4,9}[^\s/\\]+[/_][^\s/\\]+)", folder_text, flags=re.I)
            if m:
                doi = normalize_doi(m.group(1).replace("_", "/"))

        title_norm = normalize_title(title)
        folder_norm = normalize_title(article_dir.name)
        temp_index = _temp_index_from_article_dir(article_dir)

        record = {
            "text": text,
            "path": str(text_path),
            "article_dir": str(article_dir),
            "doi": doi,
            "title": title,
            "temp_index": temp_index,
            "word_count": len(tokenize(text)),
            "char_count": len(text),
        }

        keys: List[Tuple[str, str]] = []

        if doi:
            keys.append(("doi", doi))

        if title_norm:
            keys.append(("title", title_norm))

        if folder_norm:
            keys.append(("folder", folder_norm))

        if temp_index:
            keys.append(("temp", temp_index))

        keys.append(("path", str(article_dir.resolve()).lower()))

        for ktype, kval in keys:
            if kval:
                index[f"{ktype}:{kval}"] = record

    print(f"     fulltext index items: {len(index)}")

    return index


def find_fulltext_for_article(
    article: Dict[str, str],
    fulltext_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    doi = article.get("_doi_norm", "")
    title_norm = article.get("_title_norm", "")

    if doi:
        rec = fulltext_index.get(f"doi:{doi}")
        if rec:
            return rec

    if title_norm:
        rec = fulltext_index.get(f"title:{title_norm}")
        if rec:
            return rec

    # Fallback: fuzzy folder-name containment.
    # This is intentionally conservative to avoid wrong matches.
    if title_norm and len(title_norm) > 25:
        title_tokens = set(tokenize(title_norm))
        if len(title_tokens) >= 4:
            best_rec = {}
            best_overlap = 0.0

            for key, rec in fulltext_index.items():
                if not key.startswith("folder:"):
                    continue

                folder = key.replace("folder:", "", 1)
                folder_tokens = set(tokenize(folder))
                if not folder_tokens:
                    continue

                overlap = len(title_tokens & folder_tokens) / max(1, len(title_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_rec = rec

            if best_overlap >= 0.70:
                return best_rec

    return {}


# ============================================================
# SCORING
# ============================================================

def bm25_score(
    doc_tokens: List[str],
    query_tokens: List[str],
    query_token_counts: Counter,
    idf: Dict[str, float],
    avgdl: float,
) -> float:
    if not doc_tokens or not query_tokens:
        return 0.0

    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0

    for term in query_tokens:
        if term not in tf:
            continue

        term_tf = tf[term]
        term_idf = idf.get(term, 0.0)

        denom = term_tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / max(avgdl, 1.0))
        term_score = term_idf * ((term_tf * (BM25_K1 + 1.0)) / max(denom, 1e-9))

        # If a token appears multiple times in config, give it small extra importance.
        q_weight = 1.0 + math.log1p(query_token_counts.get(term, 1)) * 0.25
        score += term_score * q_weight

    return score


def phrase_match_score(text_norm: str, phrases: List[str]) -> Tuple[float, int]:
    if not text_norm or not phrases:
        return 0.0, 0

    score = 0.0
    matched = 0

    for phrase in phrases:
        phrase = normalize_phrase(phrase)
        if not phrase:
            continue

        # Skip very long query strings as exact phrases; their tokens are already in BM25.
        n_tokens = len(phrase.split())
        if n_tokens > 12:
            continue

        if phrase in text_norm:
            matched += 1

            if n_tokens >= 4:
                score += 3.0
            elif n_tokens == 3:
                score += 2.0
            elif n_tokens == 2:
                score += 1.25
            else:
                score += 0.5

    return score, matched


def exclusion_penalty(
    title_norm: str,
    abstract_norm: str,
    fulltext_norm: str,
    negative_phrases: List[str],
) -> Tuple[float, int]:
    if not negative_phrases:
        return 0.0, 0

    penalty = 0.0
    hits = 0

    for phrase in negative_phrases:
        phrase = normalize_phrase(phrase)
        if not phrase:
            continue

        if phrase in title_norm:
            penalty += EXCLUSION_TITLE_PENALTY
            hits += 1

        if phrase in abstract_norm:
            penalty += EXCLUSION_ABSTRACT_PENALTY
            hits += 1

        if fulltext_norm and phrase in fulltext_norm:
            penalty += EXCLUSION_FULLTEXT_PENALTY
            hits += 1

    return penalty, hits


def build_article_fields(
    article: Dict[str, str],
    fulltext_record: Dict[str, Any],
) -> Dict[str, Any]:
    title = get_first_existing(article, ["title", "Title", "article_title", "display_name"])
    abstract = get_first_existing(article, ["abstract", "Abstract", "summary", "description"])
    journal = get_first_existing(article, ["journal", "Journal", "container_title", "source", "venue"])
    article_type = get_first_existing(article, ["type", "article_type", "publication_type"])
    source_db = get_first_existing(article, ["source_db", "database", "source_database"])
    year = get_first_existing(article, ["year", "publication_year", "published_year", "date"])

    metadata = " ".join([journal, article_type, source_db, year])

    fulltext = clean_text(fulltext_record.get("text", "")) if fulltext_record else ""
    has_fulltext = bool(fulltext)

    if fulltext and len(fulltext) > MAX_FULLTEXT_CHARS:
        fulltext = fulltext[:MAX_FULLTEXT_CHARS]

    return {
        "title": title,
        "abstract": abstract,
        "metadata": metadata,
        "fulltext": fulltext,
        "has_fulltext": has_fulltext,
        "fulltext_path": fulltext_record.get("path", "") if fulltext_record else "",
        "fulltext_word_count": fulltext_record.get("word_count", 0) if fulltext_record else 0,
        "fulltext_char_count": fulltext_record.get("char_count", 0) if fulltext_record else 0,
    }


def score_articles(
    articles: List[Dict[str, str]],
    profile: Dict[str, Any],
    fulltext_index: Dict[str, Dict[str, Any]],
    project_name: str = "",
) -> List[Dict[str, Any]]:
    query_tokens: List[str] = profile["query_tokens"]
    query_token_counts: Counter = profile["query_token_counts"]
    positive_phrases: List[str] = profile["positive_phrases"]
    negative_phrases: List[str] = profile["negative_phrases"]

    prepared = []

    # First pass: prepare weighted document tokens for IDF/BM25.
    doc_freq = defaultdict(int)
    doc_lengths = []

    label = project_name or "project"

    for article in tqdm(
        articles,
        desc=f"{label} | preparing BM25 documents",
        unit="article",
        leave=True,
        dynamic_ncols=True,
    ):
        fulltext_record = find_fulltext_for_article(article, fulltext_index)
        fields = build_article_fields(article, fulltext_record)

        title_tokens = tokenize(fields["title"])
        abstract_tokens = tokenize(fields["abstract"])
        metadata_tokens = tokenize(fields["metadata"])
        fulltext_tokens = tokenize(fields["fulltext"])

        # Weighted representation by repeating tokens.
        weighted_tokens = []
        weighted_tokens.extend(title_tokens * int(TITLE_WEIGHT))
        weighted_tokens.extend(abstract_tokens * int(ABSTRACT_WEIGHT))
        weighted_tokens.extend(metadata_tokens * int(max(1, METADATA_WEIGHT)))
        weighted_tokens.extend(fulltext_tokens * int(max(1, FULLTEXT_WEIGHT)))

        # For non-integer weights, the above is enough for simple BM25.
        # We also calculate separate field scores below.
        if not weighted_tokens:
            weighted_tokens = title_tokens + abstract_tokens + metadata_tokens + fulltext_tokens

        unique_doc_terms = set(weighted_tokens)
        for tok in unique_doc_terms:
            doc_freq[tok] += 1

        doc_lengths.append(len(weighted_tokens))

        prepared.append({
            "article": article,
            "fields": fields,
            "title_tokens": title_tokens,
            "abstract_tokens": abstract_tokens,
            "metadata_tokens": metadata_tokens,
            "fulltext_tokens": fulltext_tokens,
            "weighted_tokens": weighted_tokens,
        })

    n_docs = max(1, len(prepared))
    avgdl = sum(doc_lengths) / max(1, len(doc_lengths)) if doc_lengths else 1.0

    idf = {}
    for term in query_tokens:
        df = doc_freq.get(term, 0)
        idf[term] = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

    scored_rows = []

    for item in tqdm(
        prepared,
        desc=f"{label} | scoring BM25 articles",
        unit="article",
        leave=True,
        dynamic_ncols=True,
    ):
        article = item["article"]
        fields = item["fields"]

        title = fields["title"]
        abstract = fields["abstract"]
        metadata = fields["metadata"]
        fulltext = fields["fulltext"]
        has_fulltext = fields["has_fulltext"]

        title_norm = normalize_phrase(title)
        abstract_norm = normalize_phrase(abstract)
        fulltext_norm = normalize_phrase(fulltext)
        combined_norm = normalize_phrase(" ".join([title, abstract, metadata, fulltext]))

        # Main BM25 score over weighted document tokens.
        main_bm25 = bm25_score(
            item["weighted_tokens"],
            query_tokens,
            query_token_counts,
            idf,
            avgdl,
        )

        # Field-specific BM25, so title/abstract do not get drowned by full text.
        title_bm25 = bm25_score(
            item["title_tokens"],
            query_tokens,
            query_token_counts,
            idf,
            max(1.0, len(item["title_tokens"])),
        )

        abstract_bm25 = bm25_score(
            item["abstract_tokens"],
            query_tokens,
            query_token_counts,
            idf,
            max(1.0, len(item["abstract_tokens"])),
        )

        fulltext_bm25 = bm25_score(
            item["fulltext_tokens"],
            query_tokens,
            query_token_counts,
            idf,
            max(1.0, len(item["fulltext_tokens"])),
        )

        # Phrase boosts.
        phrase_score_all, phrase_hits_all = phrase_match_score(combined_norm, positive_phrases)
        phrase_score_title, phrase_hits_title = phrase_match_score(title_norm, positive_phrases)
        phrase_score_abstract, phrase_hits_abstract = phrase_match_score(abstract_norm, positive_phrases)
        phrase_score_fulltext, phrase_hits_fulltext = phrase_match_score(fulltext_norm, positive_phrases)

        # Exclusion penalty.
        excl_penalty, excl_hits = exclusion_penalty(
            title_norm=title_norm,
            abstract_norm=abstract_norm,
            fulltext_norm=fulltext_norm,
            negative_phrases=negative_phrases,
        )

        # Coverage score.
        article_token_set = set(item["title_tokens"]) | set(item["abstract_tokens"]) | set(item["metadata_tokens"]) | set(item["fulltext_tokens"])
        query_token_set = set(query_tokens)

        if query_token_set:
            token_coverage = len(article_token_set & query_token_set) / max(1, len(query_token_set))
        else:
            token_coverage = 0.0

        coverage_score = token_coverage * 5.0

        fulltext_adjustment = HAS_FULLTEXT_BOOST if has_fulltext else -MISSING_FULLTEXT_PENALTY

        final_score = (
            main_bm25
            + (TITLE_WEIGHT * 0.80 * title_bm25)
            + (ABSTRACT_WEIGHT * 0.50 * abstract_bm25)
            + (FULLTEXT_WEIGHT * 0.15 * fulltext_bm25)
            + coverage_score
            + (phrase_score_title * 1.50)
            + (phrase_score_abstract * 1.00)
            + (phrase_score_fulltext * 0.35)
            + (phrase_score_all * 0.10)
            + fulltext_adjustment
            - excl_penalty
        )

        doi = article.get("_doi_norm", "")
        title_out = title
        abstract_out = abstract

        scored_rows.append({
            "rank": "",
            "score": round(final_score, 8),
            "main_bm25": round(main_bm25, 8),
            "title_bm25": round(title_bm25, 8),
            "abstract_bm25": round(abstract_bm25, 8),
            "fulltext_bm25": round(fulltext_bm25, 8),
            "coverage_score": round(coverage_score, 8),
            "token_coverage": round(token_coverage, 8),
            "phrase_score_title": round(phrase_score_title, 8),
            "phrase_score_abstract": round(phrase_score_abstract, 8),
            "phrase_score_fulltext": round(phrase_score_fulltext, 8),
            "phrase_hits_title": phrase_hits_title,
            "phrase_hits_abstract": phrase_hits_abstract,
            "phrase_hits_fulltext": phrase_hits_fulltext,
            "phrase_hits_all": phrase_hits_all,
            "exclusion_penalty": round(excl_penalty, 8),
            "exclusion_hits": excl_hits,
            "has_full_text": int(has_fulltext),
            "fulltext_word_count": fields["fulltext_word_count"],
            "fulltext_char_count": fields["fulltext_char_count"],
            "fulltext_path": fields["fulltext_path"],
            "doi": doi,
            "title": title_out,
            "abstract": abstract_out,
            "journal": get_first_existing(article, ["journal", "Journal", "container_title", "source", "venue"]),
            "year": get_first_existing(article, ["year", "publication_year", "published_year", "date"]),
            "source_db": get_first_existing(article, ["source_db", "database", "source_database"]),
            "input_rank_source": article.get("_input_rank_source", ""),
            "input_rank": article.get("_input_rank", ""),
            "input_rank_score": article.get("_input_rank_score", ""),
        })

    scored_rows.sort(key=lambda r: safe_float(r.get("score", 0.0)), reverse=True)

    for i, row in enumerate(scored_rows, start=1):
        row["rank"] = i

    return scored_rows


# ============================================================
# EVALUATION
# ============================================================

def evaluate_common_at_k(
    ranked_rows: List[Dict[str, Any]],
    reference_dois: Set[str],
    top_k_values: List[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    ranked_dois = [normalize_doi(r.get("doi", "")) for r in ranked_rows]
    ranked_dois = [d for d in ranked_dois if d]

    refs = set(reference_dois)
    n_refs = len(refs)

    for k in top_k_values:
        top_dois = set(ranked_dois[:k])
        common = len(top_dois & refs)

        out[f"C@{k}"] = common
        out[f"Coverage@{k}"] = round(common / n_refs, 6) if n_refs else 0.0

    return out


# ============================================================
# MAIN PER PROJECT
# ============================================================

def process_project(
    project_dir: Path,
    database_dir: Path,
    input_rank_rel: Path,
    top_n: int,
    force: bool = False,
) -> Dict[str, Any]:
    project_name = project_dir.name
    rankings_dir = project_dir / DEFAULT_RANKINGS_DIR
    out_path = rankings_dir / DEFAULT_RANK_FILE

    if out_path.exists() and not force:
        print(f"SKIP | {project_name} | exists: {out_path}")
        return {
            "project": project_name,
            "status": "skipped_exists",
            "ranked": 0,
        }

    config_path = project_dir / "config.json"
    config = read_json(config_path, {})

    if not config:
        print(f"FAIL | {project_name} | missing/invalid config.json")
        return {
            "project": project_name,
            "status": "missing_config",
            "ranked": 0,
        }

    articles, articles_source, input_rank_rows_loaded = load_articles(
        project_dir=project_dir,
        input_rank_rel=input_rank_rel,
        top_n=top_n,
    )
    if not articles:
        print(f"FAIL | {project_name} | no ranked input articles found: {articles_source}")
        return {
            "project": project_name,
            "status": "no_ranked_input_articles",
            "ranked": 0,
            "articles_source": articles_source,
        }

    reference_dois = load_reference_dois(project_dir)

    profile = build_config_profile(config)
    fulltext_index = build_fulltext_index(database_dir, project_name)

    ranked_rows = score_articles(
        articles=articles,
        profile=profile,
        fulltext_index=fulltext_index,
        project_name=project_name,
    )

    eval_metrics = evaluate_common_at_k(ranked_rows, reference_dois, TOP_K_VALUES)

    # Add reference flag after scoring.
    refs = set(reference_dois)
    for row in tqdm(
        ranked_rows,
        desc=f"{project_name} | flagging references",
        unit="article",
        leave=True,
        dynamic_ncols=True,
    ):
        doi = normalize_doi(row.get("doi", ""))
        row["is_reference"] = int(bool(doi and doi in refs))

    fields = [
        "rank",
        "score",
        "main_bm25",
        "title_bm25",
        "abstract_bm25",
        "fulltext_bm25",
        "coverage_score",
        "token_coverage",
        "phrase_score_title",
        "phrase_score_abstract",
        "phrase_score_fulltext",
        "phrase_hits_title",
        "phrase_hits_abstract",
        "phrase_hits_fulltext",
        "phrase_hits_all",
        "exclusion_penalty",
        "exclusion_hits",
        "has_full_text",
        "fulltext_word_count",
        "fulltext_char_count",
        "is_reference",
        "doi",
        "title",
        "abstract",
        "journal",
        "year",
        "source_db",
        "input_rank",
        "input_rank_score",
        "input_rank_source",
        "fulltext_path",
    ]

    write_csv(out_path, ranked_rows, fields)

    n_fulltext = sum(1 for r in ranked_rows if int(r.get("has_full_text", 0)) == 1)
    n_missing_fulltext = len(ranked_rows) - n_fulltext

    summary = {
        "project": project_name,
        "status": "ok",
        "ranked": len(ranked_rows),
        "articles_input": len(articles),
        "articles_source": articles_source,
        "input_rank_rows_loaded": input_rank_rows_loaded,
        "input_top_n": top_n,
        "refs_doi": len(reference_dois),
        "fulltext_index_items": len(fulltext_index),
        "with_full_text": n_fulltext,
        "missing_full_text": n_missing_fulltext,
        "query_tokens": profile["n_query_tokens"],
        "positive_phrases": profile["n_positive_phrases"],
        "negative_phrases": profile["n_negative_phrases"],
        "rank_file": str(out_path),
    }
    summary.update(eval_metrics)

    # Print compact per-project line.
    metric_text = " | ".join([f"C@{k}={summary.get(f'C@{k}', 0)}" for k in TOP_K_VALUES])
    coverage_text = " | ".join([f"Cov@{k}={summary.get(f'Coverage@{k}', 0):.3f}" for k in TOP_K_VALUES])

    print(
        f"OK | {project_name} | ranked={len(ranked_rows)} | "
        f"refs_doi={len(reference_dois)} | fulltext={n_fulltext} | "
        f"missing_fulltext={n_missing_fulltext} | {metric_text}"
    )
    print(f"     {coverage_text}")
    print(f"     file={out_path}")

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step14 FullRank1: project-specific full-text-aware Config BM25 ranking."
    )

    parser.add_argument(
        "--projects-dir",
        type=str,
        default=str(DEFAULT_PROJECTS_DIR),
        help="Path to Projects directory.",
    )

    parser.add_argument(
        "--database-dir",
        type=str,
        default=str(DEFAULT_DATABASE_DIR),
        help="Path to extracted full-text database directory.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="Run only one project folder name.",
    )

    parser.add_argument(
        "--input-rank-file",
        type=str,
        default=str(DEFAULT_INPUT_RANK_REL),
        help="Per-project ranking CSV to use as Step14 input, relative to each project folder.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N_FROM_INPUT_RANK,
        help="Only rerank the top N rows from the input ranking file.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing ranking files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    projects_dir = Path(args.projects_dir)
    database_dir = Path(args.database_dir)
    input_rank_rel = Path(args.input_rank_file)
    top_n = max(0, int(args.top_n))

    projects = discover_projects(projects_dir, args.project)

    print("=" * 120)
    print("Step14 FullRank1: Config BM25 full-text-aware ranking")
    print(f"Projects directory : {projects_dir}")
    print(f"Database directory : {database_dir}")
    print(f"Projects found     : {len(projects)}")
    print(f"Force overwrite    : {args.force}")
    print(f"Input rank file    : Projects/<project>/{input_rank_rel}")
    print(f"Input top N        : {top_n}")
    print(f"Top-K values       : {TOP_K_VALUES}")
    print("Uses APIs          : NO")
    print("Uses LLM           : NO")
    print("Uses PDFs download : NO")
    print("Full text required : NO, fallback to title/abstract")
    print("=" * 120)

    if not projects:
        print("No projects found.")
        return

    summaries: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(
        tqdm(projects, desc="All projects", unit="project", leave=True, dynamic_ncols=True),
        start=1,
    ):
        print()
        print(f"[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(
                project_dir=project_dir,
                database_dir=database_dir,
                input_rank_rel=input_rank_rel,
                top_n=top_n,
                force=args.force,
            )
            summaries.append(summary)

        except Exception as e:
            print(f"FAIL | {project_dir.name} | {type(e).__name__}: {e}")
            summaries.append({
                "project": project_dir.name,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "ranked": 0,
            })

    # Save global summary.
    summary_dir = projects_dir / DEFAULT_RANKINGS_DIR
    summary_path = summary_dir / DEFAULT_SUMMARY_FILE

    summary_fields = [
        "project",
        "status",
        "ranked",
        "articles_input",
        "input_rank_rows_loaded",
        "input_top_n",
        "refs_doi",
        "fulltext_index_items",
        "with_full_text",
        "missing_full_text",
        "query_tokens",
        "positive_phrases",
        "negative_phrases",
    ]

    for k in TOP_K_VALUES:
        summary_fields.append(f"C@{k}")
        summary_fields.append(f"Coverage@{k}")

    summary_fields.extend(["articles_source", "rank_file", "error"])

    write_csv(summary_path, summaries, summary_fields)

    print()
    print("=" * 120)
    print(f"Saved summary: {summary_path}")
    print("=" * 120)

    # Print final table.
    header = (
        f"{'Project':45s} "
        f"{'Ranked':>8s} "
        f"{'Refs':>6s} "
        f"{'FT':>6s} "
        f"{'MissFT':>7s} "
        + " ".join([f"C@{k:>4}" for k in TOP_K_VALUES])
    )

    print(header)
    print("-" * len(header))

    for s in summaries:
        project = str(s.get("project", ""))[:45]
        ranked = int(s.get("ranked", 0) or 0)
        refs = int(s.get("refs_doi", 0) or 0)
        ft = int(s.get("with_full_text", 0) or 0)
        miss = int(s.get("missing_full_text", 0) or 0)

        cvals = " ".join([f"{int(s.get(f'C@{k}', 0) or 0):>6d}" for k in TOP_K_VALUES])

        print(f"{project:45s} {ranked:8d} {refs:6d} {ft:6d} {miss:7d} {cvals}")

    print("=" * 120)


if __name__ == "__main__":
    main()
