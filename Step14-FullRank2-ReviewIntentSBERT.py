#!/usr/bin/env python3
# Step14-FullRank2-ReviewIntentSBERT.py
#
# Full-text-aware Rank Strategy 2:
#   Project-specific Review Intent SBERT semantic ranking.
#
# What this script does:
#   1. Reads each project's:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Builds one project-specific review-intent text from config:
#        - title
#        - objective
#        - research questions
#        - concepts_keywords
#        - generated_search_queries
#        - generated_search_keywords
#        - generated_abstract_include_keywords
#        - generated_full_text_include_keywords
#        - prisma_s_search_keywords
#        - prisma_s_full_search_strings
#        - prisma_full_text_include_keywords
#        - prisma_full_text_must_have_keywords
#        - quality/context keywords
#
#   3. For every article:
#        - uses title + abstract + metadata
#        - adds extracted full text if available
#        - if full text missing, still ranks article using title/abstract
#
#   4. Ranks by semantic similarity:
#        - default backend: sentence-transformers
#        - fallback backend: TF-IDF cosine similarity
#
#   5. Adds soft exclusion penalty from config:
#        - generated_abstract_exclude_keywords
#        - generated_full_text_exclude_keywords
#        - prisma_title_abstract_exclude_keywords
#        - prisma_full_text_exclude_keywords
#        - prisma_eligibility_exclusion
#
#   6. Saves:
#        Projects/<project>/Rankings/full_rank_strategy_2.csv
#        Projects/Rankings/full_rank_strategy_2_summary.csv
#
#   7. Prints:
#        C@10, C@20, C@50, C@100, C@200, C@500, C@1000
#
# Notes:
#   - This script does NOT download PDFs.
#   - This script does NOT call APIs.
#   - This script does NOT use LLMs.
#   - Full text is optional.
#   - References are used only for evaluation, not ranking.
#   - Reference DOIs are loaded from references.json / references_resolved.json / references_gold.json, with CSV fallback.
#
# Requirements:
#   Preferred:
#       pip install sentence-transformers torch tqdm numpy
#
#   Fallback:
#       pip install scikit-learn tqdm numpy
#
# Usage:
#   python Step14-FullRank2-ReviewIntentSBERT.py --projects-dir Projects --database-dir DataBase_Strategy3_Top1000 --force
#
# One project:
#   python Step14-FullRank2-ReviewIntentSBERT.py --projects-dir Projects --database-dir DataBase_Strategy3_Top1000 --project PROJECT_NAME --force
#
# If SBERT/torch is broken:
#   python Step14-FullRank2-ReviewIntentSBERT.py --backend tfidf --force

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PROJECTS_DIR = Path("Projects")
DEFAULT_DATABASE_DIR = Path("DataBase_Strategy3_Top1000")

DEFAULT_ARTICLES_CLEAN_REL = Path("Search") / "articles_clean_doi_only.csv"
DEFAULT_ARTICLES_RAW_REL = Path("Search") / "articles.csv"

# IMPORTANT:
# Step14 full-text reranking should normally rerank the same selected/downloaded
# article set used by Step11/Step13. By default, this is the top 1000 from
# abstract/metadata Strategy 3.
DEFAULT_INPUT_RANK_REL = Path("Rankings") / "rank_strategy_3.csv"
DEFAULT_TOP_N_FROM_INPUT_RANK = 1000

DEFAULT_RANKINGS_DIR = Path("Rankings")
DEFAULT_RANK_FILE = "full_rank_strategy_2.csv"
DEFAULT_SUMMARY_FILE = "full_rank_strategy_2_summary.csv"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_VALUES = [10, 20, 50, 100, 200, 500, 1000]

MAX_FULLTEXT_CHARS = 120_000
MAX_EMBED_CHARS = 18_000

MISSING_FULLTEXT_PENALTY = 0.02
HAS_FULLTEXT_BOOST = 0.02

EXCLUSION_TITLE_PENALTY = 0.12
EXCLUSION_ABSTRACT_PENALTY = 0.08
EXCLUSION_FULLTEXT_PENALTY = 0.03


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
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def flatten_values(value: Any) -> List[str]:
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


def get_first_existing(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in row and clean_text(row.get(k, "")):
            return clean_text(row.get(k, ""))
    return ""


def simple_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", clean_text(text)))


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

def build_review_intent_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    positive_fields = [
        "title",
        "review_type",
        "objective",
        "research_questions",
        "concepts_keywords",

        "prisma_eligibility_inclusion",
        "prisma_title_abstract_include_keywords",
        "prisma_title_abstract_must_have_keywords",
        "prisma_title_abstract_uncertain_keywords",
        "prisma_full_text_include_keywords",
        "prisma_full_text_must_have_keywords",
        "prisma_full_text_uncertain_keywords",

        "prisma_s_full_search_strings",
        "prisma_s_search_queries_by_database",
        "prisma_s_search_keywords",
        "prisma_s_controlled_vocabulary_terms",
        "prisma_s_free_text_terms",

        "generated_search_queries",
        "generated_search_keywords",
        "generated_abstract_include_keywords",
        "generated_full_text_include_keywords",

        "generated_quality_assessment_keywords",
        "generated_quality_assessment_questions",
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

    parts: List[str] = []
    negative_phrases: List[str] = []

    for field in positive_fields:
        vals = flatten_values(config.get(field, []))
        if not vals:
            continue

        if field == "title":
            parts.append("Review title: " + " ".join(vals))
        elif field == "objective":
            parts.append("Review objective: " + " ".join(vals))
        elif field == "research_questions":
            parts.append("Research questions: " + " ".join(vals))
        elif field == "concepts_keywords":
            parts.append("Key concepts: " + "; ".join(vals))
        elif "full_text" in field:
            parts.append("Full text inclusion concepts: " + "; ".join(vals))
        elif "search" in field:
            parts.append("Search strategy terms: " + "; ".join(vals))
        elif "quality" in field:
            parts.append("Quality and methodology context: " + "; ".join(vals))
        else:
            parts.append("; ".join(vals))

    for field in negative_fields:
        vals = flatten_values(config.get(field, []))
        negative_phrases.extend(vals)

    review_intent = clean_text(" ".join(parts))
    negative_phrases = unique_keep_order(negative_phrases)
    negative_norm_phrases = [normalize_phrase(x) for x in negative_phrases if normalize_phrase(x)]

    return {
        "review_intent": review_intent,
        "negative_phrases": negative_norm_phrases,
        "n_intent_words": simple_word_count(review_intent),
        "n_negative_phrases": len(negative_norm_phrases),
    }


# ============================================================
# ARTICLE / REFERENCE LOADING
# ============================================================

def sort_ranked_input_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Sort rows from an existing Step9 ranking file.

    Supports common rank/score columns from your previous ranking scripts.
    """
    def rank_key(row: Dict[str, str]) -> Tuple[int, float]:
        rank_value = get_first_existing(row, [
            "rank_strategy_3",
            "rank_strategy_2",
            "rank_strategy_1",
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
            "rank_strategy_2_score",
            "rank_strategy_1_score",
            "score_title_objective_sentence_transformer_raw",
            "score",
            "semantic_score",
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
    """
    Load the selected articles to rerank.

    Default behavior:
        Projects/<project>/Rankings/rank_strategy_3.csv
        top 1000 rows

    This makes FullRank2 comparable to FullRank1 and to your download set.
    If the input ranking file is missing, fallback to Search/articles_clean_doi_only.csv.
    """
    rank_path = project_dir / input_rank_rel
    rows = read_csv_rows(rank_path)
    source = str(rank_path)

    if rows:
        rows = sort_ranked_input_rows(rows)
        input_rows_loaded = len(rows)
        if top_n > 0:
            rows = rows[:top_n]
    else:
        clean_path = project_dir / DEFAULT_ARTICLES_CLEAN_REL
        raw_path = project_dir / DEFAULT_ARTICLES_RAW_REL

        rows = read_csv_rows(clean_path)
        source = str(clean_path)

        if not rows:
            rows = read_csv_rows(raw_path)
            source = str(raw_path)

        input_rows_loaded = len(rows)
        if top_n > 0:
            rows = rows[:top_n]

    deduped: List[Dict[str, str]] = []
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
        row["_input_rank"] = get_first_existing(row, [
            "rank_strategy_3",
            "rank_strategy_2",
            "rank_strategy_1",
            "rank",
            "Rank",
        ])
        row["_input_rank_score"] = get_first_existing(row, [
            "rank_strategy_3_score",
            "rank_strategy_2_score",
            "rank_strategy_1_score",
            "score_title_objective_sentence_transformer_raw",
            "score",
            "semantic_score",
        ])

        deduped.append(row)

    return deduped, source, input_rows_loaded


def possible_reference_files(project_dir: Path) -> List[Path]:
    return [
        project_dir / "References" / "references_clean.csv",
        project_dir / "References" / "references_clean_doi_only.csv",
        project_dir / "References" / "references.csv",
        project_dir / "references_clean.csv",
        project_dir / "references.csv",
        project_dir / "Search" / "references_clean.csv",
        project_dir / "Search" / "references.csv",
    ]


def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
    """
    Extract reference DOI/title records from nested JSON reference files.

    This matches the working Step9 reference-loader behavior and supports:
        Projects/<project>/references.json
        Projects/<project>/references_resolved.json
        Projects/<project>/references_gold.json

    The JSON can be a list/dict and can contain nested reference objects.
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

    Primary source is the same JSON reference files used by the working Step9 script.
    CSV paths are kept only as fallback for older project structures.
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

        dois = {
            normalize_doi(x.get("doi", ""))
            for x in records
            if normalize_doi(x.get("doi", ""))
        }

        if dois:
            return dois

    # Fallback: CSV reference files from older layouts.
    ref_dois: Set[str] = set()

    for path in possible_reference_files(project_dir):
        if not path.exists():
            continue

        rows = read_csv_rows(path)
        for row in rows:
            doi = normalize_doi(get_first_existing(row, ["doi", "DOI", "article_doi", "reference_doi"]))
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


def _tokens_for_folder_match(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_title(text)))


def find_matching_database_project_folder(database_dir: Path, project_name: str) -> Optional[Path]:
    """
    Robustly match:
        Projects/<project_name>
    to:
        DataBase_Strategy3_Top1000/<truncated_project_name>_<8char_hash>

    Example:
        project:
            health_outcomes_associated_with_reallocations_of_time_between_sleep_sedentary_be

        database:
            health_outcomes_associated_with_reallocations_of_time_betwee_0cd0b897
    """
    if not database_dir.exists():
        return None

    direct = database_dir / project_name
    if direct.exists() and direct.is_dir():
        return direct

    candidates = [p for p in database_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    project_norm = normalize_title(project_name).replace(" ", "_")
    project_words = _tokens_for_folder_match(project_name)

    # Exact-ish prefix match first.
    for p in candidates:
        if p.name == project_name or p.name.startswith(f"{project_name}_"):
            return p

    best: Optional[Path] = None
    best_score = 0.0

    for p in candidates:
        db_name = p.name.lower()
        db_base = re.sub(r"_[a-f0-9]{8}$", "", db_name)
        db_norm = normalize_title(db_base).replace(" ", "_")
        db_words = _tokens_for_folder_match(db_base)

        score = 0.0

        # Truncated DB folder base should be a prefix of project name.
        if db_norm and project_norm.startswith(db_norm):
            score = max(score, 0.98)

        if project_norm and db_norm.startswith(project_norm):
            score = max(score, 0.98)

        if project_words and db_words:
            overlap_project = len(project_words & db_words) / max(1, len(project_words))
            overlap_db = len(project_words & db_words) / max(1, len(db_words))
            score = max(score, min(overlap_project, overlap_db))

        if score > best_score:
            best_score = score
            best = p

    if best is not None and best_score >= 0.50:
        return best

    return None


def build_fulltext_index(database_dir: Path, project_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a full-text index for exactly one project.

    FIXED:
      - matches truncated/hash database folders
      - scans only that project's Temp_* folders
      - avoids scanning the entire database root and accidentally mixing projects
    """
    index: Dict[str, Dict[str, Any]] = {}

    if not database_dir.exists():
        print(f"     database missing: {database_dir}")
        return index

    text_file_names = [
        "Article_Text.txt",
        "COMBINED.txt",
        "full_text.txt",
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

    temp_dirs = [
        p for p in matched_project_dir.iterdir()
        if p.is_dir() and re.match(r"^Temp_\d+$", p.name)
    ]

    temp_dirs = sorted(
        temp_dirs,
        key=lambda p: int(p.name.replace("Temp_", "")) if p.name.replace("Temp_", "").isdigit() else 10**9,
    )

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

        metadata: Dict[str, Any] = {}

        for meta_name in metadata_names:
            mp = article_dir / meta_name
            if mp.exists():
                try:
                    metadata.update(read_json(mp, {}))
                except Exception:
                    pass

        # Your extraction_metadata.json may store the original downloader metadata
        # inside base_metadata. Merge that in too.
        base_metadata = metadata.get("base_metadata", {})
        if isinstance(base_metadata, dict):
            merged = dict(base_metadata)
            merged.update(metadata)
            metadata = merged

        doi = normalize_doi(
            metadata.get("doi")
            or metadata.get("DOI")
            or metadata.get("article_doi")
            or metadata.get("reference_doi")
            or ""
        )

        title = clean_text(
            metadata.get("title")
            or metadata.get("article_title")
            or metadata.get("display_name")
            or ""
        )

        if not doi:
            folder_text = str(article_dir)
            m = re.search(r"(10\.\d{4,9}[^\s/\\]+[/_][^\s/\\]+)", folder_text, flags=re.I)
            if m:
                doi_guess = m.group(1).replace("_", "/")
                doi = normalize_doi(doi_guess)

        title_norm = normalize_title(title)
        folder_norm = normalize_title(article_dir.name)

        keys: List[Tuple[str, str]] = []

        if doi:
            keys.append(("doi", doi))
        if title_norm:
            keys.append(("title", title_norm))
        if folder_norm:
            keys.append(("folder", folder_norm))

        keys.append(("path", str(article_dir.resolve()).lower()))

        record = {
            "text": text,
            "path": str(text_path),
            "article_dir": str(article_dir),
            "doi": doi,
            "title": title,
            "word_count": simple_word_count(text),
            "char_count": len(text),
        }

        for ktype, kval in keys:
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

    if title_norm and len(title_norm) > 25:
        title_tokens = set(re.findall(r"[a-z0-9]+", title_norm))
        if len(title_tokens) >= 4:
            best_rec = {}
            best_overlap = 0.0

            for key, rec in fulltext_index.items():
                if not key.startswith("folder:"):
                    continue

                folder = key.replace("folder:", "", 1)
                folder_tokens = set(re.findall(r"[a-z0-9]+", folder))
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
# ARTICLE TEXT BUILDING
# ============================================================

def build_article_text(
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

    # Keep title and abstract repeated because they are more important than noisy full text.
    article_text = clean_text(
        " ".join([
            "Title:", title,
            "Title:", title,
            "Abstract:", abstract,
            "Abstract:", abstract,
            "Metadata:", metadata,
            "Full text:", fulltext,
        ])
    )

    if len(article_text) > MAX_EMBED_CHARS:
        article_text = article_text[:MAX_EMBED_CHARS]

    return {
        "text": article_text,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "source_db": source_db,
        "has_fulltext": has_fulltext,
        "fulltext_path": fulltext_record.get("path", "") if fulltext_record else "",
        "fulltext_word_count": fulltext_record.get("word_count", 0) if fulltext_record else 0,
        "fulltext_char_count": fulltext_record.get("char_count", 0) if fulltext_record else 0,
        "title_norm": normalize_phrase(title),
        "abstract_norm": normalize_phrase(abstract),
        "fulltext_norm": normalize_phrase(fulltext[:MAX_EMBED_CHARS]),
    }


# ============================================================
# EXCLUSION PENALTY
# ============================================================

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


# ============================================================
# SEMANTIC BACKENDS
# ============================================================

def semantic_scores_sbert(
    query_text: str,
    documents: List[str],
    model_name: str,
    batch_size: int = 32,
) -> List[float]:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)

    query_emb = model.encode(
        [query_text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    doc_embs = model.encode(
        documents,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    scores = np.dot(doc_embs, query_emb)
    return [float(x) for x in scores]


def semantic_scores_tfidf(
    query_text: str,
    documents: List[str],
) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [query_text] + documents

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=200_000,
    )

    matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    return [float(x) for x in sims]


# ============================================================
# SCORING
# ============================================================

def score_articles(
    articles: List[Dict[str, str]],
    profile: Dict[str, Any],
    fulltext_index: Dict[str, Dict[str, Any]],
    backend: str,
    model_name: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    review_intent = profile["review_intent"]
    negative_phrases = profile["negative_phrases"]

    prepared = []

    for article in articles:
        fulltext_record = find_fulltext_for_article(article, fulltext_index)
        fields = build_article_text(article, fulltext_record)

        prepared.append({
            "article": article,
            "fields": fields,
            "document_text": fields["text"],
        })

    documents = [x["document_text"] for x in prepared]

    if not documents:
        return []

    if backend == "tfidf":
        raw_scores = semantic_scores_tfidf(review_intent, documents)
        used_backend = "tfidf"
    else:
        try:
            raw_scores = semantic_scores_sbert(
                review_intent,
                documents,
                model_name=model_name,
                batch_size=batch_size,
            )
            used_backend = "sbert"
        except Exception as e:
            print(f"WARNING | SBERT failed: {type(e).__name__}: {e}")
            print("WARNING | Falling back to TF-IDF semantic similarity.")
            raw_scores = semantic_scores_tfidf(review_intent, documents)
            used_backend = "tfidf"

    scored_rows = []

    for item, raw_score in zip(prepared, raw_scores):
        article = item["article"]
        fields = item["fields"]

        excl_penalty, excl_hits = exclusion_penalty(
            title_norm=fields["title_norm"],
            abstract_norm=fields["abstract_norm"],
            fulltext_norm=fields["fulltext_norm"],
            negative_phrases=negative_phrases,
        )

        fulltext_adjustment = HAS_FULLTEXT_BOOST if fields["has_fulltext"] else -MISSING_FULLTEXT_PENALTY

        final_score = raw_score + fulltext_adjustment - excl_penalty

        doi = article.get("_doi_norm", "")

        scored_rows.append({
            "rank": "",
            "score": round(final_score, 8),
            "semantic_score": round(raw_score, 8),
            "backend": used_backend,
            "exclusion_penalty": round(excl_penalty, 8),
            "exclusion_hits": excl_hits,
            "has_full_text": int(fields["has_fulltext"]),
            "fulltext_word_count": fields["fulltext_word_count"],
            "fulltext_char_count": fields["fulltext_char_count"],
            "doi": doi,
            "title": fields["title"],
            "abstract": fields["abstract"],
            "journal": fields["journal"],
            "year": fields["year"],
            "source_db": fields["source_db"],
            "input_rank": article.get("_input_rank", ""),
            "input_rank_score": article.get("_input_rank_score", ""),
            "input_rank_source": article.get("_input_rank_source", ""),
            "fulltext_path": fields["fulltext_path"],
        })

    scored_rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

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
    backend: str,
    model_name: str,
    batch_size: int,
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
    profile = build_review_intent_profile(config)
    fulltext_index = build_fulltext_index(database_dir, project_name)

    if not profile["review_intent"]:
        print(f"FAIL | {project_name} | empty review intent")
        return {
            "project": project_name,
            "status": "empty_review_intent",
            "ranked": 0,
        }

    ranked_rows = score_articles(
        articles=articles,
        profile=profile,
        fulltext_index=fulltext_index,
        backend=backend,
        model_name=model_name,
        batch_size=batch_size,
    )

    eval_metrics = evaluate_common_at_k(ranked_rows, reference_dois, TOP_K_VALUES)

    refs = set(reference_dois)
    for row in ranked_rows:
        doi = normalize_doi(row.get("doi", ""))
        row["is_reference"] = int(bool(doi and doi in refs))

    fields = [
        "rank",
        "score",
        "semantic_score",
        "backend",
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
        "fulltext_path",
    ]

    write_csv(out_path, ranked_rows, fields)

    n_fulltext = sum(1 for r in ranked_rows if int(r.get("has_full_text", 0)) == 1)
    n_missing_fulltext = len(ranked_rows) - n_fulltext

    used_backend = ranked_rows[0].get("backend", backend) if ranked_rows else backend

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
        "backend": used_backend,
        "model": model_name if used_backend == "sbert" else "tfidf",
        "intent_words": profile["n_intent_words"],
        "negative_phrases": profile["n_negative_phrases"],
        "rank_file": str(out_path),
    }
    summary.update(eval_metrics)

    metric_text = " | ".join([f"C@{k}={summary.get(f'C@{k}', 0)}" for k in TOP_K_VALUES])
    coverage_text = " | ".join([f"Cov@{k}={summary.get(f'Coverage@{k}', 0):.3f}" for k in TOP_K_VALUES])

    print(
        f"OK | {project_name} | ranked={len(ranked_rows)} | "
        f"refs_doi={len(reference_dois)} | fulltext={n_fulltext} | "
        f"missing_fulltext={n_missing_fulltext} | backend={used_backend} | {metric_text}"
    )
    print(f"     {coverage_text}")
    print(f"     file={out_path}")

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step14 FullRank2: project-specific full-text-aware Review Intent SBERT ranking."
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
        "--backend",
        type=str,
        default="sbert",
        choices=["sbert", "tfidf"],
        help="Semantic backend. Use tfidf if torch/sentence-transformers is broken.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
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

    # Reduce tokenizer parallelism warnings on clusters.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    projects = discover_projects(projects_dir, args.project)

    print("=" * 120)
    print("Step14 FullRank2: Review Intent SBERT full-text-aware ranking")
    print(f"Projects directory : {projects_dir}")
    print(f"Database directory : {database_dir}")
    print(f"Projects found     : {len(projects)}")
    print(f"Input rank file    : Projects/<project>/{input_rank_rel}")
    print(f"Input top N        : {top_n}")
    print(f"Backend            : {args.backend}")
    print(f"Model              : {args.model}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Force overwrite    : {args.force}")
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

    for i, project_dir in enumerate(projects, start=1):
        print()
        print(f"[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(
                project_dir=project_dir,
                database_dir=database_dir,
                input_rank_rel=input_rank_rel,
                top_n=top_n,
                backend=args.backend,
                model_name=args.model,
                batch_size=args.batch_size,
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

    summary_dir = projects_dir / DEFAULT_RANKINGS_DIR
    summary_path = summary_dir / DEFAULT_SUMMARY_FILE

    summary_fields = [
        "project",
        "status",
        "ranked",
        "articles_input",
        "articles_source",
        "input_rank_rows_loaded",
        "input_top_n",
        "refs_doi",
        "fulltext_index_items",
        "with_full_text",
        "missing_full_text",
        "backend",
        "model",
        "intent_words",
        "negative_phrases",
    ]

    for k in TOP_K_VALUES:
        summary_fields.append(f"C@{k}")
        summary_fields.append(f"Coverage@{k}")

    summary_fields.extend(["rank_file", "error"])

    write_csv(summary_path, summaries, summary_fields)

    print()
    print("=" * 120)
    print(f"Saved summary: {summary_path}")
    print("=" * 120)

    header = (
        f"{'Project':45s} "
        f"{'Ranked':>8s} "
        f"{'Refs':>6s} "
        f"{'FT':>6s} "
        f"{'MissFT':>7s} "
        f"{'Backend':>8s} "
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
        backend = str(s.get("backend", ""))[:8]

        cvals = " ".join([f"{int(s.get(f'C@{k}', 0) or 0):>6d}" for k in TOP_K_VALUES])

        print(f"{project:45s} {ranked:8d} {refs:6d} {ft:6d} {miss:7d} {backend:>8s} {cvals}")

    print("=" * 120)


if __name__ == "__main__":
    main()