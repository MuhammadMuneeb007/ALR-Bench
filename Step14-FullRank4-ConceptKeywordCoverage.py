#!/usr/bin/env python3
# Step14-FullRank4-ConceptKeywordCoverage.py
#
# Full-text-aware Rank Strategy 4:
#   Project-specific concept + keyword coverage ranking.
#
# What this script does:
#   1. Reads each project's:
#        Projects/<project>/config.json
#        Projects/<project>/Search/articles_clean_doi_only.csv
#      Fallback:
#        Projects/<project>/Search/articles.csv
#
#   2. Builds concept/keyword groups from config:
#        - title
#        - objective
#        - research_questions
#        - concepts_keywords
#        - generated_search_keywords
#        - generated_search_queries
#        - generated_abstract_include_keywords
#        - generated_full_text_include_keywords
#        - prisma_s_search_keywords
#        - prisma_s_full_search_strings
#        - prisma_title_abstract_include_keywords
#        - prisma_title_abstract_must_have_keywords
#        - prisma_full_text_include_keywords
#        - prisma_full_text_must_have_keywords
#
#   3. Builds soft exclusion penalty from:
#        - generated_abstract_exclude_keywords
#        - generated_full_text_exclude_keywords
#        - prisma_title_abstract_exclude_keywords
#        - prisma_full_text_exclude_keywords
#        - prisma_eligibility_exclusion
#
#   4. Scores each article using:
#        - exact phrase coverage
#        - token coverage
#        - field-weighted coverage
#        - concept group coverage
#        - must-have/inclusion coverage
#        - exclusion penalty
#
#   5. Uses full text if available.
#      If full text is missing, article is still ranked using title/abstract.
#
#   6. Saves:
#        Projects/<project>/Rankings/full_rank_strategy_4.csv
#        Projects/Rankings/full_rank_strategy_4_summary.csv
#
#   7. Prints:
#        C@10, C@20, C@50, C@100, C@200, C@500, C@1000
#
# Notes:
#   - No API calls.
#   - No LLM calls.
#   - No PDF download.
#   - References are used only for evaluation.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
DEFAULT_RANK_FILE = "full_rank_strategy_4.csv"
DEFAULT_SUMMARY_FILE = "full_rank_strategy_4_summary.csv"

TOP_K_VALUES = [10, 20, 50, 100, 200, 500, 1000]

MIN_TOKEN_LEN = 2
MAX_FULLTEXT_CHARS = 250_000

TITLE_WEIGHT = 4.0
ABSTRACT_WEIGHT = 2.5
FULLTEXT_WEIGHT = 1.0
METADATA_WEIGHT = 0.5

HAS_FULLTEXT_BOOST = 0.05
MISSING_FULLTEXT_PENALTY = 0.05

EXCLUSION_TITLE_PENALTY = 4.0
EXCLUSION_ABSTRACT_PENALTY = 2.5
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

def make_term_group(name: str, values: List[str], weight: float) -> Dict[str, Any]:
    phrases = unique_keep_order(values)
    norm_phrases = []
    tokens = []

    for p in phrases:
        np = normalize_phrase(p)
        if np:
            norm_phrases.append(np)
            tokens.extend(tokenize(np))

    token_set = sorted(set(tokens))

    return {
        "name": name,
        "weight": weight,
        "phrases": norm_phrases,
        "tokens": token_set,
        "n_phrases": len(norm_phrases),
        "n_tokens": len(token_set),
    }


def build_concept_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    groups = []

    groups.append(make_term_group(
        "title_objective_rq",
        flatten_values(config.get("title", []))
        + flatten_values(config.get("objective", []))
        + flatten_values(config.get("research_questions", [])),
        1.40,
    ))

    groups.append(make_term_group(
        "concepts_keywords",
        flatten_values(config.get("concepts_keywords", [])),
        1.50,
    ))

    groups.append(make_term_group(
        "generated_search_keywords",
        flatten_values(config.get("generated_search_keywords", [])),
        1.25,
    ))

    groups.append(make_term_group(
        "prisma_s_search_keywords",
        flatten_values(config.get("prisma_s_search_keywords", [])),
        1.20,
    ))

    groups.append(make_term_group(
        "generated_search_queries",
        flatten_values(config.get("generated_search_queries", []))
        + flatten_values(config.get("prisma_s_full_search_strings", []))
        + flatten_values(config.get("prisma_s_search_queries_by_database", [])),
        1.00,
    ))

    groups.append(make_term_group(
        "abstract_include",
        flatten_values(config.get("generated_abstract_include_keywords", []))
        + flatten_values(config.get("prisma_title_abstract_include_keywords", []))
        + flatten_values(config.get("prisma_title_abstract_must_have_keywords", []))
        + flatten_values(config.get("prisma_title_abstract_uncertain_keywords", [])),
        1.15,
    ))

    groups.append(make_term_group(
        "fulltext_include",
        flatten_values(config.get("generated_full_text_include_keywords", []))
        + flatten_values(config.get("prisma_full_text_include_keywords", []))
        + flatten_values(config.get("prisma_full_text_must_have_keywords", []))
        + flatten_values(config.get("prisma_full_text_uncertain_keywords", [])),
        1.35,
    ))

    groups.append(make_term_group(
        "inclusion_eligibility",
        flatten_values(config.get("prisma_eligibility_inclusion", [])),
        0.90,
    ))

    groups.append(make_term_group(
        "quality_context",
        flatten_values(config.get("generated_quality_assessment_keywords", []))
        + flatten_values(config.get("generated_quality_assessment_questions", []))
        + flatten_values(config.get("prisma_quality_assessment_keywords", []))
        + flatten_values(config.get("prisma_quality_assessment_questions", [])),
        0.50,
    ))

    negative_terms = (
        flatten_values(config.get("generated_abstract_exclude_keywords", []))
        + flatten_values(config.get("generated_full_text_exclude_keywords", []))
        + flatten_values(config.get("prisma_title_abstract_exclude_keywords", []))
        + flatten_values(config.get("prisma_full_text_exclude_keywords", []))
        + flatten_values(config.get("prisma_eligibility_exclusion", []))
    )

    negative_phrases = [normalize_phrase(x) for x in unique_keep_order(negative_terms)]
    negative_phrases = [x for x in negative_phrases if x]

    all_tokens = set()
    all_phrases = set()

    for g in groups:
        all_tokens.update(g["tokens"])
        all_phrases.update(g["phrases"])

    groups = [g for g in groups if g["n_phrases"] > 0 or g["n_tokens"] > 0]

    return {
        "groups": groups,
        "all_tokens": sorted(all_tokens),
        "all_phrases": sorted(all_phrases),
        "negative_phrases": negative_phrases,
        "n_groups": len(groups),
        "n_tokens": len(all_tokens),
        "n_phrases": len(all_phrases),
        "n_negative_phrases": len(negative_phrases),
    }


# ============================================================
# ARTICLE / REFERENCE LOADING
# ============================================================


def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
    """
    Extract DOI/title reference records from nested JSON reference files.

    Supports the same reference files used by Step9:
        references.json
        references_resolved.json
        references_gold.json
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


def sort_ranked_input_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Sort the input ranking file so Step14 reranks the same top-N pool
    selected by Step9/Step11. Default expected input:
        Projects/<project>/Rankings/rank_strategy_3.csv
    """
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
    """
    Load the SAME candidate pool used for full-text download/reranking.

    By default this reads:
        Projects/<project>/Rankings/rank_strategy_3.csv

    This is important because Step14 is a second-stage reranker, not a fresh
    search over all articles. If this file is missing, it falls back to the
    old Search/articles_clean_doi_only.csv or Search/articles.csv behavior.
    """
    rank_path = project_dir / input_rank_rel
    rows = read_csv_rows(rank_path)
    source = str(rank_path)

    if rows:
        rows = sort_ranked_input_rows(rows)
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
        if top_n > 0:
            rows = rows[:top_n]

    rows_loaded = len(rows)

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
        row["_input_rank"] = get_first_existing(row, ["rank_strategy_3", "rank", "Rank", "input_rank"])
        row["_input_rank_score"] = get_first_existing(row, ["rank_strategy_3_score", "score"])
        deduped.append(row)

    return deduped, source, rows_loaded


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


def load_reference_dois(project_dir: Path) -> Set[str]:
    """
    Load reference DOIs from JSON first, matching the working Step9 logic,
    then fall back to older CSV reference files.

    JSON priority:
        Projects/<project>/references.json
        Projects/<project>/references_resolved.json
        Projects/<project>/references_gold.json
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

    # CSV fallback for older project structures.
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


def find_matching_database_project_folder(database_dir: Path, project_name: str) -> Optional[Path]:
    """
    Match Projects/<project_name> to DataBase_Strategy3_Top1000/<project_key_hash>.

    Handles truncated database folder names, e.g.
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
    project_words = set(tokenize(project_name))

    # Exact/prefix checks first.
    for p in candidates:
        if p.name == project_name or p.name.startswith(f"{project_name}_"):
            return p

    best: Optional[Path] = None
    best_score = 0.0

    for p in candidates:
        db_name = p.name.lower()
        db_base = re.sub(r"_[a-f0-9]{8}$", "", db_name)
        db_norm = normalize_title(db_base).replace(" ", "_")
        db_words = set(tokenize(db_base))

        score = 0.0

        if project_norm.startswith(db_norm) or db_norm.startswith(project_norm):
            score = max(score, 0.95)

        if project_words and db_words:
            # Symmetric enough for truncated names.
            overlap_project = len(project_words & db_words) / max(1, len(project_words))
            overlap_db = len(project_words & db_words) / max(1, len(db_words))
            score = max(score, overlap_project, overlap_db)

        if score > best_score:
            best_score = score
            best = p

    if best is not None and best_score >= 0.50:
        return best

    return None


def build_fulltext_index(database_dir: Path, project_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Build full-text index for ONE project only.

    Fixed behavior:
      - correctly matches truncated/hash database project folders
      - scans only Temp_* folders under the matched project folder
      - avoids accidentally scanning all 14 projects and mixing records
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

    for article_dir in temp_dirs:
        chosen: Optional[Path] = None
        for name in text_file_names:
            candidate = article_dir / name
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                chosen = candidate
                break
        if chosen:
            unique_files.append(chosen)

    print(f"     fulltext files selected: {len(unique_files)}")

    for text_path in unique_files:
        article_dir = text_path.parent

        text = safe_read_text(text_path)
        if not text:
            continue

        metadata: Dict[str, Any] = {}
        for meta_name in metadata_names:
            mp = article_dir / meta_name
            if mp.exists():
                loaded = read_json(mp, {})
                if isinstance(loaded, dict):
                    metadata.update(loaded)

        # Step5 extraction_metadata.json can store original article metadata inside base_metadata.
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

        keys = []
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
# ARTICLE FIELDS
# ============================================================

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
        "journal": journal,
        "year": year,
        "source_db": source_db,
    }


# ============================================================
# SCORING
# ============================================================

def count_phrase_hits(text_norm: str, phrases: List[str]) -> Tuple[int, float]:
    hits = 0
    score = 0.0

    if not text_norm or not phrases:
        return hits, score

    for phrase in phrases:
        phrase = normalize_phrase(phrase)
        if not phrase:
            continue

        n_tokens = len(phrase.split())

        # Very long search strings are handled by token coverage.
        if n_tokens > 12:
            continue

        if phrase in text_norm:
            hits += 1
            if n_tokens >= 4:
                score += 3.0
            elif n_tokens == 3:
                score += 2.0
            elif n_tokens == 2:
                score += 1.25
            else:
                score += 0.5

    return hits, score


def token_coverage_score(text_tokens: Set[str], target_tokens: List[str]) -> Tuple[float, int, int]:
    if not target_tokens:
        return 0.0, 0, 0

    target_set = set(target_tokens)
    matched = len(text_tokens & target_set)
    total = len(target_set)

    coverage = matched / max(1, total)
    return coverage, matched, total


def group_score_for_field(
    text_norm: str,
    text_tokens: Set[str],
    group: Dict[str, Any],
) -> Dict[str, Any]:
    phrase_hits, phrase_score = count_phrase_hits(text_norm, group["phrases"])
    token_cov, token_hits, token_total = token_coverage_score(text_tokens, group["tokens"])

    # Group score balances exact phrase and token coverage.
    group_score = (
        phrase_score * 1.25
        + token_cov * 5.0
        + math.log1p(token_hits) * 0.75
    ) * float(group.get("weight", 1.0))

    return {
        "score": group_score,
        "phrase_hits": phrase_hits,
        "phrase_score": phrase_score,
        "token_coverage": token_cov,
        "token_hits": token_hits,
        "token_total": token_total,
    }


def exclusion_penalty(
    title_norm: str,
    abstract_norm: str,
    fulltext_norm: str,
    negative_phrases: List[str],
) -> Tuple[float, int]:
    penalty = 0.0
    hits = 0

    if not negative_phrases:
        return penalty, hits

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


def score_one_article(fields: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    title_norm = normalize_phrase(fields["title"])
    abstract_norm = normalize_phrase(fields["abstract"])
    metadata_norm = normalize_phrase(fields["metadata"])
    fulltext_norm = normalize_phrase(fields["fulltext"])

    title_tokens = set(tokenize(fields["title"]))
    abstract_tokens = set(tokenize(fields["abstract"]))
    metadata_tokens = set(tokenize(fields["metadata"]))
    fulltext_tokens = set(tokenize(fields["fulltext"]))

    groups = profile["groups"]

    title_score = 0.0
    abstract_score = 0.0
    metadata_score = 0.0
    fulltext_score = 0.0

    total_phrase_hits = 0
    total_token_hits = 0
    group_covered_count = 0
    group_total_count = len(groups)

    group_detail_parts = []

    for group in groups:
        g_title = group_score_for_field(title_norm, title_tokens, group)
        g_abstract = group_score_for_field(abstract_norm, abstract_tokens, group)
        g_metadata = group_score_for_field(metadata_norm, metadata_tokens, group)
        g_fulltext = group_score_for_field(fulltext_norm, fulltext_tokens, group)

        title_score += g_title["score"] * TITLE_WEIGHT
        abstract_score += g_abstract["score"] * ABSTRACT_WEIGHT
        metadata_score += g_metadata["score"] * METADATA_WEIGHT
        fulltext_score += g_fulltext["score"] * FULLTEXT_WEIGHT

        group_phrase_hits = (
            g_title["phrase_hits"]
            + g_abstract["phrase_hits"]
            + g_metadata["phrase_hits"]
            + g_fulltext["phrase_hits"]
        )

        group_token_hits = max(
            g_title["token_hits"],
            g_abstract["token_hits"],
            g_metadata["token_hits"],
            g_fulltext["token_hits"],
        )

        total_phrase_hits += group_phrase_hits
        total_token_hits += group_token_hits

        group_is_covered = bool(group_phrase_hits > 0 or group_token_hits > 0)
        if group_is_covered:
            group_covered_count += 1

        group_detail_parts.append(
            f"{group['name']}:{1 if group_is_covered else 0}"
        )

    all_article_tokens = title_tokens | abstract_tokens | metadata_tokens | fulltext_tokens
    all_tokens = profile["all_tokens"]
    all_token_cov, all_token_hits, all_token_total = token_coverage_score(all_article_tokens, all_tokens)

    group_coverage = group_covered_count / max(1, group_total_count)

    coverage_score = (
        all_token_cov * 10.0
        + group_coverage * 8.0
        + math.log1p(total_phrase_hits) * 2.0
        + math.log1p(total_token_hits) * 1.0
    )

    excl_penalty, excl_hits = exclusion_penalty(
        title_norm=title_norm,
        abstract_norm=abstract_norm,
        fulltext_norm=fulltext_norm,
        negative_phrases=profile["negative_phrases"],
    )

    fulltext_adjustment = HAS_FULLTEXT_BOOST if fields["has_fulltext"] else -MISSING_FULLTEXT_PENALTY

    final_score = (
        title_score
        + abstract_score
        + metadata_score
        + fulltext_score
        + coverage_score
        + fulltext_adjustment
        - excl_penalty
    )

    return {
        "score": final_score,
        "title_score": title_score,
        "abstract_score": abstract_score,
        "metadata_score": metadata_score,
        "fulltext_score": fulltext_score,
        "coverage_score": coverage_score,
        "all_token_coverage": all_token_cov,
        "all_token_hits": all_token_hits,
        "all_token_total": all_token_total,
        "group_coverage": group_coverage,
        "groups_covered": group_covered_count,
        "groups_total": group_total_count,
        "phrase_hits_total": total_phrase_hits,
        "token_hits_total": total_token_hits,
        "exclusion_penalty": excl_penalty,
        "exclusion_hits": excl_hits,
        "group_details": ";".join(group_detail_parts),
    }


def score_articles(
    articles: List[Dict[str, str]],
    profile: Dict[str, Any],
    fulltext_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scored_rows = []

    for article in articles:
        fulltext_record = find_fulltext_for_article(article, fulltext_index)
        fields = build_article_fields(article, fulltext_record)

        scores = score_one_article(fields, profile)

        doi = article.get("_doi_norm", "")

        scored_rows.append({
            "rank": "",
            "score": round(scores["score"], 8),
            "title_score": round(scores["title_score"], 8),
            "abstract_score": round(scores["abstract_score"], 8),
            "metadata_score": round(scores["metadata_score"], 8),
            "fulltext_score": round(scores["fulltext_score"], 8),
            "coverage_score": round(scores["coverage_score"], 8),
            "all_token_coverage": round(scores["all_token_coverage"], 8),
            "all_token_hits": scores["all_token_hits"],
            "all_token_total": scores["all_token_total"],
            "group_coverage": round(scores["group_coverage"], 8),
            "groups_covered": scores["groups_covered"],
            "groups_total": scores["groups_total"],
            "phrase_hits_total": scores["phrase_hits_total"],
            "token_hits_total": scores["token_hits_total"],
            "exclusion_penalty": round(scores["exclusion_penalty"], 8),
            "exclusion_hits": scores["exclusion_hits"],
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
            "group_details": scores["group_details"],
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
        print(f"FAIL | {project_name} | no articles found")
        return {
            "project": project_name,
            "status": "no_articles",
            "ranked": 0,
            "articles_source": articles_source,
        }

    reference_dois = load_reference_dois(project_dir)
    profile = build_concept_profile(config)
    fulltext_index = build_fulltext_index(database_dir, project_name)

    ranked_rows = score_articles(
        articles=articles,
        profile=profile,
        fulltext_index=fulltext_index,
    )

    eval_metrics = evaluate_common_at_k(ranked_rows, reference_dois, TOP_K_VALUES)

    refs = set(reference_dois)
    for row in ranked_rows:
        doi = normalize_doi(row.get("doi", ""))
        row["is_reference"] = int(bool(doi and doi in refs))

    fields = [
        "rank",
        "score",
        "title_score",
        "abstract_score",
        "metadata_score",
        "fulltext_score",
        "coverage_score",
        "all_token_coverage",
        "all_token_hits",
        "all_token_total",
        "group_coverage",
        "groups_covered",
        "groups_total",
        "phrase_hits_total",
        "token_hits_total",
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
        "group_details",
    ]

    write_csv(out_path, ranked_rows, fields)

    n_fulltext = sum(1 for r in ranked_rows if int(r.get("has_full_text", 0)) == 1)
    n_missing_fulltext = len(ranked_rows) - n_fulltext

    summary = {
        "project": project_name,
        "status": "ok",
        "ranked": len(ranked_rows),
        "articles_input": len(articles),
        "input_rank_rows_loaded": input_rank_rows_loaded,
        "input_top_n": top_n,
        "articles_source": articles_source,
        "refs_doi": len(reference_dois),
        "fulltext_index_items": len(fulltext_index),
        "with_full_text": n_fulltext,
        "missing_full_text": n_missing_fulltext,
        "groups": profile["n_groups"],
        "tokens": profile["n_tokens"],
        "phrases": profile["n_phrases"],
        "negative_phrases": profile["n_negative_phrases"],
        "rank_file": str(out_path),
    }
    summary.update(eval_metrics)

    metric_text = " | ".join([f"C@{k}={summary.get(f'C@{k}', 0)}" for k in TOP_K_VALUES])
    coverage_text = " | ".join([f"Cov@{k}={summary.get(f'Coverage@{k}', 0):.3f}" for k in TOP_K_VALUES])

    print(
        f"OK | {project_name} | ranked={len(ranked_rows)} | "
        f"refs_doi={len(reference_dois)} | fulltext={n_fulltext} | "
        f"missing_fulltext={n_missing_fulltext} | groups={profile['n_groups']} | {metric_text}"
    )
    print(f"     {coverage_text}")
    print(f"     file={out_path}")

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step14 FullRank4: project-specific full-text-aware concept keyword coverage ranking."
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
    print("Step14 FullRank4: Concept + Keyword Coverage full-text-aware ranking")
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

    for i, project_dir in enumerate(projects, start=1):
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
        "groups",
        "tokens",
        "phrases",
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