#!/usr/bin/env python3
# Step14-FullRank10-HybridFusion.py
#
# Full-text-aware Rank Strategy 10:
#   Hybrid fusion / ensemble ranking for Step14 FullRank1-9.
#
# This script:
#   1. Reads per-project full-text-aware ranking files:
#        Projects/<project>/Rankings/full_rank_strategy_1.csv
#        ...
#        Projects/<project>/Rankings/full_rank_strategy_9.csv
#
#   2. Combines them using:
#        - weighted Reciprocal Rank Fusion, RRF
#        - optional normalized score fusion
#
#   3. Does NOT use references to rank.
#      References are used only for evaluation.
#
#   4. Saves:
#        Projects/<project>/Rankings/full_rank_strategy_10.csv
#        Projects/Rankings/full_rank_strategy_10_summary.csv
#
#   5. Prints:
#        C@10, C@20, C@50, C@100, C@200, C@500, C@1000
#
# No API calls. No LLM calls. No PDF download.

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
DEFAULT_RANKINGS_DIR = Path("Rankings")

DEFAULT_RANK_FILE = "full_rank_strategy_10.csv"
DEFAULT_SUMMARY_FILE = "full_rank_strategy_10_summary.csv"

TOP_K_VALUES = [10, 20, 50, 100, 200, 500, 1000]

# Main default ensemble.
# Rank6 can be noisy, Rank3 can be narrow, but both still add evidence.
DEFAULT_STRATEGIES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# RRF constant. 60 is standard and stable.
DEFAULT_RRF_K = 60

# Minimum available strategy files required per project.
DEFAULT_MIN_STRATEGIES = 2

# FullRank weights.
# Updated after fixing Step14 FullRank1-9:
#   - FullRank2 is boosted because the ReviewIntentSBERT full-text reranker
#     improved early/mid recovery compared with FullRank1.
#   - Lexical/query methods are kept because they add complementary evidence.
#   - Very noisy methods are not removed by default; RRF makes the ensemble robust.
DEFAULT_STRATEGY_WEIGHTS = {
    1: 1.00,  # FullRank1 ConfigBM25
    2: 1.45,  # FullRank2 ReviewIntentSBERT
    3: 1.10,  # FullRank3 TitleObjectiveSentenceTransformer
    4: 1.20,  # FullRank4 ConceptKeywordCoverage
    5: 1.10,  # FullRank5 GeneratedQueryMatch
    6: 0.95,  # FullRank6 BooleanSoftMatch
    7: 1.20,  # FullRank7 FieldWeightedLexical
    8: 1.05,  # FullRank8 SciBERTSemantic
    9: 1.05,  # FullRank9 SPECTER2Semantic
}

STRATEGY_NAMES = {
    1: "FullConfigBM25",
    2: "FullReviewIntentSBERT",
    3: "FullTitleObjectiveSentenceTransformer",
    4: "FullConceptKeywordCoverage",
    5: "FullGeneratedQueryMatch",
    6: "FullBooleanSoftMatch",
    7: "FullFieldWeightedLexical",
    8: "FullSciBERTSemantic",
    9: "FullSPECTER2Semantic",
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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
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


def get_first_existing(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in row and clean_text(row.get(k, "")):
            return clean_text(row.get(k, ""))
    return ""


def article_key(row: Dict[str, Any]) -> str:
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi:{doi}"

    title = normalize_title(row.get("title", ""))
    if title:
        return f"title:{title}"

    return ""


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
# REFERENCES
# ============================================================


def extract_reference_records_from_json(data: Any) -> List[Dict[str, str]]:
    """
    Extract DOI/title reference records from nested JSON reference files.

    Supports the same reference files used by the working Step9 scripts:
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
# RANKING INPUTS
# ============================================================

def parse_strategy_list(x: str) -> List[int]:
    if not x:
        return DEFAULT_STRATEGIES[:]

    out = []
    for part in x.split(","):
        part = part.strip()
        if not part:
            continue

        try:
            n = int(part)
        except Exception:
            continue

        if 1 <= n <= 9:
            out.append(n)

    return out or DEFAULT_STRATEGIES[:]


def parse_weight_overrides(x: str) -> Dict[int, float]:
    """
    Format:
        1=1.5,7=1.4,9=1.2
    """
    weights = DEFAULT_STRATEGY_WEIGHTS.copy()

    if not x:
        return weights

    for part in x.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue

        k, v = part.split("=", 1)

        try:
            strategy = int(k.strip())
            weight = float(v.strip())
        except Exception:
            continue

        if 1 <= strategy <= 9:
            weights[strategy] = weight

    return weights


def load_strategy_ranking(project_dir: Path, strategy: int) -> List[Dict[str, Any]]:
    path = project_dir / DEFAULT_RANKINGS_DIR / f"full_rank_strategy_{strategy}.csv"
    rows = read_csv_rows(path)

    cleaned = []
    seen = set()

    for i, row in enumerate(rows, start=1):
        key = article_key(row)
        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        rank = safe_int(row.get("rank", i), i)
        score = safe_float(row.get("score", 0.0), 0.0)

        row["_strategy"] = strategy
        row["_strategy_name"] = STRATEGY_NAMES.get(strategy, f"Strategy{strategy}")
        row["_rank"] = rank
        row["_score"] = score
        row["_key"] = key
        row["_source_file"] = str(path)

        cleaned.append(row)

    cleaned.sort(key=lambda r: safe_int(r.get("_rank", 10**9), 10**9))

    return cleaned


def normalize_scores(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Min-max normalize scores per strategy.
    Returns key -> normalized score.
    """
    if not rows:
        return {}

    vals = [safe_float(r.get("_score", 0.0), 0.0) for r in rows]
    lo = min(vals)
    hi = max(vals)

    out = {}

    for r in rows:
        key = r["_key"]
        score = safe_float(r.get("_score", 0.0), 0.0)

        if hi > lo:
            norm = (score - lo) / (hi - lo)
        else:
            norm = 0.0

        out[key] = norm

    return out


# ============================================================
# FUSION
# ============================================================

def fuse_rankings(
    strategy_rows: Dict[int, List[Dict[str, Any]]],
    strategies: List[int],
    weights: Dict[int, float],
    rrf_k: int,
    use_score_fusion: bool,
    score_fusion_weight: float,
) -> List[Dict[str, Any]]:
    article_records: Dict[str, Dict[str, Any]] = {}

    # Precompute per-strategy normalized scores.
    normalized_by_strategy: Dict[int, Dict[str, float]] = {}
    for strategy, rows in strategy_rows.items():
        normalized_by_strategy[strategy] = normalize_scores(rows)

    for strategy in strategies:
        rows = strategy_rows.get(strategy, [])
        if not rows:
            continue

        weight = weights.get(strategy, 1.0)
        strategy_name = STRATEGY_NAMES.get(strategy, f"Strategy{strategy}")
        norm_scores = normalized_by_strategy.get(strategy, {})

        for row in rows:
            key = row.get("_key", "")
            if not key:
                continue

            rank = safe_int(row.get("_rank", 10**9), 10**9)
            raw_score = safe_float(row.get("_score", 0.0), 0.0)
            norm_score = safe_float(norm_scores.get(key, 0.0), 0.0)

            rrf_component = weight * (1.0 / (rrf_k + rank))

            score_component = 0.0
            if use_score_fusion:
                score_component = weight * norm_score * score_fusion_weight

            fused_add = rrf_component + score_component

            if key not in article_records:
                article_records[key] = {
                    "key": key,
                    "doi": normalize_doi(row.get("doi", "")),
                    "title": clean_text(row.get("title", "")),
                    "abstract": clean_text(row.get("abstract", "")),
                    "journal": clean_text(row.get("journal", "")),
                    "year": clean_text(row.get("year", "")),
                    "source_db": clean_text(row.get("source_db", "")),
                    "has_full_text": safe_int(row.get("has_full_text", 0), 0),
                    "fulltext_word_count": safe_int(row.get("fulltext_word_count", 0), 0),
                    "fulltext_char_count": safe_int(row.get("fulltext_char_count", 0), 0),
                    "fulltext_path": clean_text(row.get("fulltext_path", "")),
                    "fusion_score": 0.0,
                    "rrf_score": 0.0,
                    "score_fusion_score": 0.0,
                    "strategies_found": 0,
                    "strategy_list": [],
                    "rank_details": [],
                    "score_details": [],
                    "best_input_rank": rank,
                    "best_input_strategy": strategy_name,
                    "mean_input_rank": 0.0,
                    "input_rank_sum": 0.0,
                }

            rec = article_records[key]

            # Fill missing metadata from later strategy rows if earlier row had blanks.
            for field in ["doi", "title", "abstract", "journal", "year", "source_db", "fulltext_path"]:
                if not clean_text(rec.get(field, "")) and clean_text(row.get(field, "")):
                    if field == "doi":
                        rec[field] = normalize_doi(row.get(field, ""))
                    else:
                        rec[field] = clean_text(row.get(field, ""))

            rec["has_full_text"] = max(safe_int(rec.get("has_full_text", 0)), safe_int(row.get("has_full_text", 0)))
            rec["fulltext_word_count"] = max(
                safe_int(rec.get("fulltext_word_count", 0)),
                safe_int(row.get("fulltext_word_count", 0)),
            )
            rec["fulltext_char_count"] = max(
                safe_int(rec.get("fulltext_char_count", 0)),
                safe_int(row.get("fulltext_char_count", 0)),
            )

            rec["fusion_score"] += fused_add
            rec["rrf_score"] += rrf_component
            rec["score_fusion_score"] += score_component
            rec["strategies_found"] += 1
            rec["strategy_list"].append(str(strategy))
            rec["rank_details"].append(f"{strategy}:{rank}")
            rec["score_details"].append(f"{strategy}:{raw_score:.8f}:{norm_score:.6f}")
            rec["input_rank_sum"] += rank

            if rank < rec["best_input_rank"]:
                rec["best_input_rank"] = rank
                rec["best_input_strategy"] = strategy_name

    fused_rows = []

    for rec in article_records.values():
        n = max(1, safe_int(rec.get("strategies_found", 0), 0))
        rec["mean_input_rank"] = rec["input_rank_sum"] / n

        rec["strategy_list"] = ",".join(rec["strategy_list"])
        rec["rank_details"] = ";".join(rec["rank_details"])
        rec["score_details"] = ";".join(rec["score_details"])

        fused_rows.append(rec)

    fused_rows.sort(
        key=lambda r: (
            safe_float(r.get("fusion_score", 0.0), 0.0),
            safe_int(r.get("strategies_found", 0), 0),
            -safe_float(r.get("mean_input_rank", 10**9), 10**9),
        ),
        reverse=True,
    )

    for i, row in enumerate(fused_rows, start=1):
        row["rank"] = i
        row["fusion_score"] = round(safe_float(row.get("fusion_score", 0.0)), 10)
        row["rrf_score"] = round(safe_float(row.get("rrf_score", 0.0)), 10)
        row["score_fusion_score"] = round(safe_float(row.get("score_fusion_score", 0.0)), 10)
        row["mean_input_rank"] = round(safe_float(row.get("mean_input_rank", 0.0)), 4)

    return fused_rows


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
    strategies: List[int],
    weights: Dict[int, float],
    rrf_k: int,
    use_score_fusion: bool,
    score_fusion_weight: float,
    require_all: bool,
    min_strategies: int,
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

    strategy_rows: Dict[int, List[Dict[str, Any]]] = {}
    available_strategies = []
    missing_strategies = []
    strategy_input_counts: Dict[int, int] = {}

    for strategy in strategies:
        rows = load_strategy_ranking(project_dir, strategy)

        if rows:
            strategy_rows[strategy] = rows
            available_strategies.append(strategy)
            strategy_input_counts[strategy] = len(rows)
        else:
            missing_strategies.append(strategy)
            strategy_input_counts[strategy] = 0

    if require_all and missing_strategies:
        print(f"FAIL | {project_name} | missing strategies: {missing_strategies}")
        return {
            "project": project_name,
            "status": "missing_required_strategies",
            "ranked": 0,
            "available_strategies": ",".join(map(str, available_strategies)),
            "missing_strategies": ",".join(map(str, missing_strategies)),
        }

    if not available_strategies:
        print(f"FAIL | {project_name} | no ranking files found")
        return {
            "project": project_name,
            "status": "no_rankings",
            "ranked": 0,
            "available_strategies": "",
            "missing_strategies": ",".join(map(str, missing_strategies)),
            "strategy_input_counts": ",".join([f"{k}:0" for k in strategies]),
        }

    if len(available_strategies) < min_strategies:
        print(f"FAIL | {project_name} | only {len(available_strategies)} strategy file(s) available; min required={min_strategies}")
        return {
            "project": project_name,
            "status": "too_few_strategies",
            "ranked": 0,
            "available_strategies": ",".join(map(str, available_strategies)),
            "missing_strategies": ",".join(map(str, missing_strategies)),
            "strategy_input_counts": ",".join([f"{k}:{strategy_input_counts.get(k, 0)}" for k in strategies]),
        }

    fused_rows = fuse_rankings(
        strategy_rows=strategy_rows,
        strategies=available_strategies,
        weights=weights,
        rrf_k=rrf_k,
        use_score_fusion=use_score_fusion,
        score_fusion_weight=score_fusion_weight,
    )

    reference_dois = load_reference_dois(project_dir)
    eval_metrics = evaluate_common_at_k(fused_rows, reference_dois, TOP_K_VALUES)

    refs = set(reference_dois)

    for row in fused_rows:
        doi = normalize_doi(row.get("doi", ""))
        row["is_reference"] = int(bool(doi and doi in refs))

    fields = [
        "rank",
        "fusion_score",
        "rrf_score",
        "score_fusion_score",
        "strategies_found",
        "strategy_list",
        "best_input_rank",
        "best_input_strategy",
        "mean_input_rank",
        "rank_details",
        "score_details",
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
        "key",
    ]

    write_csv(out_path, fused_rows, fields)

    n_fulltext = sum(1 for r in fused_rows if safe_int(r.get("has_full_text", 0), 0) == 1)
    n_missing_fulltext = len(fused_rows) - n_fulltext

    summary = {
        "project": project_name,
        "status": "ok",
        "ranked": len(fused_rows),
        "refs_doi": len(reference_dois),
        "with_full_text": n_fulltext,
        "missing_full_text": n_missing_fulltext,
        "available_strategies": ",".join(map(str, available_strategies)),
        "missing_strategies": ",".join(map(str, missing_strategies)),
        "n_available_strategies": len(available_strategies),
        "n_missing_strategies": len(missing_strategies),
        "strategy_input_counts": ",".join([f"{k}:{strategy_input_counts.get(k, 0)}" for k in strategies]),
        "weights_used": ",".join([f"{k}:{weights.get(k, 1.0)}" for k in available_strategies]),
        "rrf_k": rrf_k,
        "use_score_fusion": int(use_score_fusion),
        "score_fusion_weight": score_fusion_weight,
        "rank_file": str(out_path),
    }
    summary.update(eval_metrics)

    metric_text = " | ".join([f"C@{k}={summary.get(f'C@{k}', 0)}" for k in TOP_K_VALUES])
    coverage_text = " | ".join([f"Cov@{k}={summary.get(f'Coverage@{k}', 0):.3f}" for k in TOP_K_VALUES])

    print(
        f"OK | {project_name} | ranked={len(fused_rows)} | "
        f"refs_doi={len(reference_dois)} | fulltext={n_fulltext} | "
        f"missing_fulltext={n_missing_fulltext} | strategies={summary['available_strategies']} | {metric_text}"
    )
    print(f"     {coverage_text}")
    print(f"     input counts: {', '.join([f'{k}:{strategy_input_counts.get(k, 0)}' for k in strategies])}")
    if missing_strategies:
        print(f"     missing strategies: {missing_strategies}")
    print(f"     file={out_path}")

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step14 FullRank10: hybrid fusion over FullRank1-9."
    )

    parser.add_argument(
        "--projects-dir",
        type=str,
        default=str(DEFAULT_PROJECTS_DIR),
        help="Path to Projects directory.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="Run only one project folder name.",
    )

    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(map(str, DEFAULT_STRATEGIES)),
        help="Comma-separated strategies to fuse, e.g. 1,2,4,7,8,9",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Optional weight overrides, e.g. 1=1.5,7=1.4,9=1.2",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant. Default 60.",
    )

    parser.add_argument(
        "--score-fusion",
        action="store_true",
        help="Also include normalized score fusion along with RRF.",
    )

    parser.add_argument(
        "--score-fusion-weight",
        type=float,
        default=0.05,
        help="Weight for normalized score fusion if --score-fusion is used.",
    )

    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail a project if any requested strategy file is missing.",
    )

    parser.add_argument(
        "--min-strategies",
        type=int,
        default=DEFAULT_MIN_STRATEGIES,
        help="Minimum number of available strategy files required per project. Default 2.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing full_rank_strategy_10.csv files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    strategies = parse_strategy_list(args.strategies)
    weights = parse_weight_overrides(args.weights)

    print("=" * 120)
    print("Step14 FullRank10: Hybrid Fusion / Ensemble Ranking")
    print(f"Projects directory     : {projects_dir}")
    print(f"Projects found         : {len(projects)}")
    print(f"Strategies             : {strategies}")
    print(f"Weights                : {', '.join([f'{k}={weights.get(k, 1.0)}' for k in strategies])}")
    print(f"RRF k                  : {args.rrf_k}")
    print(f"Use score fusion       : {args.score_fusion}")
    print(f"Score fusion weight    : {args.score_fusion_weight}")
    print(f"Require all strategies : {args.require_all}")
    print(f"Minimum strategies     : {args.min_strategies}")
    print(f"Force overwrite        : {args.force}")
    print(f"Top-K values           : {TOP_K_VALUES}")
    print("Uses references to rank: NO")
    print("Uses APIs              : NO")
    print("Uses LLM               : NO")
    print("Uses PDF download      : NO")
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
                strategies=strategies,
                weights=weights,
                rrf_k=args.rrf_k,
                use_score_fusion=args.score_fusion,
                score_fusion_weight=args.score_fusion_weight,
                require_all=args.require_all,
                min_strategies=args.min_strategies,
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
        "refs_doi",
        "with_full_text",
        "missing_full_text",
        "available_strategies",
        "missing_strategies",
        "n_available_strategies",
        "n_missing_strategies",
        "strategy_input_counts",
        "weights_used",
        "rrf_k",
        "use_score_fusion",
        "score_fusion_weight",
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
        f"{'Strat':>7s} "
        + " ".join([f"C@{k:>4}" for k in TOP_K_VALUES])
    )

    print(header)
    print("-" * len(header))

    for s in summaries:
        project = str(s.get("project", ""))[:45]
        ranked = safe_int(s.get("ranked", 0), 0)
        refs = safe_int(s.get("refs_doi", 0), 0)
        ft = safe_int(s.get("with_full_text", 0), 0)
        miss = safe_int(s.get("missing_full_text", 0), 0)
        nstrat = safe_int(s.get("n_available_strategies", 0), 0)

        cvals = " ".join([f"{safe_int(s.get(f'C@{k}', 0), 0):>6d}" for k in TOP_K_VALUES])

        print(f"{project:45s} {ranked:8d} {refs:6d} {ft:6d} {miss:7d} {nstrat:7d} {cvals}")

    print("=" * 120)


if __name__ == "__main__":
    main()