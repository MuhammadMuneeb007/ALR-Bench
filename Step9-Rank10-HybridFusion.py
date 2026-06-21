# Step9-Rank10-HybridFusion.py
#
# Rank Strategy 10:
#   Hybrid fusion / ensemble ranking.
#
# What this script does:
#   1. Reads per-project ranking files:
#        Projects/<project>/Rankings/rank_strategy_1.csv
#        Projects/<project>/Rankings/rank_strategy_2.csv
#        ...
#        Projects/<project>/Rankings/rank_strategy_9.csv
#
#   2. Combines rankings using:
#        - Reciprocal Rank Fusion, RRF
#        - optional normalized score fusion
#        - optional reference flags for evaluation only
#
#   3. Saves:
#        Projects/<project>/Rankings/rank_strategy_10.csv
#        Projects/Rankings/rank_strategy_10_summary.csv
#
#   4. Prints:
#        Common@10, Common@20, Common@50, Common@100
#
# Notes:
#   - This is the final ensemble ranking.
#   - It does NOT use references to rank articles.
#   - References are used only for evaluation.
#   - No API calls.
#   - No PDF download.
#   - No full-text download.
#   - No LLM call.
#
# Usage:
#   python Step9-Rank10-HybridFusion.py --projects-dir Projects --force
#
# One project:
#   python Step9-Rank10-HybridFusion.py --projects-dir Projects --project PROJECT_NAME --force
#
# Use only selected strategies:
#   python Step9-Rank10-HybridFusion.py --projects-dir Projects --strategies 1,2,4,5,7,8,9 --force

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
DEFAULT_RANK_FILE = "rank_strategy_10.csv"

TOP_K_VALUES = [10, 20, 50, 100]

# Default final ensemble based on the locked plan.
# Rank6 is useful but can be noisy for configs without true Boolean strings.
# Rank3 is useful but partly redundant with Rank2.
DEFAULT_STRATEGIES = [1, 2, 4, 5, 7, 8, 9]

# RRF constant. 60 is standard and stable.
DEFAULT_RRF_K = 60

# Strategy weights. You can tune later based on validation results.
DEFAULT_STRATEGY_WEIGHTS = {
    1: 1.20,  # ConfigWeightedBM25
    2: 1.20,  # ReviewIntentSBERT
    3: 0.80,  # TitleObjectiveSBERT
    4: 1.00,  # ConceptKeywordCoverage
    5: 0.90,  # GeneratedQueryMatch
    6: 0.80,  # BooleanSoftMatch
    7: 1.00,  # FieldWeightedLexical
    8: 1.10,  # SciBERTSemantic
    9: 1.30,  # SPECTER2Semantic
}

STRATEGY_NAMES = {
    1: "ConfigWeightedBM25",
    2: "ReviewIntentSBERT",
    3: "TitleObjectiveSBERT",
    4: "ConceptKeywordCoverage",
    5: "GeneratedQueryMatch",
    6: "BooleanSoftMatch",
    7: "FieldWeightedLexical",
    8: "SciBERTSemantic",
    9: "SPECTER2Semantic",
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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        text = str(x).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        text = str(x).strip()
        if not text:
            return default
        return int(float(text))
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


def parse_strategy_list(text: str) -> List[int]:
    if not clean_text(text):
        return list(DEFAULT_STRATEGIES)

    out: List[int] = []

    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        n = safe_int(part, default=-1)
        if 1 <= n <= 9 and n not in out:
            out.append(n)

    return out or list(DEFAULT_STRATEGIES)


def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    if not raw_scores:
        return {}

    vals = list(raw_scores.values())
    min_s = min(vals)
    max_s = max(vals)

    if max_s <= min_s:
        return {k: 0.0 for k in raw_scores}

    return {
        k: (v - min_s) / (max_s - min_s)
        for k, v in raw_scores.items()
    }


def make_article_key(row: Dict[str, Any]) -> str:
    doi = normalize_doi(row.get("doi", ""))
    if doi:
        return f"doi::{doi}"

    title = normalize_title(row.get("title", ""))
    if title:
        return f"title::{title}"

    # Very rare fallback.
    return f"row::{hash(clean_text(row))}"


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
# RANK FILE LOADING
# ============================================================

def strategy_rank_path(project_dir: Path, strategy_id: int) -> Path:
    return project_dir / DEFAULT_RANKINGS_DIR / f"rank_strategy_{strategy_id}.csv"


def strategy_score_field(strategy_id: int) -> str:
    return f"rank_strategy_{strategy_id}_score"


def load_strategy_rows(
    project_dir: Path,
    strategy_id: int,
) -> Tuple[List[Dict[str, str]], Path]:
    path = strategy_rank_path(project_dir, strategy_id)
    return read_csv_rows(path), path


def collect_available_strategy_rows(
    project_dir: Path,
    strategy_ids: List[int],
) -> Tuple[Dict[int, List[Dict[str, str]]], Dict[int, str], List[int], List[int]]:
    data: Dict[int, List[Dict[str, str]]] = {}
    paths: Dict[int, str] = {}
    available: List[int] = []
    missing: List[int] = []

    for sid in strategy_ids:
        rows, path = load_strategy_rows(project_dir, sid)

        if rows:
            data[sid] = rows
            paths[sid] = str(path)
            available.append(sid)
        else:
            missing.append(sid)

    return data, paths, available, missing


def choose_best_metadata(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the row with richer metadata as the canonical article row.
    """
    if not existing:
        return dict(candidate)

    def richness(row: Dict[str, Any]) -> int:
        score = 0
        score += 100 if normalize_doi(row.get("doi", "")) else 0
        score += 50 if clean_text(row.get("title", "")) else 0
        score += min(len(clean_text(row.get("abstract", ""))), 2000)
        score += 20 if clean_text(row.get("journal", "")) else 0
        score += 20 if clean_text(row.get("authors", "")) else 0
        score += 10 if clean_text(row.get("year", "")) else 0
        score += 10 if clean_text(row.get("url", "")) else 0
        return score

    if richness(candidate) > richness(existing):
        return dict(candidate)

    return existing


# ============================================================
# FUSION
# ============================================================

def compute_rrf_score(rank: int, rrf_k: int) -> float:
    if rank <= 0:
        return 0.0
    return 1.0 / (rrf_k + rank)


def build_fusion_records(
    strategy_rows: Dict[int, List[Dict[str, str]]],
    available_strategies: List[int],
    ref_dois: Set[str],
    ceiling_dois: Set[str],
    rrf_k: int,
    score_fusion_weight: float,
) -> List[Dict[str, Any]]:

    article_map: Dict[str, Dict[str, Any]] = {}

    rrf_by_article: Dict[str, float] = {}
    score_by_article: Dict[str, float] = {}
    strategy_count_by_article: Dict[str, int] = {}
    best_rank_by_article: Dict[str, int] = {}

    per_strategy_rank: Dict[str, Dict[int, int]] = {}
    per_strategy_score: Dict[str, Dict[int, float]] = {}

    # First pass: normalize score values within each strategy.
    normalized_strategy_scores: Dict[int, Dict[str, float]] = {}

    for sid in available_strategies:
        rows = strategy_rows.get(sid, [])

        raw_scores: Dict[str, float] = {}

        for idx, row in enumerate(rows, start=1):
            key = make_article_key(row)
            score_field = strategy_score_field(sid)

            # If score column missing, fallback to inverse rank.
            score = safe_float(row.get(score_field, ""), default=0.0)
            if score == 0.0 and not clean_text(row.get(score_field, "")):
                rank_val = safe_int(row.get("rank", idx), default=idx)
                score = 1.0 / max(rank_val, 1)

            raw_scores[key] = score

        normalized_strategy_scores[sid] = normalize_scores(raw_scores)

    # Second pass: RRF + score fusion.
    for sid in available_strategies:
        rows = strategy_rows.get(sid, [])
        weight = DEFAULT_STRATEGY_WEIGHTS.get(sid, 1.0)

        for idx, row in enumerate(rows, start=1):
            key = make_article_key(row)

            rank_val = safe_int(row.get("rank", idx), default=idx)
            if rank_val <= 0:
                rank_val = idx

            norm_score = normalized_strategy_scores.get(sid, {}).get(key, 0.0)

            article_map[key] = choose_best_metadata(article_map.get(key, {}), row)

            rrf_by_article[key] = rrf_by_article.get(key, 0.0) + weight * compute_rrf_score(rank_val, rrf_k)
            score_by_article[key] = score_by_article.get(key, 0.0) + weight * norm_score
            strategy_count_by_article[key] = strategy_count_by_article.get(key, 0) + 1

            if key not in best_rank_by_article:
                best_rank_by_article[key] = rank_val
            else:
                best_rank_by_article[key] = min(best_rank_by_article[key], rank_val)

            per_strategy_rank.setdefault(key, {})[sid] = rank_val
            per_strategy_score.setdefault(key, {})[sid] = norm_score

    # Normalize global RRF and score components.
    rrf_norm = normalize_scores(rrf_by_article)
    score_norm = normalize_scores(score_by_article)

    output_rows: List[Dict[str, Any]] = []

    n_available = len(available_strategies)

    for key, base in article_map.items():
        doi = normalize_doi(base.get("doi", ""))
        title = clean_text(base.get("title", ""))

        rrf_component = rrf_norm.get(key, 0.0)
        score_component = score_norm.get(key, 0.0)

        strategy_count = strategy_count_by_article.get(key, 0)
        strategy_coverage = strategy_count / n_available if n_available else 0.0

        best_rank = best_rank_by_article.get(key, 999999)

        # Final hybrid fusion:
        #   mostly RRF, some score fusion, small coverage reward.
        final_score = (
            (1.0 - score_fusion_weight) * rrf_component
            + score_fusion_weight * score_component
            + 0.05 * strategy_coverage
        )

        out = dict(base)

        out["doi"] = doi
        out["title"] = title
        out["rank_strategy_10_score"] = round(final_score, 8)
        out["score_rrf_norm"] = round(rrf_component, 8)
        out["score_fusion_norm"] = round(score_component, 8)
        out["score_strategy_coverage"] = round(strategy_coverage, 8)
        out["strategy_count"] = strategy_count
        out["best_rank_any_strategy"] = best_rank
        out["available_strategies_used"] = ",".join(str(x) for x in available_strategies)

        # Per-strategy rank and score columns.
        for sid in range(1, 10):
            out[f"rank_s{sid}"] = per_strategy_rank.get(key, {}).get(sid, "")
            s = per_strategy_score.get(key, {}).get(sid, "")
            out[f"score_s{sid}_norm"] = round(s, 8) if isinstance(s, float) else ""

        contributing = [
            f"{sid}:{STRATEGY_NAMES.get(sid, f'Rank{sid}')}"
            for sid in available_strategies
            if sid in per_strategy_rank.get(key, {})
        ]

        out["contributing_strategies"] = "; ".join(contributing)
        out["is_reference_match"] = bool(doi and doi in ref_dois)
        out["is_retrievable_reference_match"] = bool(doi and doi in ceiling_dois)

        output_rows.append(out)

    output_rows.sort(
        key=lambda r: (
            float(r.get("rank_strategy_10_score", 0.0) or 0.0),
            int(r.get("strategy_count", 0) or 0),
            -int(r.get("best_rank_any_strategy", 999999) or 999999),
            clean_text(r.get("title", "")),
        ),
        reverse=True,
    )

    for idx, row in enumerate(output_rows, start=1):
        row["rank"] = idx

    return output_rows


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
    strategy_ids: List[int],
    rrf_k: int,
    score_fusion_weight: float,
    require_all: bool,
) -> Dict[str, Any]:

    ranking_dir = project_dir / DEFAULT_RANKINGS_DIR
    rank_path = ranking_dir / DEFAULT_RANK_FILE

    if rank_path.exists() and not force:
        return {
            "project": project_dir.name,
            "status": "skipped_existing_ranking",
            "rank_file": str(rank_path),
        }

    strategy_rows, strategy_paths, available, missing = collect_available_strategy_rows(
        project_dir=project_dir,
        strategy_ids=strategy_ids,
    )

    if require_all and missing:
        return {
            "project": project_dir.name,
            "status": "failed_missing_required_strategy_files",
            "missing_strategies": ",".join(str(x) for x in missing),
            "available_strategies": ",".join(str(x) for x in available),
        }

    if not available:
        return {
            "project": project_dir.name,
            "status": "failed_no_strategy_files",
            "missing_strategies": ",".join(str(x) for x in missing),
        }

    ref_dois, ref_source, references_total = load_reference_dois(project_dir)
    ceiling_dois = load_retrievable_ceiling_dois(project_dir)

    ranked_rows = build_fusion_records(
        strategy_rows=strategy_rows,
        available_strategies=available,
        ref_dois=ref_dois,
        ceiling_dois=ceiling_dois,
        rrf_k=rrf_k,
        score_fusion_weight=score_fusion_weight,
    )

    base_fields = list(ranked_rows[0].keys()) if ranked_rows else [
        "rank",
        "rank_strategy_10_score",
        "doi",
        "title",
        "abstract",
        "year",
        "journal",
        "source_database",
    ]

    preferred_first = [
        "rank",
        "rank_strategy_10_score",
        "score_rrf_norm",
        "score_fusion_norm",
        "score_strategy_coverage",
        "strategy_count",
        "best_rank_any_strategy",
        "available_strategies_used",
        "contributing_strategies",
    ]

    for sid in range(1, 10):
        preferred_first.append(f"rank_s{sid}")
        preferred_first.append(f"score_s{sid}_norm")

    preferred_first.extend([
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
    ])

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
        "rank_file": str(rank_path),
        "articles_ranked": len(ranked_rows),
        "references_source": ref_source,
        "references_total": references_total,
        "references_with_doi": len(ref_dois),
        "retrievable_ceiling": len(ceiling_dois) if ceiling_dois else "",
        "strategies_requested": ",".join(str(x) for x in strategy_ids),
        "strategies_available": ",".join(str(x) for x in available),
        "strategies_missing": ",".join(str(x) for x in missing),
        "n_strategies_available": len(available),
        "rrf_k": rrf_k,
        "score_fusion_weight": score_fusion_weight,
    }

    for sid in available:
        summary[f"rank_strategy_{sid}_file"] = strategy_paths.get(sid, "")

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
    "strategies_requested",
    "strategies_available",
    "strategies_missing",
    "n_strategies_available",
    "rrf_k",
    "score_fusion_weight",
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
    out_path = projects_dir / "Rankings" / "rank_strategy_10_summary.csv"
    write_csv(out_path, rows, SUMMARY_FIELDS)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create final hybrid fusion ranking from previous strategy outputs."
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
        "--strategies",
        default=",".join(str(x) for x in DEFAULT_STRATEGIES),
        help="Comma-separated ranking strategy IDs to fuse, e.g. 1,2,4,5,7,8,9.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant. Standard value is 60.",
    )

    parser.add_argument(
        "--score-fusion-weight",
        type=float,
        default=0.25,
        help="Weight given to normalized score fusion. RRF receives 1 - this weight.",
    )

    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail project if any requested strategy file is missing.",
    )

    args = parser.parse_args()

    projects_dir = Path(args.projects_dir)
    projects = discover_projects(projects_dir, args.project)

    strategy_ids = parse_strategy_list(args.strategies)

    score_fusion_weight = max(0.0, min(float(args.score_fusion_weight), 1.0))
    rrf_k = max(1, int(args.rrf_k))

    print("=" * 140)
    print("STEP 9: RANK STRATEGY 10 — HYBRID FUSION / ENSEMBLE RANKING")
    print("=" * 140)
    print(f"Projects directory : {projects_dir.resolve()}")
    print(f"Projects found     : {len(projects)}")
    print(f"Force overwrite    : {args.force}")
    print(f"Top-K values       : {TOP_K_VALUES}")
    print(f"Strategies fused   : {strategy_ids}")
    print(f"RRF k              : {rrf_k}")
    print(f"Score fusion weight: {score_fusion_weight}")
    print(f"Require all files  : {args.require_all}")
    print("Uses APIs          : NO")
    print("Uses LLM           : NO")
    print("Uses PDFs/fulltext : NO")
    print("=" * 140)

    summaries: List[Dict[str, Any]] = []

    for i, project_dir in enumerate(projects, start=1):
        print(f"\n[{i}/{len(projects)}] {project_dir.name}")

        try:
            summary = process_project(
                project_dir=project_dir,
                force=args.force,
                strategy_ids=strategy_ids,
                rrf_k=rrf_k,
                score_fusion_weight=score_fusion_weight,
                require_all=args.require_all,
            )

            summaries.append(summary)

            if summary.get("status") == "success":
                print(
                    f"OK | ranked={summary.get('articles_ranked')} | "
                    f"strategies={summary.get('strategies_available')} | "
                    f"refs_doi={summary.get('references_with_doi')} | "
                    f"C@10={summary.get('common_doi_at_10')} | "
                    f"C@20={summary.get('common_doi_at_20')} | "
                    f"C@50={summary.get('common_doi_at_50')} | "
                    f"C@100={summary.get('common_doi_at_100')} | "
                    f"file={summary.get('rank_file')}"
                )
            else:
                print(
                    f"{summary.get('status')} | "
                    f"available={summary.get('available_strategies', summary.get('strategies_available', ''))} | "
                    f"missing={summary.get('missing_strategies', summary.get('strategies_missing', ''))}"
                )

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
    print("  Projects/<project>/Rankings/rank_strategy_10.csv")


if __name__ == "__main__":
    main()