import os
import json
import re
from pathlib import Path
import pandas as pd

# =========================
# SETTINGS
# =========================

# IMPORTANT:
# Change this if your real folder is DataBase_Strategy3_Top1000
DB_DIR = "DataBase"
DB_DIR = "DataBase_Strategy3_Top1000"

OUTPUT_DIR = "Output"

SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "project_download_summary.csv")
DETAIL_OUTPUT = os.path.join(OUTPUT_DIR, "project_article_download_status.csv")
OVERALL_OUTPUT = os.path.join(OUTPUT_DIR, "overall_download_summary.csv")
SOURCE_STATS_OUTPUT = os.path.join(OUTPUT_DIR, "download_source_stats_from_log.csv")
RETRIEVAL_SOURCE_STATS_OUTPUT = os.path.join(OUTPUT_DIR, "download_retrieval_source_stats.csv")
QUERY_ORIGIN_STATS_OUTPUT = os.path.join(OUTPUT_DIR, "download_query_origin_stats.csv")
METHOD_BY_RETRIEVAL_SOURCE_OUTPUT = os.path.join(OUTPUT_DIR, "download_method_by_retrieval_source.csv")
PROVENANCE_COVERAGE_OUTPUT = os.path.join(OUTPUT_DIR, "download_provenance_coverage.csv")

DOWNLOAD_LOG = os.path.join(OUTPUT_DIR, "download_log.csv")

MIN_PDF_SIZE = 8000

PROVENANCE_FIELDS = [
    "source_database",
    "source_query",
    "query_origin",
    "rendered_query",
    "url",
    "openalex_id",
    "pmid",
    "pmcid",
    "semantic_scholar_id",
    "rank_strategy_3_score",
    "score_backend",
]


# =========================
# HELPERS
# =========================

def is_valid_pdf(pdf_path):
    """
    Check if document.pdf exists and looks like a valid PDF.
    """
    try:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return False

        if pdf_path.stat().st_size < MIN_PDF_SIZE:
            return False

        with open(pdf_path, "rb") as f:
            header = f.read(4)

        return header == b"%PDF"

    except Exception:
        return False


def find_project_folders(db_dir):
    """
    Find project folders inside DataBase.
    A valid project folder should contain project_candidates_selected.csv.
    """
    db_path = Path(db_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Database folder not found: {db_dir}")

    project_folders = []

    for folder in db_path.iterdir():
        if folder.is_dir():
            candidate_file = folder / "project_candidates_selected.csv"
            if candidate_file.exists():
                project_folders.append(folder)

    return sorted(project_folders)


def get_pdf_size_kb(pdf_path):
    try:
        pdf_path = Path(pdf_path)
        if pdf_path.exists():
            return round(pdf_path.stat().st_size / 1024, 2)
    except Exception:
        pass
    return 0


def normalize_doi(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().strip(".,;:()[]{}<>")


def normalize_title(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_semicolon_values(value):
    text = str(value or "").strip()
    if not text:
        return []
    out = []
    seen = set()
    for part in re.split(r"\s*;\s*", text):
        item = part.strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def read_metadata(project_folder, rank):
    """
    Try to read Temp_x/metadata.json if present.
    """
    try:
        metadata_path = project_folder / f"Temp_{rank}" / "metadata.json"
        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        return data if isinstance(data, dict) else {}

    except Exception:
        return {}


def read_method_from_metadata(project_folder, rank):
    return read_metadata(project_folder, rank).get("method", "")


def known_original_download_method(method):
    method = str(method or "").strip()
    if method in {"", "previously_downloaded", "already_exists", "failed"}:
        return ""
    return method


def classify_pdf_method(metadata, row):
    method_this_run = first_nonempty(
        metadata.get("method_this_run", ""),
        row.get("method_this_run", ""),
    )
    original_method = first_nonempty(
        metadata.get("original_download_method", ""),
        row.get("original_download_method", ""),
    )
    recorded_method = first_nonempty(
        original_method,
        metadata.get("method", ""),
        row.get("method", ""),
    )

    known_method = known_original_download_method(recorded_method)
    if known_method:
        return known_method, method_this_run or recorded_method, "known_original_pdf_method"

    if recorded_method == "previously_downloaded" or method_this_run == "previously_downloaded":
        return "previously_downloaded_original_method_unknown", "previously_downloaded", "existing_pdf_original_method_unknown"

    return recorded_method, method_this_run, "pdf_method_missing"


_RANKED_LOOKUP_CACHE = {}


def build_ranked_lookup(source_ranked_csv):
    """
    Load ranking CSV once and index rows by DOI/title so Step12 can recover
    source_database/source_query/query_origin even if Step11 selected manifest
    did not preserve them.
    """
    source_ranked_csv = str(source_ranked_csv or "").strip()
    if not source_ranked_csv:
        return {}

    path = Path(source_ranked_csv)
    if not path.is_absolute():
        path = Path(source_ranked_csv)

    cache_key = str(path)
    if cache_key in _RANKED_LOOKUP_CACHE:
        return _RANKED_LOOKUP_CACHE[cache_key]

    lookup = {}
    try:
        if path.exists():
            ranked_df = pd.read_csv(path).fillna("")
        else:
            ranked_df = pd.DataFrame()

        for _, ranked_row in ranked_df.iterrows():
            data = {k: ranked_row.get(k, "") for k in ranked_df.columns}
            doi = normalize_doi(data.get("doi", ""))
            title_key = normalize_title(data.get("title", ""))
            if doi:
                lookup[f"doi::{doi}"] = data
            if title_key:
                lookup[f"title::{title_key}"] = data
    except Exception:
        lookup = {}

    _RANKED_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def recover_ranked_row(candidate_row):
    source_ranked_csv = candidate_row.get("source_ranked_csv", "")
    lookup = build_ranked_lookup(source_ranked_csv)
    if not lookup:
        return {}

    doi = normalize_doi(candidate_row.get("doi", ""))
    title_key = normalize_title(candidate_row.get("title", ""))

    if doi and f"doi::{doi}" in lookup:
        return lookup[f"doi::{doi}"]
    if title_key and f"title::{title_key}" in lookup:
        return lookup[f"title::{title_key}"]
    return {}


def first_nonempty(*values):
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return ""


def get_provenance(candidate_row, metadata, ranked_row):
    provenance = {}
    for field in PROVENANCE_FIELDS:
        provenance[field] = first_nonempty(
            candidate_row.get(field, ""),
            metadata.get(field, ""),
            ranked_row.get(field, ""),
        )
    provenance["provenance_recovered_from"] = (
        "metadata_json" if any(metadata.get(f, "") for f in PROVENANCE_FIELDS)
        else "source_ranked_csv" if any(ranked_row.get(f, "") for f in PROVENANCE_FIELDS)
        else ""
    )
    return provenance


def make_source_stats_from_download_log():
    """
    If Output/download_log.csv exists, calculate source/method counts.
    """
    if not os.path.exists(DOWNLOAD_LOG):
        return None

    try:
        log_df = pd.read_csv(DOWNLOAD_LOG).fillna("")

        if "success" not in log_df.columns or "method" not in log_df.columns:
            return None

        # Normalize success column
        log_df["success_bool"] = log_df["success"].astype(str).str.lower().isin(
            ["true", "1", "yes", "downloaded", "ok"]
        )

        source_stats = (
            log_df[log_df["success_bool"]]
            .groupby("method")
            .size()
            .reset_index(name="downloaded_count")
            .sort_values("downloaded_count", ascending=False)
        )

        source_stats.to_csv(SOURCE_STATS_OUTPUT, index=False)
        return source_stats

    except Exception as e:
        print(f"WARNING: Could not read source stats from {DOWNLOAD_LOG}: {e}")
        return None


# =========================
# MAIN
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    project_folders = find_project_folders(DB_DIR)

    summary_rows = []
    detail_rows = []

    total_should_download = 0
    total_downloaded = 0
    total_pending = 0

    for project_folder in project_folders:
        candidates_file = project_folder / "project_candidates_selected.csv"

        df = pd.read_csv(candidates_file).fillna("")

        project_key = project_folder.name

        if "project_name" in df.columns and len(df) > 0:
            project_name = str(df["project_name"].iloc[0])
        else:
            project_name = project_key

        should_download_count = len(df)
        downloaded_count = 0
        pending_count = 0

        for idx, row in df.iterrows():
            if "rank_in_project" in row:
                try:
                    rank = int(row.get("rank_in_project", idx))
                except Exception:
                    rank = idx
            else:
                rank = idx

            pdf_path = project_folder / f"Temp_{rank}" / "document.pdf"

            downloaded = is_valid_pdf(pdf_path)
            pdf_size_kb = get_pdf_size_kb(pdf_path)

            metadata = read_metadata(project_folder, rank)
            ranked_row = recover_ranked_row(row)
            provenance = get_provenance(row, metadata, ranked_row)

            pdf_method, method_this_run, pdf_method_provenance = classify_pdf_method(metadata, row)

            if downloaded:
                downloaded_count += 1
                status = "Downloaded"
            else:
                pending_count += 1
                status = "Pending"

            detail_rows.append({
                "project_name": project_name,
                "project_key": project_key,
                "rank_in_project": rank,
                "title": row.get("title", ""),
                "doi": row.get("doi", ""),
                "pdf_path": str(pdf_path),
                "pdf_size_kb": pdf_size_kb,
                "pdf_download_method_recorded": pdf_method,
                "method_this_run": method_this_run,
                "pdf_method_provenance": pdf_method_provenance,
                "method_from_metadata": pdf_method,
                "status": status,
                **provenance,
            })

        success_rate = round(
            downloaded_count * 100 / should_download_count, 2
        ) if should_download_count > 0 else 0

        summary_rows.append({
            "project_name": project_name,
            "project_key": project_key,
            "should_download": should_download_count,
            "downloaded": downloaded_count,
            "pending": pending_count,
            "success_rate_percent": success_rate
        })

        total_should_download += should_download_count
        total_downloaded += downloaded_count
        total_pending += pending_count

    overall_success_rate = round(
        total_downloaded * 100 / total_should_download, 2
    ) if total_should_download > 0 else 0

    overall_df = pd.DataFrame([{
        "total_should_download": total_should_download,
        "total_downloaded": total_downloaded,
        "total_pending": total_pending,
        "overall_success_rate_percent": overall_success_rate
    }])

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["success_rate_percent", "downloaded"],
            ascending=[False, False]
        )

    summary_df.to_csv(SUMMARY_OUTPUT, index=False)
    detail_df.to_csv(DETAIL_OUTPUT, index=False)
    overall_df.to_csv(OVERALL_OUTPUT, index=False)

    source_stats = make_source_stats_from_download_log()

    retrieval_source_stats = None
    query_origin_stats = None
    method_by_retrieval_source = None
    provenance_coverage = None

    if not detail_df.empty:
        expanded_source_rows = []
        expanded_origin_rows = []
        for _, row in detail_df.iterrows():
            sources = split_semicolon_values(row.get("source_database", "")) or ["unknown"]
            origins = split_semicolon_values(row.get("query_origin", "")) or ["unknown"]

            for source in sources:
                item = row.to_dict()
                item["retrieval_source_database"] = source
                expanded_source_rows.append(item)

            for origin in origins:
                item = row.to_dict()
                item["retrieval_query_origin"] = origin
                expanded_origin_rows.append(item)

        if expanded_source_rows:
            source_detail_df = pd.DataFrame(expanded_source_rows)
            retrieval_source_stats = (
                source_detail_df
                .groupby(["retrieval_source_database", "status"])
                .size()
                .reset_index(name="article_count")
                .sort_values(["status", "article_count"], ascending=[True, False])
            )
            retrieval_source_stats.to_csv(RETRIEVAL_SOURCE_STATS_OUTPUT, index=False)

            method_by_retrieval_source = (
                source_detail_df[source_detail_df["status"] == "Downloaded"]
                .groupby(["retrieval_source_database", "pdf_download_method_recorded", "pdf_method_provenance"])
                .size()
                .reset_index(name="downloaded_count")
                .sort_values(["retrieval_source_database", "downloaded_count"], ascending=[True, False])
            )
            method_by_retrieval_source.to_csv(METHOD_BY_RETRIEVAL_SOURCE_OUTPUT, index=False)

        provenance_coverage = (
            detail_df
            .groupby(["status", "provenance_recovered_from", "pdf_method_provenance"])
            .size()
            .reset_index(name="article_count")
            .sort_values(["status", "article_count"], ascending=[True, False])
        )
        provenance_coverage.to_csv(PROVENANCE_COVERAGE_OUTPUT, index=False)

        if expanded_origin_rows:
            origin_detail_df = pd.DataFrame(expanded_origin_rows)
            query_origin_stats = (
                origin_detail_df
                .groupby(["retrieval_query_origin", "status"])
                .size()
                .reset_index(name="article_count")
                .sort_values(["status", "article_count"], ascending=[True, False])
            )
            query_origin_stats.to_csv(QUERY_ORIGIN_STATS_OUTPUT, index=False)

    print("\n" + "=" * 90)
    print("OVERALL DOWNLOAD SUMMARY")
    print("=" * 90)
    print(f"Total articles should be downloaded : {total_should_download}")
    print(f"Total articles downloaded           : {total_downloaded}")
    print(f"Total articles pending/missing      : {total_pending}")
    print(f"Overall success rate                : {overall_success_rate}%")

    print("\n" + "=" * 90)
    print("PROJECT DOWNLOAD SUMMARY")
    print("=" * 90)

    if summary_df.empty:
        print("No project folders found.")
    else:
        print(summary_df.to_string(index=False))

    if source_stats is not None and not source_stats.empty:
        print("\n" + "=" * 90)
        print("DOWNLOADS BY SOURCE / METHOD FROM download_log.csv")
        print("=" * 90)
        print(source_stats.to_string(index=False))
    else:
        print("\nNo source/method stats found because Output/download_log.csv was not available or incomplete.")

    if retrieval_source_stats is not None and not retrieval_source_stats.empty:
        print("\n" + "=" * 90)
        print("DOWNLOAD STATUS BY ORIGINAL RETRIEVAL DATABASE")
        print("=" * 90)
        print(retrieval_source_stats.to_string(index=False))

    if query_origin_stats is not None and not query_origin_stats.empty:
        print("\n" + "=" * 90)
        print("DOWNLOAD STATUS BY CONFIGURATION QUERY FIELD")
        print("=" * 90)
        print(query_origin_stats.to_string(index=False))

    if method_by_retrieval_source is not None and not method_by_retrieval_source.empty:
        print("\n" + "=" * 90)
        print("RECORDED PDF METHOD BY ORIGINAL RETRIEVAL DATABASE")
        print("=" * 90)
        print("Note: 'previously_downloaded_original_method_unknown' means the PDF existed before this run,")
        print("      but the original PDF download method was not preserved in metadata/logs.")
        print(method_by_retrieval_source.to_string(index=False))

    if provenance_coverage is not None and not provenance_coverage.empty:
        print("\n" + "=" * 90)
        print("PROVENANCE COVERAGE AUDIT")
        print("=" * 90)
        print(provenance_coverage.to_string(index=False))

    print("\n" + "=" * 90)
    print("SAVED FILES")
    print("=" * 90)
    print("Overall summary:")
    print(OVERALL_OUTPUT)

    print("\nProject summary:")
    print(SUMMARY_OUTPUT)

    print("\nArticle-level status:")
    print(DETAIL_OUTPUT)

    if source_stats is not None:
        print("\nSource/method stats:")
        print(SOURCE_STATS_OUTPUT)

    if retrieval_source_stats is not None:
        print("\nRetrieval source stats:")
        print(RETRIEVAL_SOURCE_STATS_OUTPUT)

    if query_origin_stats is not None:
        print("\nQuery origin stats:")
        print(QUERY_ORIGIN_STATS_OUTPUT)

    if method_by_retrieval_source is not None:
        print("\nDownload method by retrieval source:")
        print(METHOD_BY_RETRIEVAL_SOURCE_OUTPUT)

    if provenance_coverage is not None:
        print("\nProvenance coverage audit:")
        print(PROVENANCE_COVERAGE_OUTPUT)


if __name__ == "__main__":
    main()
