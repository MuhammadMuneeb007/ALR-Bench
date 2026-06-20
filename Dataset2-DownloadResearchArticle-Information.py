# Step2-SaveEvaluationDatasetReport.py
from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import requests

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("benchmark_reviews")
PROJECTS_DIR = Path("Projects")
OUTPUT_CSV = BASE_DIR / "evaluation_dataset.csv"
OUTPUT_TXT = BASE_DIR / "evaluation_dataset_report.txt"
OUTPUT_BIB = BASE_DIR / "evaluation_dataset_merged.bib"
OUTPUT_JSON = BASE_DIR / "evaluation_dataset.json"

REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ============================================================
# HELPERS
# ============================================================

def clean_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip()

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()

def slugify_project_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", clean_text(text).lower())
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:max_len].strip("_") or "untitled_project"

def load_n_gold_dois(row: Dict) -> int:
    project_dir = PROJECTS_DIR / slugify_project_name(clean_text(row.get("title", "")))
    gold_path = project_dir / "references_gold.json"
    if not gold_path.exists():
        return 0
    try:
        payload = json.loads(gold_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    dois = payload.get("dois") or []
    if isinstance(dois, list):
        return len([d for d in dois if clean_text(d)])
    return int(payload.get("n_with_doi", 0) or 0)

def latex_escape(text: str) -> str:
    if text is None:
        return ""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out

def make_cite_key(authors: str, year: str, title: str) -> str:
    # First author's surname-ish token
    first = "paper"
    if authors:
        first_author = authors.split(";")[0].strip()
        parts = re.split(r"\s+", first_author)
        if parts:
            first = re.sub(r"[^A-Za-z0-9]", "", parts[-1].lower()) or "paper"

    year = clean_text(year) or "nodate"

    # first informative title token
    title_tokens = re.findall(r"[A-Za-z0-9]+", title.lower())
    stop = {
        "a", "an", "the", "on", "of", "for", "and", "in", "to", "with",
        "systematic", "literature", "review", "reviews", "study"
    }
    word = "paper"
    for t in title_tokens:
        if t not in stop and len(t) > 2:
            word = t
            break

    return f"{first}{year}{word}"

def try_crossref_bibtex(doi: str) -> str:
    doi = clean_text(doi)
    if not doi:
        return ""

    url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and r.text.strip().startswith("@"):
            return r.text.strip()
    except Exception:
        pass
    return ""

def fallback_bibtex(row: Dict, cite_key: str) -> str:
    authors = clean_text(row.get("authors", ""))
    title = clean_text(row.get("title", ""))
    journal = clean_text(row.get("journal", ""))
    year = clean_text(row.get("year", ""))
    doi = clean_text(row.get("doi", ""))
    url = clean_text(row.get("landing_page_url", ""))

    fields = [
        f"  author = {{{authors}}}",
        f"  title = {{{title}}}",
        f"  journal = {{{journal}}}",
        f"  year = {{{year}}}",
    ]
    if doi:
        fields.append(f"  doi = {{{doi}}}")
    if url:
        fields.append(f"  url = {{{url}}}")

    return "@article{" + cite_key + ",\n" + ",\n".join(fields) + "\n}"

def replace_bibtex_key(bibtex: str, new_key: str) -> str:
    return re.sub(r"^@\w+\{[^,]+,", lambda m: m.group(0).split("{")[0] + "{" + new_key + ",", bibtex, count=1)

def detect_download_method(row: Dict) -> str:
    """
    Best-effort method label for reporting.
    """
    oa_pdf_url = clean_text(row.get("oa_pdf_url", ""))
    landing = clean_text(row.get("landing_page_url", ""))
    doi = clean_text(row.get("doi", ""))

    if oa_pdf_url:
        if "europepmc" in oa_pdf_url.lower() or "pmc" in oa_pdf_url.lower():
            return "Open-access PDF via Europe PMC / PMC"
        return "Open-access PDF via source-provided OA URL"

    if landing:
        if "doi.org" in landing.lower() and doi:
            return "Downloaded via DOI landing / OA resolution"
        return "Downloaded via publisher landing page"

    if doi:
        return "Downloaded via DOI-based resolution"

    return "Downloaded from available open-access source"

def load_downloaded_records(base_dir: Path) -> List[Dict]:
    rows = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        meta_path = folder / "metadata.json"
        pdf_path = folder / "paper.pdf"

        if not meta_path.exists():
            continue
        if not pdf_path.exists():
            continue
        if pdf_path.stat().st_size <= 1000:
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # include only actual downloaded PDFs
        if clean_text(meta.get("download_status", "")).lower() != "downloaded":
            continue

        meta["pdf_path"] = str(pdf_path)
        meta["folder_name"] = folder.name
        meta["download_method"] = detect_download_method(meta)
        meta["n_gold_dois"] = load_n_gold_dois(meta)
        rows.append(meta)

    return rows

def build_topic_summary(rows: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        topic = clean_text(r.get("topic", "unknown")) or "unknown"
        out[topic] = out.get(topic, 0) + 1
    return dict(sorted(out.items(), key=lambda x: x[0]))

def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width))

# ============================================================
# REPORT BUILDERS
# ============================================================

def build_manuscript_paragraph(rows: List[Dict]) -> str:
    n = len(rows)
    topic_summary = build_topic_summary(rows)

    topic_parts = []
    for topic, count in topic_summary.items():
        topic_parts.append(f"{topic.replace('_', ' ')} (n={count})")
    topic_text = ", ".join(topic_parts)

    paragraph = (
        f"For pipeline evaluation, we constructed a diverse benchmark dataset of {n} "
        f"downloaded review articles with accessible full-text PDFs. The dataset was assembled "
        f"to cover multiple topical areas, including {topic_text}. For each article, we retained "
        f"the title, authors, year, journal, DOI, OpenAlex identifier, abstract, landing-page URL, "
        f"download status, and local PDF path. Only records with successfully downloaded PDFs were "
        f"included in the evaluation dataset to ensure that the downstream reproduction pipeline "
        f"could operate on complete source documents. The benchmark was designed to support testing "
        f"of retrieval, full-text processing, structured extraction, synthesis, and evaluation "
        f"components under a multi-domain setting."
    )
    return wrap(paragraph)

def build_methods_paragraph(rows: List[Dict]) -> str:
    n = len(rows)
    topic_summary = build_topic_summary(rows)
    topic_text = ", ".join(
        f"{topic.replace('_', ' ')} (n={count})"
        for topic, count in topic_summary.items()
    )
    cite_keys = [r["cite_key"] for r in rows]
    cite_block = ", ".join([f"\\cite{{{k}}}" for k in cite_keys])

    paragraph = (
        f"Evaluation articles were collected from open-access sources and retained only when a valid "
        f"PDF was successfully downloaded. The resulting evaluation dataset comprised {n} downloaded "
        f"review articles across {len(topic_summary)} topic groups: {topic_text}. Metadata for each "
        f"retained paper included topic, title, year, journal, DOI, OpenAlex identifier, OpenAlex "
        f"referenced-works list, authorship information, abstract, source, download method, and local "
        f"file path. This enabled stress-testing of the proposed literature-review reproduction "
        f"pipeline on heterogeneous inputs. The final benchmark articles were: {cite_block}."
    )
    return wrap(paragraph)

def build_markdown_table(rows: List[Dict]) -> str:
    header = "| Topic | Paper | Journal | Year | DOI | Gold DOIs | Cite |\n|---|---|---|---:|---|---:|---|"
    lines = [header]
    for r in rows:
        title = clean_text(r["title"])
        journal = clean_text(r["journal"])
        year = clean_text(r["year"])
        doi = clean_text(r["doi"])
        n_gold_dois = clean_text(r.get("n_gold_dois", 0))
        topic = clean_text(r["topic"]).replace("_", " ")
        cite_cmd = f"\\cite{{{r['cite_key']}}}"
        lines.append(f"| {topic} | {title} | {journal} | {year} | {doi} | {n_gold_dois} | {cite_cmd} |")
    return "\n".join(lines)

def build_latex_table(rows: List[Dict]) -> str:
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{p{2.0cm} p{5.8cm} p{2.5cm} p{0.8cm} p{3.0cm} p{0.9cm} p{1.5cm}}",
        r"\hline",
        r"\textbf{Topic} & \textbf{Paper} & \textbf{Journal} & \textbf{Year} & \textbf{DOI} & \textbf{Gold DOIs} & \textbf{Citation} \\",
        r"\hline",
    ]

    for r in rows:
        topic = latex_escape(clean_text(r["topic"]).replace("_", " "))
        title = latex_escape(clean_text(r["title"]))
        journal = latex_escape(clean_text(r["journal"]))
        year = latex_escape(clean_text(r["year"]))
        doi = latex_escape(clean_text(r["doi"]))
        n_gold_dois = latex_escape(clean_text(r.get("n_gold_dois", 0)))
        cite_cmd = rf"\cite{{{r['cite_key']}}}"

        lines.append(
            f"{topic} & {title} & {journal} & {year} & {doi} & {n_gold_dois} & {cite_cmd} \\\\"
        )

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Evaluation dataset of downloaded review articles used to test the literature-review reproduction pipeline. Only articles with successfully downloaded PDFs were retained.}",
        r"\label{tab:evaluation_dataset}",
        r"\end{table*}",
    ])
    return "\n".join(lines)

def build_download_log_table(rows: List[Dict]) -> str:
    header = "| Folder | Topic | Download method | PDF path |\n|---|---|---|---|"
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['folder_name']} | {clean_text(r['topic'])} | "
            f"{clean_text(r['download_method'])} | {clean_text(r['pdf_path'])} |"
        )
    return "\n".join(lines)

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    rows = load_downloaded_records(BASE_DIR)

    if not rows:
        print("No downloaded PDFs found. Nothing to save.")
        return

    # Add cite keys and bibtex
    used_keys = set()
    merged_bib = []

    for row in rows:
        key = make_cite_key(
            authors=clean_text(row.get("authors", "")),
            year=clean_text(row.get("year", "")),
            title=clean_text(row.get("title", "")),
        )

        original_key = key
        suffix = 1
        while key in used_keys:
            suffix += 1
            key = f"{original_key}{suffix}"
        used_keys.add(key)

        row["cite_key"] = key

        bibtex = try_crossref_bibtex(clean_text(row.get("doi", "")))
        if bibtex:
            bibtex = replace_bibtex_key(bibtex, key)
        else:
            bibtex = fallback_bibtex(row, key)

        row["bibtex"] = bibtex
        merged_bib.append(bibtex)

    # Sort for stable output
    rows = sorted(rows, key=lambda x: (clean_text(x.get("topic", "")), clean_text(x.get("year", "")), clean_text(x.get("title", ""))))

    # Save CSV
    csv_fields = [
        "topic",
        "title",
        "year",
        "journal",
        "doi",
        "openalex_id",
        "referenced_works",
        "n_gold_dois",
        "authors",
        "abstract",
        "type_crossref",
        "is_oa",
        "oa_pdf_url",
        "landing_page_url",
        "source",
        "download_status",
        "download_method",
        "pdf_path",
        "folder_name",
        "cite_key",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in csv_fields})

    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save merged BibTeX
    OUTPUT_BIB.write_text("\n\n".join(merged_bib), encoding="utf-8")

    # Save text report
    topic_summary = build_topic_summary(rows)
    topic_summary_text = "\n".join([f"- {k.replace('_', ' ')}: {v}" for k, v in topic_summary.items()])

    report_sections = [
        "============================================================",
        "EVALUATION DATASET REPORT",
        "============================================================",
        "",
        f"Total downloaded review articles included: {len(rows)}",
        "",
        "Topic distribution:",
        topic_summary_text,
        "",
        "MANUSCRIPT-READY DATASET PARAGRAPH",
        "------------------------------------------------------------",
        build_manuscript_paragraph(rows),
        "",
        "METHODS PARAGRAPH WITH CITATIONS",
        "------------------------------------------------------------",
        build_methods_paragraph(rows),
        "",
        "MARKDOWN TABLE",
        "------------------------------------------------------------",
        build_markdown_table(rows),
        "",
        "LATEX TABLE",
        "------------------------------------------------------------",
        build_latex_table(rows),
        "",
        "DOWNLOAD LOG TABLE",
        "------------------------------------------------------------",
        build_download_log_table(rows),
        "",
        "MERGED BIBTEX FILE",
        "------------------------------------------------------------",
        f"Saved separately to: {OUTPUT_BIB}",
        "",
        "ARTICLE DETAILS",
        "------------------------------------------------------------",
    ]

    for i, row in enumerate(rows, start=1):
        report_sections.extend([
            f"[{i}] {clean_text(row['title'])}",
            f"Topic           : {clean_text(row['topic'])}",
            f"Authors         : {clean_text(row['authors'])}",
            f"Year            : {clean_text(row['year'])}",
            f"Journal         : {clean_text(row['journal'])}",
            f"DOI             : {clean_text(row['doi'])}",
            f"OpenAlex ID     : {clean_text(row['openalex_id'])}",
            f"Gold DOIs       : {clean_text(row.get('n_gold_dois', 0))}",
            f"Source          : {clean_text(row['source'])}",
            f"Download method : {clean_text(row['download_method'])}",
            f"Landing page    : {clean_text(row['landing_page_url'])}",
            f"PDF path        : {clean_text(row['pdf_path'])}",
            f"Citation key    : {clean_text(row['cite_key'])}",
            "Abstract:",
            wrap(clean_text(row["abstract"]), 100),
            "",
        ])

    OUTPUT_TXT.write_text("\n".join(report_sections), encoding="utf-8")

    print("Saved:")
    print(f"  CSV   : {OUTPUT_CSV}")
    print(f"  TXT   : {OUTPUT_TXT}")
    print(f"  BIB   : {OUTPUT_BIB}")
    print(f"  JSON  : {OUTPUT_JSON}")
    print(f"  Total : {len(rows)} downloaded PDF articles included")

if __name__ == "__main__":
    main()
