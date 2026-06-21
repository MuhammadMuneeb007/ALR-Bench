#!/usr/bin/env python3
"""
================================================================================
  Step 5: Article Text Extractor / Full-Text Converter
  PROJECT-AWARE + INDEX-AWARE VERSION

  PURPOSE
  -------
  Run this AFTER the PDF download step.

  This script scans the database folder, finds article folders containing PDFs,
  validates each PDF, then extracts:

      1) PDF main text
      2) embedded figures/images
      3) OCR text from figures/images
      4) combined Article_Text.txt / COMBINED.txt

  NEW INDEX MODE
  --------------
  Convert one article by Temp index:

      python Step5-FullTextConversion.py --index 25

  Convert one article by Temp index inside a specific project:

      python Step5-FullTextConversion.py --project project_folder_name --index 25

  Convert one article by global discovered index:

      python Step5-FullTextConversion.py --global-index 10

  Show available indices:

      python Step5-FullTextConversion.py --list-indices

  Convert a Temp index range:

      python Step5-FullTextConversion.py --start-index 0 --end-index 100

  Force reprocess selected articles:

      python Step5-FullTextConversion.py --index 25 --reset

  Change DB folder:

      python Step5-FullTextConversion.py --db DataBase_Strategy3_Top1000 --index 25

  OUTPUT PER ARTICLE FOLDER
  -------------------------
      full_text.txt
      figures/
      figures_ocr.txt
      COMBINED.txt
      Article_Text.txt
      extraction_metadata.json

  GLOBAL OUTPUT
  -------------
      DataBase/extraction_summary.csv
      DataBase/pdf_validation_summary.csv
================================================================================
"""

from __future__ import annotations

import os
import sys
import re
import json
import csv
import time
import logging
import threading
import traceback
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DB_DIR = "DataBase_Strategy3_Top1000"

_RESET = False

MAX_WORKERS = 8
SAVE_EVERY = 25

# PDF validation
VALIDATE_WITH_PYPDF = True
REQUIRE_PYPDF_VALIDATION = True
QUARANTINE_BAD_PDFS = False
MIN_PDF_SIZE = 4

# Tesseract
TESSERACT_CMD = ""
TESSERACT_LANG = "eng"
TESSERACT_PSM = 6

# Image filters
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Text quality
MIN_USEFUL_WORDS = 50

# Thread-safe progress
_progress_lock = threading.Lock()
_progress_done = 0

# Dependency status
_PYPDF_AVAILABLE = False
_PYMUPDF_AVAILABLE = False
_TESSERACT_AVAILABLE = None

# PDF validation rows
_pdf_validation_rows = []
_pdf_validation_lock = threading.Lock()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger("Step5FullTextConverter")


# ============================================================
# BASIC UTILITY
# ============================================================

def clean(value) -> str:
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "", "n/a") else s


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_references(text: str) -> str:
    """
    Remove reference section from extracted main text.
    This prevents references from dominating downstream full-text QA/ranking.
    """
    if not text:
        return ""

    patterns = [
        r"\n\s*References\s*\n",
        r"\n\s*Bibliography\s*\n",
        r"\n\s*Works Cited\s*\n",
        r"\n\s*Literature Cited\s*\n",
    ]

    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            # Use the last/late occurrence only if it appears after some article body.
            for m in matches:
                if m.start() > len(text) * 0.35:
                    return text[:m.start()].strip()

    return text.strip()


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


def _read_header(path: Path, n: int = 20) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except Exception:
        return b""


def _temp_index_from_folder(folder: Path) -> int | None:
    m = re.match(r"^Temp_(\d+)$", folder.name)
    if not m:
        return None
    return int(m.group(1))


# ============================================================
# DEPENDENCY CHECKS
# ============================================================

def check_dependencies() -> dict:
    global _PYPDF_AVAILABLE, _PYMUPDF_AVAILABLE

    status = {}

    try:
        import pypdf
        from pypdf import PdfReader  # noqa

        _PYPDF_AVAILABLE = True
        status["pypdf"] = {
            "ok": True,
            "version": getattr(pypdf, "__version__", "unknown"),
            "error": "",
        }
        log.info(f"    ✓ pypdf ({status['pypdf']['version']})")

    except Exception as e:
        _PYPDF_AVAILABLE = False
        status["pypdf"] = {
            "ok": False,
            "version": "",
            "error": str(e),
        }
        log.warning(f"    ✗ pypdf not installed/working: {e}")

    try:
        import fitz  # noqa

        _PYMUPDF_AVAILABLE = True
        status["fitz"] = {
            "ok": True,
            "version": "installed",
            "error": "",
        }
        log.info("    ✓ PyMuPDF / fitz")

    except Exception as e:
        _PYMUPDF_AVAILABLE = False
        status["fitz"] = {
            "ok": False,
            "version": "",
            "error": str(e),
        }
        log.warning(f"    ✗ PyMuPDF / fitz not installed: {e}")

    try:
        import pytesseract  # noqa

        status["pytesseract"] = {
            "ok": True,
            "version": "installed",
            "error": "",
        }
        log.info("    ✓ pytesseract")

    except Exception as e:
        status["pytesseract"] = {
            "ok": False,
            "version": "",
            "error": str(e),
        }
        log.warning(f"    ✗ pytesseract not installed: {e}")

    try:
        import PIL  # noqa

        status["PIL"] = {
            "ok": True,
            "version": "installed",
            "error": "",
        }
        log.info("    ✓ Pillow")

    except Exception as e:
        status["PIL"] = {
            "ok": False,
            "version": "",
            "error": str(e),
        }
        log.warning(f"    ✗ Pillow not installed: {e}")

    return status


def _check_tesseract() -> bool:
    global _TESSERACT_AVAILABLE

    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE

    try:
        import pytesseract

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        version = pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        log.info(f"    Tesseract ready ({version})")

    except ImportError:
        log.warning("    pytesseract not installed")
        _TESSERACT_AVAILABLE = False

    except Exception as e:
        log.warning(f"    Tesseract not available: {e}")
        _TESSERACT_AVAILABLE = False

    return _TESSERACT_AVAILABLE


# ============================================================
# PDF VALIDATION
# ============================================================

def validate_pdf_with_pypdf(pdf_path: Path) -> dict:
    result = {
        "pdf_path": str(pdf_path),
        "exists": False,
        "size_bytes": 0,
        "header": "",
        "header_ok": False,
        "pypdf_available": _PYPDF_AVAILABLE,
        "pypdf_ok": False,
        "page_count": 0,
        "valid": False,
        "error": "",
    }

    try:
        if not pdf_path.exists():
            result["error"] = "file does not exist"
            return result

        result["exists"] = True
        result["size_bytes"] = _safe_file_size(pdf_path)

        if result["size_bytes"] < MIN_PDF_SIZE:
            result["error"] = f"file too small: {result['size_bytes']} bytes"
            return result

        header = _read_header(pdf_path, 20)
        result["header"] = repr(header)
        result["header_ok"] = header.startswith(b"%PDF")

        if not result["header_ok"]:
            result["error"] = f"not a PDF header: {repr(header[:20])}"
            return result

        if not VALIDATE_WITH_PYPDF:
            result["valid"] = True
            return result

        if not _PYPDF_AVAILABLE:
            if REQUIRE_PYPDF_VALIDATION:
                result["error"] = "pypdf not available"
                return result

            result["valid"] = True
            return result

        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)

        result["page_count"] = page_count

        if page_count <= 0:
            result["error"] = "pypdf opened file but found 0 pages"
            return result

        _ = reader.pages[0]

        result["pypdf_ok"] = True
        result["valid"] = True

        return result

    except Exception as e:
        result["error"] = repr(e)
        return result


def quarantine_bad_pdf(pdf_path: Path, reason: str) -> Path | None:
    if not pdf_path.exists():
        return None

    bad_path = pdf_path.with_name(
        f"{pdf_path.stem}.bad_pdf_validation_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"{pdf_path.suffix}"
    )

    try:
        pdf_path.rename(bad_path)
        log.warning(f"    Bad PDF moved aside: {bad_path} ({reason})")
        return bad_path

    except Exception as e:
        log.warning(f"    Could not quarantine bad PDF: {pdf_path} ({e})")
        return None


def record_pdf_validation(
    project_name: str,
    folder: Path,
    pdf_path: Path,
    validation: dict,
):
    row = dict(validation)
    row.update({
        "project_name": project_name,
        "article_folder": folder.name,
        "temp_index": _temp_index_from_folder(folder),
    })

    with _pdf_validation_lock:
        _pdf_validation_rows.append(row)


def save_pdf_validation_summary(db_dir: str):
    out_csv = Path(db_dir) / "pdf_validation_summary.csv"

    fields = [
        "project_name",
        "article_folder",
        "temp_index",
        "pdf_path",
        "exists",
        "size_bytes",
        "header",
        "header_ok",
        "pypdf_available",
        "pypdf_ok",
        "page_count",
        "valid",
        "error",
    ]

    try:
        rows = list(_pdf_validation_rows)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fields})

        log.info(f"  Saved PDF validation summary: {out_csv}")

    except Exception as e:
        log.error(f"  Could not save PDF validation summary: {e}")


# ============================================================
# DISCOVERY
# ============================================================

def _find_pdf(folder: Path) -> Path | None:
    """
    Find a likely PDF in one article folder.

    Priority:
      1) document.pdf
      2) any .pdf with %PDF header
    """
    doc_pdf = folder / "document.pdf"

    if doc_pdf.exists():
        header = _read_header(doc_pdf, 4)
        if header == b"%PDF":
            return doc_pdf

    try:
        files = sorted(folder.iterdir(), key=lambda x: x.name.lower())
    except Exception:
        return None

    for f in files:
        if f.is_file() and f.suffix.lower() == ".pdf":
            header = _read_header(f, 4)
            if header == b"%PDF":
                return f

    return None


def _discover_article_folders(db_path: Path) -> list[tuple[Path, Path, str]]:
    """
    Returns list of:
        (article_folder, pdf_path, project_name)

    Supports:
        DataBase/<project>/Temp_x/
        DataBase/Temp_x/
    """
    found = []

    if not db_path.exists():
        return found

    for item in sorted(db_path.iterdir(), key=natural_sort_key):
        if not item.is_dir():
            continue

        if item.name.startswith("_"):
            continue

        # Legacy single-level:
        # DataBase/Temp_0/document.pdf
        if re.match(r"^Temp_\d+$", item.name):
            pdf = _find_pdf(item)
            if pdf:
                found.append((item, pdf, "legacy"))
            continue

        # Project-aware:
        # DataBase/<project>/Temp_0/document.pdf
        try:
            subs = sorted(item.iterdir(), key=natural_sort_key)
        except Exception:
            continue

        for sub in subs:
            if sub.is_dir() and re.match(r"^Temp_\d+$", sub.name):
                pdf = _find_pdf(sub)
                if pdf:
                    found.append((sub, pdf, item.name))

    return found


def _filter_papers_by_index(
    papers: list[tuple[Path, Path, str]],
    project: str = "",
    index: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    global_index: int | None = None,
) -> list[tuple[Path, Path, str]]:
    """
    --index refers to Temp_X index.
    --global-index refers to sorted discovered position, 1-based.
    """
    selected = list(papers)

    project = clean(project)

    if project:
        selected = [
            x for x in selected
            if x[2] == project
            or project.lower() in x[2].lower()
        ]

    if global_index is not None:
        gi = int(global_index)

        if gi < 1 or gi > len(selected):
            log.error(f"  --global-index {gi} is outside valid range 1..{len(selected)}")
            return []

        return [selected[gi - 1]]

    if index is not None:
        ti = int(index)
        return [
            x for x in selected
            if _temp_index_from_folder(x[0]) == ti
        ]

    if start_index is not None or end_index is not None:
        si = 0 if start_index is None else int(start_index)
        ei = 10**18 if end_index is None else int(end_index)

        return [
            x for x in selected
            if _temp_index_from_folder(x[0]) is not None
            and si <= _temp_index_from_folder(x[0]) <= ei
        ]

    return selected


def _print_index_preview(
    papers: list[tuple[Path, Path, str]],
    max_rows: int = 300,
):
    log.info("")
    log.info("=" * 120)
    log.info("DISCOVERED PDF ARTICLES")
    log.info("=" * 120)
    log.info(
        f"{'GLOBAL':>8}  "
        f"{'TEMP_IDX':>8}  "
        f"{'PROJECT':<55}  "
        f"{'FOLDER':<15}  "
        f"PDF"
    )
    log.info("-" * 120)

    for global_i, (folder, pdf_path, project_name) in enumerate(papers, 1):
        temp_i = _temp_index_from_folder(folder)
        temp_s = "" if temp_i is None else str(temp_i)

        log.info(
            f"{global_i:>8}  "
            f"{temp_s:>8}  "
            f"{project_name[:55]:<55}  "
            f"{folder.name:<15}  "
            f"{pdf_path.name}"
        )

        if global_i >= max_rows and len(papers) > max_rows:
            log.info(f"... showing first {max_rows}/{len(papers)} only")
            break

    log.info("=" * 120)
    log.info("")


# ============================================================
# PDF TEXT EXTRACTION — PyMuPDF
# ============================================================

def extract_text_pymupdf(pdf_path: Path, out_folder: Path) -> str:
    import fitz

    txt_out = out_folder / "full_text.txt"

    if txt_out.exists() and txt_out.stat().st_size > 100 and not _RESET:
        log.info("      full_text.txt already exists, loading...")
        return txt_out.read_text(encoding="utf-8", errors="ignore")

    log.info("    [TEXT] PyMuPDF text extraction...")

    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []

        for page_i, page in enumerate(doc, 1):
            try:
                page_text = page.get_text("text")
            except Exception as e:
                log.debug(f"      Page {page_i} text error: {e}")
                page_text = ""

            if page_text and page_text.strip():
                pages_text.append(page_text)

        doc.close()

        full_text = "\n\n".join(pages_text)
        full_text = _clean_text(full_text)

        word_count = len(full_text.split())

        if word_count < MIN_USEFUL_WORDS:
            log.warning(f"      Only {word_count} words extracted, possibly scanned PDF")

        txt_out.write_text(full_text, encoding="utf-8")

        log.info(
            f"      Saved full_text.txt "
            f"({len(full_text):,} chars / {word_count:,} words)"
        )

        return full_text

    except Exception as e:
        log.error(f"      PyMuPDF text error: {e}")
        log.debug(traceback.format_exc())
        return ""


# ============================================================
# FIGURE EXTRACTION — PyMuPDF
# ============================================================

def extract_figures(pdf_path: Path, out_folder: Path) -> list[dict]:
    import fitz

    log.info("    [FIGS] Extracting figures...")

    figures_dir = out_folder / "figures"
    figures_dir.mkdir(exist_ok=True)

    extracted = []

    try:
        doc = fitz.open(str(pdf_path))
        total = 0
        skipped = 0

        for page_num, page in enumerate(doc):
            try:
                images = page.get_images(full=True)
            except Exception as e:
                log.debug(f"      Could not get images on page {page_num + 1}: {e}")
                continue

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)

                    img_bytes = base_image["image"]
                    img_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        skipped += 1
                        continue

                    fname = f"fig_p{page_num + 1:03d}_img{img_idx:02d}.{img_ext}"
                    fpath = figures_dir / fname

                    if fpath.exists() and fpath.stat().st_size > 0 and not _RESET:
                        extracted.append({
                            "path": fpath,
                            "page": page_num + 1,
                            "width": width,
                            "height": height,
                            "fname": fname,
                        })
                        total += 1
                        continue

                    with open(fpath, "wb") as f:
                        f.write(img_bytes)

                    extracted.append({
                        "path": fpath,
                        "page": page_num + 1,
                        "width": width,
                        "height": height,
                        "fname": fname,
                    })

                    total += 1

                except Exception as img_err:
                    log.debug(f"      xref {xref} error: {img_err}")

        doc.close()

        log.info(f"      {total} figures extracted/available ({skipped} tiny images skipped)")

    except Exception as e:
        log.error(f"      Figure extraction error: {e}")
        log.debug(traceback.format_exc())

    return extracted


# ============================================================
# OCR — Tesseract
# ============================================================

def _ocr_tesseract(image_path: Path) -> str:
    if not _check_tesseract():
        return ""

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    try:
        img = Image.open(str(image_path)).convert("L")
        config = f"--oem 3 --psm {TESSERACT_PSM} -l {TESSERACT_LANG}"

        text = pytesseract.image_to_string(img, config=config).strip()

        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    except Exception as e:
        log.debug(f"        Tesseract error on {image_path.name}: {e}")
        return ""


def extract_figures_ocr(figures: list[dict], out_folder: Path) -> str:
    ocr_out = out_folder / "figures_ocr.txt"

    if ocr_out.exists() and ocr_out.stat().st_size > 50 and not _RESET:
        log.info("    [OCR] figures_ocr.txt already exists, loading...")
        return ocr_out.read_text(encoding="utf-8", errors="ignore")

    if not figures:
        log.info("    [OCR] No figures to OCR.")
        ocr_out.write_text("", encoding="utf-8")
        return ""

    log.info(f"    [OCR] Tesseract on {len(figures)} figures...")

    ocr_lines = []
    success = 0

    for fig in figures:
        text = _ocr_tesseract(fig["path"])

        ocr_lines.append(
            f"-- {fig['fname']} "
            f"(p{fig['page']}, {fig['width']}x{fig['height']}px) --"
        )

        if text:
            ocr_lines.append(text[:5000])
            success += 1
        else:
            ocr_lines.append("[no text detected]")

        ocr_lines.append("")

    ocr_full = "\n".join(ocr_lines).strip()
    ocr_out.write_text(ocr_full, encoding="utf-8")

    log.info(f"      OCR done — {success}/{len(figures)} figures had text")

    return ocr_full


# ============================================================
# COMBINE OUTPUTS
# ============================================================

def combine_outputs(
    out_folder: Path,
    main_text: str,
    ocr_text: str,
    folder_name: str,
    pdf_path: Path,
) -> str:
    combined_out = out_folder / "COMBINED.txt"
    article_out = out_folder / "Article_Text.txt"

    metadata = _safe_read_json(out_folder / "metadata.json")
    title = clean(metadata.get("title", ""))
    doi = clean(metadata.get("doi", ""))

    sections = []

    sections.append("=" * 80)
    sections.append(f"ARTICLE FOLDER: {folder_name}")
    sections.append(f"PDF FILE      : {pdf_path.name}")
    if title:
        sections.append(f"TITLE         : {title}")
    if doi:
        sections.append(f"DOI           : {doi}")
    sections.append(f"EXTRACTED AT  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append("=" * 80)
    sections.append("")

    sections.append("MAIN PDF TEXT")
    sections.append("-" * 80)

    if main_text:
        sections.append(_remove_references(main_text))
    else:
        sections.append("[Text extraction failed or empty PDF text]")

    if ocr_text:
        sections.append("")
        sections.append("=" * 80)
        sections.append("FIGURE OCR TEXT")
        sections.append("-" * 80)
        sections.append(ocr_text)

    full = "\n".join(sections).strip() + "\n"

    combined_out.write_text(full, encoding="utf-8")
    article_out.write_text(full, encoding="utf-8")

    return full


# ============================================================
# METADATA
# ============================================================

def write_extraction_metadata(
    paper_folder: Path,
    project_name: str,
    pdf_path: Path,
    result: dict,
    pdf_validation: dict | None = None,
) -> None:
    meta_path = paper_folder / "extraction_metadata.json"
    existing = _safe_read_json(paper_folder / "metadata.json")

    payload = {
        "project_name": project_name,
        "article_folder": paper_folder.name,
        "temp_index": _temp_index_from_folder(paper_folder),
        "pdf_file": pdf_path.name,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "status": result.get("status", ""),
        "text_chars": result.get("text_chars", 0),
        "figures": result.get("figures", 0),
        "ocr_figures": result.get("ocr_figures", 0),
        "pdf_valid": result.get("pdf_valid", False),
        "pdf_pages": result.get("pdf_pages", 0),
        "error": result.get("error", ""),
        "pdf_validation": pdf_validation or {},
        "base_metadata": existing,
    }

    meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# PROCESS ONE PAPER
# ============================================================

def process_paper(
    paper_folder: Path,
    pdf_path: Path,
    project_name: str,
) -> dict:
    folder_name = paper_folder.name

    result = {
        "project_name": project_name,
        "folder": folder_name,
        "temp_index": _temp_index_from_folder(paper_folder),
        "pdf": pdf_path.name,
        "status": "FAILED",
        "text_chars": 0,
        "figures": 0,
        "ocr_figures": 0,
        "error": "",
        "pdf_valid": False,
        "pdf_pages": 0,
    }

    validation = validate_pdf_with_pypdf(pdf_path)

    result["pdf_valid"] = bool(validation.get("valid", False))
    result["pdf_pages"] = int(validation.get("page_count", 0) or 0)

    record_pdf_validation(project_name, paper_folder, pdf_path, validation)

    if not validation["valid"]:
        result["status"] = "BAD_PDF"
        result["error"] = validation.get("error", "invalid PDF")

        log.warning(
            f"    BAD PDF: {project_name}/{folder_name}/{pdf_path.name} | "
            f"{result['error']}"
        )

        if QUARANTINE_BAD_PDFS:
            quarantine_bad_pdf(pdf_path, result["error"])

        write_extraction_metadata(
            paper_folder,
            project_name,
            pdf_path,
            result,
            pdf_validation=validation,
        )

        return result

    combined = paper_folder / "COMBINED.txt"

    if not _RESET and combined.exists() and combined.stat().st_size > 500:
        log.info("    Already done: COMBINED.txt exists, skipping")

        result["status"] = "Already done"
        result["text_chars"] = combined.stat().st_size

        write_extraction_metadata(
            paper_folder,
            project_name,
            pdf_path,
            result,
            pdf_validation=validation,
        )

        return result

    try:
        main_text = extract_text_pymupdf(pdf_path, paper_folder)

        result["text_chars"] = len(main_text)

        figures = extract_figures(pdf_path, paper_folder)

        result["figures"] = len(figures)

        ocr_text = extract_figures_ocr(figures, paper_folder)

        # Count number of figure blocks that had OCR text.
        ocr_figures = 0
        if ocr_text:
            blocks = re.split(r"\n-- ", "\n" + ocr_text)
            for block in blocks:
                if not block.strip():
                    continue
                if "[no text detected]" not in block:
                    ocr_figures += 1

        result["ocr_figures"] = ocr_figures

        combined_text = combine_outputs(
            paper_folder,
            main_text,
            ocr_text,
            folder_name,
            pdf_path,
        )

        result["status"] = "Done" if combined_text.strip() else "Empty"

        write_extraction_metadata(
            paper_folder,
            project_name,
            pdf_path,
            result,
            pdf_validation=validation,
        )

        log.info(
            f"    ✓ {project_name}/{folder_name} — "
            f"{result['text_chars']:,} chars, "
            f"{result['figures']} figs, "
            f"{result['pdf_pages']} pages"
        )

    except KeyboardInterrupt:
        raise

    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)

        write_extraction_metadata(
            paper_folder,
            project_name,
            pdf_path,
            result,
            pdf_validation=validation,
        )

        log.error(f"    ✗ {project_name}/{folder_name}: {e}")
        log.debug(traceback.format_exc())

    return result


# ============================================================
# SUMMARY
# ============================================================

def _print_and_save_summary(results: list[dict], db_dir: str):
    summary_csv = Path(db_dir) / "extraction_summary.csv"

    fields = [
        "project_name",
        "folder",
        "temp_index",
        "pdf",
        "status",
        "pdf_valid",
        "pdf_pages",
        "text_chars",
        "figures",
        "ocr_figures",
        "error",
    ]

    total = len(results)
    done = sum(1 for r in results if r.get("status") in ("Done", "Already done"))
    bad_pdf = sum(1 for r in results if r.get("status") == "BAD_PDF")
    failed = sum(1 for r in results if r.get("status") == "FAILED")
    empty = sum(1 for r in results if r.get("status") == "Empty")

    try:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for r in sorted(
                results,
                key=lambda x: (
                    clean(x.get("project_name", "")),
                    int(x.get("temp_index") or -1),
                    clean(x.get("folder", "")),
                ),
            ):
                writer.writerow({k: r.get(k, "") for k in fields})

        log.info("")
        log.info("=" * 80)
        log.info("EXTRACTION SUMMARY")
        log.info("=" * 80)
        log.info(f"  Total processed : {total}")
        log.info(f"  Done/skipped    : {done}")
        log.info(f"  Bad PDFs        : {bad_pdf}")
        log.info(f"  Empty           : {empty}")
        log.info(f"  Failed          : {failed}")
        log.info(f"  Summary CSV     : {summary_csv}")
        log.info("=" * 80)

    except Exception as e:
        log.error(f"Could not save extraction summary: {e}")


# ============================================================
# MAIN RUNNER
# ============================================================

def run(
    db_dir: str,
    reset: bool = False,
    list_indices: bool = False,
    project: str = "",
    index: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    global_index: int | None = None,
    workers: int | None = None,
):
    global DB_DIR, _RESET, _progress_done, _pdf_validation_rows

    DB_DIR = db_dir
    _RESET = bool(reset)
    _progress_done = 0
    _pdf_validation_rows = []

    db_path = Path(db_dir)

    if not db_path.exists():
        log.error(f"Database directory not found: {db_dir}")
        return

    log.info("")
    log.info("=" * 80)
    log.info("STEP 5: FULL-TEXT CONVERSION")
    log.info("=" * 80)
    log.info(f"Database dir : {db_path.resolve()}")
    log.info(f"Reset        : {_RESET}")
    log.info("")

    log.info("Dependency checks:")
    deps = check_dependencies()

    if not deps.get("fitz", {}).get("ok"):
        log.error("")
        log.error("PyMuPDF / fitz is required.")
        log.error("Install:")
        log.error("    pip install -U pymupdf")
        return

    if REQUIRE_PYPDF_VALIDATION and not deps.get("pypdf", {}).get("ok"):
        log.error("")
        log.error("pypdf validation is enabled, but pypdf is missing.")
        log.error("Install:")
        log.error("    pip install -U pypdf")
        return

    all_papers = _discover_article_folders(db_path)

    if list_indices:
        _print_index_preview(all_papers)
        return

    papers = _filter_papers_by_index(
        papers=all_papers,
        project=project,
        index=index,
        start_index=start_index,
        end_index=end_index,
        global_index=global_index,
    )

    log.info("")
    log.info(f"Total article folders with PDFs found : {len(all_papers)}")
    log.info(f"Selected article folders to process  : {len(papers)}")

    if project:
        log.info(f"Project filter                       : {project}")

    if index is not None:
        log.info(f"Temp index filter                    : Temp_{index}")

    if global_index is not None:
        log.info(f"Global index filter                  : {global_index}")

    if start_index is not None or end_index is not None:
        log.info(
            f"Temp index range                     : "
            f"{start_index if start_index is not None else 0} to "
            f"{end_index if end_index is not None else 'END'}"
        )

    log.info("")

    if len(all_papers) == 0:
        log.warning("No PDFs found. Run the download step first.")
        return

    if len(papers) == 0:
        log.warning("No article matched the requested filter/index.")
        log.warning("Use this command to inspect available indices:")
        log.warning(f"    python {Path(sys.argv[0]).name} --db {db_dir} --list-indices")
        return

    already_done = 0

    if not _RESET:
        already_done = sum(
            1 for folder, _, _ in papers
            if (folder / "COMBINED.txt").exists()
            and (folder / "COMBINED.txt").stat().st_size > 500
        )

    n_pending = len(papers) - already_done

    log.info(f"Already done : {already_done}")
    log.info(f"Pending      : {n_pending}")

    if _RESET:
        log.info("Reset mode   : selected articles will be reprocessed")

    log.info("")

    if n_pending == 0:
        log.info("Selected article(s) already extracted. Done.")
        return

    _check_tesseract()

    single_article_mode = (
        index is not None
        or global_index is not None
        or len(papers) == 1
    )

    if workers is None:
        selected_workers = 1 if single_article_mode else min(MAX_WORKERS, max(1, n_pending))
    else:
        selected_workers = max(1, int(workers))

    if single_article_mode:
        selected_workers = 1

    log.info(f"Using worker threads: {selected_workers}")
    log.info("")

    results = []
    results_lock = threading.Lock()
    start_time = time.time()
    interrupted = False

    def _worker(
        paper_folder: Path,
        pdf_path: Path,
        project_name: str,
        order_idx: int,
    ) -> dict:
        global _progress_done

        temp_idx = _temp_index_from_folder(paper_folder)
        temp_label = f"Temp_{temp_idx}" if temp_idx is not None else paper_folder.name

        log.info(
            f"\n[{order_idx}/{len(papers)}] "
            f"{project_name}/{temp_label}/{pdf_path.name}"
        )

        try:
            result = process_paper(paper_folder, pdf_path, project_name)

        except Exception as e:
            result = {
                "project_name": project_name,
                "folder": paper_folder.name,
                "temp_index": temp_idx,
                "pdf": pdf_path.name,
                "status": "FAILED",
                "pdf_valid": False,
                "pdf_pages": 0,
                "text_chars": 0,
                "figures": 0,
                "ocr_figures": 0,
                "error": str(e),
            }

            log.error(f"{project_name}/{paper_folder.name} failed: {e}")

        with _progress_lock:
            _progress_done += 1

        return result

    try:
        if selected_workers == 1:
            for order_idx, (folder, pdf_path, project_name) in enumerate(papers, 1):
                result = _worker(folder, pdf_path, project_name, order_idx)

                with results_lock:
                    results.append(result)

                    if len(results) % SAVE_EVERY == 0:
                        _print_and_save_summary(results, db_dir)
                        save_pdf_validation_summary(db_dir)

        else:
            with ThreadPoolExecutor(
                max_workers=selected_workers,
                thread_name_prefix="step5_fulltext",
            ) as pool:
                future_map = {}

                for order_idx, (folder, pdf_path, project_name) in enumerate(papers, 1):
                    fut = pool.submit(
                        _worker,
                        folder,
                        pdf_path,
                        project_name,
                        order_idx,
                    )
                    future_map[fut] = (project_name, folder.name, pdf_path.name)

                for fut in as_completed(future_map):
                    try:
                        result = fut.result()

                    except Exception as e:
                        project_name, folder_name, pdf_name = future_map[fut]

                        result = {
                            "project_name": project_name,
                            "folder": folder_name,
                            "temp_index": None,
                            "pdf": pdf_name,
                            "status": "FAILED",
                            "pdf_valid": False,
                            "pdf_pages": 0,
                            "text_chars": 0,
                            "figures": 0,
                            "ocr_figures": 0,
                            "error": str(e),
                        }

                        log.error(f"Thread error [{project_name}/{folder_name}]: {e}")

                    with results_lock:
                        results.append(result)

                        if len(results) % SAVE_EVERY == 0:
                            _print_and_save_summary(results, db_dir)
                            save_pdf_validation_summary(db_dir)

    except KeyboardInterrupt:
        log.warning("")
        log.warning("Interrupted. Saving progress...")
        interrupted = True

    elapsed = time.time() - start_time
    rate = len(results) / elapsed if elapsed > 0 else 0

    log.info("")
    log.info(f"Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    log.info(f"Rate: {rate:.2f} articles/sec")

    _print_and_save_summary(results, db_dir)
    save_pdf_validation_summary(db_dir)

    if interrupted:
        remaining = len(papers) - len(results)
        log.warning(f"Interrupted — {remaining} remaining. Re-run to continue.")
        raise KeyboardInterrupt("Step 5 interrupted")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 5 full-text conversion with index-based processing."
    )

    parser.add_argument(
        "--db",
        default=DB_DIR,
        help=f"Database directory. Default: {DB_DIR}",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Force reprocess selected article(s).",
    )

    parser.add_argument(
        "--list-indices",
        action="store_true",
        help="List discovered global indices and Temp indices, then exit.",
    )

    parser.add_argument(
        "--project",
        default="",
        help="Project folder name or substring. Use with --index to select one project/article.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Process specific Temp_X index, e.g. --index 25 means Temp_25.",
    )

    parser.add_argument(
        "--global-index",
        type=int,
        default=None,
        help="Process specific discovered global index, 1-based.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start Temp index for range processing.",
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End Temp index for range processing, inclusive.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads. Single-index mode always uses 1.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Prevent confusing combinations.
    selected_modes = sum([
        args.index is not None,
        args.global_index is not None,
        args.start_index is not None or args.end_index is not None,
    ])

    if selected_modes > 1:
        log.error(
            "Use only one selection mode at a time: "
            "--index OR --global-index OR --start-index/--end-index"
        )
        sys.exit(1)

    run(
        db_dir=args.db,
        reset=args.reset,
        list_indices=args.list_indices,
        project=args.project,
        index=args.index,
        start_index=args.start_index,
        end_index=args.end_index,
        global_index=args.global_index,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()