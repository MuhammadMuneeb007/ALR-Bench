# Step3-PrepareProjectsFromDownloadedArticles.py
from __future__ import annotations

import json
import re
import shutil
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = Path("benchmark_reviews")
PROJECTS_DIR = Path("Projects")

# Process articles one by one, in order.
PROCESS_ONE_BY_ONE = True

# If True, overwrite existing project files
OVERWRITE = True

# Minimum image size to save/OCR
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# OCR settings
DO_IMAGE_OCR = True
OCR_LANG = "eng"

# Windows example:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = ""

# For figure/diagram text, sparse text mode is usually better.
# If results are bad, try "--psm 6"
TESSERACT_CONFIG = "--psm 11"

MIN_OCR_CHARS = 3


# ============================================================
# OPTIONAL OCR IMPORTS
# ============================================================

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# ============================================================
# HELPERS
# ============================================================

def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", text.strip().lower())
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:max_len].strip("_") or "untitled_project"


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def find_downloaded_articles(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Returns [(metadata.json path, paper.pdf path), ...]
    only for folders with a real downloaded PDF.
    """
    items = []

    if not source_dir.exists():
        return items

    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir():
            continue

        meta_path = folder / "metadata.json"
        pdf_path = folder / "paper.pdf"

        if not meta_path.exists() or not pdf_path.exists():
            continue

        try:
            meta = read_json(meta_path)
        except Exception:
            continue

        if str(meta.get("download_status", "")).lower() != "downloaded":
            continue

        if pdf_path.stat().st_size < 1000:
            continue

        items.append((meta_path, pdf_path))

    return items


# ============================================================
# OCR HELPERS
# ============================================================

def setup_tesseract() -> None:
    if pytesseract is None:
        return

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def ocr_image_file(image_path: Path) -> Tuple[str, str]:
    """
    OCR one extracted image.

    Returns:
        ocr_text, ocr_status
    """
    if not DO_IMAGE_OCR:
        return "", "image_ocr_disabled"

    if pytesseract is None:
        return "", "pytesseract_not_installed"

    if Image is None:
        return "", "pillow_not_installed"

    try:
        setup_tesseract()

        img = Image.open(image_path)

        text = pytesseract.image_to_string(
            img,
            lang=OCR_LANG,
            config=TESSERACT_CONFIG,
        )

        text = clean_text(text)

        if len(text) < MIN_OCR_CHARS:
            return "", "empty_or_too_short"

        return text, "ok"

    except Exception as e:
        return "", f"ocr_failed: {type(e).__name__}: {e}"


def build_image_ocr_markdown(images: List[Dict]) -> str:
    """
    Build Markdown block containing OCR text extracted from images.

    This is inserted into article_markdown.md directly.
    """
    blocks = []

    for img in images:
        ocr_text = clean_text(img.get("ocr_text", ""))

        if not ocr_text:
            continue

        page = img.get("page", "")
        image_index = img.get("image_index_on_page", "")
        file_name = img.get("file_name", "")
        width = img.get("width", "")
        height = img.get("height", "")

        block = (
            f"### Image OCR: page {page}, image {image_index}\n\n"
            f"**Image file:** `{file_name}`  \n"
            f"**Size:** {width} x {height}\n\n"
            f"{ocr_text}"
        )

        blocks.append(block)

    if not blocks:
        return ""

    return clean_text(
        "## Image / Figure Internal OCR Text\n\n"
        + "\n\n".join(blocks)
    )


def inject_image_ocr_into_markdown(markdown_text: str, images: List[Dict]) -> str:
    """
    Insert image OCR into the SAME Markdown output.

    No new markdown file is created.
    article_markdown.md becomes:
        Docling markdown
        +
        Image/Figure OCR text
    """
    image_ocr_md = build_image_ocr_markdown(images)

    if not image_ocr_md:
        return clean_text(markdown_text)

    return clean_text(markdown_text + "\n\n\n" + image_ocr_md)


# ============================================================
# PDF → MARKDOWN / TEXT
# ============================================================

_DOCLING_CONVERTER = None


def get_docling_converter():
    """
    Lazily build one shared Docling converter.
    """
    global _DOCLING_CONVERTER

    if _DOCLING_CONVERTER is None:
        from docling.document_converter import DocumentConverter
        _DOCLING_CONVERTER = DocumentConverter()

    return _DOCLING_CONVERTER


def pdf_to_markdown_docling(pdf_path: Path) -> Tuple[str, str]:
    """
    Convert PDF to Markdown using Docling only.

    Returns:
        markdown_text, backend_used
    """
    converter = get_docling_converter()
    result = converter.convert(str(pdf_path))
    markdown_text = result.document.export_to_markdown()

    if not markdown_text or not markdown_text.strip():
        raise RuntimeError("Docling returned empty Markdown.")

    return clean_text(markdown_text), "docling"


def build_plain_text_from_markdown(md: str) -> str:
    """
    Make a simpler text version from markdown.
    This keeps the OCR text because OCR is injected into markdown first.
    """
    txt = md
    txt = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", txt)          # remove markdown images
    txt = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", txt)      # links -> text
    txt = re.sub(r"^#{1,6}\s*", "", txt, flags=re.M)        # headings markers
    txt = re.sub(r"`{1,3}", "", txt)
    txt = re.sub(r"\*\*", "", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return clean_text(txt)


def extract_images_pymupdf(pdf_path: Path, output_dir: Path) -> List[Dict]:
    """
    Extract images from PDF using PyMuPDF.
    Saves images into output_dir and returns metadata.

    Updated:
    OCR is run on each saved image and stored inside the same image metadata.
    """
    ensure_dir(output_dir)

    doc = fitz.open(str(pdf_path))
    saved = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images, start=1):
            xref = img[0]

            try:
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base.get("ext", "png")

                width = int(base.get("width", 0) or 0)
                height = int(base.get("height", 0) or 0)

                if width and height:
                    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                        continue

                out_name = f"page_{page_index + 1:03d}_img_{img_idx:02d}.{ext}"
                out_path = output_dir / out_name
                out_path.write_bytes(img_bytes)

                ocr_text, ocr_status = ocr_image_file(out_path)

                saved.append({
                    "page": page_index + 1,
                    "image_index_on_page": img_idx,
                    "file_name": out_name,
                    "file_path": str(out_path),
                    "width": width,
                    "height": height,
                    "ocr_text": ocr_text,
                    "ocr_status": ocr_status,
                })

            except Exception as e:
                saved.append({
                    "page": page_index + 1,
                    "image_index_on_page": img_idx,
                    "file_name": "",
                    "file_path": "",
                    "width": 0,
                    "height": 0,
                    "ocr_text": "",
                    "ocr_status": f"extract_failed: {type(e).__name__}: {e}",
                })
                continue

    doc.close()
    return saved


# ============================================================
# PROJECT PROCESSING
# ============================================================

@dataclass
class ProcessResult:
    project_name: str
    source_pdf: str
    success: bool
    backend_used: str
    error: str = ""


def process_one_article(meta_path: Path, pdf_path: Path) -> ProcessResult:
    meta = read_json(meta_path)
    title = meta.get("title", pdf_path.stem)
    project_name = slugify(title)
    project_dir = PROJECTS_DIR / project_name
    images_dir = project_dir / "extracted_images"

    try:
        ensure_dir(project_dir)
        ensure_dir(images_dir)

        # 1) Save/copy PDF into project folder
        target_pdf = project_dir / "article.pdf"

        if OVERWRITE or not target_pdf.exists():
            shutil.copy2(pdf_path, target_pdf)

        # Also save metadata
        target_meta = project_dir / "article_metadata.json"

        if OVERWRITE or not target_meta.exists():
            write_json(target_meta, meta)

        # 2) Convert PDF to Markdown using Docling
        markdown_text, backend_used = pdf_to_markdown_docling(target_pdf)

        # 3) Extract images and OCR them
        images = extract_images_pymupdf(target_pdf, images_dir)

        # 4) Insert OCR text into SAME Markdown
        markdown_text = inject_image_ocr_into_markdown(markdown_text, images)

        # 5) Create plain text from OCR-enriched Markdown
        plain_text = build_plain_text_from_markdown(markdown_text)

        # Keep the same file names as your original code
        write_text(project_dir / "article_markdown.md", markdown_text)
        write_text(project_dir / "article_text.txt", plain_text)

        n_images_ocr_ok = sum(1 for img in images if img.get("ocr_status") == "ok")

        preprocessing_summary = {
            "project_name": project_name,
            "title": title,
            "source_pdf": str(pdf_path),
            "project_pdf": str(target_pdf),
            "backend_used_for_markdown": backend_used,
            "image_ocr_enabled": DO_IMAGE_OCR,
            "image_ocr_method": "pymupdf_extract_images_plus_tesseract_ocr_inserted_into_article_markdown",
            "files_created": {
                "article_pdf": str(target_pdf),
                "article_metadata_json": str(target_meta),
                "article_markdown_md": str(project_dir / "article_markdown.md"),
                "article_text_txt": str(project_dir / "article_text.txt"),
                "extracted_images_dir": str(images_dir),
            },
            "text_statistics": {
                "markdown_characters": len(markdown_text),
                "plain_text_characters": len(plain_text),
                "plain_text_words": len(plain_text.split()),
                "n_extracted_images": len(images),
                "n_images_ocr_ok": n_images_ocr_ok,
            },
            "images": images,
        }

        write_json(project_dir / "preprocessing_summary.json", preprocessing_summary)

        return ProcessResult(
            project_name=project_name,
            source_pdf=str(pdf_path),
            success=True,
            backend_used=backend_used,
            error="",
        )

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        try:
            ensure_dir(project_dir)
            write_text(project_dir / "processing_error.txt", err)
        except Exception:
            pass

        return ProcessResult(
            project_name=project_name,
            source_pdf=str(pdf_path),
            success=False,
            backend_used="",
            error=err,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(PROJECTS_DIR)

    articles = find_downloaded_articles(SOURCE_DIR)

    if not articles:
        print("No downloaded articles found in benchmark_reviews.")
        return

    print(f"Found {len(articles)} downloaded articles.")
    print(f"Creating projects in: {PROJECTS_DIR.resolve()}")
    print("Using one-by-one sequential processing")
    print("Converter: Docling")
    print("Output files unchanged: article_markdown.md and article_text.txt")
    print(f"Image OCR enabled: {DO_IMAGE_OCR}")

    results: List[ProcessResult] = []

    for meta_path, pdf_path in articles:
        try:
            result = process_one_article(meta_path, pdf_path)
        except Exception as e:
            result = ProcessResult(
                project_name=pdf_path.stem,
                source_pdf=str(pdf_path),
                success=False,
                backend_used="",
                error=str(e),
            )

        results.append(result)

        status = "OK" if result.success else "FAILED"
        backend = f" [{result.backend_used}]" if result.backend_used else ""
        print(f"{status}: {result.project_name}{backend}")

    summary = {
        "n_total": len(results),
        "n_success": sum(1 for r in results if r.success),
        "n_failed": sum(1 for r in results if not r.success),
        "results": [asdict(r) for r in sorted(results, key=lambda x: x.project_name)],
    }

    write_json(PROJECTS_DIR / "projects_preparation_summary.json", summary)

    print("\nDone.")
    print(f"Summary saved to: {PROJECTS_DIR / 'projects_preparation_summary.json'}")


if __name__ == "__main__":
    main()