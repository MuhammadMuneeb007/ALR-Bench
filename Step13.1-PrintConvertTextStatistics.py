#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

DB_DIR = "DataBase_Strategy3_Top1000"
OUTPUT_CSV = "fulltext_count_per_project.csv"

MIN_CHARS = 500

TEXT_FILES_TO_CHECK = [
    "Article_Text.txt",
    "COMBINED.txt",
    "full_text.txt",
]


# ============================================================
# HELPERS
# ============================================================

def has_full_text(temp_folder: Path):
    """
    Check whether Temp_X folder has usable full text.
    Priority:
      1. Article_Text.txt
      2. COMBINED.txt
      3. full_text.txt
    """
    for fname in TEXT_FILES_TO_CHECK:
        fpath = temp_folder / fname

        if fpath.exists():
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) >= MIN_CHARS:
                    return True, fname, len(text)
            except Exception:
                pass

    return False, "", 0


def main():
    db_path = Path(DB_DIR)

    if not db_path.exists():
        raise FileNotFoundError(f"Database folder not found: {DB_DIR}")

    rows = []

    print("\n" + "=" * 100)
    print("FULL TEXT COUNT PER PROJECT")
    print("=" * 100)

    for project_dir in sorted(db_path.iterdir()):
        if not project_dir.is_dir():
            continue

        if project_dir.name.startswith("_"):
            continue

        temp_folders = sorted(
            [p for p in project_dir.iterdir() if p.is_dir() and p.name.startswith("Temp_")],
            key=lambda x: int(x.name.replace("Temp_", "")) if x.name.replace("Temp_", "").isdigit() else 999999
        )

        total_articles = len(temp_folders)
        fulltext_count = 0
        missing_count = 0

        article_text_count = 0
        combined_count = 0
        full_text_count = 0

        total_chars = 0

        for temp_folder in temp_folders:
            ok, source_file, chars = has_full_text(temp_folder)

            if ok:
                fulltext_count += 1
                total_chars += chars

                if source_file == "Article_Text.txt":
                    article_text_count += 1
                elif source_file == "COMBINED.txt":
                    combined_count += 1
                elif source_file == "full_text.txt":
                    full_text_count += 1
            else:
                missing_count += 1

        fulltext_percent = round(
            fulltext_count * 100 / total_articles, 2
        ) if total_articles else 0

        avg_chars = round(
            total_chars / fulltext_count, 2
        ) if fulltext_count else 0

        rows.append({
            "project": project_dir.name,
            "total_articles": total_articles,
            "fulltext_available": fulltext_count,
            "missing_fulltext": missing_count,
            "fulltext_percent": fulltext_percent,
            "article_text_txt": article_text_count,
            "combined_txt": combined_count,
            "full_text_txt": full_text_count,
            "avg_chars_per_fulltext": avg_chars,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No project folders found.")
        return

    df = df.sort_values(
        by=["fulltext_available", "fulltext_percent"],
        ascending=[False, False]
    )

    print(df.to_string(index=False))

    total_articles = int(df["total_articles"].sum())
    total_fulltext = int(df["fulltext_available"].sum())
    total_missing = int(df["missing_fulltext"].sum())

    overall_percent = round(
        total_fulltext * 100 / total_articles, 2
    ) if total_articles else 0

    print("\n" + "=" * 100)
    print("OVERALL")
    print("=" * 100)
    print(f"Total articles          : {total_articles}")
    print(f"Full text available     : {total_fulltext}")
    print(f"Missing full text       : {total_missing}")
    print(f"Full text percentage    : {overall_percent}%")

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()