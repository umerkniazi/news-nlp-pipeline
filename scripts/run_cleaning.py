import csv
import re
from datetime import datetime

INPUT_FILE = "articles.csv"
OUTPUT_FILE = "articles_clean.csv"


def clean_body(text: str) -> str:
    if not text:
        return ""

    text = re.sub(
        r"(Published|Updated)\s+\d{1,2}\s+\w+,\s+\d{4}.*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_date(d: str):
    try:
        return datetime.strptime(d, "%Y-%m-%d")
    except Exception:
        return datetime.max


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        row["body"] = clean_body(row.get("body", ""))

    rows.sort(key=lambda x: parse_date(x.get("published_at", "")))

    for i, row in enumerate(rows, start=1):
        row["id"] = i

    fieldnames = ["id", "published_at", "headline", "category", "body", "url"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved cleaned + sorted CSV → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()