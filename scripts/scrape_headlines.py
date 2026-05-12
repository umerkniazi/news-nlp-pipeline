import argparse
import concurrent.futures
import datetime as dt
import logging
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DB_FILE = "dawn_news.db"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scraper.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            published_at TEXT,
            headline TEXT,
            category TEXT,
            body TEXT,
            url TEXT UNIQUE
        )
    """)
    conn.commit()
    return conn


def insert_articles(articles: List[Dict]) -> int:

    if not articles:
        return 0

    conn = get_connection()

    inserted = 0

    with conn:

        for article in articles:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO articles (
                        published_at,
                        headline,
                        category,
                        body,
                        url
                    )
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    article["published_at"],
                    article["headline"],
                    article["category"],
                    article["body"],
                    article["url"],
                ))

                if conn.total_changes > inserted:
                    inserted += 1

            except sqlite3.Error as exc:
                logger.error(f"DB insert failed: {exc}")

    conn.close()

    return inserted


def get_session() -> requests.Session:

    session = requests.Session()

    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    ]

    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    })

    return session


def request_with_retry(
    session: requests.Session,
    url: str,
    timeout: int = 20,
    attempts: int = 4,
    base_backoff: float = 1.5,
) -> Optional[requests.Response]:

    for attempt in range(1, attempts + 1):

        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()

            logger.info(f"Fetched: {url}")

            return response

        except requests.exceptions.RequestException as exc:

            if attempt == attempts:
                logger.error(f"Failed after {attempts} attempts: {url} | {exc}")
                return None

            sleep_seconds = (
                (base_backoff * attempt)
                + random.uniform(0.3, 1.2)
            )

            logger.warning(
                f"Retry {attempt}/{attempts} for {url} "
                f"in {sleep_seconds:.2f}s"
            )

            time.sleep(sleep_seconds)

    return None


def extract_date_from_article(
    article: BeautifulSoup,
    fallback_date: str
) -> str:

    time_tag = article.find("time")

    candidates = []

    if time_tag:
        candidates.extend([
            time_tag.get("datetime", ""),
            time_tag.get("title", ""),
            time_tag.get_text(" ", strip=True),
        ])

    for raw in candidates:

        if not raw:
            continue

        match = re.search(r"(\d{4}-\d{2}-\d{2})", raw)

        if match:
            return match.group(1)

    return fallback_date


def extract_archive_article(
    article: BeautifulSoup,
    fallback_date: str
) -> Optional[Dict]:

    link_tag = (
        article.select_one("h2 a")
        or article.select_one(".story__title a")
    )

    if not link_tag:
        return None

    headline = link_tag.get_text(strip=True)

    if not headline:
        return None

    article_url = link_tag.get("href", "").strip()

    if not article_url:
        return None

    if not article_url.startswith("http"):
        article_url = f"https://www.dawn.com{article_url}"

    badge_tag = article.select_one("span.badge")

    category = (
        badge_tag.get_text(" ", strip=True)
        if badge_tag
        else "Unknown"
    )

    excerpt_tag = article.select_one("div.story__excerpt")

    excerpt = (
        excerpt_tag.get_text(" ", strip=True)[:300]
        if excerpt_tag
        else ""
    )

    article_date = extract_date_from_article(article, fallback_date)

    return {
        "published_at": article_date,
        "headline": headline,
        "category": category,
        "body": excerpt,
        "url": article_url,
    }


def scrape_archive_page(
    session: requests.Session,
    date_str: str,
    min_delay: float = 2.5,
    max_delay: float = 5.0,
) -> List[Dict]:

    archive_url = f"https://www.dawn.com/archive/{date_str}"

    delay = random.uniform(min_delay, max_delay)

    logger.info(f"{date_str} sleeping {delay:.2f}s")

    time.sleep(delay)

    response = request_with_retry(session, archive_url)

    if response is None:
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("article")

    results = []

    for article in articles:

        extracted = extract_archive_article(article, date_str)

        if extracted:
            results.append(extracted)

    logger.info(f"{date_str} -> {len(results)} articles")

    return results


def generate_dates(start_year: int, end_year: int) -> List[str]:

    dates = []

    for year in range(start_year, end_year + 1):

        start = dt.date(year, 1, 1)
        end = dt.date(year, 12, 31)

        for i in range((end - start).days + 1):
            dates.append((start + dt.timedelta(days=i)).isoformat())

    return dates


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int)

    parser.add_argument("--workers", type=int, default=10)

    parser.add_argument("--min-delay", type=float, default=2.5)
    parser.add_argument("--max-delay", type=float, default=5.0)

    args = parser.parse_args()

    end_year = args.end_year or args.start_year

    dates = generate_dates(args.start_year, end_year)

    logger.info(f"Total dates: {len(dates)}")

    tls = threading.local()

    def one_day(date_str: str):

        if not hasattr(tls, "session"):
            tls.session = get_session()

        articles = scrape_archive_page(
            tls.session,
            date_str,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
        )

        inserted = insert_articles(articles)

        logger.info(
            f"{date_str} inserted={inserted} scraped={len(articles)}"
        )

        return inserted

    total_inserted = 0

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(one_day, date_str): date_str
            for date_str in dates
        }

        for idx, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1
        ):

            date_str = futures[future]

            try:
                inserted = future.result()

                total_inserted += inserted

                logger.info(
                    f"Completed {idx}/{len(dates)} | "
                    f"{date_str} | total={total_inserted}"
                )

            except Exception as exc:
                logger.error(f"{date_str} failed: {exc}")

            if idx % 50 == 0:
                logger.info("Cooldown 30s")
                time.sleep(30)

    logger.info(f"Done | total inserted={total_inserted}")


if __name__ == "__main__":
    main()