import argparse
import hashlib
import logging
import socket
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote

import dateutil
import feedparser
import pandas as pd
import requests
from newspaper import Article, configuration
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from tqdm import tqdm

# This script scrapes news articles from various sources using Google News RSS feeds.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
)
#with new list

socket.setdefaulttimeout(30)


DEFAULT_SOURCES = [
    "nationalpost.com",
    "theglobeandmail.com",
    "cnn.com",
    "foxnews.com",
    "cbsnews.com",
    "abcnews.go.com",
    "nytimes.com",
    "washingtonpost.com",
    "bbc.com",
    "usatoday.com",
    "wsj.com",  # paywalled
    "apnews.com",
    "politico.com",
    "nypost.com",
    "forbes.com",
    "reuters.com",
    "bloomberg.com",  # paywalled, cookies popup
    "aljazeera.com",
    "pbs.org/newshour",
    "theguardian.com",
    "huffpost.com",
    "newsmax.com",
    "cnbc.com",
    "c-span.org",
    "economist.com",
    "ft.com",
    "time.com",
    "newsweek.com",
    "theatlantic.com",
    "newyorker.com",
    "thehill.com",
    "propublica.org",
    "axios.com",
    "nationalreview.com",
    "thedailybeast.com",
    "dailykos.com",
    "washingtonexaminer.com",
    "thefederalist.com",
    "oann.com",
    "dailycaller.com",
    "breitbart.com",
    "cbc.ca",
    "torontosun.com",
    "globalnews.ca",
]

# TODO add BBC pidgin links
BLACKLIST = [
    "https://news.google.com/rss/articles/CBMiTWh0dHBzOi8vd3d3LmJiYy5jb20vcGlkZ2luL3Jlc291cmNlcy9pZHQtMDJkOTA2MGUtMTVkYy00MjZjLWJmZTAtODZhNjQzN2U1MjM00gEA?oc=5",  # BBC 100 Women 2023: Who dey on di list dis year?
    "https://news.google.com/rss/articles/CBMiKmh0dHBzOi8vd3d3Lm5ld3N3ZWVrLmNvbS90b3BpYy91cy1wb2xpdGljc9IBAA?oc=5",  # newsweek us politics page
    "https://www.bbc.com/news/politics",
    "https://news.google.com/rss/articles/CBMiTmh0dHBzOi8vd3d3LmJiYy5jb20vYWZyaXF1ZS9yZXNvdXJjZXMvaWR0LTAyZDkwNjBlLTE1ZGMtNDI2Yy1iZmUwLTg2YTY0MzdlNTIzNNIBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiPWh0dHBzOi8vd3d3LmNic25ld3MuY29tL3NhY3JhbWVudG8vbGF0ZXN0L2Nicy13ZWVrZW5kZXIvMzg5My_SAQA?oc=5",
    "https://news.google.com/rss/articles/CBMiPGh0dHBzOi8vd3d3LmNic25ld3MuY29tL2NvbG9yYWRvL2xhdGVzdC9leWUtb24tYW1lcmljYS8xMjMxL9IBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiNmh0dHBzOi8vd3d3LmNic25ld3MuY29tL2RldHJvaXQvdGFnL3dlc3QtcGhpbGFkZWxwaGlhL9IBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiHWh0dHBzOi8vd3d3LmNic25ld3MuY29tL3RlYW0v0gEA?oc=5",
    "https://news.google.com/rss/articles/CBMiNmh0dHBzOi8vd3d3LmNic25ld3MuY29tL3NhY3JhbWVudG8vbGF0ZXN0L3VwbGlmdC8zMjA4L9IBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiN2h0dHBzOi8vd3d3LmNic25ld3MuY29tL3NhbmZyYW5jaXNjby9sYXRlc3Qvc3BhY2UvNTQ2Ny_SAQA?oc=5",
    "https://news.google.com/rss/articles/CBMiNGh0dHBzOi8vd3d3LmNic25ld3MuY29tL2xhdGVzdC90aGUtZGFpbHktcmVwb3J0Lzk3Ni_SAQA?oc=5",
    "https://news.google.com/rss/articles/CBMiTWh0dHBzOi8vYWJjbmV3cy5nby5jb20vVVMvd2lyZVN0b3J5L2VkaXRvcmlhbC1yb3VuZHVwLXVuaXRlZC1zdGF0ZXMtMTA5MDA4MzA10gFRaHR0cHM6Ly9hYmNuZXdzLmdvLmNvbS9hbXAvVVMvd2lyZVN0b3J5L2VkaXRvcmlhbC1yb3VuZHVwLXVuaXRlZC1zdGF0ZXMtMTA5MDA4MzA1?oc=5",
    "https://news.google.com/rss/articles/CBMie2h0dHBzOi8vd3d3LmFsamF6ZWVyYS5jb20vbmV3cy9saXZlYmxvZy8yMDIzLzExLzE2L2lzcmFlbC1oYW1hcy13YXItbGl2ZS1pc3JhZWwtcmVqZWN0cy11bi1zZWN1cml0eS1jb3VuY2lsLWdhemEtcmVzb2x1dGlvbtIBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiRmh0dHBzOi8vd3d3LnVzYXRvZGF5LmNvbS9lbGVjdGlvbnMvdm90ZXItZ3VpZGUvMjAyNC0wNC0wMi9yaG9kZS1pc2xhbmTSAQA?oc=5",
    "https://news.google.com/rss/articles/CBMiQmh0dHBzOi8vd3d3LnVzYXRvZGF5LmNvbS9lbGVjdGlvbnMvdm90ZXItZ3VpZGUvMjAyNC0wNC0wMi9uZXcteW9ya9IBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiQ2h0dHBzOi8vd3d3LnVzYXRvZGF5LmNvbS9lbGVjdGlvbnMvdm90ZXItZ3VpZGUvMjAyNC0wNC0wMi93aXNjb25zaW7SAQA?oc=5",
    "https://news.google.com/rss/articles/CBMiRWh0dHBzOi8vd3d3LnVzYXRvZGF5LmNvbS9lbGVjdGlvbnMvdm90ZXItZ3VpZGUvMjAyNC0wNC0wMi9jb25uZWN0aWN1dNIBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiP2h0dHBzOi8vd3d3LnVzYXRvZGF5LmNvbS9lbGVjdGlvbnMvdm90ZXItZ3VpZGUvMjAyNC0wMy0wMi9pZGFob9IBAA?oc=5",
    "https://news.google.com/rss/articles/CBMiVWh0dHBzOi8vd3d3LmVjb25vbWlzdC5jb20vdG9waWNzL2VsZWN0aW9ucz9hZnRlcj0xNzcxNGFkNC05NzQ5LTRkZDYtOTRmNi02YWY5NmM5MjE0NTPSAQA?oc=5",
]

NEWSPAPER_CONFIG = configuration.Configuration().update(
    min_word_count=300,  # num of words in text
    min_sent_count=7,  # num of sentences
    max_title=200,  # num of chars
    max_text=100000,  # num of chars
    max_authors=10,  # num strings in list
)


class NewsSource:
    """
    Represents a news source with methods to fetch articles and images.
    """
    def __init__(
        self, name: str, base_url: str, image_prefixes: Optional[List[str]] = None
    ):
        self.name = name
        self.base_url = base_url
        self.image_prefixes = image_prefixes

    def get_article_for_entry_link(self, entry_link: str) -> Article:
        """
        Fetches the article for a given entry link using Newspaper3k or Selenium if necessary.
        Args:
            entry_link (str): The link to the article.
        Returns:
            Article: The parsed article object.
        """
        article = NewsSource._parse_article_with_selenium(entry_link)

        return article

    @classmethod
    def _parse_article_with_selenium(cls, entry_link: str) -> Article:
        """
        Parses the article using Selenium to handle JavaScript-rendered content.
        Args:
            entry_link (str): The link to the article.
        Returns:
            Article: The parsed article object.
        """
        options = Options()
        options.add_argument("--headless")
        browser = webdriver.Firefox(options=options)
        browser.get(entry_link)
        time.sleep(3)  # wait for the page to load

        html_source = browser.page_source
        article = Article(browser.current_url, config=NEWSPAPER_CONFIG)
        article.download(input_html=html_source)
        article.parse()

        browser.close()
        browser.quit()

        return article

    def get_image_links_from_article(self, article: Article) -> List[str]:
        """
        Extracts image URLs from the article, filtering them based on specified prefixes.
        Args:
            article (Article): The article object containing images.
        Returns:
            List[str]: A list of filtered image URLs.
        """
        if not hasattr(article, "images"):
            logging.error("Article has no 'images' attribute.")
            return []

        all_image_urls = list(article.images)
        if self.image_prefixes is None:
            return all_image_urls

        filtered_image_urls = []
        for image_url in all_image_urls:
            if any(
                [
                    image_url.startswith(image_prefix)
                    for image_prefix in self.image_prefixes
                ]
            ):
                filtered_image_urls.append(image_url)

        if len(filtered_image_urls) == 0:
            logging.error(f"All {len(all_image_urls)} images have been filtered out!")
        else:
            logging.info(
                f"Filtered out {len(all_image_urls) - len(filtered_image_urls)} images ({len(all_image_urls)} total)."
            )

        return filtered_image_urls

    def __repr__(self):
        return f"{self.name} ({self.base_url})"


def create_unique_id(string: str) -> str:
    """Create a unique identifier based on the hash of the given string."""
    return hashlib.sha256(string.encode("utf-8")).hexdigest()[:10]


def make_url(
    source: str,
    keywords: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """
    Create a Google News RSS feed URL based on the source and keywords.
    Args:
        source (str): The news source domain.
        keywords (List[str]): List of keywords to search for.
        start_date (Optional[str]): Start date for the search in YYYY-MM-DD format.
        end_date (Optional[str]): End date for the search in YYYY-MM-DD format.
    Returns:
        str: The constructed RSS feed URL.
    """
    query = f"site:{source}"

    # Define a custom date range for the RSS feed
    if start_date is not None:
        query += f"+after:{start_date}"
    if end_date is not None:
        query += f"+before:{end_date}"

    if len(keywords) == 1:
        query += f"+topic={keywords[0]}"
    else:
        query += "+" + "|".join(keywords)

    return f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def download_image(image_url: str, images_path: Path, image_name: str) -> Optional[str]:
    """
    Download an image from a given URL and save it to the specified path.
    Args:
        image_url (str): The URL of the image to download.
        images_path (Path): The directory where the image will be saved.
        image_name (str): The name to save the image as (without extension).
    Returns:
        Optional[str]: The full name of the saved image file, or None if the download failed.
    """
    if image_url is None or len(image_url) == 0:
        logging.error("Image is None or empty.")
        return None

    image_full_name = f"{image_name}.jpg"
    url = image_url
    try:
        urllib.request.urlretrieve(url, str(images_path / image_full_name))
    except Exception as e:
        # Try to download it differently
        url_keyword = "?url="
        url_index = url.find(url_keyword)
        if url_index != -1:
            url = unquote(url[url_index + len(url_keyword) :])

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
        except Exception as e2:
            logging.error(f"Failed to download image (requests): {e2}")
            raise e2

        content_type = response.headers["Content-type"]
        if content_type in ["image/jpeg", "image/png"]:
            image_extension = content_type.replace("image/", "")
            image_full_name = f"{image_name}.{image_extension}"

        elif content_type == "image/svg+xml":
            image_full_name = f"{image_name}.svg"

        elif content_type == "image/webp":
            image_full_name = f"{image_name}.webp"

        else:
            logging.error(f"Failed to download image (urlretrieve): {e}")
            raise e

        with open(images_path / image_full_name, "wb") as file:
            file.write(response.content)

    return image_full_name


def get_entries_balanced_per_month(
    news_source: NewsSource,
    keywords: List[str],
    start_date_str: Optional[str],
    end_date_str: Optional[str],
    blacklist: List[str],
) -> List[feedparser.util.FeedParserDict]:
    """
    Get entries from the RSS feed, balancing the number of articles per month.
    Args:
        news_source (NewsSource): The news source object.
        keywords (List[str]): List of keywords to search for.
        start_date_str (Optional[str]): Start date in YYYY-MM-DD format.
        end_date_str (Optional[str]): End date in YYYY-MM-DD format.
        blacklist (List[str]): List of blacklisted article links.
    Returns:
        List[feedparser.util.FeedParserDict]: A list of feed entries.
    """
    all_entries = []
    set_of_links = set([])

    def _parse_rss_feed(
        url: str,
        keywords: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> feedparser.util.FeedParserDict:
        url = make_url(news_source.base_url, keywords, start_date, end_date)

        return feedparser.parse(
            url,
            referrer="https://www.google.com/",
            request_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
                "Accept-Encoding": "gzip",
                "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
                "Upgrade-Insecure-Requests": "1",
                "Cookie": f"CONSENT=YES+cb.{datetime.now().isoformat().split('T')[0].replace('-', '')}-04-p0.en-GB+FX+667",
            },
        )

    if start_date_str is None and end_date_str is None:  # date range not specified
        feed = _parse_rss_feed(news_source.base_url, keywords)

        collected = 0
        for entry in feed.entries:
            if entry.link in blacklist:
                logging.debug(f"Skipping blacklisted entry: {entry.link}")
                continue
            if entry.link in set_of_links:
                logging.debug(f"Skipping repeated entry: {entry.link}")
                continue

            collected += 1
            set_of_links.add(entry.link)
            all_entries.append(entry)

        logging.info(f"Collected {collected} articles.")
        return all_entries

    if start_date_str is not None:  # only start_date is specified
        start_date = dateutil.parser.parse(start_date_str)

        if end_date_str is None:
            end_date_str = datetime.now().isoformat().split("T")[0]

    if end_date_str is not None:  # only end_date is specified
        end_date = dateutil.parser.parse(end_date_str)
        if start_date_str is None:
            # start_date is a year before the end_date
            start_date = end_date - timedelta(days=365)

    amount_of_days = (end_date - start_date).days + 1
    all_days = [start_date + timedelta(n) for n in range(amount_of_days)]

    progress_bar = tqdm(range(len(all_days)))
    for i in progress_bar:
        # getting articles 2 days at a time
        if i % 2 != 0:
            continue

        end_index = min(i + 1, len(all_days) - 1)

        current_start_date = all_days[i].strftime("%Y-%m-%d")
        current_end_date = all_days[end_index].strftime("%Y-%m-%d")

        feed = _parse_rss_feed(
            news_source, keywords, current_start_date, current_end_date
        )

        if len(feed.entries) == 0:
            logging.error(
                f"News source returned 0 entries for date range {current_start_date} to {current_end_date}."
            )
            continue

        collected = 0
        for entry in feed.entries:
            if entry.link in blacklist:
                logging.debug(f"Skipping blacklisted entry: {entry.link}")
                continue
            if entry.link in set_of_links:
                logging.debug(f"Skipping repeated entry: {entry.link}")
                continue

            collected += 1
            set_of_links.add(entry.link)
            all_entries.append(entry)

        progress_bar.set_description(
            f"Collected {collected} articles out of {len(feed.entries)} (total: {len(set_of_links)})"
        )

    logging.info(f"Final count: {len(all_entries)} articles.")
    return all_entries


def download_all_data(
    news_sources: List[str],
    keywords: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
    output_directory: Path,
    output_csv_name: str,
    dedupe_csv_name: str,
    blacklist: List[str] = BLACKLIST,
) -> List[Dict[str, Any]]:
    """
    Download articles from specified news sources based on keywords and date range.
    Args:
        news_sources (List[str]): List of news sources to scrape.
        keywords (List[str]): List of keywords to search for.
        start_date (Optional[str]): Start date for the search in YYYY-MM-DD format.
        end_date (Optional[str]): End date for the search in YYYY-MM-DD format.
        output_directory (Path): Directory to save the downloaded data.
        output_csv_name (str): Name of the CSV file to save the data.
        dedupe_csv_name (str): Name of the CSV file to save the deduped data.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the downloaded articles.
    """
    if (
        len(news_sources) == 1
    ):  # common mistake to pass a single string instead of a list
        news_sources = news_sources[0].split(",")

    if len(keywords) == 1:
        keywords = keywords[0].split(",")

    encoded_keywords = [quote(kw) for kw in keywords]
    output_csv_path = output_directory.joinpath(output_csv_name)

    if output_csv_path.exists():
        logging.info("Loading existing CSV data.")
        df = pd.read_csv(str(output_csv_path))
    else:
        logging.info("Making a new CSV file.")
        df = pd.DataFrame(
            columns=[
                "unique_id",
                "outlet",
                "title",
                "date_published",
                "first_paragraph",
                "content",
                "top_image",
                "article_url",
                "canonical_link",
                "source_url",
            ]
        )

    dedupe_csv_path = output_directory.joinpath(dedupe_csv_name)
    if dedupe_csv_path.exists():
        logging.info("Loading dedupe CSV data.")
        dedupe_df = pd.read_csv(str(dedupe_csv_path))
    else:
        logging.info("No dedupe CSV file.")
        dedupe_df = None

    # create directory for storing images
    output_images_dir = output_directory.joinpath("images")
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # create news source objects
    news_source_objects = [
        NewsSource(name=source.split(".")[0], base_url=source)
        for source in news_sources
    ]

    for news_source in news_source_objects:
        logging.info(
            "******************************************************************************"
        )
        logging.info(f"Looking into news source {news_source}")

        entries = get_entries_balanced_per_month(
            news_source, encoded_keywords, start_date, end_date, blacklist
        )

        count = 0
        for entry in entries:
            count += 1
            unique_id = create_unique_id(entry.title)

            if len(df[df.unique_id == unique_id]) > 0:
                logging.info(
                    f"Article with unique id {unique_id} has already been processed. Skipping..."
                )
                continue

            if (
                dedupe_df is not None
                and len(dedupe_df[dedupe_df.unique_id == unique_id]) > 0
            ):
                logging.info(
                    f"Skipping, it already exists in the dedupe csv under unique id {unique_id}."
                )
                continue

            logging.info(
                f"Downloading article '[{entry.source.title} {count}/{len(entries)}] {entry.title}' ({entry.published})"
            )

            try:
                article = news_source.get_article_for_entry_link(entry.link)
            except Exception as e:
                logging.exception(f"Error processing article: {e}")
                continue

            content: str = article.text
            logging.info(f"Content length: {len(content)}")
            if len(content) == 0:
                logging.error("Article is empty. Skipping...")
                continue

            if article.is_media_news():
                logging.info("Article is a media news. Skipping...")
                continue

            if not article.is_valid_body():
                logging.info(
                    "Article does not meet the configuration criteria. Skipping..."
                )
                continue

            logging.info(f"Downloading top image {article.top_image}")
            try:
                top_image = download_image(
                    article.top_image, output_images_dir, unique_id
                )
                logging.info(
                    f"Downloaded top image to {output_images_dir.joinpath(top_image)}"
                )
            except Exception as ex:
                top_image = None
                logging.exception(
                    f"Failed to download top image at {article.top_image}: {ex}"
                )

            first_paragraph = " ".join(content.split("\n")[:5]).strip()
            data_entry = {
                "unique_id": unique_id,
                "outlet": entry.source.title,
                "title": entry.title,
                "date_published": article.publish_date,
                "first_paragraph": first_paragraph,
                "content": content,
                "top_image": None
                if top_image is None
                else str(output_images_dir.joinpath(top_image)),
                "article_url": article.url,  # The google article URL
                "canonical_link": article.canonical_link,  # The original article URL
                "source_url": entry.source.href,
            }

            df = pd.concat(
                [pd.DataFrame([data_entry], columns=df.columns), df], ignore_index=True
            )
            df.to_csv(str(output_csv_path), index=False)

            logging.info(
                "------------------------------------------------------------------------"
            )


def _valid_date(date_str: Optional[str]) -> str:
    """
    Validate the date string format and convert it to a standardized string.
    """
    if date_str is None:
        return None
    try:
        return str(datetime.strptime(date_str, "%Y-%m-%d").date())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--news_sources",
        type=str,
        nargs="+",
        default=DEFAULT_SOURCES,
        help="List of news sources to scrape.",
    )
    parser.add_argument(
        "--start_date",
        type=_valid_date,
        default=None,
        help="Start date for the search, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end_date",
        type=_valid_date,
        default=None,
        help="End date for the search, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=["politics"],
        help="List of topics to search for.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the downloaded data.",
    )
    parser.add_argument(
        "--output_csv_name",
        type=str,
        default="scrapped_data.csv",
        help="Name of the CSV file to save the data.",
    )
    parser.add_argument(
        "--dedupe_csv_name",
        type=str,
        default="deduped.csv",
        help="Name of the CSV file to save the deduped data.",
    )

    args = parser.parse_args()

    download_all_data(
        news_sources=args.news_sources,
        keywords=args.keywords,
        start_date=args.start_date,
        end_date=args.end_date,
        output_directory=args.output_dir,
        output_csv_name=args.output_csv_name,
        dedupe_csv_name=args.dedupe_csv_name,
    )
