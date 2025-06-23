import socket
import requests
from datetime import datetime, timedelta
from urllib.parse import quote
import urllib.request
import hashlib
from pathlib import Path
import logging
from typing import List, Optional
import feedparser
from newspaper import Article
import pandas as pd
import random
import time
from fake_useragent import UserAgent
import argparse

# This script scrapes news articles from various sources using Google News RSS feeds.

# Define or import NewsSource and ScrapingMethod
class ScrapingMethod:
    """Enum-like class to represent different scraping methods."""
    NEWSPAPER = "NEWSPAPER"
    SELENIUM = "SELENIUM"

class NewsSource:
    """Class to represent a news source with its name, URL, and scraping method."""
    def __init__(self, name, url, method):
        self.name = name
        self.url = url
        self.method = method

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
)

socket.setdefaulttimeout(30)

OUTPUT_DIRECTORY = Path("bad_data_2")
# START_DATE = "2024-01-01"
START_DATE = "2023-05-01"
END_DATE = "2023-12-01"
# END_DATE = "2024-06-27"

 
#if any other 

ua = UserAgent()

NEWS_SOURCES = [
    # NewsSource("The Onion", "theonion.com", ScrapingMethod.NEWSPAPER),
    # NewsSource("BitChute", "bitchute.com", ScrapingMethod.SELENIUM),
    NewsSource("WND", "wnd.com", ScrapingMethod.NEWSPAPER),
    NewsSource("The Babylon Bee", "babylonbee.com", ScrapingMethod.NEWSPAPER),
    NewsSource("thepeoplesvoice", "thepeoplesvoice.tv", ScrapingMethod.SELENIUM),
    NewsSource("clickhole", "clickhole.com", ScrapingMethod.SELENIUM),
    NewsSource("newyorker", "newyorker.com", ScrapingMethod.SELENIUM)
    # NewsSource("InfoWars", "infowars.com", ScrapingMethod.SELENIUM),
    # NewsSource("Breitbart", "breitbart.com", ScrapingMethod.SELENIUM),
    # NewsSource("News Punch", "newspunch.com", ScrapingMethod.SELENIUM)
]
# #https://thepeoplesvoice.tv/
# https://clickhole.com/
# https://www.newyorker.com/humor



def create_unique_id(string: str) -> str:
    """Create a unique ID based on the SHA-256 hash of the input string."""
    return hashlib.sha256(string.encode("utf-8")).hexdigest()[:10]

def make_url(source: str, start_date: str, end_date: str) -> str:
    """Generate a Google News RSS URL for the given source and date range."""
    query = f"site:{source}+after:{start_date}+before:{end_date}"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    logging.info(f"Generated URL: {url}")
    return url

def download_image(image_url: str, images_path: Path, image_name: str) -> Optional[str]:
    """Download an image from the given URL and save it to the specified path."""
    if not image_url:
        logging.error("Image URL is empty.")
        return None

    image_full_name = f"{image_name}.jpg"
    try:
        urllib.request.urlretrieve(image_url, str(images_path / image_full_name))
        logging.info(f"Image downloaded and saved as {image_full_name}")
    except Exception as e:
        logging.error(f"Failed to download image: {e}")
        return None
    return image_full_name

def get_entries(source: str, start_date_str: str, end_date_str: str) -> List:
    """Fetch entries from a news source RSS feed within the specified date range."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    all_entries = []
    # Reverse the date range to start from the latest date
    for dt in pd.date_range(end_date, start_date, freq='-4D'):
        current_start_date = dt.strftime("%Y-%m-%d")
        current_end_date = (dt - timedelta(days=3)).strftime("%Y-%m-%d")

        url = make_url(source, current_start_date, current_end_date)
        
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            logging.error(f"Failed to fetch URL {url} with status code {response.status_code}")
            continue

        feed = feedparser.parse(response.text)

        if not feed.entries:
            logging.warning(f"No entries found for {current_start_date} to {current_end_date}.")
            continue

        all_entries.extend(feed.entries)
        
        # Introduce a random delay between requests
        time.sleep(random.uniform(5, 10))

    logging.info(f"Total articles collected: {len(all_entries)}")
    return all_entries


def scrape_data():
    """Main function to scrape data from the defined news sources."""
    output_directory = OUTPUT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    content_dir = output_directory / "content"
    content_dir.mkdir(parents=True, exist_ok=True)

    for news_source in NEWS_SOURCES:
        logging.info(f"Processing source: {news_source.name}")
        output_csv_path = output_directory / f"{news_source.name.lower().replace(' ', '_')}_data.csv"

        # Check if CSV file already exists
        if output_csv_path.exists():
            df = pd.read_csv(output_csv_path)
            logging.info(f"Loaded existing CSV file for {news_source.name}")
        else:
            df = pd.DataFrame(columns=["title", "article_url", 
                                       "date_published", "first_paragraph", 
                                       "top_image", "content", "unique_id"])
            logging.info(f"Created new DataFrame for {news_source.name}")

        entries = get_entries(news_source.url, START_DATE, END_DATE)

        if not entries:
            logging.warning(f"No articles found for source: {news_source.name}")
            continue

        for entry in entries:
            unique_id = create_unique_id(entry.title)
            logging.info(f"Processing article with unique ID: {unique_id}")

            # Check if the article has already been processed
            if df[df.unique_id == unique_id].empty:
                try:
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    logging.info(f"Downloaded and parsed article: {entry.link}")
                except Exception as e:
                    logging.error(f"Failed to process article: {e}")
                    continue

                content = article.text
                if not content:
                    logging.error("Article content is empty, skipping.")
                    continue

                date_published = article.publish_date if article.publish_date else entry.published
                if isinstance(date_published, str):
                    try:
                        date_published = datetime.strptime(date_published, '%a, %d %b %Y %H:%M:%S %Z')
                    except ValueError:
                        logging.error(f"Invalid date format: {date_published}")
                        date_published = None

                first_paragraph = ' '.join(content.split('\n')[:5]).strip()

                top_image = download_image(article.top_image, images_dir, unique_id)

                content_file_path = content_dir / f"{unique_id}.txt"
                with open(content_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Content saved to {content_file_path}")

                data_entry = {
                    "title": entry.title,
                    "article_url": entry.link,
                    "canonical_link": article.canonical_link,  # The original article URL
                    "date_published": date_published,
                    "first_paragraph": first_paragraph,
                    "top_image": top_image,
                    "content": str(content_file_path),  # Store the path to the content file
                    "outlet": entry.source.title,
                    "source_url": entry.source.href,
                    "unique_id": unique_id,
                }
                
                df = pd.concat([pd.DataFrame([data_entry]), df], ignore_index=True)

                # Sort by date_published in descending order before saving
                df['date_published'] = pd.to_datetime(df['date_published'], errors='coerce')
                df = df.sort_values(by='date_published', ascending=False)

                # Save the DataFrame iteratively after each article is processed
                df.to_csv(output_csv_path, index=False)
                logging.info(f"Processed and saved article with date: {date_published}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", default=START_DATE, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", default=END_DATE, help="End date in YYYY-MM-DD format")
    parser.add_argument("--output_dir", default="bad_data_2", help="Output directory for scraped data")
    args = parser.parse_args()
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTPUT_DIRECTORY = Path(args.output_dir)
    scrape_data()


# To run the script, use:
# python newsscrapper-otherlist.py --start_date 2023-05-01 --end_date 2023-12-01 --output_dir bad_data_2