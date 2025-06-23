# 🕸️ News Scraper

A robust web scraping toolkit to collect news articles from mainstream and fringe media sources using Google News RSS feeds, Newspaper3k, and Selenium. Outputs structured data with optional image downloads and deduplication.

---

### Modules

| File                       | Description                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `news_scrapper.py`         | Scrapes **mainstream media** using RSS + Selenium + Newspaper3k with filtering, deduplication, and image downloading. |
| `newsscraper-otherlist.py` | Targets **satirical or fringe news sources** using RSS feeds and lightweight scraping with random user agents.        |
| `utils.py`                 | Contains utility functions for decoding and cleaning Base64-encoded or malformed article URLs.                        |

---

### Features

* RSS-based crawling by **domain and keyword/topic**
* **Time-window slicing** for date-balanced article retrieval
* Auto-deduplication using **SHA-256 hashes**
* Handles **JavaScript-rendered content** via Selenium
* Downloads article content, top image, and metadata
* Stores images and article text locally by `unique_id`
