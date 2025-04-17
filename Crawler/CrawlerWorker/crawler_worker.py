#!/usr/bin/env python3
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
import asyncio
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, BestFirstCrawlingStrategy, KeywordRelevanceScorer
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN
import aiohttp

# Logging setup (same as original)
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s | %(levelname)-8s | %(message)s%(extra_info)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        record.extra_info = ""
        if record.__dict__.get('extra'):
            extra_str = " | " + ", ".join(f"{k}: {v}" for k, v in record.__dict__['extra'].items())
            record.extra_info = extra_str
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("crawler_worker.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

# Lock management
LOCK_FILE = os.path.join(tempfile.gettempdir(), "crawler_worker.lock")

def acquire_lock():
    if os.path.exists(LOCK_FILE):
        logger.info("Another crawler instance is running, exiting.")
        sys.exit(0)
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
        logger.debug("Acquired lock", extra={"pid": os.getpid(), "lock_file": LOCK_FILE})
    except IOError as e:
        logger.error("Failed to acquire lock", extra={"error": str(e)})
        sys.exit(1)

def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logger.debug("Released lock", extra={"lock_file": LOCK_FILE})
    except IOError as e:
        logger.error("Failed to release lock", extra={"error": str(e)})

# API interaction functions (same as original)
async def fetch_scraping_targets() -> list[dict[str, any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{BOINGO_API_URL}/scraping-target/all"
            logger.debug("Fetching targets", extra={"url": url})
            async with session.get(url, headers=headers, timeout=30) as response:
                response.raise_for_status()
                response_data = await response.json()
                targets = response_data.get("data", [])
                logger.info("Fetched targets", extra={"count": len(targets)})
                return targets
        except aiohttp.ClientError as e:
            logger.error("Failed to fetch targets", extra={"error": str(e)})
            return []

async def post_to_scraping_results(data: dict[str, any]) -> dict[str, any] | None:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Posting result", extra={"url": data.get("source_url")})
            async with session.post(f"{BOINGO_API_URL}/scraping-results/", headers=headers, json=data, timeout=30) as response:
                response.raise_for_status()
                response_data = await response.json()
                result_id = response_data["data"]["id"]
                logger.info("Posted result", extra={"id": result_id})
                return {"id": result_id, "data": response_data["data"]}
        except aiohttp.ClientError as e:
            logger.error("Failed to post result", extra={"error": str(e)})
            return None

async def update_list_extracted(target_id: str) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    payload = {"id": target_id, "list_extracted": True}
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Updating list_extracted", extra={"target_id": target_id})
            async with session.put(f"{BOINGO_API_URL}/scraping-target/update-list-extracted", headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                logger.info("Updated list_extracted", extra={"target_id": target_id})
                return True
        except aiohttp.ClientError as e:
            logger.error("Failed to update list_extracted", extra={"error": str(e)})
            return False

# Crawling logic
async def scrape_and_process(url: str, target_id: str, listing_format: str, crawler: AsyncWebCrawler, search_range: int, max_properties: int):
    scorer = KeywordRelevanceScorer(keywords=["property", "sale", "house"])
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(max_depth=search_range, max_pages=max_properties, url_scorer=scorer),
        cache_mode="BYPASS",
        verbose=True,
        page_timeout=30000,
        wait_until="domcontentloaded",
        stream=True
    )
    logger.debug("Starting crawl", extra={"url": url, "target_id": target_id})
    
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    initial_payload_template = {
        "data": {
            "address": {"country": "", "region": "", "city": ""},
            "property": {"lat": "0.0", "lng": "0.0"},
            "listing": {"listing_title": "", "description": "", "price": "", "currency": "", "status": "", "listing_type": "", "category": ""},
            "features": [],
            "files": [],
            "contact": {"first_name": "", "last_name": "", "phone_number": "", "email": "", "company": ""}
        },
        "large_result": "",
        "progress": 0,
        "status": "In Progress",
        "scraped_at": now,
        "target_id": target_id,
        "agent_status": [
            {"agent_name": "Scraping Agent", "status": "Queued", "start_time": now, "end_time": now},
            {"agent_name": "Cleaning Agent", "status": "Queued", "start_time": now, "end_time": now},
            {"agent_name": "Extracting Agent", "status": "Queued", "start_time": now, "end_time": now}
        ]
    }

    posted_ids = {}
    async for result in await crawler.arun(url=url, config=run_config):
        if not result.url.startswith(listing_format) or not result.success or not result.markdown or not result.markdown.strip():
            logger.debug("Skipping invalid result", extra={"url": result.url})
            continue

        logger.debug("Found valid listing", extra={"url": result.url})
        initial_payload = initial_payload_template.copy()
        initial_payload["source_url"] = result.url
        initial_payload["large_result"] = result.markdown

        post_result = await post_to_scraping_results(initial_payload)
        if not post_result:
            logger.error("Failed to queue listing", extra={"url": result.url})
            continue
        
        scraping_result_id = post_result["id"]
        posted_ids[result.url] = scraping_result_id
        logger.info("Queued listing", extra={"url": result.url, "id": scraping_result_id})

    if posted_ids and await update_list_extracted(target_id):
        logger.info("Marked target extracted", extra={"target_id": target_id})

async def scraper_loop(crawler: AsyncWebCrawler):
    while True:
        targets = await fetch_scraping_targets()
        valid_targets = [t for t in targets if not t.get("list_extracted", True)]
        
        if not valid_targets:
            logger.info("No targets, exiting")
            break

        for target in valid_targets:
            url = target.get("website_url")
            target_id = target.get("id")
            listing_format = target.get("listing_url_format")
            search_range = target.get("search_range", 3)
            max_properties = target.get("max_properties", 20)

            if not url or not target_id or not listing_format:
                logger.error("Invalid target", extra={"url": url, "id": target_id})
                continue

            try:
                await scrape_and_process(url, target_id, listing_format, crawler, search_range, max_properties)
            except Exception as e:
                logger.error("Scrape failed", extra={"url": url, "error": str(e)})
                continue

        await asyncio.sleep(30)

async def main():
    acquire_lock()
    try:
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=True,
            use_persistent_context=True,
            user_data_dir="browser_data",
            extra_args=["--no-sandbox", "--disable-setuid-sandbox"],
            cdp_url=None  # Address ECONNREFUSED issue
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            await scraper_loop(crawler)
    finally:
        release_lock()

if __name__ == "__main__":
    asyncio.run(main())