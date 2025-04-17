import logging
import requests
import json
from datetime import datetime, timezone
import asyncio
import os
import queue
import tiktoken
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, BestFirstCrawlingStrategy, KeywordRelevanceScorer
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from .queue_manager import acquire_lock, release_lock
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Pydantic Models
class Address(BaseModel):
    country: str
    region: str
    city: str
    district: str

class Property(BaseModel):
    lat: Optional[str] = None
    lng: Optional[str] = None

class Listing(BaseModel):
    listing_title: str
    description: str
    price: str
    currency: str
    status: str
    listing_type: str
    category: str

class Feature(BaseModel):
    feature: str
    value: str

class Contact(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: str
    email_address: Optional[str] = None
    company: Optional[str] = None

class ListingData(BaseModel):
    address: Address
    property: Property
    listing: Listing
    features: List[Feature] = Field(default_factory=list)
    files: List[str]
    contact: Contact

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

def split_into_chunks(content: str, max_tokens: int = 7000) -> List[str]:
    """Split content into chunks based on token count, preserving logical sections."""
    lines = content.split("\n")
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    logger.debug(f"Split content into {len(chunks)} chunks")
    return chunks

def merge_chunk_results(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple chunk results into a single ListingData-compliant dictionary."""
    logger.debug(f"Merging {len(chunks)} chunks")
    merged = {
        "address": {"country": "", "region": "", "city": "", "district": ""},
        "property": {"lat": None, "lng": None},
        "listing": {"listing_title": "", "description": "", "price": "", "currency": "", "status": "", "listing_type": "", "category": ""},
        "features": [],
        "files": [],
        "contact": {"phone_number": "", "first_name": None, "last_name": None, "email_address": None, "company": None}
    }
    try:
        for chunk in chunks:
            logger.debug(f"Processing chunk: {json.dumps(chunk, default=str)}")
            for section in ["address", "property", "listing", "contact"]:
                if section in chunk and isinstance(chunk[section], dict):
                    for key, value in chunk[section].items():
                        if key in merged[section] and value and (merged[section][key] is None or merged[section][key] == ""):
                            merged[section][key] = value
                            logger.debug(f"Updated {section}.{key} with {value}")
            if "features" in chunk and isinstance(chunk["features"], list):
                existing_features = {item["feature"]: item for item in merged["features"]}
                for item in chunk["features"]:
                    if item["feature"] not in existing_features:
                        merged["features"].append(item)
                        logger.debug(f"Added to features: {item}")
            if "amenities" in chunk and isinstance(chunk["amenities"], list):
                existing_features = {item["feature"]: item for item in merged["features"]}
                for amenity in chunk["amenities"]:
                    feature_item = {"feature": amenity["amenity"], "value": amenity.get("value", "Yes")}
                    if feature_item["feature"] not in existing_features:
                        merged["features"].append(feature_item)
                        logger.debug(f"Converted amenity to feature: {feature_item}")
            if "files" in chunk and isinstance(chunk["files"], list):
                merged["files"].extend(chunk["files"])
                merged["files"] = list(set(merged["files"]))
                logger.debug(f"Merged files: {merged['files']}")
        logger.debug(f"Merged result: {json.dumps(merged, default=str)}")
        return merged
    except Exception as e:
        logger.error(f"Error merging chunks: {str(e)}", exc_info=True)
        raise

async def extract_with_openai(content: str, schema: Dict[str, Any], instruction: str, max_retries: int = 3) -> str:
    """Extract data from content using OpenAI API with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Content:\n{content}\n\nSchema:\n{json.dumps(schema, indent=2)}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt * 10
                logger.warning(f"Rate limit hit, retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"OpenAI extraction failed: {str(e)}")
                return "{}"
    logger.error(f"Max retries ({max_retries}) exceeded for OpenAI extraction")
    return "{}"

async def process_page(result, target_id: str) -> Optional[Dict[str, Any]]:
    """Process a crawled page and return validated listing data."""
    logger.debug(f"Processing page: {result.url}")
    if not result.success:
        logger.debug(f"Page crawl failed: {result.url}")
        return None

    lines = result.markdown.split("\n")
    top_level_title_count = sum(1 for line in lines if line.strip().startswith("# ") and not line.strip().startswith("##"))
    if top_level_title_count != 1:
        logger.debug(f"Skipping {result.url}: Incorrect title count ({top_level_title_count})")
        return None

    if not result.extracted_content:
        logger.debug(f"No extracted content for {result.url}")
        return None

    try:
        extracted_data = json.loads(result.extracted_content)
        logger.debug(f"Raw LLM extracted data: {json.dumps(extracted_data, default=str)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {result.url}: {str(e)}", exc_info=True)
        return None

    if isinstance(extracted_data, dict) and not extracted_data:
        logger.debug(f"Empty object extracted for {result.url}")
        return None
    if isinstance(extracted_data, list):
        if len(extracted_data) > 1:
            merged_data = merge_chunk_results(extracted_data)
        elif len(extracted_data) == 1:
            merged_data = extracted_data[0]
        else:
            logger.debug(f"Empty list extracted for {result.url}")
            return None
    else:
        merged_data = extracted_data

    if not isinstance(merged_data, dict):
        logger.debug(f"Invalid merged data type for {result.url}: {type(merged_data)}")
        return None

    try:
        validated_data = ListingData(**merged_data)
        final_data = validated_data.model_dump(exclude_none=True)
        logger.debug(f"Validated data: {json.dumps(final_data, default=str)}")
    except ValidationError as e:
        logger.error(f"Validation error for {result.url}: {str(e)}", exc_info=True)
        if (merged_data.get("listing", {}).get("listing_title") and 
            merged_data.get("listing", {}).get("price") and 
            merged_data.get("files")):
            final_data = merged_data
            logger.debug(f"Using raw merged data due to optional fields missing")
        else:
            return None

    if "files" in final_data:
        final_data["files"] = list(set(''.join(url.split()) for url in final_data["files"]))
        logger.debug(f"Cleaned files: {final_data['files']}")

    if not (final_data.get("listing", {}).get("listing_title", "").strip() and 
            final_data.get("listing", {}).get("price", "").strip() and 
            final_data.get("files", [])):
        logger.debug(f"Invalid listing data for {result.url}: Missing required fields")
        return None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output = {
        "source_url": result.url,
        "data": final_data,
        "progress": 33,
        "status": "In Progress",
        "scraped_at": now,
        "target_id": target_id,
        "agent_status": [
            {"agent_name": "Scraping Agent", "status": "Success", "start_time": now, "end_time": now},
            {"agent_name": "Cleaning Agent", "status": "Queued", "start_time": now, "end_time": now},
            {"agent_name": "Extracting Agent", "status": "Queued", "start_time": now, "end_time": now}
        ]
    }
    logger.info(f"Processed valid listing at {result.url}: {final_data['listing']['listing_title']}")
    return output

async def scrape_website(crawler: AsyncWebCrawler, url: str, target_id: str, listing_format: str, retries: int = 2, rate_limit_delay: float = 3.0, task_queue=None, processing_lock: asyncio.Lock = None) -> Optional[Dict]:
    """Scrape a website, process individual listings sequentially, and queue internal links."""
    for attempt in range(retries + 1):
        try:
            scorer = KeywordRelevanceScorer(keywords=["property", "sale", "house", "price"])
            instruction = """Extract detailed information from the provided markdown content about a single property listing. Return the data in JSON format matching the provided schema. Focus on extracting:
            - Address details (country, region, city, district)
            - Property coordinates. You have to look for property coordinates from the page. You will guess if you don't find it. Use the map center first, almost all listings are bound to have it. You will only guess( based on description and location ) when there isn't a specific mapping listed on the page.  (latitude and longitude, if not available, guess using location data. I want you to always guess with whatever information that you have. The latitude and longitude must not be empty fields. They must be entered into property. But guess if and only if you can't find the data from the mapping. )
            - Listing details (title, description, price, currency, status, type, category)
            - Features (e.g., bedrooms, bathrooms, pool, garage - include all property attributes here, no separate amenities. I want to include property size in here too. It should be converted into square kilometer.)
            - File URLs (e.g., images, documents). I want you to get every image that's possible and to list them accordingly. You have to make sure that every image is extracted and added to the JSON. 
            - Contact information (phone number, name, email, company)
            If critical data (title, price, files) is missing, return an empty object."""

            run_config = CrawlerRunConfig(
                deep_crawl_strategy=BestFirstCrawlingStrategy(max_depth=2, max_pages=20, url_scorer=scorer),
                cache_mode="BYPASS",
                verbose=True,
                page_timeout=60000,
                exclude_external_links=True,
                exclude_social_media_links=True,
            )
            
            results = await crawler.arun(url=url, config=run_config)
            logger.debug(f"Crawled {len(results)} pages: {[r.url for r in results]}")
            if not results or not isinstance(results, list) or len(results) == 0:
                logger.error(f"No results returned for {url}")
                return None

            # Process each result
            processed_any = False
            base_domain = urlparse(url).netloc  # e.g., "www.properstar.com"

            for result in results:
                logger.debug(f"Checking URL: {result.url}")
                # Use listing_format from the scraping target to identify individual listings
                is_individual = result.url.startswith(listing_format)
                logger.debug(f"Is individual (based on {listing_format}): {is_individual}")

                # Queue internal links from all pages (hubs and listings)
                internal_links = [link for link in (result.links or []) if urlparse(link).netloc == base_domain]
                if task_queue and internal_links:
                    for link in internal_links[:10]:
                        task_queue.put({"website_url": link, "target_id": target_id, "listing_format": listing_format})
                        logger.debug(f"Queued internal link: {link}")

                if not is_individual:
                    logger.debug(f"Non-individual page (hub): {result.url}, queued {len(internal_links)} internal links")
                    continue  # Skip to next result

                # Process individual listing sequentially with lock
                logger.debug(f"Individual listing detected: {result.url}")
                if processing_lock:
                    async with processing_lock:
                        markdown_content = result.markdown or ""
                        total_tokens = count_tokens(markdown_content)
                        logger.debug(f"Total tokens in markdown: {total_tokens}")
                        max_input_tokens = 7000

                        if total_tokens > max_input_tokens:
                            chunks = split_into_chunks(markdown_content, max_tokens=max_input_tokens)
                            extracted_chunks = []
                            for i, chunk in enumerate(chunks):
                                logger.debug(f"Processing chunk {i+1}/{len(chunks)} with {count_tokens(chunk)} tokens")
                                chunk_result = await extract_with_openai(chunk, ListingData.model_json_schema(), instruction)
                                extracted_chunks.append(json.loads(chunk_result))
                                if i < len(chunks) - 1:
                                    await asyncio.sleep(rate_limit_delay)
                            result.extracted_content = json.dumps(extracted_chunks)
                        else:
                            result.extracted_content = await extract_with_openai(markdown_content, ListingData.model_json_schema(), instruction)

                        processed_data = await process_page(result, target_id)
                        if processed_data:
                            scraping_result_id = post_to_scraping_results(processed_data)
                            if scraping_result_id:
                                logger.info(f"Successfully posted {result.url} to scraping-results: {scraping_result_id}")
                                processed_any = True
                            else:
                                logger.error(f"Failed to post scraping result for {result.url}")
                        else:
                            logger.debug(f"No valid data processed for {result.url}")

            # Return None if no listings were processed, indicating this was a hub crawl
            return processed_data if processed_any else None

        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{retries + 1} failed for {url}: {str(e)}")
            if attempt == retries:
                logger.error(f"All retries exhausted for {url}")
                return None
            await asyncio.sleep(5)

def post_to_scraping_results(data):
    """Post scraped data to Boingo API."""
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    response = requests.post(f"{BOINGO_API_URL}/scraping-results/", headers=headers, json=data)
    if response.status_code == 201:
        scraping_result_id = response.json()["data"]["id"]
        logger.info(f"Posted to scraping-results: {scraping_result_id}")
        return scraping_result_id
    logger.error(f"Failed to post: {response.status_code} - {response.text}")
    return None

def fetch_scraping_targets():
    """Fetch URLs from the scraping-target endpoint and return a list of tasks."""
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        response = requests.get(f"{BOINGO_API_URL}/scraping-target", headers=headers, timeout=30)
        if response.status_code != 200:
            logger.error(f"Failed to fetch scraping targets: {response.status_code} - {response.text}")
            return []
        data = response.json()
        if data.get("status") != 200 or "data" not in data or "rows" not in data["data"]:
            logger.error(f"Invalid response format: {json.dumps(data, indent=2)}")
            return []
        tasks = [
            {"website_url": row["website_url"], "target_id": row["id"], "listing_format": row.get("listing_url_format", "")}
            for row in data["data"]["rows"]
            if row.get("status") == "Active" and "listing_url_format" in row
        ]
        logger.info(f"Fetched {len(tasks)} active scraping targets")
        return tasks
    except Exception as e:
        logger.error(f"Error fetching scraping targets: {str(e)}", exc_info=True)
        return []

async def process_scraping_tasks():
    """Fetch scraping targets, queue them, and process each task sequentially."""
    if not acquire_lock():
        logger.info("Another process is running, skipping...")
        return

    try:
        tasks = fetch_scraping_targets()
        if not tasks:
            logger.info("No active scraping targets found")
            return

        task_queue = queue.Queue()
        for task in tasks:
            task_queue.put(task)
            logger.debug(f"Queued task: {task['website_url']} (Target ID: {task['target_id']})")

        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=True,
            use_persistent_context=True,
            user_data_dir="browser_data",
            extra_args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        processing_lock = asyncio.Lock()

        async with AsyncWebCrawler(config=browser_config) as crawler:
            while not task_queue.empty():
                task = task_queue.get()
                url = task["website_url"]
                target_id = task["target_id"]
                listing_format = task["listing_format"]
                logger.info(f"Scraping {url} (Target ID: {target_id})")
                try:
                    result = await scrape_website(
                        crawler,
                        url,
                        target_id,
                        listing_format,
                        retries=2,
                        rate_limit_delay=3.0,
                        task_queue=task_queue,
                        processing_lock=processing_lock
                    )
                    if result:
                        logger.info(f"Scraping completed for {url}, posted to scraping-results")
                    else:
                        logger.debug(f"No data processed for {url} (likely a hub or invalid)")
                except Exception as e:
                    logger.error(f"Scraping failed for {url}: {str(e)}", exc_info=True)
                finally:
                    task_queue.task_done()
                await asyncio.sleep(3.0)

    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Error processing scraping tasks: {str(e)}", exc_info=True)
    finally:
        release_lock()

if __name__ == "__main__":
    asyncio.run(process_scraping_tasks())