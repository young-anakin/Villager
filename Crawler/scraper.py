import logging
import requests
import json
from datetime import datetime
import asyncio
import os
from crawl4ai import AsyncWebCrawler, LLMConfig, CrawlerRunConfig, LLMExtractionStrategy, BestFirstCrawlingStrategy, KeywordRelevanceScorer
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from queue_manager import get_next_task, acquire_lock, release_lock
from ..app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY  # Adjusted path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    features: List[Feature] = Field(default_factory=list)  # Amenities will now go here
    files: List[str]
    contact: Contact

def merge_chunk_results(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple chunk results into a single ListingData-compliant dictionary."""
    logger.debug(f"Merging {len(chunks)} chunks")
    merged = {
        "address": {"country": "", "region": "", "city": "", "district": ""},
        "property": {"lat": None, "lng": None},
        "listing": {"listing_title": "", "description": "", "price": "", "currency": "", "status": "", "listing_type": "", "category": ""},
        "features": [],  # Will now include both features and former amenities
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
            # Merge features
            if "features" in chunk and isinstance(chunk["features"], list):
                existing_features = {item["feature"]: item for item in merged["features"]}
                for item in chunk["features"]:
                    if item["feature"] not in existing_features:
                        merged["features"].append(item)
                        logger.debug(f"Added to features: {item}")
            # Merge amenities into features
            if "amenities" in chunk and isinstance(chunk["amenities"], list):
                existing_features = {item["feature"]: item for item in merged["features"]}
                for amenity in chunk["amenities"]:
                    feature_item = {"feature": amenity["amenity"], "value": amenity.get("value", "Yes")}
                    if feature_item["feature"] not in existing_features:
                        merged["features"].append(feature_item)
                        logger.debug(f"Converted amenity to feature: {feature_item}")
            # Merge files
            if "files" in chunk and isinstance(chunk["files"], list):
                merged["files"].extend(chunk["files"])
                merged["files"] = list(set(merged["files"]))
                logger.debug(f"Merged files: {merged['files']}")
        logger.debug(f"Merged result: {json.dumps(merged, default=str)}")
        return merged
    except Exception as e:
        logger.error(f"Error merging chunks: {str(e)}", exc_info=True)
        raise

async def process_page(result, target_id: str) -> Optional[Dict[str, Any]]:
    """Process a crawled page, save markdown versions, and return validated listing data."""
    logger.debug(f"Processing page: {result.url}")
    if not result.success:
        logger.debug(f"Page crawl failed: {result.url}")
        return None

    safe_url = "".join(c if c.isalnum() or c in "-_" else "_" for c in result.url)
    markdown_dir = f"markdown_{target_id}"
    os.makedirs(markdown_dir, exist_ok=True)
    
    markdown_types = {
        "fit_markdown": result.markdown,
        "markdown_with_citation": getattr(result, "markdown_with_citation", ""),
        "raw_markdown": getattr(result, "raw_markdown", "")
    }
    
    for markdown_type, content in markdown_types.items():
        if content:
            file_path = os.path.join(markdown_dir, f"{markdown_type}_{safe_url}.md")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.debug(f"Saved {markdown_type} to {file_path} (Length: {len(content)} characters)")
            except Exception as e:
                logger.error(f"Failed to save {markdown_type} to {file_path}: {str(e)}", exc_info=True)
        else:
            logger.debug(f"No content for {markdown_type} at {result.url}")

    lines = result.markdown.split("\n")
    top_level_title_count = sum(1 for line in lines if line.strip().startswith("# ") and not line.strip().startswith("##"))
    if top_level_title_count != 1:
        logger.debug(f"Skipping {result.url}: Incorrect title count ({top_level_title_count})")
        return None

    metadata = result.metadata
    if metadata:
        multi_listing_keywords = ["listings", "properties", "results", "search", "for sale"]
        if any(kw in metadata.get("og:title", "").lower() for kw in multi_listing_keywords) or \
           any(kw in metadata.get("og:description", "").lower() for kw in multi_listing_keywords) or \
           any(kw in metadata.get("title", "").lower() for kw in multi_listing_keywords):
            logger.debug(f"Skipping {result.url}: Metadata suggests multiple listings")
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
        final_data = validated_data.dict(exclude_none=True)
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

    now = datetime.utcnow().isoformat() + "Z"
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

async def scrape_website(url: str, target_id: str):
    """Scrape a website and return processed data."""
    scorer = KeywordRelevanceScorer(keywords=["property", "sale", "house", "price"])
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider="openai/gpt-4", api_token=OPENAI_API_KEY),
        schema=ListingData.model_json_schema(),
        extraction_type="schema",
        instruction="""Extract detailed information from the provided markdown content about a single property listing. Return the data in JSON format matching the provided schema. Focus on extracting:
        - Address details (country, region, city, district)
        - Property coordinates (latitude and longitude, if available)
        - Listing details (title, description, price, currency, status, type, category)
        - Features (e.g., bedrooms, bathrooms, pool, garage - include all property attributes here, no separate amenities)
        - File URLs (e.g., images, documents)
        - Contact information (phone number, name, email, company)
        If critical data (title, price, files) is missing, return an empty object."""
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(max_depth=1, max_pages=1, url_scorer=scorer),
        extraction_strategy=llm_strategy
    )
    async with AsyncWebCrawler(verbose=True) as crawler:
        results = await crawler.arun(url=url, config=config)
        if not results or not isinstance(results, list) or len(results) == 0:
            logger.error(f"No results returned for {url}")
            return None
        result = results[0]  # Take the first result
        logger.debug(f"Crawled URL: {result.url}")
    return await process_page(result, target_id)

def post_to_scraping_results(data):
    """Post scraped data to Boingo API."""
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    response = requests.post(f"{BOINGO_API_URL}/scraping-results/", headers=headers, json=data)
    if response.status_code == 201:
        scraping_result_id = response.json()["data"]["id"]
        logger.info(f"Posted to scraping-results: {scraping_result_id}")
        return scraping_result_id
    logger.error(f"Failed to post: {response.text}")
    return None

def process_scraping_task():
    """Process the next scraping task from the queue."""
    if not acquire_lock():
        logger.info("Another process is running, skipping...")
        return
    try:
        task = get_next_task("scraping")
        if task:
            url = task["website_url"]
            target_id = task["target_id"]
            logger.info(f"Scraping {url}")
            result = asyncio.run(scrape_website(url, target_id))
            if result:
                scraping_result_id = post_to_scraping_results(result)
                if scraping_result_id:
                    logger.info(f"Scraping completed for {url}, result ID: {scraping_result_id}")
                else:
                    logger.error(f"Failed to post scraping result for {url}")
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}", exc_info=True)
    finally:
        release_lock()

if __name__ == "__main__":
    process_scraping_task()