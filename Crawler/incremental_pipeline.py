import logging
import requests
import json
from datetime import datetime, timezone
import asyncio
import tiktoken
from openai import AsyncOpenAI
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, BestFirstCrawlingStrategy, KeywordRelevanceScorer
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.FileHandler("incremental_pipeline.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Set up requests session with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Pydantic Models
class Address(BaseModel):
    country: str
    region: str
    city: str
    district: Optional[str] = None

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
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    company: Optional[str] = None

class ListingData(BaseModel):
    address: Address
    property: Optional[Property] = None
    listing: Listing
    features: List[Feature] = Field(default_factory=list)
    files: List[str] = Field(default_factory=list)
    contact: Optional[Contact] = None

# Chunking and Merging Logic
def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        return 0

def split_into_chunks(content: str, max_tokens: int = 7000) -> List[str]:
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
    logger.debug(f"Merging {len(chunks)} chunks")
    merged = {
        "address": {"country": "", "region": "", "city": "", "district": None},
        "property": {"lat": None, "lng": None},
        "listing": {"listing_title": "", "description": "", "price": "", "currency": "", "status": "", "listing_type": "", "category": ""},
        "features": [],
        "files": [],
        "contact": {"phone_number": None, "first_name": None, "last_name": None, "email_address": None, "company": None}
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
            if "files" in chunk and isinstance(chunk["files"], list):
                merged["files"].extend(chunk["files"])
                merged["files"] = list(set(merged["files"]))
                logger.debug(f"Merged files: {merged['files']}")
        logger.debug(f"Merged result: {json.dumps(merged, default=str)}")
        return merged
    except Exception as e:
        logger.error(f"Error merging chunks: {str(e)}")
        raise

# OpenAI Extraction with Retry Logic
async def extract_with_openai(content: str, schema: Dict[str, Any], instruction: str, max_retries: int = 3) -> str:
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

# Scraping Logic
async def scrape_listing(content: str) -> Dict[str, Any]:
    instruction = """Extract detailed information from the provided markdown content about a single property listing. Return the data in JSON format matching the provided schema. Focus on extracting:
    - Address details (country, region, city, district)
    - Property coordinates (latitude and longitude, guess if not available based on location data)
    - Listing details (title, description, price, currency, status, type, category)
    - Features (e.g., bedrooms, bathrooms, include all property attributes)
    - File URLs (e.g., images, documents)
    - Contact information (phone number, name, email, company)
    If critical data (title, price, files) is missing, return an empty object."""

    total_tokens = count_tokens(content)
    max_input_tokens = 7000
    if total_tokens > max_input_tokens:
        chunks = split_into_chunks(content, max_tokens=max_input_tokens)
        extracted_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} with {count_tokens(chunk)} tokens")
            chunk_result = await extract_with_openai(chunk, ListingData.model_json_schema(), instruction)
            extracted_chunks.append(json.loads(chunk_result))
            if i < len(chunks) - 1:
                await asyncio.sleep(3.0)
        merged_data = merge_chunk_results(extracted_chunks)
    else:
        extracted_data = await extract_with_openai(content, ListingData.model_json_schema(), instruction)
        merged_data = json.loads(extracted_data)

    if not merged_data or not merged_data.get("listing", {}).get("listing_title"):
        logger.debug(f"Invalid extracted data: {json.dumps(merged_data, default=str)}")
        return {}

    try:
        validated_data = ListingData(**merged_data)
        final_data = validated_data.model_dump(exclude_none=True)
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        if (merged_data.get("listing", {}).get("listing_title") and 
            merged_data.get("listing", {}).get("price") and 
            merged_data.get("files")):
            final_data = merged_data
        else:
            return {}

    if "files" in final_data:
        final_data["files"] = list(set(''.join(url.split()) for url in final_data["files"]))
    return final_data

# Cleaning Logic
async def clean_listing(data: Dict[str, Any]) -> Dict[str, Any]:
    instruction = """Translate the following text fields to English if needed, and refine them to be clear and concise. Return as JSON with the same structure."""
    text_fields = {
        "address": data.get("address", {}),
        "listing": data.get("listing", {}),
        "contact": data.get("contact", {}),
        "features": data.get("features", [])
    }
    prompt = f"{instruction}\n\nInput data:\n{json.dumps(text_fields, default=str)}"
    
    cleaned_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    cleaned_data = json.loads(cleaned_text)
    data.update(cleaned_data)
    return data

# Formatting Logic
async def format_listing(data: Dict[str, Any]) -> Dict[str, Any]:
    instruction = """Enhance these fields to be polished and appealing in English. Add lat/lng to 'property' if missing, guessing based on address. Return as JSON with the same structure plus 'property' with 'lat' and 'lng'."""
    text_fields = {
        "address": data.get("address", {}),
        "listing": data.get("listing", {}),
        "contact": data.get("contact", {}),
        "features": data.get("features", [])
    }
    prompt = f"{instruction}\n\nInput data:\n{json.dumps(text_fields, default=str)}"
    
    formatted_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    formatted_data = json.loads(formatted_text)
    data.update(formatted_data)
    if "property" not in data or not data["property"].get("lat"):
        data["property"] = formatted_data.get("property", {"lat": "0.0", "lng": "0.0"})
    return data

# API Interaction Functions
def post_to_scraping_results(data: Dict[str, Any]) -> Optional[str]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        response = session.post(f"{BOINGO_API_URL}/scraping-results/", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["data"]["id"]
    except Exception as e:
        logger.error(f"Failed to post to /scraping-results/: {str(e)}")
        return None

def update_scraping_results(scraping_result_id: str, data: Dict[str, Any]) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    data["id"] = scraping_result_id
    try:
        response = session.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=data)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to update /scraping-results/{scraping_result_id}: {str(e)}")
        return False

# Process a Single Listing with Pauses
async def process_listing(result, target_id: str) -> Optional[Dict[str, Any]]:
    if not result.success or not result.markdown:
        logger.debug(f"Skipping {result.url}: No valid content")
        return None

    # Step 1: Scrape (33%)
    try:
        raw_data = await scrape_listing(result.markdown)
        if not raw_data:
            logger.debug(f"No valid data scraped for {result.url}")
            return None

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        base_payload = {
            "source_url": result.url,
            "data": raw_data,
            "progress": 33,
            "status": "In Progress",
            "scraped_at": now,
            "target_id": target_id,
            "agent_status": [
                {"agent_name": "Scraping Agent", "status": "Success", "start_time": now, "end_time": now},
                {"agent_name": "Cleaning Agent", "status": "Queued", "start_time": now, "end_time": now},
                {"agent_name": "Formatting Agent", "status": "Queued", "start_time": now, "end_time": now}
            ]
        }

        scraping_result_id = post_to_scraping_results(base_payload)
        if not scraping_result_id:
            logger.error(f"Failed to post scraped data for {result.url}")
            return None
        logger.info(f"Scraped: Posted {result.url} to /scraping-results: {scraping_result_id}")
    except Exception as e:
        logger.error(f"Error scraping listing {result.url}: {str(e)}")
        return None

    # Pause before cleaning
    await asyncio.sleep(3.0)

    # Step 2: Clean (66%)
    try:
        cleaned_data = await clean_listing(raw_data.copy())
        base_payload["data"] = cleaned_data
        base_payload["progress"] = 66
        base_payload["agent_status"][1]["status"] = "Success"
        base_payload["agent_status"][1]["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not update_scraping_results(scraping_result_id, base_payload):
            logger.error(f"Failed to update cleaned data for {result.url}")
            return None
        logger.info(f"Cleaned: Updated {result.url} in /scraping-results: {scraping_result_id}")
    except Exception as e:
        logger.error(f"Error cleaning listing {result.url}: {str(e)}")
        return None

    # Pause before formatting
    await asyncio.sleep(3.0)

    # Step 3: Format (100%)
    try:
        formatted_data = await format_listing(cleaned_data.copy())
        base_payload["data"] = formatted_data
        base_payload["progress"] = 100
        base_payload["status"] = "Success"
        base_payload["agent_status"][2]["status"] = "Success"
        base_payload["agent_status"][2]["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if not update_scraping_results(scraping_result_id, base_payload):
            logger.error(f"Failed to update formatted data for {result.url}")
            return None
        logger.info(f"Formatted: Updated {result.url} in /scraping-results: {scraping_result_id}")

        return base_payload
    except Exception as e:
        logger.error(f"Error formatting listing {result.url}: {str(e)}")
        return None

# Main Scrape and Process Function
async def scrape_and_process(url: str, target_id: str, listing_format: str):
    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        scorer = KeywordRelevanceScorer(keywords=["property", "sale", "house"])
        run_config = CrawlerRunConfig(
            deep_crawl_strategy=BestFirstCrawlingStrategy(max_depth=2, max_pages=20, url_scorer=scorer),
            cache_mode="BYPASS"
        )
        
        # Step 1: Crawl and collect matching URLs
        results = await crawler.arun(url=url, config=run_config)
        matching_listings = [result for result in results if result.url.startswith(listing_format)]
        logger.info(f"Found {len(matching_listings)} listings matching {listing_format}")

        # Step 2: Process each listing one by one with full pauses
        for result in matching_listings:
            logger.debug(f"Processing {result.url}")
            await process_listing(result, target_id)
            # Pause between listings
            await asyncio.sleep(3.0)

if __name__ == "__main__":
    asyncio.run(scrape_and_process("https://example.com", "abc123", "https://example.com/listing/"))