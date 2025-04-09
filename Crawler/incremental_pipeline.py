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
from playwright.async_api import async_playwright

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
    email: Optional[str] = None
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
        "address": {"country": "", "region": "", "city": ""},
        "property": {"lat": None, "lng": None},
        "listing": {"listing_title": "", "description": "", "price": "", "currency": "", "status": "", "listing_type": "", "category": ""},
        "features": [],
        "files": [],
        "contact": {"first_name": None, "last_name": None, "phone_number": None, "email": None, "company": None}
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
    - Address details (country, region, city). Do not include 'district'.
    - Property coordinates (latitude and longitude, guess if not available based on location data)
    - Listing details (title, description, price, currency, status, type, category)
    - Features (e.g., bedrooms, bathrooms, include all property attributes)
    - File URLs (e.g., images, documents)
    - Contact information (phone number, first_name, last_name, email, company)
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
        if "contact" in final_data and "email_address" in final_data["contact"]:
            final_data["contact"]["email"] = final_data["contact"].pop("email_address")
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return {}

    if "files" in final_data:
        final_data["files"] = list(set(''.join(url.split()) for url in final_data["files"]))
    return final_data

# Cleaning Logic
async def clean_listing(data: Dict[str, Any]) -> Dict[str, Any]:
    instruction = """Translate the following text fields to English, and refine them to be clear and concise. Do not include 'district' in 'address'. Return as JSON with the same structure."""
    text_fields = {
        "address": data.get("address", {"country": "", "region": "", "city": ""}),
        "listing": data.get("listing", {}),
        "contact": data.get("contact", {}),
        "features": data.get("features", [])
    }
    prompt = f"{instruction}\n\nInput data:\n{json.dumps(text_fields, default=str)}"
    
    cleaned_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    cleaned_data = json.loads(cleaned_text)
    if "contact" in cleaned_data and "email_address" in cleaned_data["contact"]:
        cleaned_data["contact"]["email"] = cleaned_data["contact"].pop("email_address")
    data.update(cleaned_data)
    return data

# Formatting Logic
async def format_listing(data: Dict[str, Any]) -> Dict[str, Any]:
    instruction = """Enhance these fields to be polished and appealing in English. Add lat/lng to 'property' if missing, guessing based on address. Do not include 'district' in 'address'. Return as JSON with the same structure plus 'property' with 'lat' and 'lng'."""
    text_fields = {
        "address": data.get("address", {"country": "", "region": "", "city": ""}),
        "listing": data.get("listing", {}),
        "contact": data.get("contact", {}),
        "features": data.get("features", [])
    }
    prompt = f"{instruction}\n\nInput data:\n{json.dumps(text_fields, default=str)}"
    
    formatted_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    formatted_data = json.loads(formatted_text)
    if "contact" in formatted_data and "email_address" in formatted_data["contact"]:
        formatted_data["contact"]["email"] = formatted_data["contact"].pop("email_address")
    data.update(formatted_data)
    if "property" not in data or not data["property"].get("lat"):
        data["property"] = formatted_data.get("property", {"lat": "0.0", "lng": "0.0"})
    return data

# API Interaction Functions
def fetch_scraping_targets() -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        logger.debug(f"Fetching targets from {BOINGO_API_URL}/scraping-target/")
        response = session.get(f"{BOINGO_API_URL}/scraping-target/", headers=headers, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        
        if not isinstance(response_data, dict) or "data" not in response_data or "rows" not in response_data["data"]:
            logger.error(f"Unexpected response format: {response_data}")
            return []
        
        targets = response_data["data"]["rows"]
        logger.info(f"Fetched {len(targets)} targets")
        return targets
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch targets: {str(e)}")
        return []

def fetch_queued_listings() -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        logger.debug(f"Fetching queued listings from {BOINGO_API_URL}/scraping-results/queued-agent-status")
        response = session.get(f"{BOINGO_API_URL}/scraping-results/queued-agent-status", headers=headers, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        
        if response_data.get("status") != 200 or "data" not in response_data or "rows" not in response_data["data"]:
            logger.error(f"Unexpected response format: {response_data}")
            return []
        
        queued_listings = response_data["data"]["rows"]
        logger.info(f"Fetched {len(queued_listings)} queued listings")
        return queued_listings
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch queued listings: {str(e)}")
        return []

def fetch_single_queued_listing(scraping_result_id: str) -> Optional[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        response = session.get(f"{BOINGO_API_URL}/scraping-results/queued-agent-status", headers=headers, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        queued_listings = response_data["data"]["rows"]
        for listing in queued_listings:
            if listing["id"] == scraping_result_id:
                logger.debug(f"Fetched queued listing for ID {scraping_result_id}: {json.dumps(listing, default=str)}")
                return listing
        logger.error(f"No queued listing found for ID {scraping_result_id}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch queued listing {scraping_result_id}: {str(e)}")
        return None

def debug_payload(payload: Dict[str, Any]) -> None:
    logger.debug("Debugging payload structure and types:")
    for key, value in payload.items():
        if isinstance(value, dict):
            logger.debug(f"  {key} (dict):")
            for sub_key, sub_value in value.items():
                logger.debug(f"    {sub_key}: {sub_value} (type: {type(sub_value).__name__})")
        elif isinstance(value, list):
            logger.debug(f"  {key} (list of {len(value)} items):")
            for i, item in enumerate(value):
                logger.debug(f"    Item {i}: {item} (type: {type(item).__name__})")
        else:
            logger.debug(f"  {key}: {value} (type: {type(value).__name__})")

def post_to_scraping_results(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        debug_payload(data)
        logger.debug(f"Posting to {BOINGO_API_URL}/scraping-results/: {json.dumps(data, default=str)}")
        response = session.post(f"{BOINGO_API_URL}/scraping-results/", headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        result_id = response_data["data"]["id"]
        logger.debug(f"Successfully posted, got ID: {result_id}, Response: {json.dumps(response_data, default=str)}")
        return {"id": result_id, "data": response_data["data"]}
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post to /scraping-results/: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"API response: {e.response.text}")
        return None

def update_scraping_results(scraping_result_id: str, data: Dict[str, Any]) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": scraping_result_id,
        "source_url": data.get("source_url", ""),
        "data": {
            "address": {
                "country": data.get("data", {}).get("address", {}).get("country", ""),
                "region": data.get("data", {}).get("address", {}).get("region", ""),
                "city": data.get("data", {}).get("address", {}).get("city", "")
            },
            "property": {
                "lat": data.get("data", {}).get("property", {}).get("lat", "0.0"),
                "lng": data.get("data", {}).get("property", {}).get("lng", "0.0")
            },
            "listing": {
                "listing_title": data.get("data", {}).get("listing", {}).get("listing_title", ""),
                "description": data.get("data", {}).get("listing", {}).get("description", ""),
                "price": data.get("data", {}).get("listing", {}).get("price", ""),
                "currency": data.get("data", {}).get("listing", {}).get("currency", ""),
                "status": data.get("data", {}).get("listing", {}).get("status", ""),
                "listing_type": data.get("data", {}).get("listing", {}).get("listing_type", ""),
                "category": data.get("data", {}).get("listing", {}).get("category", "")
            },
            "features": data.get("data", {}).get("features", []),
            "files": data.get("data", {}).get("files", []),
            "contact": {
                "first_name": data.get("data", {}).get("contact", {}).get("first_name", ""),
                "last_name": data.get("data", {}).get("contact", {}).get("last_name", ""),
                "phone_number": data.get("data", {}).get("contact", {}).get("phone_number", ""),
                "email": data.get("data", {}).get("contact", {}).get("email", ""),
                "company": data.get("data", {}).get("contact", {}).get("company", "")
            }
        },
        "progress": data.get("progress", 0),  # Fixed: Use top-level progress directly
        "status": data.get("status", "In Progress"),
        "target_id": data.get("target_id", ""),
        "scraped_at": data.get("scraped_at", now),
        "last_updated": now
    }
    try:
        logger.debug(f"Updating {BOINGO_API_URL}/scraping-results with: {json.dumps(payload, default=str)}")
        response = session.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully updated scraping-results for ID {scraping_result_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to update /scraping-results/{scraping_result_id}: {str(e)}")
        return False

def update_agent_status(agent: Dict[str, Any], scraping_result_id: str) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": agent["id"],
        "agent_name": agent["agent_name"],
        "status": "Success",
        "start_time": agent["start_time"],
        "end_time": now,
        "scraping_result_id": scraping_result_id
    }
    try:
        logger.debug(f"Updating agent-status for {agent['agent_name']}: {json.dumps(payload, default=str)}")
        response = session.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully updated agent-status for {agent['agent_name']} (ID: {agent['id']})")
        return True
    except Exception as e:
        logger.error(f"Failed to update agent-status for {agent['agent_name']}: {str(e)}")
        return False

# Process a Single Listing
async def process_listing(result, target_id: str, scraping_result_id: str, initial_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not result.success or not result.markdown:
        logger.debug(f"Skipping {result.url}: No valid content")
        return None

    base_payload = initial_payload.copy()
    base_payload["source_url"] = result.url
    base_payload["target_id"] = target_id

    # Step 1: Scrape (33%)
    try:
        logger.debug(f"Scraping {result.url} for ID {scraping_result_id}")
        raw_data = await scrape_listing(result.markdown)
        if not raw_data:
            logger.debug(f"No valid data scraped for {result.url}")
            return None

        base_payload["data"] = raw_data
        base_payload["progress"] = 33
        if not update_scraping_results(scraping_result_id, base_payload):
            logger.error(f"Failed to update scraped data for {result.url}")
            return None
        
        scraping_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Scraping Agent"), None)
        if not scraping_agent:
            logger.error(f"No Scraping Agent found in agent_status: {json.dumps(base_payload['agent_status'], default=str)}")
            return None
        if "id" not in scraping_agent:
            logger.error(f"Scraping Agent missing 'id' in agent_status: {json.dumps(base_payload['agent_status'], default=str)}")
            return None
        if not update_agent_status(scraping_agent, scraping_result_id):
            logger.error(f"Failed to update Scraping Agent status for {result.url}")
            return None
        logger.info(f"Scraped: Updated {result.url} to progress 33")
    except Exception as e:
        logger.error(f"Error scraping listing {result.url}: {str(e)}", exc_info=True)
        return None

    await asyncio.sleep(3.0)

    # Step 2: Clean (66%)
    try:
        logger.debug(f"Cleaning data for {result.url} (ID: {scraping_result_id})")
        cleaned_data = await clean_listing(raw_data.copy())
        base_payload["data"] = cleaned_data
        base_payload["progress"] = 66
        if not update_scraping_results(scraping_result_id, base_payload):
            logger.error(f"Failed to update cleaned data for {result.url}")
            return None
        
        cleaning_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Cleaning Agent"), None)
        if not cleaning_agent or "id" not in cleaning_agent:
            logger.error(f"No valid Cleaning Agent with ID found in agent_status: {json.dumps(base_payload['agent_status'], default=str)}")
            return None
        if not update_agent_status(cleaning_agent, scraping_result_id):
            logger.error(f"Failed to update Cleaning Agent status for {result.url}")
            return None
        logger.info(f"Cleaned: Updated {result.url} to progress 66")
    except Exception as e:
        logger.error(f"Error cleaning listing {result.url}: {str(e)}", exc_info=True)
        return None

    await asyncio.sleep(3.0)

    # Step 3: Format (100%)
    try:
        logger.debug(f"Formatting data for {result.url} (ID: {scraping_result_id})")
        formatted_data = await format_listing(cleaned_data.copy())
        base_payload["data"] = formatted_data
        base_payload["progress"] = 100
        base_payload["status"] = "Success"
        if not update_scraping_results(scraping_result_id, base_payload):
            logger.error(f"Failed to update formatted data for {result.url}")
            return None
        
        formatting_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Extracting Agent"), None)
        if not formatting_agent or "id" not in formatting_agent:
            logger.error(f"No valid Extracting Agent with ID found in agent_status: {json.dumps(base_payload['agent_status'], default=str)}")
            return None
        if not update_agent_status(formatting_agent, scraping_result_id):
            logger.error(f"Failed to update Extracting Agent status for {result.url}")
            return None
        logger.info(f"Formatted: Updated {result.url} to progress 100")
        return base_payload
    except Exception as e:
        logger.error(f"Error formatting listing {result.url}: {str(e)}", exc_info=True)
        return None

# Main Scrape and Process Function
async def scrape_and_process(url: str, target_id: str, listing_format: str, crawler: AsyncWebCrawler):
    scorer = KeywordRelevanceScorer(keywords=["property", "sale", "house"])
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(max_depth=3, max_pages=20, url_scorer=scorer),
        cache_mode="BYPASS"
    )
    
    logger.debug(f"Starting crawl at {url} with listing_format {listing_format}")
    results = await crawler.arun(url=url, config=run_config)
    logger.debug(f"Crawled {len(results)} pages: {[r.url for r in results]}")
    
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

    processed_ids = set()
    for result in results:
        if result.url.startswith(listing_format) and result.success and result.markdown:
            logger.debug(f"Found valid listing: {result.url}, pausing crawl to process")
            initial_payload = initial_payload_template.copy()
            initial_payload["source_url"] = result.url
            post_result = post_to_scraping_results(initial_payload)
            
            if post_result:
                scraping_result_id = post_result["id"]
                if scraping_result_id in processed_ids:
                    logger.debug(f"Skipping already processed ID {scraping_result_id}")
                    continue
                
                queued_listing = fetch_single_queued_listing(scraping_result_id)
                if queued_listing and "agents" in queued_listing:
                    initial_payload["agent_status"] = queued_listing["agents"]
                    logger.debug(f"Updated agent_status with IDs: {json.dumps(initial_payload['agent_status'], default=str)}")
                else:
                    logger.error(f"Failed to fetch agent IDs for {scraping_result_id}, using initial payload")
                
                logger.info(f"Queued listing {result.url} with ID {scraping_result_id}")
                result_data = await process_listing(result, target_id, scraping_result_id, initial_payload)
                if result_data:
                    processed_ids.add(scraping_result_id)
                    logger.info(f"Completed processing {result.url} (ID: {scraping_result_id})")
                else:
                    logger.error(f"Processing failed for {result.url}, continuing to next listing")
            else:
                logger.error(f"Failed to queue listing {result.url}, skipping processing")
            
            await asyncio.sleep(1.0)

    logger.info(f"Completed crawling {url}, processed {len(processed_ids)} listings")

    queued_listings = fetch_queued_listings()
    if not queued_listings:
        logger.warning("No additional queued listings found to process")
    else:
        for queued_listing in queued_listings:
            scraping_result_id = queued_listing["id"]
            if scraping_result_id in processed_ids:
                logger.debug(f"Skipping already processed queued listing ID {scraping_result_id}")
                continue
            
            result = next((r for r in results if r.url == queued_listing["source_url"]), None)
            if result:
                logger.debug(f"Processing queued listing {result.url} with ID {scraping_result_id}")
                queued_listing["agent_status"] = queued_listing["agents"]
                result_data = await process_listing(result, target_id, scraping_result_id, queued_listing)
                if result_data:
                    processed_ids.add(scraping_result_id)
                    logger.info(f"Completed processing queued listing {result.url} (ID: {scraping_result_id})")
                await asyncio.sleep(3.0)

    logger.info(f"Finished processing all listings for {url}, total processed: {len(processed_ids)}")

# Main Function to Fetch and Process Targets
async def main():
    targets = fetch_scraping_targets()
    if not targets:
        logger.error("No targets fetched, exiting.")
        return

    browser_config = BrowserConfig(
        browser_type="chromium",
        headless=True,
        verbose=True,
        use_persistent_context=True,
        user_data_dir="browser_data",
        extra_args=["--no-sandbox", "--disable-setuid-sandbox"]
    )
    
    async with async_playwright() as p:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for target in targets:
                url = target.get("website_url")
                target_id = target.get("id")
                listing_format = target.get("listing_url_format")

                if not url or not target_id or not listing_format:
                    logger.error(f"Invalid target: {json.dumps(target)}, skipping.")
                    continue

                logger.info(f"Processing target: URL={url}, Target ID={target_id}, Listing Format={listing_format}")
                try:
                    await scrape_and_process(url, target_id, listing_format, crawler)
                except Exception as e:
                    logger.error(f"Failed to process target {target_id}: {str(e)}", exc_info=True)
                    continue

                logger.info(f"Completed processing target {target_id}, pausing 5s")
                await asyncio.sleep(5.0)

if __name__ == "__main__":
    asyncio.run(main())