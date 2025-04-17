#!/usr/bin/env python3
import openai
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
import asyncio
import tiktoken
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY
import aiohttp
import re
import httpx

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
        logging.FileHandler("processor_worker.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

# Lock management
LOCK_FILE = os.path.join(tempfile.gettempdir(), "processor_worker.lock")

def acquire_lock():
    if os.path.exists(LOCK_FILE):
        logger.info("Another processor instance is running, exiting.")
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

# OpenAI client
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=30.0, read=30.0, write=30.0)
    )
)

# Pydantic models (same as original)
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

# Validation, chunking, and OpenAI extraction functions (same as original)
def is_valid_large_result(large_result: str) -> bool:
    if not large_result or not isinstance(large_result, str) or len(large_result.strip()) < 10:
        logger.debug("Skipping validation, content empty or too short")
        return False
    http_codes = ["403", "404", "500", "502", "503", "429"]
    error_context = ["error", "forbidden", "not found", "server", "failed", "blocked", "denied", "timeout", "unavailable"]
    for code in http_codes:
        if re.search(rf"\b{code}\b.*?(?:{'|'.join(error_context)})", large_result.lower(), re.IGNORECASE):
            logger.debug("Detected HTTP error", extra={"http_code": code})
            return False
    return True

def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error("Failed to count tokens", extra={"error": str(e)})
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
    logger.debug("Split into chunks", extra={"count": len(chunks)})
    return chunks

def merge_chunk_results(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            for section in ["address", "property", "listing", "contact"]:
                if section in chunk and isinstance(chunk[section], dict):
                    for key, value in chunk[section].items():
                        if key in merged[section] and value and (merged[section][key] is None or merged[section][key] == ""):
                            merged[section][key] = value
            if "features" in chunk and isinstance(chunk["features"], list):
                existing_features = {item["feature"]: item for item in merged["features"]}
                for item in chunk["features"]:
                    if item["feature"] not in existing_features:
                        merged["features"].append(item)
            if "files" in chunk and isinstance(chunk["files"], list):
                merged["files"].extend(chunk["files"])
                merged["files"] = list(set(merged["files"]))
        return merged
    except Exception as e:
        logger.error("Failed to merge chunks", extra={"error": str(e)})
        raise

async def extract_with_openai(content: str, schema: Dict[str, Any], instruction: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Content:\n{content}\n\nSchema:\n{schema}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            logger.debug("Extracted data", extra={"attempt": attempt + 1})
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt * 10
                logger.warning("Rate limit, retrying", extra={"attempt": attempt + 1, "wait": wait_time})
                await asyncio.sleep(wait_time)
            else:
                logger.error("Extraction failed", extra={"error": str(e)})
                return "{}"
    logger.error("Max retries exceeded")
    return "{}"

# Processing functions (same as original)
async def scrape_listing(content: str) -> Dict[str, Any]:
    instruction = """Extract detailed information from the provided markdown content about a single property listing. Return the data in JSON format matching the provided schema. Focus on extracting:
    - Address details (country, region, city). Do not include 'district'.
    - Property coordinates (latitude and longitude, guess if not available)
    - Listing details (title, description, price, currency, status, type, category)
    - Features (e.g., bedrooms, bathrooms, include all attributes)
    - ALL file URLs (e.g., images, documents) without limit
    - Contact information (phone number, first_name, last_name, email, company)
    If critical data (title, price, files) is missing, return an empty object."""
    
    total_tokens = count_tokens(content)
    if total_tokens > 7000:
        chunks = split_into_chunks(content)
        extracted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_result = await extract_with_openai(chunk, ListingData.model_json_schema(), instruction)
            extracted_chunks.append(json.loads(chunk_result))
        merged_data = merge_chunk_results(extracted_chunks)
    else:
        extracted_data = await extract_with_openai(content, ListingData.model_json_schema(), instruction)
        merged_data = json.loads(extracted_data)

    if not merged_data or not merged_data.get("listing", {}).get("listing_title"):
        logger.debug("No valid data extracted")
        return {}

    try:
        validated_data = ListingData(**merged_data)
        final_data = validated_data.model_dump(exclude_none=True)
        if "contact" in final_data and "email_address" in final_data["contact"]:
            final_data["contact"]["email"] = final_data["contact"].pop("email_address")
        logger.debug("Validated listing", extra={"title": final_data["listing"]["listing_title"]})
    except ValidationError as e:
        logger.error("Validation failed", extra={"error": str(e)})
        return {}

    if "files" in final_data:
        final_data["files"] = list(set(''.join(url.split()) for url in final_data["files"]))
    return final_data

async def clean_listing(data: Dict[str, Any]) -> Dict[str, Any]:
    instruction = """Translate the following text fields to English, and refine them to be clear and concise. Do not include 'district' in 'address'. Return as JSON with the same structure."""
    text_fields = {
        "address": data.get("address", {"country": "", "region": "", "city": ""}),
        "listing": data.get("listing", {}),
        "contact": data.get("contact", {}),
        "features": data.get("features", [])
    }
    prompt = f"{instruction}\n\nInput data:\n{text_fields}"
    
    cleaned_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    cleaned_data = json.loads(cleaned_text)
    if "contact" in cleaned_data and "email_address" in cleaned_data["contact"]:
        cleaned_data["contact"]["email"] = cleaned_data["contact"].pop("email_address")
    data.update(cleaned_data)
    logger.debug("Cleaned listing", extra={"title": data["listing"]["listing_title"]})
    return data

async def format_listing(data: Dict[str, Any]) -> Dict[str, Any]:
    instruction = """Enhance these fields to be polished and appealing in English. Add lat/lng to 'property' if missing, guessing based on address. Do not include 'district' in 'address'. Return as JSON with the same structure plus 'property' with 'lat' and 'lng'."""
    text_fields = {
        "address": data.get("address", {"country": "", "region": "", "city": ""}),
        "listing": data.get("listing", {}),
        "contact": data.get("contact", {}),
        "features": data.get("features", [])
    }
    prompt = f"{instruction}\n\nInput data:\n{text_fields}"
    
    formatted_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    formatted_data = json.loads(formatted_text)
    if "contact" in formatted_data and "email_address" in formatted_data["contact"]:
        formatted_data["contact"]["email"] = formatted_data["contact"].pop("email_address")
    data.update(formatted_data)
    if "property" not in data or not data["property"].get("lat"):
        data["property"] = formatted_data.get("property", {"lat": "0.0", "lng": "0.0"})
    logger.debug("Formatted listing", extra={"title": data["listing"]["listing_title"]})
    return data

# API interaction functions (subset from original)
async def fetch_queued_listings() -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{BOINGO_API_URL}/scraping-results/queued-agent-status"
            logger.debug("Fetching queued listings", extra={"url": url})
            async with session.get(url, headers=headers, timeout=30) as response:
                response.raise_for_status()
                response_data = await response.json()
                if response_data.get("status") != 200 or "data" not in response_data or "rows" not in response_data["data"]:
                    logger.error("Invalid response", extra={"error": "Missing data"})
                    return []
                queued_listings = response_data["data"]["rows"]
                for listing in queued_listings:
                    if "data" in listing and "large_result" in listing["data"]:
                        listing["large_result"] = listing["data"]["large_result"]
                logger.info("Fetched listings", extra={"count": len(queued_listings)})
                return queued_listings
        except aiohttp.ClientError as e:
            logger.error("Failed to fetch listings", extra={"error": str(e)})
            return []

async def update_scraping_results(scraping_result_id: str, data: Dict[str, Any]) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": scraping_result_id,
        "source_url": data.get("source_url", ""),
        "data": {
            "address": data.get("data", {}).get("address", {"country": "", "region": "", "city": ""}),
            "property": data.get("data", {}).get("property", {"lat": "0.0", "lng": "0.0"}),
            "listing": data.get("data", {}).get("listing", {"listing_title": "", "description": "", "price": "", "currency": "", "status": "", "listing_type": "", "category": ""}),
            "features": data.get("data", {}).get("features", []),
            "files": data.get("data", {}).get("files", []),
            "contact": data.get("data", {}).get("contact", {"first_name": "", "last_name": "", "phone_number": "", "email": "", "company": ""})
        },
        "progress": data.get("progress", 0),
        "status": data.get("status", "In Progress"),
        "target_id": data.get("target_id", ""),
        "scraped_at": data.get("scraped_at", now),
        "last_updated": now
    }
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Updating result", extra={"id": scraping_result_id, "progress": payload["progress"]})
            async with session.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                logger.info("Updated result", extra={"id": scraping_result_id})
                return True
        except aiohttp.ClientError as e:
            logger.error("Failed to update result", extra={"error": str(e)})
            return False

async def update_agent_status(agent: Dict[str, Any], scraping_result_id: str) -> bool:
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
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Updating agent", extra={"name": agent["agent_name"]})
            async with session.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                logger.info("Updated agent", extra={"name": agent["agent_name"]})
                return True
        except aiohttp.ClientError as e:
            logger.error("Failed to update agent", extra={"error": str(e)})
            return False

async def delete_scraping_result(scraping_result_id: str) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    payload = {"id": scraping_result_id, "force": True}
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Deleting result", extra={"id": scraping_result_id})
            async with session.delete(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=payload, timeout=30) as response:
                response.raise_for_status()
                logger.info("Deleted result", extra={"id": scraping_result_id})
                return True
        except aiohttp.ClientError as e:
            logger.error("Failed to delete result", extra={"error": str(e)})
            return False

# Processing logic
async def process_listing(queued_listing: Dict[str, Any], target_id: str, scraping_result_id: str) -> Optional[Dict[str, Any]]:
    base_payload = queued_listing.copy()
    source_url = base_payload.get("source_url", "")
    large_result = base_payload.get("large_result", "")

    logger.debug("Processing listing", extra={"url": source_url, "id": scraping_result_id})

    if not isinstance(large_result, str) or not large_result.strip() or not is_valid_large_result(large_result):
        logger.debug("Invalid content", extra={"url": source_url})
        await delete_scraping_result(scraping_result_id)
        return None

    base_payload["target_id"] = target_id

    # Step 1: Scrape
    try:
        logger.debug("Scraping", extra={"url": source_url})
        raw_data = await scrape_listing(large_result)
        if not raw_data:
            logger.debug("No data scraped", extra={"url": source_url})
            return None
        base_payload["data"] = raw_data
        base_payload["progress"] = 33
        if not await update_scraping_results(scraping_result_id, base_payload):
            logger.error("Failed to update scraped data", extra={"url": source_url})
            return None
        scraping_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Scraping Agent"), None)
        if scraping_agent and "id" in scraping_agent:
            await update_agent_status(scraping_agent, scraping_result_id)
        logger.info("Scraped", extra={"url": source_url, "progress": 33})
    except Exception as e:
        logger.error("Scraping error", extra={"url": source_url, "error": str(e)})
        return None

    # Step 2: Clean
    try:
        logger.debug("Cleaning", extra={"url": source_url})
        cleaned_data = await clean_listing(raw_data.copy())
        base_payload["data"] = cleaned_data
        base_payload["progress"] = 66
        if not await update_scraping_results(scraping_result_id, base_payload):
            logger.error("Failed to update cleaned data", extra={"url": source_url})
            return None
        cleaning_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Cleaning Agent"), None)
        if cleaning_agent and "id" in cleaning_agent:
            await update_agent_status(cleaning_agent, scraping_result_id)
        logger.info("Cleaned", extra={"url": source_url, "progress": 66})
    except Exception as e:
        logger.error("Cleaning error", extra={"url": source_url, "error": str(e)})
        return None

    # Step 3: Format
    try:
        logger.debug("Formatting", extra={"url": source_url})
        formatted_data = await format_listing(cleaned_data.copy())
        base_payload["data"] = formatted_data
        base_payload["progress"] = 100
        base_payload["status"] = "Success"
        if not await update_scraping_results(scraping_result_id, base_payload):
            logger.error("Failed to update formatted data", extra={"url": source_url})
            return None
        formatting_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Extracting Agent"), None)
        if formatting_agent and "id" in formatting_agent:
            await update_agent_status(formatting_agent, scraping_result_id)
        logger.info("Formatted", extra={"url": source_url, "progress": 100})
        return base_payload
    except Exception as e:
        logger.error("Formatting error", extra={"url": source_url, "error": str(e)})
        return None

async def fetch_queued_listings_AP():
    queued_listings = await fetch_queued_listings()
    if not queued_listings:
        logger.warning("No queued listings")
        return

    processed_ids = set()
    batch_size = 7
    for i in range(0, len(queued_listings), batch_size):
        batch = queued_listings[i:i + batch_size]
        tasks = []
        for queued_listing in batch:
            scraping_result_id = queued_listing.get("id", "")
            source_url = queued_listing.get("source_url", "")
            target_id = queued_listing.get("target_id", "")
            large_result = queued_listing.get("large_result", "")

            if not scraping_result_id or not target_id or not source_url or not isinstance(large_result, str) or not large_result.strip():
                logger.error("Invalid listing", extra={"id": scraping_result_id})
                await delete_scraping_result(scraping_result_id)
                continue
            if not is_valid_large_result(large_result):
                logger.error("Invalid content", extra={"id": scraping_result_id})
                await delete_scraping_result(scraping_result_id)
                continue

            logger.info("Processing listing", extra={"url": source_url, "id": scraping_result_id})
            if "agents" in queued_listing:
                queued_listing["agent_status"] = queued_listing["agents"]
            elif "agent_status" not in queued_listing:
                queued_listing["agent_status"] = []
            tasks.append(process_listing(queued_listing, target_id, scraping_result_id))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result, task_listing in zip(results, batch):
                scraping_result_id = task_listing.get("id", "")
                source_url = task_listing.get("source_url", "")
                if isinstance(result, Exception):
                    logger.error("Processing failed", extra={"url": source_url, "error": str(result)})
                elif result is None:
                    logger.debug("Processing skipped", extra={"url": source_url})
                else:
                    processed_ids.add(scraping_result_id)
                    logger.info("Processed listing", extra={"url": source_url, "id": scraping_result_id})

        await asyncio.sleep(2)

    logger.info("Finished processing", extra={"count": len(processed_ids)})

async def processor_loop():
    while True:
        await fetch_queued_listings_AP()
        await asyncio.sleep(10)

async def main():
    acquire_lock()
    try:
        await processor_loop()
    finally:
        release_lock()

if __name__ == "__main__":
    asyncio.run(main())