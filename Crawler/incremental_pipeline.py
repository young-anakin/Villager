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
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, BestFirstCrawlingStrategy, KeywordRelevanceScorer
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY
import aiohttp
from playwright.async_api import async_playwright
import re
import httpx
import uuid

openai.log = "info"

# Configure logging with a colorful and structured format
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
        logging.FileHandler("incremental_pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

# Use a platform-independent temporary directory for the lock file
LOCK_FILE = os.path.join(tempfile.gettempdir(), "incremental_pipeline.lock")

# Lock management functions
def acquire_lock():
    if os.path.exists(LOCK_FILE):
        logger.info("Another instance is running, lock file exists, exiting.")
        sys.exit(0)
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
        logger.debug("Acquired lock for process", extra={"pid": os.getpid(), "lock_file": LOCK_FILE})
    except IOError as e:
        logger.error(
            "Failed to acquire lock file",
            extra={"error_type": "IOError", "details": str(e), "action": "Exiting process"}
        )
        sys.exit(1)

def release_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logger.debug("Released lock", extra={"lock_file": LOCK_FILE})
    except IOError as e:
        logger.error(
            "Failed to release lock file",
            extra={"error_type": "IOError", "details": str(e), "action": "Continuing without lock release"}
        )

# Initialize OpenAI async client with 30-second timeout
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=30.0, read=30.0, write=30.0)
    )
)

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

# Validate large_result for HTTP status code errors
def is_valid_large_result(large_result: str) -> bool:
    if not large_result or not isinstance(large_result, str) or len(large_result.strip()) < 10:
        logger.debug("Skipping validation, content is empty or too short")
        return False

    http_codes = ["403", "404", "500", "502", "503", "429"]
    error_context = [
        "error", "forbidden", "not found", "server", "failed", "blocked",
        "denied", "timeout", "unavailable", "could not be satisfied", "try again"
    ]
    price_context = [
        "price", "cost", "rental", "sale", "usd", "mxn", "dollar", "peso",
        "k", "m", "thousand", "million", "bedroom", "bathroom", "square"
    ]

    for code in http_codes:
        error_pattern = rf"\b{code}\b.*?(?:{'|'.join(error_context)})"
        if re.search(error_pattern, large_result.lower(), re.IGNORECASE):
            price_pattern = rf"\b{code}\b\s*(?:{'|'.join(price_context)}|\$|€|£|\d+)"
            if not re.search(price_pattern, large_result.lower(), re.IGNORECASE):
                logger.debug("Detected HTTP error in content", extra={"http_code": code})
                return False

    general_error_phrases = [
        "access denied", "request blocked", "server error", "gateway timeout",
        "service unavailable", "connection refused"
    ]
    for phrase in general_error_phrases:
        if phrase in large_result.lower():
            logger.debug("Detected error phrase in content", extra={"phrase": phrase})
            return False

    logger.debug("Content validated successfully, no errors detected")
    return True

# Chunking and Merging Logic
def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(
            "Failed to count tokens",
            extra={"error_type": type(e).__name__, "details": str(e), "action": "Returning 0 tokens"}
        )
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

    logger.debug("Split content into chunks", extra={"chunk_count": len(chunks)})
    return chunks

def merge_chunk_results(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.debug("Merging chunk results", extra={"chunk_count": len(chunks)})
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
                            logger.debug("Updated section field", extra={"section": section, "field": key, "value": value})
            if "features" in chunk and isinstance(chunk["features"], list):
                existing_features = {item["feature"]: item for item in merged["features"]}
                for item in chunk["features"]:
                    if item["feature"] not in existing_features:
                        merged["features"].append(item)
                        logger.debug("Added feature", extra={"feature": item["feature"]})
            if "files" in chunk and isinstance(chunk["files"], list):
                merged["files"].extend(chunk["files"])
                merged["files"] = list(set(merged["files"]))
                logger.debug("Merged files", extra={"file_count": len(merged["files"])})
        return merged
    except Exception as e:
        logger.error(
            "Failed to merge chunk results",
            extra={"error_type": type(e).__name__, "details": str(e), "action": "Raising exception"}
        )
        raise

# OpenAI Extraction with Retry Logic
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
            logger.debug("Extracted data with OpenAI", extra={"attempt": attempt + 1})
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e):
                wait_time = 2 ** attempt * 10
                logger.warning(
                    "OpenAI rate limit hit, retrying",
                    extra={"attempt": f"{attempt + 1}/{max_retries}", "wait_time": f"{wait_time} seconds"}
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "OpenAI extraction failed",
                    extra={"error_type": type(e).__name__, "details": str(e), "action": "Returning empty result"}
                )
                return "{}"
    logger.error(
        "Max retries exceeded for OpenAI extraction",
        extra={"max_retries": max_retries, "action": "Returning empty result"}
    )
    return "{}"

# Scraping Logic
async def scrape_listing(content: str) -> Dict[str, Any]:
    instruction = """Extract detailed information from the provided markdown content about a single property listing. Return the data in JSON format matching the provided schema. Focus on extracting:
    - Address details (country, region, city). Do not include 'district'.
    - Property coordinates (latitude and longitude, guess if not available based on location data)
    - Listing details (title, description, price, currency, status, type, category)
    - Features (e.g., bedrooms, bathrooms, include all property attributes)
    - ALL file URLs (e.g., images, documents) present in the content, without any limit
    - Contact information (phone number, first_name, last_name, email, company)
    If critical data (title, price, files) is missing, return an empty object."""
    
    total_tokens = count_tokens(content)
    max_input_tokens = 7000
    if total_tokens > max_input_tokens:
        chunks = split_into_chunks(content, max_tokens=max_input_tokens)
        extracted_chunks = []
        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk", extra={"chunk_number": i + 1, "total_chunks": len(chunks), "tokens": count_tokens(chunk)})
            chunk_result = await extract_with_openai(chunk, ListingData.model_json_schema(), instruction)
            extracted_chunks.append(json.loads(chunk_result))
        merged_data = merge_chunk_results(extracted_chunks)
    else:
        logger.debug("Extracting single content block", extra={"tokens": total_tokens})
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
        logger.debug("Validated and prepared listing data", extra={"title": final_data["listing"]["listing_title"]})
    except ValidationError as e:
        logger.error(
            "Data validation failed",
            extra={"error_type": "ValidationError", "details": str(e), "action": "Returning empty result"}
        )
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
    prompt = f"{instruction}\n\nInput data:\n{text_fields}"
    
    cleaned_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    cleaned_data = json.loads(cleaned_text)
    if "contact" in cleaned_data and "email_address" in cleaned_data["contact"]:
        cleaned_data["contact"]["email"] = cleaned_data["contact"].pop("email_address")
    data.update(cleaned_data)
    logger.debug("Cleaned listing data", extra={"title": data["listing"]["listing_title"]})
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
    prompt = f"{instruction}\n\nInput data:\n{text_fields}"
    
    formatted_text = await extract_with_openai(prompt, ListingData.model_json_schema(), instruction)
    formatted_data = json.loads(formatted_text)
    if "contact" in formatted_data and "email_address" in formatted_data["contact"]:
        formatted_data["contact"]["email"] = formatted_data["contact"].pop("email_address")
    data.update(formatted_data)
    if "property" not in data or not data["property"].get("lat"):
        data["property"] = formatted_data.get("property", {"lat": "0.0", "lng": "0.0"})
    logger.debug("Formatted listing data", extra={"title": data["listing"]["listing_title"]})
    return data

# Async API Interaction Functions
async def fetch_scraping_targets() -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{BOINGO_API_URL}/scraping-target/all"
            logger.debug("Fetching scraping targets", extra={"url": url})
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if not isinstance(response_data, dict) or "data" not in response_data:
                    logger.error(
                        "Invalid API response format",
                        extra={"error_type": "DataFormatError", "details": "Expected dict with 'data' key", "action": "Returning empty list"}
                    )
                    return []
                
                targets = response_data["data"]
                if not isinstance(targets, list):
                    logger.error(
                        "Invalid target data type",
                        extra={"error_type": "TypeError", "details": f"Expected list, got {type(targets).__name__}", "action": "Returning empty list"}
                    )
                    return []
                
                logger.info("Fetched scraping targets", extra={"target_count": len(targets)})
                return targets
        except aiohttp.ClientError as e:
            logger.error(
                "Failed to fetch scraping targets",
                extra={"error_type": type(e).__name__, "details": str(e), "action": "Returning empty list"}
            )
            return []

async def fetch_queued_listings() -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{BOINGO_API_URL}/scraping-results/queued-agent-status"
            logger.debug("Fetching queued listings", extra={"url": url})
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                response_data = await response.json()
                
                if response_data.get("status") != 200 or "data" not in response_data or "rows" not in response_data["data"]:
                    logger.error(
                        "Invalid queued listings response",
                        extra={"error_type": "DataFormatError", "details": "Missing status, data, or rows", "action": "Returning empty list"}
                    )
                    return []
                
                queued_listings = response_data["data"]["rows"]
                for listing in queued_listings:
                    if "data" in listing and "large_result" in listing["data"]:
                        listing["large_result"] = listing["data"]["large_result"]
                        logger.debug("Extracted large_result", extra={"source_url": listing["source_url"]})
                
                logger.info("Fetched queued listings", extra={"listing_count": len(queued_listings)})
                return queued_listings
        except aiohttp.ClientError as e:
            logger.error(
                "Failed to fetch queued listings",
                extra={"error_type": type(e).__name__, "details": str(e), "action": "Returning empty list"}
            )
            return []

async def fetch_single_queued_listing(scraping_result_id: str) -> Optional[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Fetching single queued listing", extra={"scraping_result_id": scraping_result_id})
            async with session.get(f"{BOINGO_API_URL}/scraping-results/queued-agent-status", headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                response_data = await response.json()
                queued_listings = response_data["data"]["rows"]
                for listing in queued_listings:
                    if listing["id"] == scraping_result_id:
                        logger.debug("Found queued listing", extra={"scraping_result_id": scraping_result_id})
                        return listing
                logger.error(
                    "Queued listing not found",
                    extra={"error_type": "NotFoundError", "details": f"No listing with ID {scraping_result_id}", "action": "Returning None"}
                )
                return None
        except aiohttp.ClientError as e:
            logger.error(
                "Failed to fetch queued listing",
                extra={"error_type": type(e).__name__, "details": str(e), "action": "Returning None"}
            )
            return None

async def post_to_scraping_results(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Posting to scraping results", extra={"source_url": data.get("source_url")})
            async with session.post(f"{BOINGO_API_URL}/scraping-results/", headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                response_data = await response.json()
                result_id = response_data["data"]["id"]
                logger.info("Posted scraping result", extra={"result_id": result_id, "source_url": data.get("source_url")})
                return {"id": result_id, "data": response_data["data"]}
        except aiohttp.ClientError as e:
            error_details = {"error_type": type(e).__name__, "details": str(e), "action": "Returning None"}
            if hasattr(e, 'response') and e.response is not None:
                error_details["api_response"] = await e.response.text()
            logger.error("Failed to post scraping result", extra=error_details)
            return None

async def update_scraping_results(scraping_result_id: str, data: Dict[str, Any]) -> bool:
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
        "progress": data.get("progress", 0),
        "status": data.get("status", "In Progress"),
        "target_id": data.get("target_id", ""),
        "scraped_at": data.get("scraped_at", now),
        "last_updated": now
    }
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Updating scraping result", extra={"scraping_result_id": scraping_result_id, "progress": payload["progress"]})
            async with session.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                logger.info("Updated scraping result", extra={"scraping_result_id": scraping_result_id, "progress": payload["progress"]})
                return True
        except aiohttp.ClientError as e:
            error_details = {"error_type": type(e).__name__, "details": str(e), "action": "Returning False"}
            if hasattr(e, 'response') and e.response is not None:
                error_details["api_response"] = await e.response.text()
            logger.error("Failed to update scraping result", extra=error_details)
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
            logger.debug("Updating agent status", extra={"agent_name": agent["agent_name"], "scraping_result_id": scraping_result_id})
            async with session.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                logger.info("Updated agent status", extra={"agent_name": agent["agent_name"], "status": "Success"})
                return True
        except aiohttp.ClientError as e:
            logger.error(
                "Failed to update agent status",
                extra={"error_type": type(e).__name__, "details": str(e), "action": "Returning False"}
            )
            return False

async def delete_scraping_result(scraping_result_id: str) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "id": scraping_result_id,
        "force": True
    }
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Deleting invalid listing", extra={"scraping_result_id": scraping_result_id})
            async with session.delete(
                f"{BOINGO_API_URL}/scraping-results",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                logger.info("Deleted invalid listing", extra={"scraping_result_id": scraping_result_id})
                return True
        except aiohttp.ClientError as e:
            error_details = {"error_type": type(e).__name__, "details": str(e), "action": "Returning False"}
            if hasattr(e, 'response') and e.response is not None:
                error_details["api_response"] = await e.response.text()
            logger.error("Failed to delete invalid listing", extra=error_details)
            return False

async def update_list_extracted(target_id: str) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "id": target_id,
        "list_extracted": True
    }
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug("Updating list_extracted status", extra={"target_id": target_id})
            async with session.put(
                f"{BOINGO_API_URL}/scraping-target/update-list-extracted",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                logger.info("Updated list_extracted to True", extra={"target_id": target_id})
                return True
        except aiohttp.ClientError as e:
            error_details = {"error_type": type(e).__name__, "details": str(e), "action": "Returning False"}
            if hasattr(e, 'response') and e.response is not None:
                error_details["api_response"] = await e.response.text()
            logger.error("Failed to update list_extracted", extra=error_details)
            return False

# Process a Single Listing
async def process_listing(queued_listing: Dict[str, Any], target_id: str, scraping_result_id: str) -> Optional[Dict[str, Any]]:
    base_payload = queued_listing.copy()
    source_url = base_payload.get("source_url", "")
    large_result = base_payload.get("large_result", "")

    logger.debug("Starting listing processing", extra={"source_url": source_url, "scraping_result_id": scraping_result_id})

    if not isinstance(large_result, str) or not large_result.strip():
        logger.debug("Skipping listing, no valid content", extra={"source_url": source_url})
        return None

    if not is_valid_large_result(large_result):
        logger.debug("Skipping listing, content contains errors", extra={"source_url": source_url})
        return None

    base_payload["target_id"] = target_id

    # Step 1: Scrape (33%)
    try:
        logger.debug("Scraping listing data", extra={"source_url": source_url})
        raw_data = await scrape_listing(large_result)
        if not raw_data:
            logger.debug("No data scraped", extra={"source_url": source_url})
            return None

        base_payload["data"] = raw_data
        base_payload["progress"] = 33
        if not await update_scraping_results(scraping_result_id, base_payload):
            logger.error(
                "Failed to update scraped data",
                extra={"error_type": "APIError", "details": f"Update rejected for {source_url}", "action": "Skipping listing"}
            )
            return None
        
        scraping_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Scraping Agent"), None)
        if not scraping_agent or "id" not in scraping_agent:
            logger.warning(
                "Invalid Scraping Agent",
                extra={"error_type": "AgentError", "details": f"No valid agent for {source_url}", "action": "Continuing without agent update"}
            )
        elif not await update_agent_status(scraping_agent, scraping_result_id):
            logger.error(
                "Failed to update Scraping Agent status",
                extra={"error_type": "APIError", "details": f"Status update failed for {source_url}", "action": "Continuing"}
            )
        logger.info("Completed scraping", extra={"source_url": source_url, "progress": 33})
    except Exception as e:
        logger.error(
            "Scraping error",
            extra={"error_type": type(e).__name__, "details": str(e), "action": f"Skipping listing {source_url}"}
        )
        return None

    # Step 2: Clean (66%)
    try:
        logger.debug("Cleaning listing data", extra={"source_url": source_url})
        cleaned_data = await clean_listing(raw_data.copy())
        base_payload["data"] = cleaned_data
        base_payload["progress"] = 66
        if not await update_scraping_results(scraping_result_id, base_payload):
            logger.error(
                "Failed to update cleaned data",
                extra={"error_type": "APIError", "details": f"Update rejected for {source_url}", "action": "Skipping listing"}
            )
            return None
        
        cleaning_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Cleaning Agent"), None)
        if not cleaning_agent or "id" not in cleaning_agent:
            logger.warning(
                "Invalid Cleaning Agent",
                extra={"error_type": "AgentError", "details": f"No valid agent for {source_url}", "action": "Continuing without agent update"}
            )
        elif not await update_agent_status(cleaning_agent, scraping_result_id):
            logger.error(
                "Failed to update Cleaning Agent status",
                extra={"error_type": "APIError", "details": f"Status update failed for {source_url}", "action": "Continuing"}
            )
        logger.info("Completed cleaning", extra={"source_url": source_url, "progress": 66})
    except Exception as e:
        logger.error(
            "Cleaning error",
            extra={"error_type": type(e).__name__, "details": str(e), "action": f"Skipping listing {source_url}"}
        )
        return None

    # Step 3: Format (100%)
    try:
        logger.debug("Formatting listing data", extra={"source_url": source_url})
        formatted_data = await format_listing(cleaned_data.copy())
        base_payload["data"] = formatted_data
        base_payload["progress"] = 100
        base_payload["status"] = "Success"
        if not await update_scraping_results(scraping_result_id, base_payload):
            logger.error(
                "Failed to update formatted data",
                extra={"error_type": "APIError", "details": f"Update rejected for {source_url}", "action": "Skipping listing"}
            )
            return None
        
        formatting_agent = next((agent for agent in base_payload["agent_status"] if agent["agent_name"] == "Extracting Agent"), None)
        if not formatting_agent or "id" not in formatting_agent:
            logger.warning(
                "Invalid Extracting Agent",
                extra={"error_type": "AgentError", "details": f"No valid agent for {source_url}", "action": "Continuing without agent update"}
            )
        elif not await update_agent_status(formatting_agent, scraping_result_id):
            logger.error(
                "Failed to update Extracting Agent status",
                extra={"error_type": "APIError", "details": f"Status update failed for {source_url}", "action": "Continuing"}
            )
        logger.info("Completed formatting", extra={"source_url": source_url, "progress": 100})
        return base_payload
    except Exception as e:
        logger.error(
            "Formatting error",
            extra={"error_type": type(e).__name__, "details": str(e), "action": f"Skipping listing {source_url}"}
        )
        return None

# Modified Scrape and Process Function
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
            logger.debug("Skipping invalid crawl result", extra={"url": result.url})
            continue

        logger.debug("Found valid listing", extra={"url": result.url})
        initial_payload = initial_payload_template.copy()
        initial_payload["source_url"] = result.url
        initial_payload["large_result"] = result.markdown

        post_result = await post_to_scraping_results(initial_payload)
        if not post_result:
            logger.error(
                "Failed to queue listing",
                extra={"error_type": "APIError", "details": f"Could not post {result.url}", "action": "Skipping listing"}
            )
            continue
        
        scraping_result_id = post_result["id"]
        posted_ids[result.url] = scraping_result_id
        logger.info("Queued listing", extra={"url": result.url, "scraping_result_id": scraping_result_id})
    if posted_ids:
        if await update_list_extracted(target_id):
            logger.info("Marked target as extracted", extra={"target_id": target_id})
        else:
            logger.error(
                "Failed to mark target as extracted",
                extra={"error_type": "APIError", "details": f"Update failed for target {target_id}", "action": "Continuing"}
            )

    logger.info("Finished crawling", extra={"url": url, "queued_count": len(posted_ids)})

async def fetch_queued_listings_AP():
    queued_listings = await fetch_queued_listings()
    if not queued_listings:
        logger.warning("No queued listings found")
        return

    processed_ids = set()
    batch_size = 7  # Process up to 7 listings concurrently

    # Process listings in batches of 7
    for i in range(0, len(queued_listings), batch_size):
        batch = queued_listings[i:i + batch_size]
        tasks = []

        for queued_listing in batch:
            scraping_result_id = queued_listing.get("id", "")
            source_url = queued_listing.get("source_url", "")
            target_id = queued_listing.get("target_id", "")
            large_result = queued_listing.get("large_result", "")

            # Validate listing data
            if not scraping_result_id:
                logger.error(
                    "Invalid listing",
                    extra={"error_type": "MissingData", "details": "No scraping_result_id", "action": "Skipping listing"}
                )
                continue
            if not target_id:
                logger.error(
                    "Invalid listing",
                    extra={"error_type": "MissingData", "details": f"No target_id for {scraping_result_id}", "action": "Skipping listing"}
                )
                continue
            if not source_url:
                logger.error(
                    "Invalid listing",
                    extra={"error_type": "MissingData", "details": f"No source_url for {scraping_result_id}", "action": "Skipping listing"}
                )
                continue
            if not isinstance(large_result, str) or not large_result.strip():
                logger.error(
                    "Invalid listing",
                    extra={"error_type": "MissingData", "details": f"No valid content for {scraping_result_id}", "action": "Deleting listing"}
                )
                await delete_scraping_result(scraping_result_id)
                continue
            if not is_valid_large_result(large_result):
                logger.error(
                    "Invalid listing content",
                    extra={"error_type": "ContentError", "details": f"HTTP error in content for {scraping_result_id}", "action": "Deleting listing"}
                )
                await delete_scraping_result(scraping_result_id)
                continue

            logger.info("Starting processing for listing", extra={"source_url": source_url, "scraping_result_id": scraping_result_id})
            
            # Initialize agent status if missing
            if "agents" in queued_listing:
                queued_listing["agent_status"] = queued_listing["agents"]
            elif "agent_status" not in queued_listing:
                logger.warning("No agent status found, initializing empty", extra={"source_url": source_url})
                queued_listing["agent_status"] = []

            # Add task for processing
            tasks.append(process_listing(queued_listing, target_id, scraping_result_id))

        # Run batch concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result, task_listing in zip(results, batch):
                scraping_result_id = task_listing.get("id", "")
                source_url = task_listing.get("source_url", "")
                if isinstance(result, Exception):
                    logger.error(
                        "Processing failed for listing",
                        extra={"error_type": type(result).__name__, "details": str(result), "action": f"Skipping listing {source_url}"}
                    )
                elif result is None:
                    logger.debug("Processing skipped", extra={"source_url": source_url, "scraping_result_id": scraping_result_id})
                else:
                    processed_ids.add(scraping_result_id)
                    logger.info("Completed processing for listing", extra={"source_url": source_url, "scraping_result_id": scraping_result_id})

        # Brief delay to avoid overwhelming APIs
        await asyncio.sleep(2)

    logger.info("Finished processing all listings", extra={"processed_count": len(processed_ids), "total_listings": len(queued_listings)})

# Scraper Loop
async def scraper_loop(crawler: AsyncWebCrawler):
    while True:
        targets = await fetch_scraping_targets()
        valid_targets = [t for t in targets if not t.get("list_extracted", True)]
        
        if not valid_targets:
            logger.info("No valid targets remaining, exiting scraper loop")
            break

        logger.info("Processing targets", extra={"target_count": len(valid_targets)})
        for target in valid_targets:
            url = target.get("website_url")
            target_id = target.get("id")
            listing_format = target.get("listing_url_format")
            search_range = target.get("search_range", 3)
            max_properties = target.get("max_properties", 20)

            if not url or not target_id or not listing_format:
                logger.error(
                    "Invalid target",
                    extra={"error_type": "MissingData", "details": "Missing url, id, or format", "action": "Skipping target"}
                )
                continue

            logger.info("Scraping target", extra={"url": url, "target_id": target_id})
            try:
                await scrape_and_process(url, target_id, listing_format, crawler, search_range, max_properties)
            except Exception as e:
                logger.error(
                    "Target scraping failed",
                    extra={"error_type": type(e).__name__, "details": str(e), "action": "Continuing with next target"}
                )
                continue

        # Wait before checking for new targets to avoid overwhelming APIs
        logger.debug("Scraper sleeping before next target check", extra={"sleep_seconds": 30})
        await asyncio.sleep(30)

# Processor Loop
async def processor_loop():
    while True:
        queued_listings = await fetch_queued_listings()
        if not queued_listings:
            logger.info("No queued listings remaining, checking again after delay")
            await asyncio.sleep(30)  # Wait before checking again
            continue

        logger.info("Processing queued listings", extra={"listing_count": len(queued_listings)})
        await fetch_queued_listings_AP()  # Process all queued listings in batches

        # Brief delay to avoid immediate re-fetching
        logger.debug("Processor sleeping before next listing check", extra={"sleep_seconds": 10})
        await asyncio.sleep(10)

# Main Function
async def main():
    acquire_lock()
    try:
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            verbose=True,
            use_persistent_context=True,
            user_data_dir="browser_data",
            extra_args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        
        # Debug: Test Playwright directly
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
            page = await browser.new_page()
            await page.goto("https://example.com")
            logger.debug("Playwright test: Page title", extra={"title": await page.title()})
            await browser.close()

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Run scraper and processor concurrently
            scraper_task = asyncio.create_task(scraper_loop(crawler))
            processor_task = asyncio.create_task(processor_loop())

            # Wait for both tasks to complete
            await asyncio.gather(scraper_task, processor_task, return_exceptions=True)
                
    finally:
        release_lock()

if __name__ == "__main__":
    asyncio.run(main())