import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
import httpx
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError
import tiktoken
import openai

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("trial_test.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Configure OpenAI client
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        timeout=httpx.Timeout(120.0, connect=30.0, read=30.0, write=30.0),
    ),
)
openai.log = "info"

# Boingo API settings
BOINGO_API_URL = os.getenv("BOINGO_API_URL", "https://api.boingo.ai")
BOINGO_BEARER_TOKEN = os.getenv("BOINGO_BEARER_TOKEN")

# Pydantic models
class Address(BaseModel):
    country: str
    region: str
    city: str

class Contact(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None

class Feature(BaseModel):
    feature: str
    value: str

class Listing(BaseModel):
    listing_title: str
    description: str
    price: str
    currency: str
    status: str
    listing_type: str
    category: str

class Property(BaseModel):
    lat: Optional[str] = None
    lng: Optional[str] = None

class ListingData(BaseModel):
    address: Address
    property: Optional[Property] = None
    listing: Listing
    features: list[Feature] = []
    files: list[str] = []
    contact: Optional[Contact] = None

# Token counting
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Save data locally as fallback
def save_fallback_data(target_id: str, data: Dict[str, Any], progress: float, status: str):
    try:
        fallback = {
            "target_id": target_id,
            "progress": progress,
            "status": status,
            "data": data,
            "timestamp": "2025-04-15T21:16:16Z"
        }
        with open(f"fallback_{target_id}.json", "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2)
        logger.info(f"Saved fallback data | file=fallback_{target_id}.json")
    except Exception as e:
        logger.error(f"Failed to save fallback data | error={str(e)}")

# Check OpenAI connectivity
async def check_openai_connectivity(retries: int = 3, delay: float = 5.0) -> bool:
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(retries):
            try:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                )
                response.raise_for_status()
                logger.info(f"OpenAI API connectivity check passed | attempt={attempt+1}")
                return True
            except Exception as e:
                logger.warning(
                    f"OpenAI API connectivity check failed | attempt={attempt+1}/{retries}, "
                    f"error_type={type(e).__name__}, details={str(e)}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
        logger.error("OpenAI API connectivity check failed after all attempts")
        return False

# Update Boingo API
async def update_scraping_result(
    listing: Dict[str, Any],
    progress: float,
    status: str = "In Progress",
) -> bool:
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.put(
                f"{BOINGO_API_URL}/scraping-results",
                json=listing,
                headers=headers,
            )
            response.raise_for_status()
            logger.info(
                f"Updated scraping result | progress={progress}, status={status}, target_id={listing['target_id']}"
            )
            return True
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to update scraping result | target_id={listing['target_id']}, error={str(e)}"
            )
            save_fallback_data(listing['target_id'], listing, progress, status)
            return False

# Extract with OpenAI
async def extract_with_openai(content: str, instruction: str, retries: int = 3, delay: float = 5.0) -> str:
    for attempt in range(retries):
        try:
            logger.debug(f"Attempting OpenAI extraction | attempt={attempt+1}, tokens={count_tokens(content)}")
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Content:\n\n{content}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            result = response.choices[0].message.content
            try:
                json.loads(result)
                logger.debug(f"OpenAI response | attempt={attempt+1}, content={result}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response | attempt={attempt+1}, error={str(e)}")
                if attempt == retries - 1:
                    return "{}"
        except OpenAIError as e:
            if "400" in str(e) or "invalid_request_error" in str(e):
                logger.error(f"OpenAI request invalid | error={str(e)}, stopping retries")
                return "{}"
            logger.warning(
                f"OpenAI attempt {attempt+1}/{retries} failed | error_type={type(e).__name__}, details={str(e)}"
            )
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))
            else:
                logger.error(f"OpenAI extraction failed after {retries} attempts")
                return "{}"
    return "{}"

# Fix schema mismatches
def fix_schema(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        fixed = raw_data.copy()
        # Fix coordinates -> property
        if "coordinates" in fixed:
            fixed["property"] = {
                "lat": str(fixed["coordinates"].get("latitude", "")),
                "lng": str(fixed["coordinates"].get("longitude", ""))
            }
            del fixed["coordinates"]
        # Fix listing fields
        if "listing" in fixed:
            listing = fixed["listing"]
            if "title" in listing:
                listing["listing_title"] = listing.pop("title")
            if "price" in listing:
                listing["price"] = str(listing["price"])
            if "type" in listing:
                listing["listing_type"] = listing.pop("type")
        # Fix features
        if "features" in fixed and isinstance(fixed["features"], dict):
            features = []
            for key, value in fixed["features"].items():
                if key == "amenities":
                    for amenity in value:
                        features.append({"feature": "Amenity", "value": amenity})
                elif key not in ["view", "heating"]:
                    features.append({"feature": key.capitalize(), "value": str(value)})
                elif key == "view":
                    features.append({"feature": "View", "value": value})
                elif key == "heating":
                    features.append({"feature": "Heating", "value": value})
            fixed["features"] = features
        return fixed
    except Exception as e:
        logger.error(f"Schema fix failed | error={str(e)}")
        return raw_data

# Scrape listing (33%)
async def scrape_listing(content: str) -> Dict[str, Any]:
    instruction = (
        "Extract detailed information from the provided markdown content about a single property listing. "
        "Return the data in JSON format matching this structure:\n"
        "- address: {country, region, city}\n"
        "- property: {lat, lng} (guess coordinates if not provided, as strings)\n"
        "- listing: {listing_title, description, price (as string), currency, status, listing_type, category}\n"
        "- features: [{feature, value}] (list all attributes like bedrooms, bathrooms)\n"
        "- files: [string] (all URLs, empty if none)\n"
        "- contact: {phone_number, first_name, last_name, email, company}\n"
        "If listing_title or price is missing, return {}. Files are optional."
    )
    max_input_tokens = 7000
    total_tokens = count_tokens(content)
    
    logger.debug(f"Extracting content | tokens={total_tokens}")
    if total_tokens > max_input_tokens:
        logger.warning("Content exceeds token limit")
        return {}
    
    try:
        extracted_data = await extract_with_openai(content, instruction)
        raw_data = json.loads(extracted_data)
        if not raw_data or not raw_data.get("listing", {}).get("listing_title") or not raw_data.get("listing", {}).get("price"):
            logger.debug("Skipping listing | reason=Empty or missing critical fields")
            return {}
        fixed_data = fix_schema(raw_data)
        logger.debug(f"Extraction result | data={fixed_data}")
        return fixed_data
    except Exception as e:
        logger.error(f"Scrape failed | error_type={type(e).__name__}, details={str(e)}")
        return {}

# Clean data (66%)
async def clean_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cleaned_data = raw_data.copy()
        if cleaned_data.get("listing", {}).get("description"):
            cleaned_data["listing"]["description"] = "Cleaned: " + cleaned_data["listing"]["description"]
        if cleaned_data.get("features"):
            cleaned_data["features"] = [
                {"feature": f["feature"], "value": f["value"].strip().capitalize()}
                for f in cleaned_data["features"]
            ]
        logger.debug(f"Cleaned data | data={cleaned_data}")
        return cleaned_data
    except Exception as e:
        logger.error(f"Cleaning failed | error_type={type(e).__name__}, details={str(e)}")
        return {}

# Format data (100%)
async def format_data(cleaned_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        formatted_data = cleaned_data.copy()
        if not formatted_data.get("property") or not formatted_data["property"].get("lat"):
            formatted_data["property"] = {"lat": "20.2114", "lng": "-87.4654"}
        formatted_data["listing"]["status"] = formatted_data["listing"].get("status", "Active")
        logger.debug(f"Formatted data | data={formatted_data}")
        return formatted_data
    except Exception as e:
        logger.error(f"Formatting failed | error_type={type(e).__name__}, details={str(e)}")
        return {}

# Prepare listing for API
def prepare_listing_payload(
    listing: Dict[str, Any],
    scraped_data: Dict[str, Any],
    progress: float,
    status: str,
) -> Dict[str, Any]:
    from datetime import datetime
    return {
        "id": listing["id"],
        "source_url": listing["source_url"],
        "data": scraped_data,
        "progress": progress,
        "status": status,
        "target_id": listing["target_id"],
        "scraped_at": listing["scraped_at"],
        "last_updated": datetime.utcnow().isoformat() + "Z"
    }

# Main processing function
async def process_single_listing(listing: Dict[str, Any]):
    target_id = listing["target_id"]
    large_result = listing["large_result"]
    agent_status = listing.get("agents", [{}])[0].get("status", "Queued")

    logger.info(f"Starting processing | target_id={target_id}")

    # Check connectivity
    if not await check_openai_connectivity():
        logger.error("Aborting due to OpenAI connectivity failure")
        payload = prepare_listing_payload(listing, {}, 0.0, "Failed")
        await update_scraping_result(payload, 0.0, status="Failed")
        return

    # Scrape (33%)
    try:
        raw_data = await scrape_listing(large_result)
        if not raw_data:
            logger.debug("Skipping listing | reason=Empty scrape result")
            payload = prepare_listing_payload(listing, {}, 0.0, "Failed")
            await update_scraping_result(payload, 0.0, status="Failed")
            return
        logger.info(f"Scrape complete | data={raw_data}")
        payload = prepare_listing_payload(listing, raw_data, 33.0, "In Progress")
        if not await update_scraping_result(payload, 33.0, status="In Progress"):
            logger.warning("Continuing despite API update failure")
    except Exception as e:
        logger.error(f"Scrape stage failed | error_type={type(e).__name__}, details={str(e)}")
        payload = prepare_listing_payload(listing, {}, 0.0, "Failed")
        await update_scraping_result(payload, 0.0, status="Failed")
        return

    # Clean (66%)
    try:
        cleaned_data = await clean_data(raw_data)
        if not cleaned_data:
            logger.debug("Skipping listing | reason=Empty clean result")
            payload = prepare_listing_payload(listing, raw_data, 0.0, "Failed")
            await update_scraping_result(payload, 0.0, status="Failed")
            return
        logger.info(f"Clean complete | data={cleaned_data}")
        payload = prepare_listing_payload(listing, cleaned_data, 66.0, "In Progress")
        if not await update_scraping_result(payload, 66.0, status="In Progress"):
            logger.warning("Continuing despite API update failure")
    except Exception as e:
        logger.error(f"Clean stage failed | error_type={type(e).__name__}, details={str(e)}")
        payload = prepare_listing_payload(listing, raw_data, 0.0, "Failed")
        await update_scraping_result(payload, 0.0, status="Failed")
        return

    # Format (100%)
    try:
        formatted_data = await format_data(cleaned_data)
        if not formatted_data:
            logger.debug("Skipping listing | reason=Empty format result")
            payload = prepare_listing_payload(listing, cleaned_data, 0.0, "Failed")
            await update_scraping_result(payload, 0.0, status="Failed")
            return
        logger.info(f"Format complete | data={formatted_data}")
        payload = prepare_listing_payload(listing, formatted_data, 100.0, "Success")
        if not await update_scraping_result(payload, 100.0, status="Success"):
            logger.warning("Continuing despite API update failure")
    except Exception as e:
        logger.error(f"Format stage failed | error_type={type(e).__name__}, details={str(e)}")
        payload = prepare_listing_payload(listing, cleaned_data, 0.0, "Failed")
        await update_scraping_result(payload, 0.0, status="Failed")
        return

# Listing data
LISTING = {
    "id": "f137bff3-a746-4864-b060-1faa01f38750",
    "source_url": "https://www.properstar.com/listing/104433349",
    "destination_url": None,
    "data": {
        "address": {"country": "", "region": "", "city": ""},
        "property": {"lat": "0.0", "lng": "0.0"},
        "listing": {
            "listing_title": "",
            "description": "",
            "price": "",
            "currency": "",
            "status": "",
            "listing_type": "",
            "category": ""
        },
        "features": [],
        "files": [],
        "contact": {
            "first_name": "",
            "last_name": "",
            "phone_number": "",
            "email": "",
            "company": ""
        },
    },
    "progress": 0,
    "status": "In Progress",
    "last_updated": None,
    "scraped_at": "2025-04-15T10:33:14.000Z",
    "target_id": "fb94b628-3c5a-4651-b08d-484c71cdc5ef",
    "large_result": (
        "\nTulum / Buy\n``CTRL`+`K``\n* [Search](https://www.properstar.com/dominican-republic/buy/flat?searchId=748273356)\n"
        "* [Favorites](https://www.properstar.com/favorites)\n* [Messages](https://www.properstar.com/messages/)\n* Profile\n"
        "* ![US](https://www.properstar.com/assets/flags/1x/us.png)\nEN\nBrowsing preferences\n"
        "* ![US](https://www.properstar.com/assets/flags/1x/us.png)\nEnglish, United States\nUSD\nBack\nShare\nAdd to favorites\n!\n!\n!\n"
        "Open the gallery\n  1.\n  2. [Mexico](https://www.properstar.com/mexico/buy/flat)\n"
        "  3. [Quintana Roo](https://www.properstar.com/mexico/quintana-roo-l1/buy/flat)\n"
        "  4. [Tulum Municipality](https://www.properstar.com/mexico/tulum-l2/buy/flat)\n"
        "  5. [Tulum](https://www.properstar.com/mexico/tulum/buy/flat)\n"
        "  6. [Condo/Apartment](https://www.properstar.com/mexico/tulum/buy/flat)\n\n\n"
        "# 2 Bedroom Apartment New in Tulum - Great Price!\n"
        "Tulum, Quintana Roo, Mexico, Quintana Roo, Tulum, Region 15 Kukulcan\n"
        "Condo/Apartment • 5 room(s) • 2 bed. • 2 bath. • 915 sq ft (85 m²) • ref: RAP6784239\n"
        "$179,200\n## Description\nTranslated from Spanish.Show original\n"
        "Luna de Plata is located in the jungle of Region 15, between the area with the highest tourist influx in Tulum Pueblo (La Veleta) and the famous Hotel Zone. "
        "It's the perfect balance between the tranquility and natural beauty of the jungle while still having easy access to the fully developed and busy areas of Tulum. "
        "Conveniently located between Avenida Kukulkán (which goes directly to the beach) and Fifth Avenue it promises to be a major shopping destination and tourist corridor. "
        "Because of the central location of Luna de Plata; the areas of greatest interest in Tulum are just a few minutes drive away or a bike ride away! "
        "It is a condominium of 4 exclusive apartments with common areas where you will find a pool with a waterfall of 33 m2, designated parking lots, bicycle parking and a coexistence area with barbecue. "
        "Ground floor apartment #1 consists of 95 m2 (1005 ft2) with 2 bedrooms, 2 full bathrooms, laundry room, kitchen, living and dining room in open plan and a covered balcony overlooking the common areas. "
        "The building is already in its last stage and ready to be delivered and deeded 60 days from the date of signing the contract. "
        "Prices subjet to change without warning.\nSee more\n## Areas\n### Spaces\nRooms5\nFloors1\nBedrooms2\nBathrooms2\nParking lots (outside)1\n"
        "### Surfaces\nLiving area915 sq ft (85 m²)\nLand3767 sq ft (350 m²)\nInternal915 sq ft (85 m²)\nTotal1023 sq ft (95 m²)\n"
        "## Amenities\nAir conditioning\nBalcony\nDining room\nGarage\nInternet\nSwimming pool\nWashing machine\n## Location\n"
        "#### Quintana Roo, Tulum, Region 15 Kukulcan, Tulum\n### View and orientation\nViews and orientation extracted by AI. Know more\nForest view\n"
        "## Construction\nTypeCondo/Apartment\nConditionNew\nHeatingGas\n## Your contact\n#### Real Estate Puerto Aventuras\nPhone number\n+52 9...Show\n"
        "## Agency\n![Real Estate Puerto Aventuras](https://res.listglobally.com/accounts/6362754/9e319471054ac6ef577d15fe683d0518?width=220&height=92&mode=max&autorotate=true)\n"
        "#### Real Estate Puerto Aventuras\n+52 984 1577534\nPuerto Aventuras\nSolidaridad\n[Call the agency](tel:+52 984 1577534)\n"
        "[All listings](https://www.properstar.com/agency/real-estate-puerto-aventuras/6362754)\n  * ### Download App\n"
        "  * [!](https://apps.apple.com/app/apple-store/id1253028900?pt=118713609&ct=ps_footer_app_store&mt=8)\n"
        "  * [!](https://play.google.com/store/apps/details?id=com.listglobally.properstar&utm_source=properstar&utm_medium=website&utm_campaign=footer)\n\n\n"
        "  * ### Resources\n  * [Real estate guides](https://www.properstar.com/buying-property)\n"
        "  * [Blog](https://blog.properstar.com/en)\n  * [House price index](https://www.properstar.com/united-states/house-price)\n\n\n"
        "  * ### About us\n  * [Meet the team](https://www.properstar.com/meet-the-team)\n"
        "  * [Join our team](https://properstar.bamboohr.com/careers)\n  * [Contact us](https://www.properstar.com/contact)\n\n\n"
        "  * ### For professionals\n  * [Agents blog](https://agent.properstar.com/en)\n"
        "  * [Agent dashboard](https://dashboard.properstar.com/en-US/home?utm_medium=referrer&utm_source=properstar&utm_campaign=properstar_footer_agent_dashboard)\n"
        "  * [Add your listing](https://dashboard.properstar.com/en-US/home?utm_medium=referrer&utm_source=properstar&utm_campaign=properstar_footer_advertise)\n\n\n"
        "  * ### Legal\n  * [Terms and conditions](https://www.properstar.com/terms-and-conditions)\n"
        "  * [Privacy policy](https://www.properstar.com/privacy-policy)\n"
        "  * [Cookies policy](https://www.properstar.com/cookies-policy)\n\n\n  *\n  *\n  *\n  *\n\n\n"
        "Properstar © 2025\nRequest a tourContact agent\nRequest a tourContact agent\n"
        "Are you OK with optional cookies?\nThey let us tailor the best experience, improve our products and get more users. "
        "They’re off until you accept. [Learn more](https://www.properstar.com/cookies-policy)\nSettingsAccept\n"
    ),
    "agents": [
        {
            "id": "4a0c7ffc-74dc-46cb-9e49-96c296d69ef7",
            "agent_name": "Scraping Agent",
            "status": "Queued",
            "start_time": "2025-04-15T10:33:14.000Z",
            "end_time": "2025-04-15T10:33:14.000Z",
            "scraping_result_id": "f137bff3-a746-4864-b060-1faa01f38750",
        },
        {
            "id": "9e3cbcaf-c297-4975-b67a-9c54072b3e73",
            "agent_name": "Extracting Agent",
            "status": "Queued",
            "start_time": "2025-04-15T10:33:14.000Z",
            "end_time": "2025-04-15T10:33:14.000Z",
            "scraping_result_id": "f137bff3-a746-4864-b060-1faa01f38750",
        },
        {
            "id": "8e74d408-e161-4470-8e1c-1b33236f155b",
            "agent_name": "Cleaning Agent",
            "status": "Queued",
            "start_time": "2025-04-15T10:33:14.000Z",
            "end_time": "2025-04-15T10:33:14.000Z",
            "scraping_result_id": "f137bff3-a746-4864-b060-1faa01f38750",
        },
    ],
}

# Run the script
async def main():
    if not BOINGO_BEARER_TOKEN or not os.getenv("OPENAI_API_KEY"):
        logger.error("Missing environment variables: BOINGO_BEARER_TOKEN or OPENAI_API_KEY")
        return
    
    try:
        async with asyncio.timeout(300):
            await process_single_listing(LISTING)
    except asyncio.TimeoutError:
        logger.error("Script timed out after 5 minutes")
        payload = prepare_listing_payload(LISTING, {}, 0.0, "Failed")
        await update_scraping_result(payload, 0.0, status="Failed")

if __name__ == "__main__":
    asyncio.run(main())