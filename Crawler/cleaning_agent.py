# cleaner_agent.py
import requests
import json
from datetime import datetime, timezone
import logging
import sys
from openai import OpenAI
from .queue_manager import acquire_lock, release_lock
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("cleaning.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Force UTF-8 encoding for the console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up a requests session with retry logic
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def clean_with_openai(data):
    """Clean and translate listing data using OpenAI."""
    try:
        # Ensure the data structure has all required nested dictionaries
        data.setdefault("address", {})
        data.setdefault("listing", {})
        data.setdefault("contact", {})
        data.setdefault("features", [])

        # Define text fields based on cleaned data structure
        text_fields = {
            "address_country": data.get("address", {}).get("country", ""),
            "address_region": data.get("address", {}).get("region", ""),
            "address_city": data.get("address", {}).get("city", ""),
            "address_district": data.get("address", {}).get("district", ""),
            "listing_title": data.get("listing", {}).get("listing_title", ""),
            "listing_description": data.get("listing", {}).get("description", ""),
            "listing_price": data.get("listing", {}).get("price", ""),
            "listing_currency": data.get("listing", {}).get("currency", ""),
            "listing_status": data.get("listing", {}).get("status", ""),
            "listing_type": data.get("listing", {}).get("listing_type", ""),
            "listing_category": data.get("listing", {}).get("category", ""),
            "contact_first_name": data.get("contact", {}).get("first_name", ""),
            "contact_last_name": data.get("contact", {}).get("last_name", ""),
            "contact_phone": data.get("contact", {}).get("phone_number", ""),
            "contact_email": data.get("contact", {}).get("email_address", ""),
            "contact_company": data.get("contact", {}).get("company", ""),
            "features": json.dumps(data.get("features", []))
        }

        # Check if all fields are empty (except features, which might be an empty list)
        non_empty_fields = [value for key, value in text_fields.items() if key != "features" and value]
        if not non_empty_fields and not json.loads(text_fields["features"]):
            logger.warning("All input fields are empty; skipping OpenAI cleaning.")
            return data

        prompt = (
            "Translate the following text fields to English if they are not already in English, "
            "and refine them to be clear, concise, and grammatically correct. "
            "Ensure 'listing_title' is fully translated to English and cleaned up (e.g., fix typos like 'Cassa' to 'House'). "
            "For 'features', translate the text within the JSON structure while preserving the format. "
            "Return the results as a JSON object with the same keys as provided below.\n\n"
            "Input data:\n"
        )
        for key, value in text_fields.items():
            prompt += f"{key}: {value}\n"
        prompt += "\nReturn the cleaned data in this JSON format:\n"
        prompt += json.dumps({key: "" for key in text_fields.keys()}, indent=2)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in text refinement and translation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        cleaned_text = response.choices[0].message.content.strip()
        logger.debug(f"Raw OpenAI response: {cleaned_text}")
        
        try:
            cleaned_data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {cleaned_text}")
            raise ValueError(f"OpenAI response is not valid JSON: {str(e)}")

        # Update the data structure with cleaned values
        for key in text_fields:
            if key.startswith("address_"):
                new_key = key.replace("address_", "")
                data["address"][new_key] = cleaned_data[key] if cleaned_data[key] else None
            elif key.startswith("listing_"):
                new_key = key.replace("listing_", "")
                if new_key == "title":
                    data["listing"][new_key] = cleaned_data[key] if cleaned_data[key] else None
                else:
                    data["listing"][new_key] = cleaned_data[key] if cleaned_data[key] else None
            elif key.startswith("contact_"):
                new_key = key.replace("contact_", "")
                data["contact"][new_key] = cleaned_data[key] if cleaned_data[key] else None
            elif key == "features":
                # Check if cleaned_data[key] is already a list (OpenAI might return it as a list)
                if isinstance(cleaned_data[key], list):
                    data["features"] = cleaned_data[key]
                else:
                    data["features"] = json.loads(cleaned_data[key]) if cleaned_data[key] else []

        # Remove amenities from data if it exists (for backward compatibility)
        if "amenities" in data:
            del data["amenities"]

        return data
    except Exception as e:
        logger.error(f"Error in OpenAI cleaning: {str(e)}", exc_info=True)
        raise

def process_cleaning():
    """Process cleaning for all queued tasks."""
    if not acquire_lock():
        logger.info("Another process is running, skipping...")
        return
    
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        # Fetch all queued tasks for Cleaning Agent
        response = session.get(f"{BOINGO_API_URL}/agent-status/queued?agent_name=Cleaning Agent", headers=headers, timeout=30)
        response.raise_for_status()
        queued_tasks = response.json().get("data", {}).get("rows", {}).get("rows", [])
        if not queued_tasks:
            logger.info("No queued tasks for Cleaning Agent")
            return

        logger.info(f"Found {len(queued_tasks)} queued tasks for Cleaning Agent")

        # Process each task one by one
        for task in queued_tasks:
            agent_id = task["id"]
            scraping_result_id = task["scraping_result_id"]
            logger.info(f"Processing task with agent_id: {agent_id}, scraping_result_id: {scraping_result_id}")

            # Fetch and log the original scraping result
            result_response = session.get(f"{BOINGO_API_URL}/scraping-results/{scraping_result_id}", headers=headers, timeout=30)
            result_response.raise_for_status()
            result = result_response.json().get("data", {})
            logger.info(f"Original data from scraping-results/{scraping_result_id}: {json.dumps(result, indent=2)}")

            # Clean the data
            cleaned_data = None
            status = "Success"
            try:
                cleaned_data = clean_with_openai(result.get("data", {}).copy())
                logger.info(f"Successfully cleaned data for scraping_result_id: {scraping_result_id}")
            except Exception as e:
                status = "Error"  # Use "Error" as per Boingo API's allowed values
                logger.error(f"Cleaning failed for result ID {scraping_result_id}: {str(e)}")

            # Prepare update payload for scraping-results (without agent_statuses)
            now = datetime.now(timezone.utc).isoformat()
            update_payload = {
                "id": scraping_result_id,
                "source_url": result.get("source_url", ""),
                "data": cleaned_data if cleaned_data is not None else result.get("data", {}),
                "progress": 66 if status == "Success" else result.get("progress", 30),
                "status": "In Progress",
                "target_id": result.get("target_id", ""),
                "last_updated": now,
                "scraped_at": result.get("scraped_at", now)
            }

            update_payload["data"].setdefault("files", [])
            if update_payload["data"]["files"]:
                update_payload["data"]["files"] = update_payload["data"]["files"]

            # Update scraping-results
            logger.info(f"Sending update to {BOINGO_API_URL}/scraping-results: {json.dumps(update_payload, indent=2)}")
            response = session.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=update_payload, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully updated scraping-results for ID {scraping_result_id}")

            # Update agent-status (for the Cleaning Agent task only)
            agent_update_payload = {
                "id": agent_id,
                "agent_name": "Cleaning Agent",
                "status": status,
                "start_time": task.get("start_time", now),
                "end_time": now,
                "scraping_result_id": scraping_result_id
            }
            logger.info(f"Sending update to {BOINGO_API_URL}/agent-status: {json.dumps(agent_update_payload, indent=2)}")
            response = session.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=agent_update_payload, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully updated agent-status for ID {agent_id}")

    except requests.RequestException as e:
        logger.error(f"HTTP Request failed: {str(e)}")
        if e.response is not None:
            logger.error(f"Response body: {e.response.text}")
        else:
            logger.error("Response body: No response received")
    except Exception as e:
        logger.error(f"Cleaning process failed: {str(e)}", exc_info=True)
    finally:
        release_lock()

if __name__ == "__main__":
    process_cleaning()