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

# Configure logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("extracting.log", encoding='utf-8'),
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
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

def enhance_with_openai(data):
    """Enhance listing data using OpenAI to make it polished and appealing."""
    try:
        # Ensure the data structure has all required nested dictionaries
        data.setdefault("address", {})
        data.setdefault("listing", {})
        data.setdefault("contact", {})
        data.setdefault("features", [])

        # Define text fields based on cleaned data structure
        text_fields = {
            "address_country": data["address"].get("country", ""),
            "address_region": data["address"].get("region", ""),
            "address_city": data["address"].get("city", ""),
            "address_district": data["address"].get("district", ""),
            "listing_title": data["listing"].get("listing_title", ""),
            "listing_description": data["listing"].get("description", ""),
            "listing_price": data["listing"].get("price", ""),
            "listing_currency": data["listing"].get("currency", ""),
            "listing_status": data["listing"].get("status", ""),
            "listing_type": data["listing"].get("listing_type", ""),
            "listing_category": data["listing"].get("category", ""),
            "contact_first_name": data["contact"].get("first_name", ""),
            "contact_last_name": data["contact"].get("last_name", ""),
            "contact_phone": data["contact"].get("phone_number", ""),
            "contact_email": data["contact"].get("email_address", ""),
            "contact_company": data["contact"].get("company", ""),
            "features": json.dumps(data.get("features", []))
        }

        # Check if all fields are empty (except features, which might be an empty list)
        non_empty_fields = [value for key, value in text_fields.items() if key != "features" and value]
        if not non_empty_fields and not json.loads(text_fields["features"]):
            logger.warning("All input fields are empty; skipping OpenAI enhancement.")
            return None  # Return None to indicate no enhancement was performed

        prompt = (
            "Enhance the following text fields for a property listing to be polished, concise, and appealing in English. "
            "Rewrite each field to improve clarity and attractiveness while maintaining its meaning. "
            "For 'features', enhance the text within the JSON structure while preserving the format. "
            "If all fields are empty, return the input data unchanged in the same JSON format. "
            "Return the results in JSON format with the same keys.\n\n"
            "Input data:\n"
        )
        for key, value in text_fields.items():
            prompt += f"{key}: {value}\n"
        prompt += "\nReturn the enhanced data in this JSON format:\n"
        prompt += json.dumps({key: "" for key in text_fields.keys()}, indent=2)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a skilled writer creating polished property listing content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        enhanced_text = response.choices[0].message.content.strip()
        logger.debug(f"Raw OpenAI response: {enhanced_text}")

        try:
            enhanced_data = json.loads(enhanced_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {enhanced_text}")
            raise ValueError(f"OpenAI response is not valid JSON: {str(e)}")

        # Update the data structure with enhanced values
        for key in text_fields:
            if key.startswith("address_"):
                new_key = key.replace("address_", "")
                data["address"][new_key] = enhanced_data[key] if enhanced_data[key] else None
            elif key.startswith("listing_"):
                new_key = key.replace("listing_", "")
                if new_key == "title":
                    data["listing"][new_key] = enhanced_data[key] if enhanced_data[key] else None
                else:
                    data["listing"][new_key] = enhanced_data[key] if enhanced_data[key] else None
            elif key.startswith("contact_"):
                new_key = key.replace("contact_", "")
                data["contact"][new_key] = enhanced_data[key] if enhanced_data[key] else None
            elif key == "features":
                if isinstance(enhanced_data[key], list):
                    data["features"] = enhanced_data[key]
                else:
                    data["features"] = json.loads(enhanced_data[key]) if enhanced_data[key] else []

        # Remove amenities if present (backward compatibility)
        if "amenities" in data:
            del data["amenities"]

        return data
    except Exception as e:
        logger.error(f"Error in OpenAI enhancement: {str(e)}", exc_info=True)
        raise

def process_extracting():
    """Process extracting for queued tasks."""
    if not acquire_lock():
        logger.info("Another process is running, skipping...")
        return

    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        # Fetch queued tasks for Extracting Agent
        response = session.get(
            f"{BOINGO_API_URL}/agent-status/queued?agent_name=Extracting Agent",
            headers=headers,
            timeout=60  # Increased timeout
        )
        if response.status_code != 200:
            logger.error(f"Failed to fetch queued tasks: {response.status_code} - {response.text}")
            return
        queued_tasks = response.json().get("data", {}).get("rows", {}).get("rows", [])
        if not queued_tasks:
            logger.info("No queued tasks for Extracting Agent")
            return

        # Process the first task
        task = queued_tasks[0]
        agent_id = task["id"]
        scraping_result_id = task["scraping_result_id"]

        # Fetch the scraping result
        result_response = session.get(
            f"{BOINGO_API_URL}/scraping-results/{scraping_result_id}",
            headers=headers,
            timeout=60  # Increased timeout
        )
        if result_response.status_code != 200:
            logger.error(f"Failed to fetch scraping result {scraping_result_id}: {result_response.status_code} - {result_response.text}")
            return
        result = result_response.json().get("data", {})
        logger.info(f"Original data from scraping-results/{scraping_result_id}: {json.dumps(result, indent=2)}")

        # Skip if the data field is empty or contains no meaningful data
        data = result.get("data", {})
        if not data or (not any(data.get(key) for key in ["address", "listing", "contact"]) and not data.get("features")):
            logger.warning(f"Skipping extraction for {scraping_result_id}: 'data' field is empty or contains no meaningful data")
            # Update agent-status to mark as Error
            now = datetime.now(timezone.utc).isoformat()
            agent_update_payload = {
                "id": agent_id,
                "agent_name": "Extracting Agent",
                "status": "Error",
                "start_time": task.get("start_time", now),
                "end_time": now,
                "scraping_result_id": scraping_result_id
            }
            logger.info(f"Sending update to {BOINGO_API_URL}/agent-status: {json.dumps(agent_update_payload, indent=2)}")
            response = session.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=agent_update_payload, timeout=60)
            if response.status_code != 200:
                logger.error(f"Failed to update agent status {agent_id}: {response.status_code} - {response.text}")
            else:
                logger.info(f"Successfully updated agent-status for ID {agent_id} with status 'Error'")
            return

        # Enhance the data
        enhanced_data = None
        status = "Success"
        try:
            enhanced_data = enhance_with_openai(data.copy())
            if enhanced_data is None:  # No enhancement due to empty data
                status = "Error"
        except Exception as e:
            status = "Error"
            logger.error(f"Enhancement failed for result ID {scraping_result_id}: {str(e)}")

        # Prepare update payload for scraping-results
        now = datetime.now(timezone.utc).isoformat()
        update_payload = {
            "id": scraping_result_id,
            "source_url": result.get("source_url", ""),
            "data": enhanced_data if enhanced_data is not None else data,
            "progress": 100 if status == "Success" else result.get("progress", 66),
            "status": "Success" if status == "Success" else "Error",
            "target_id": result.get("target_id", ""),
            "last_updated": now,
            "scraped_at": result.get("scraped_at", now)
        }

        # Update scraping-results with retry logic
        logger.info(f"Sending update to {BOINGO_API_URL}/scraping-results: {json.dumps(update_payload, indent=2)}")
        try:
            response = session.put(
                f"{BOINGO_API_URL}/scraping-results",
                headers=headers,
                json=update_payload,
                timeout=120  # Increased timeout
            )
            response.raise_for_status()
            logger.info(f"Successfully updated scraping-results for ID {scraping_result_id}")
        except requests.Timeout:
            logger.error(f"Timeout error while updating scraping result {scraping_result_id}")
            status = "Error"
        except requests.RequestException as e:
            logger.error(f"Failed to update scraping result {scraping_result_id}: {str(e)}")
            status = "Error"

        # Update agent-status for Extracting Agent
        agent_update_payload = {
            "id": agent_id,
            "agent_name": "Extracting Agent",
            "status": status,
            "start_time": task.get("start_time", now),
            "end_time": now,
            "scraping_result_id": scraping_result_id
        }
        logger.info(f"Sending update to {BOINGO_API_URL}/agent-status: {json.dumps(agent_update_payload, indent=2)}")
        response = session.put(
            f"{BOINGO_API_URL}/agent-status",
            headers=headers,
            json=agent_update_payload,
            timeout=60
        )
        if response.status_code != 200:
            logger.error(f"Failed to update agent status {agent_id}: {response.status_code} - {response.text}")
        else:
            logger.info(f"Successfully updated agent-status for ID {agent_id}")

    except Exception as e:
        logger.error(f"Extracting process failed: {str(e)}", exc_info=True)
    finally:
        release_lock()

if __name__ == "__main__":
    process_extracting()