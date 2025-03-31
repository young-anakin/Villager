import requests
import json
from datetime import datetime, timezone
import logging
from openai import OpenAI
from queue_manager import acquire_lock, release_lock
from ..app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN, OPENAI_API_KEY  # Adjusted path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.FileHandler("cleaning.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def clean_with_openai(data):
    """Clean and translate listing data using OpenAI."""
    try:
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

        for key in text_fields:
            if key.startswith("contact_"):
                new_key = key.replace("contact_", "")
                data["contact"][new_key] = cleaned_data[key]
            else:
                keys = key.split("_")
                target = data
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                if key == "features" and isinstance(cleaned_data[key], str):
                    target[keys[-1]] = json.loads(cleaned_data[key])
                else:
                    target[keys[-1]] = cleaned_data[key]

        # Remove amenities from data if it exists (for backward compatibility)
        if "amenities" in data:
            del data["amenities"]

        return data
    except Exception as e:
        logger.error(f"Error in OpenAI cleaning: {str(e)}", exc_info=True)
        return data

def process_cleaning():
    """Process cleaning for queued tasks."""
    if not acquire_lock():
        logger.info("Another process is running, skipping...")
        return
    
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        response = requests.get(f"{BOINGO_API_URL}/agent-status/queued?agent_name=Cleaning Agent", headers=headers)
        response.raise_for_status()
        queued_tasks = response.json().get("data", {}).get("rows", {}).get("rows", [])
        if not queued_tasks:
            logger.info("No queued tasks for Cleaning Agent")
            return

        task = queued_tasks[0]
        agent_id = task["id"]
        scraping_result_id = task["scraping_result_id"]

        # Fetch and log the original scraping result
        result_response = requests.get(f"{BOINGO_API_URL}/scraping-results/{scraping_result_id}", headers=headers)
        result_response.raise_for_status()
        result = result_response.json().get("data", {})
        logger.info(f"Original data from scraping-results/{scraping_result_id}: {json.dumps(result, indent=2)}")

        # Clean the data
        cleaned_data = None
        status = "In Progress"
        error_message = None
        try:
            cleaned_data = clean_with_openai(result.get("data", {}).copy())
        except Exception as e:
            status = "Failed"
            error_message = str(e)
            logger.error(f"Cleaning failed for result ID {scraping_result_id}: {error_message}")

        # Prepare update payload for scraping-results (without agent_statuses)
        now = datetime.now(timezone.utc).isoformat()
        update_payload = {
            "id": scraping_result_id,
            "source_url": result.get("source_url", ""),
            "data": cleaned_data if cleaned_data is not None else result.get("data", {}),
            "progress": 66,  # Fixed to 66 as per your earlier request
            "status": status,
            "target_id": result.get("target_id", ""),
            "last_updated": now,
            "scraped_at": result.get("scraped_at", now)
        }
        if error_message:
            update_payload["error"] = error_message

        # Trim files list to test payload size
        update_payload["data"]["files"] = update_payload["data"]["files"][:2]

        # Update scraping-results
        logger.info(f"Sending update to {BOINGO_API_URL}/scraping-results: {json.dumps(update_payload, indent=2)}")
        response = requests.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=update_payload)
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
        response = requests.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=agent_update_payload)
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