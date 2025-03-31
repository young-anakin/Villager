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
    handlers=[logging.FileHandler("extracting.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

def enhance_with_openai(data):
    """Enhance listing data using OpenAI to make it polished and appealing."""
    try:
        # Define text fields based on cleaned data structure (no amenities)
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
            "Enhance the following text fields for a property listing to be polished, concise, and appealing in English. "
            "Rewrite each field to improve clarity and attractiveness while maintaining its meaning. "
            "For 'features', enhance the text within the JSON structure while preserving the format. "
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
            if key.startswith("contact_"):
                new_key = key.replace("contact_", "")
                data["contact"][new_key] = enhanced_data[key] if enhanced_data[key] else None
            else:
                keys = key.split("_")
                target = data
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                if key == "features" and isinstance(enhanced_data[key], str):
                    target[keys[-1]] = json.loads(enhanced_data[key])
                else:
                    target[keys[-1]] = enhanced_data[key]

        # Remove amenities if present (backward compatibility)
        if "amenities" in data:
            del data["amenities"]

        return data
    except Exception as e:
        logger.error(f"Error in OpenAI enhancement: {str(e)}", exc_info=True)
        return data

def process_extracting():
    """Process extracting for queued tasks."""
    if not acquire_lock():
        logger.info("Another process is running, skipping...")
        return
    
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    try:
        # Fetch queued tasks for Extracting Agent
        response = requests.get(f"{BOINGO_API_URL}/agent-status/queued?agent_name=Extracting Agent", headers=headers)
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
        result_response = requests.get(f"{BOINGO_API_URL}/scraping-results/{scraping_result_id}", headers=headers)
        if result_response.status_code != 200:
            logger.error(f"Failed to fetch scraping result {scraping_result_id}: {result_response.status_code} - {result_response.text}")
            return
        result = result_response.json().get("data", {})
        logger.info(f"Original data from scraping-results/{scraping_result_id}: {json.dumps(result, indent=2)}")

        # Enhance the data
        enhanced_data = None
        status = "Success"
        error_message = None
        try:
            enhanced_data = enhance_with_openai(result.get("data", {}).copy())
        except Exception as e:
            status = "Failed"
            error_message = str(e)
            logger.error(f"Enhancement failed for result ID {scraping_result_id}: {error_message}")

        # Prepare update payload for scraping-results
        now = datetime.now(timezone.utc).isoformat()
        update_payload = {
            "id": scraping_result_id,
            "source_url": result.get("source_url", ""),
            "data": enhanced_data if enhanced_data is not None else result.get("data", {}),
            "progress": 100,
            "status": status,
            "target_id": result.get("target_id", ""),
            "last_updated": now,
            "scraped_at": result.get("scraped_at", now)
        }
        if error_message:
            update_payload["error"] = error_message

        # Update scraping-results
        logger.info(f"Sending update to {BOINGO_API_URL}/scraping-results: {json.dumps(update_payload, indent=2)}")
        response = requests.put(f"{BOINGO_API_URL}/scraping-results", headers=headers, json=update_payload)
        if response.status_code != 200:
            logger.error(f"Failed to update scraping result {scraping_result_id}: {response.status_code} - {response.text}")
            return
        logger.info(f"Successfully updated scraping-results for ID {scraping_result_id}")

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
        response = requests.put(f"{BOINGO_API_URL}/agent-status", headers=headers, json=agent_update_payload)
        if response.status_code != 200:
            logger.error(f"Failed to update agent status {agent_id}: {response.status_code} - {response.text}")
            return
        logger.info(f"Successfully updated agent-status for ID {agent_id}")

    except Exception as e:
        logger.error(f"Extracting process failed: {str(e)}", exc_info=True)
    finally:
        release_lock()

if __name__ == "__main__":
    process_extracting()