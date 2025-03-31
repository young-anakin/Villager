import requests
import logging
from app.core.config import BOINGO_API_URL, BOINGO_BEARER_TOKEN
from Crawler.queue_manager import add_to_queue

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def submit_task():
    """Submit a task to the Boingo API and add it to the local queue."""
    headers = {"Authorization": f"Bearer {BOINGO_BEARER_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "website_url": "www.google.com",
        "location": "Mexico, USA",
        "schedule_time": "2025-03-17T09:00:00Z",
        "frequency": "Daily",
        "search_range": 0,
        "max_properties": 1
    }
    try:
        logger.debug(f"Sending POST request to {BOINGO_API_URL}/scraping-target/ with payload: {payload}")
        response = requests.post(f"{BOINGO_API_URL}/scraping-target/", headers=headers, json=payload)
        if response.status_code in [200, 201]:
            logger.info(f"Task submitted successfully: {response.json()}")
            # Add to local queue for scraper.py
            # Use website_url and an identifier (e.g., response ID if available)
            task = {
                "website_url": payload["website_url"],
                "target_id": response.json().get("data", {}).get("id", "manual-task-id")  # Adjust based on API response
            }
            add_to_queue("scraping", task)
            logger.info(f"Task added to queue.json: {task}")
        else:
            logger.error(f"Failed to submit task: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error submitting task: {str(e)}")

if __name__ == "__main__":
    submit_task()