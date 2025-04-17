import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

QUEUE_FILE = Path("queue.json")

def clear_queue():
    """Clear all tasks from queue.json."""
    empty_queues = {"scraping": [], "cleaning": [], "extracting": []}
    try:
        with open(QUEUE_FILE, "w") as f:
            json.dump(empty_queues, f, indent=2)
        logger.info(f"Cleared all tasks from {QUEUE_FILE}")
    except IOError as e:
        logger.error(f"Failed to clear {QUEUE_FILE}: {str(e)}")
        raise

if __name__ == "__main__":
    clear_queue()