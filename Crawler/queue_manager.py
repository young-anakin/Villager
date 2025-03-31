import json
import os
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("queue_manager.log")]
)
logger = logging.getLogger(__name__)

QUEUE_FILE = Path("queue.json")
LOCK_FILE = Path("processing.lock")

def init_queue():
    """Initialize queue.json if it doesn't exist."""
    if not QUEUE_FILE.exists():
        logger.debug(f"Initializing {QUEUE_FILE} with empty queues")
        try:
            with open(QUEUE_FILE, "w") as f:
                json.dump({"scraping": [], "cleaning": [], "extracting": []}, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to initialize {QUEUE_FILE}: {str(e)}")
            raise

def add_to_queue(queue_name: str, task: Dict[str, Any]):
    """Add a task to the specified queue in queue.json."""
    queue_file = QUEUE_FILE
    logger.debug(f"Attempting to read {queue_file}")
    try:
        with open(queue_file, "r") as f:
            raw_content = f.read()
            logger.debug(f"Raw queue.json content: {raw_content!r}")
            queues = json.loads(raw_content)
            logger.debug(f"Parsed queue.json: {json.dumps(queues, indent=2)}")
    except FileNotFoundError:
        logger.debug(f"{queue_file} not found, initializing new queues")
        init_queue()
        queues = {"scraping": [], "cleaning": [], "extracting": []}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode queue.json: {str(e)}")
        logger.debug(f"Error position: line {e.lineno}, column {e.colno}, char {e.pos}")
        logger.debug(f"Text around error (char {e.pos-10}:{e.pos+10}): {raw_content[max(0, e.pos-10):e.pos+10]!r}")
        raise
    
    if queue_name not in queues:
        logger.debug(f"Queue {queue_name} not found, initializing")
        queues[queue_name] = []
    
    logger.debug(f"Appending task to {queue_name}: {task}")
    queues[queue_name].append(task)
    logger.debug(f"Writing updated queues to {queue_file}")
    try:
        with open(queue_file, "w") as f:
            json.dump(queues, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to write to {queue_file}: {str(e)}")
        raise
    logger.debug(f"Added task to {queue_name} queue: {task}")

def get_next_task(task_type: str):
    """Retrieve and remove the next task from the specified queue."""
    init_queue()
    logger.debug(f"Attempting to read and update {QUEUE_FILE} for {task_type}")
    try:
        with open(QUEUE_FILE, "r+") as f:
            raw_content = f.read()
            logger.debug(f"Raw queue.json content: {raw_content!r}")
            queue = json.loads(raw_content)
            logger.debug(f"Parsed queue.json: {json.dumps(queue, indent=2)}")
            if task_type in queue and queue[task_type]:
                task = queue[task_type].pop(0)
                f.seek(0)
                json.dump(queue, f, indent=2)
                f.truncate()
                logger.debug(f"Retrieved task from {task_type}: {task}")
                return task
            logger.debug(f"No tasks in {task_type} queue")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode queue.json: {str(e)}")
        logger.debug(f"Error position: line {e.lineno}, column {e.colno}, char {e.pos}")
        logger.debug(f"Text around error (char {e.pos-10}:{e.pos+10}): {raw_content[max(0, e.pos-10):e.pos+10]!r}")
        raise
    except IOError as e:
        logger.error(f"Failed to access {QUEUE_FILE}: {str(e)}")
        raise

def acquire_lock():
    """Acquire a lock by creating a lock file."""
    logger.debug(f"Checking for lock file {LOCK_FILE}")
    if LOCK_FILE.exists():
        logger.debug("Lock file exists, cannot acquire lock")
        return False
    try:
        with open(LOCK_FILE, "w") as f:
            pid = str(os.getpid())
            f.write(pid)
            logger.debug(f"Lock acquired, wrote PID: {pid}")
        return True
    except IOError as e:
        logger.error(f"Failed to acquire lock: {str(e)}")
        raise

def release_lock():
    """Release the lock by removing the lock file."""
    logger.debug(f"Attempting to release lock file {LOCK_FILE}")
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
            logger.debug("Lock file removed")
        except OSError as e:
            logger.error(f"Failed to release lock: {str(e)}")
            raise
    else:
        logger.debug("No lock file to release")