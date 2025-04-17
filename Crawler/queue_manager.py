import json
import os
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("queue_manager.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Use absolute paths (adjust BASE_DIR to your project root)
BASE_DIR = Path(__file__).parent.parent  # Assumes queue_manager.py is in Crawler/
QUEUE_FILE = BASE_DIR / "queue.json"
LOCK_FILE = BASE_DIR / "processing.lock"

def init_queue():
    """Initialize queue.json if it doesn't exist."""
    if not QUEUE_FILE.exists():
        logger.debug(f"Initializing {QUEUE_FILE} with empty queues")
        try:
            with open(QUEUE_FILE, "w", encoding='utf-8') as f:
                json.dump({"scraping": [], "cleaning": [], "extracting": []}, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to initialize {QUEUE_FILE}: {str(e)}")
            raise

def add_to_queue(queue_name: str, task: Dict[str, Any]):
    """Add a task to the specified queue in queue.json with file locking."""
    init_queue()  # Ensure queue exists
    lock_fd = None
    try:
        lock_fd = acquire_file_lock(QUEUE_FILE)
        with open(QUEUE_FILE, "r", encoding='utf-8') as f:
            queues = json.load(f)
        
        if queue_name not in queues:
            logger.debug(f"Queue {queue_name} not found, initializing")
            queues[queue_name] = []
        
        queues[queue_name].append(task)
        with open(QUEUE_FILE, "w", encoding='utf-8') as f:
            json.dump(queues, f, indent=2)
        logger.debug(f"Added task to {queue_name} queue: {task}")
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to update {QUEUE_FILE}: {str(e)}")
        raise
    finally:
        if lock_fd:
            release_file_lock(lock_fd)

def get_next_task(task_type: str):
    """Retrieve and remove the next task from the specified queue with file locking."""
    lock_fd = None
    try:
        lock_fd = acquire_file_lock(QUEUE_FILE)
        with open(QUEUE_FILE, "r", encoding='utf-8') as f:
            queue = json.load(f)
        
        if task_type in queue and queue[task_type]:
            task = queue[task_type].pop(0)
            with open(QUEUE_FILE, "w", encoding='utf-8') as f:
                json.dump(queue, f, indent=2)
            logger.debug(f"Retrieved task from {task_type}: {task}")
            return task
        logger.debug(f"No tasks in {task_type} queue")
        return None
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to access {QUEUE_FILE}: {str(e)}")
        raise
    finally:
        if lock_fd:
            release_file_lock(lock_fd)

def acquire_lock():
    """Acquire a lock by creating a lock file atomically."""
    logger.debug(f"Attempting to acquire lock with {LOCK_FILE}")
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            pid = str(os.getpid())
            f.write(pid)
        logger.debug(f"Lock acquired, wrote PID: {pid}")
        return True
    except FileExistsError:
        logger.debug("Lock file exists, cannot acquire lock")
        return False
    except OSError as e:
        logger.error(f"Failed to acquire lock: {str(e)}")
        return False

def release_lock():
    """Release the lock by removing the lock file."""
    logger.debug(f"Attempting to release lock file {LOCK_FILE}")
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
            logger.debug("Lock file removed")
        except OSError as e:
            logger.error(f"Failed to release lock: {str(e)}")
            # Don’t raise; log and continue

def acquire_file_lock(file_path: Path):
    """Acquire an exclusive file lock for atomic queue operations."""
    lock_file = file_path.with_suffix('.lock')
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        logger.debug(f"Acquired file lock for {file_path}")
        return fd
    except FileExistsError:
        logger.debug(f"Lock exists for {file_path}, waiting not implemented")
        raise IOError(f"Cannot acquire lock for {file_path}")
    except OSError as e:
        logger.error(f"Failed to acquire file lock for {file_path}: {str(e)}")
        raise

def release_file_lock(fd):
    """Release the file lock."""
    try:
        lock_file = QUEUE_FILE.with_suffix('.lock')
        os.close(fd)
        if lock_file.exists():
            lock_file.unlink()
        logger.debug(f"Released file lock for {QUEUE_FILE}")
    except OSError as e:
        logger.error(f"Failed to release file lock: {str(e)}")
