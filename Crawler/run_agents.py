import time
import subprocess
from datetime import datetime
from croniter import croniter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ec2-user/logs/run_agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CONFIG
CRON_EXPRESSION = "0 0 * * *"  # Runs daily at midnight
SCRIPTS = [
    "/home/ec2-user/Crawler/scraper.py",
    "/home/ec2-user/Crawler/cleaner_agent.py",
    "/home/ec2-user/Crawler/extracting_agent.py"
]

def run_script(script):
    logger.info(f"Running: {script}")
    try:
        subprocess.run(["/usr/bin/python3", script], check=True)
        logger.info(f"Finished: {script}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script}: {e}")

def main():
    base_time = datetime.now()
    cron = croniter(CRON_EXPRESSION, base_time)

    logger.info(f"Scheduler started. Running daily as per '{CRON_EXPRESSION}'")

    while True:
        next_time = cron.get_next(datetime)
        wait_seconds = (next_time - datetime.now()).total_seconds()

        if wait_seconds > 0:
            logger.info(f"Sleeping for {wait_seconds:.2f} seconds until {next_time}")
            time.sleep(wait_seconds)

        for script in SCRIPTS:
            run_script(script)

if __name__ == "__main__":
    main()