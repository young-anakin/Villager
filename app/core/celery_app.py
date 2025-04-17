# app/core/celery_app.py
from celery import Celery

# Define the Celery app
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['Crawler.property_pipeline']  # Load tasks from this module
)

# Configure Celery queues and routes
celery_app.conf.task_queues = {
    'scraper_queue': {'exchange': 'scraper_queue', 'routing_key': 'scraper'},
    'cleaner_queue': {'exchange': 'cleaner_queue', 'routing_key': 'cleaner'},
}
celery_app.conf.task_routes = {
    'Crawler.property_pipeline.scrape_task': {'queue': 'scraper_queue'},
    'Crawler.property_pipeline.clean_task': {'queue': 'cleaner_queue'},
}

# Fix deprecation warning for Celery 6.0+
celery_app.conf.broker_connection_retry_on_startup = True

# Enable task events and configure logging
celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    task_send_sent_event=True,
    task_ignore_result=False,
    worker_send_task_events=True,
    task_always_eager=False,
    task_eager_propagates=True,
    task_remote_tracebacks=True,
    task_annotations={
        '*': {'rate_limit': '10/s'}
    }
)

if __name__ == '__main__':
    celery_app.start()