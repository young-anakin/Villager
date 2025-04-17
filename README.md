# Boingo API Tester

A FastAPI application with Celery for property data scraping and processing.

## Prerequisites

- Python 3.8+
- Docker
- Redis (via Docker)
- OpenAI API Key

## Setup Instructions

1. Start Docker on your system

2. Set up Redis using Docker:
   ```bash
   # Check running containers
   docker ps
   
   # Remove existing Redis container if any
   docker rm -f redis-container
   
   # Start Redis container
   docker run --name redis-container -d -p 6379:6379 redis
   ```

3. Configure OpenAI API Key:
   - Open `Crawler/cleaner_agent.py`
   - Open `Crawler/crawleragent.py`
   - Open `Crawler/formatter_agent.py`
   - In each file, replace the empty API key with your OpenAI API key:
     ```python
     client = OpenAI(api_key="your-api-key-here")
     ```

4. Start the Celery worker:
   ```bash
   python -m celery -A app.core.celery_app worker --loglevel=info --pool=solo -Q scraper_queue,cleaner_queue
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

The application will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
.
├── app/
│   ├── core/
│   ├── models/
│   └── routers/
├── Crawler/
│   ├── cleaner_agent.py
│   ├── crawleragent.py
│   └── formatter_agent.py
└── requirements.txt
```

## License

MIT

## Features

- Authentication with Boingo API
- Management of scraping targets (create, read, update, delete)
- Handling of scraping results (create, read, update, delete)
- Agent status management
- Scraping analytics

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/boingo-api-tester.git
cd boingo-api-tester
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
BOINGO_API_URL=https://api.boingo.com
BOINGO_EMAIL=your-email@example.com
BOINGO_PASSWORD=your-password
```

## Usage

1. Start the application:
```bash
uvicorn app.main:app --reload
```

2. Open your browser and navigate to [http://localhost:8000/docs](http://localhost:8000/docs) to access the Swagger UI documentation.

3. Use the interactive API documentation to test the various endpoints.

## API Endpoints

The application provides the following categories of endpoints:

- **/auth** - Authentication endpoints
- **/scraping-target** - Endpoints for managing scraping targets
- **/scraping-results** - Endpoints for managing scraping results
- **/agent-status** - Endpoints for managing agent status
- **/scraping-analytics** - Endpoints for accessing scraping analytics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 