# Property Crawler

A powerful web crawler specialized for real estate property listings. This tool automatically extracts, cleans, and formats property information from real estate websites.

## Features

- **Web Crawling**: Uses AsyncWebCrawler to scrape property listing websites
- **AI-Powered Extraction**: Leverages OpenAI's GPT-3.5 to extract structured property data
- **Data Cleaning**: Validates and standardizes property information
- **Report Generation**: Creates polished Markdown reports of property listings
- **Distributed Processing**: Uses Celery and Redis for scalable task processing
- **User-Friendly GUI**: Simple interface for configuring and launching crawls

## Components

- `crawleragent.py` - Main crawler that scrapes websites and extracts property data
- `cleaner_agent.py` - Cleans and validates the extracted property data
- `formatter_agent.py` - Formats the cleaned data into readable reports
- `property_pipeline.py` - Orchestrates the workflow using Celery

## Requirements

- Python 3.8+
- OpenAI API key
- Redis (for task queue)
- Required Python packages (see installation)

## Installation

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Set up Redis for task queue
4. Configure your OpenAI API key
5. Run the application: `python crawleragent.py`

## Usage

The easiest way to use the tool is through the GUI:

1. Launch the application with `python crawleragent.py`
2. Enter the URLs of property websites to scrape
3. Configure crawling depth and maximum pages
4. Click "Start Scraping"

For command-line usage:

```
python property_pipeline.py "url1,url2" --depth 2 --pages 5
``` 