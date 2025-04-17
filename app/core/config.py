import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Boingo API Settings
BOINGO_API_URL = os.getenv("BOINGO_API_URL", "https://api.boingo.ai")
BOINGO_BEARER_TOKEN = os.getenv("BOINGO_BEARER_TOKEN")  # No default for security
BOINGO_EMAIL = os.getenv("BOINGO_EMAIL")  # No default
BOINGO_PASSWORD = os.getenv("BOINGO_PASSWORD")  # No default

# OpenAI API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # No default

# Validate critical variables
for key, value in {
    "BOINGO_BEARER_TOKEN": BOINGO_BEARER_TOKEN,
    "OPENAI_API_KEY": OPENAI_API_KEY
}.items():
    if not value:
        raise ValueError(f"{key} is required in environment variables")