# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import auth, targets, results, agent_status, analytics
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Boingo API Tester",
    description="A FastAPI application for testing Boingo API endpoints",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Authentication API",
            "description": "Operations related to authentication with the Boingo API"
        },
        {
            "name": "Scraping Target API", 
            "description": "Operations for creating and managing scraping targets"
        },
        {
            "name": "Scraping Result API",
            "description": "Operations for managing scraped results data"
        },
        {
            "name": "Agent Status API",
            "description": "Operations for monitoring and managing agent statuses"
        },
        {
            "name": "Scraping Analytics API",
            "description": "Operations for retrieving scraping analytics and metrics"
        }
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(auth.router)
app.include_router(targets.router)
app.include_router(results.router)
app.include_router(agent_status.router)
app.include_router(analytics.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Boingo API Tester",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth",
            "targets": "/scraping-target",
            "results": "/scraping-results",
            "agent_status": "/agent-status",
            "analytics": "/scraping-analytics"
        }
    }