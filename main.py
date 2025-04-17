from fastapi import FastAPI, HTTPException, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, validator, Field
from typing import Optional, List, Dict, Any, Literal
import httpx
from datetime import datetime
import os
from dotenv import load_dotenv
import json
import redis
from celery import Celery, chain
import uuid
from datetime import datetime

# Initialize Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['Crawler.property_pipeline']
)

celery_app.conf.task_queues = {
    'scraper_queue': {'exchange': 'scraper_queue', 'routing_key': 'scraper'},
    'cleaner_queue': {'exchange': 'cleaner_queue', 'routing_key': 'cleaner'},
}
celery_app.conf.task_routes = {
    'Crawler.property_pipeline.scrape_task': {'queue': 'scraper_queue'},
    'Crawler.property_pipeline.clean_task': {'queue': 'cleaner_queue'},
}
celery_app.conf.broker_connection_retry_on_startup = True
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
    task_annotations={'*': {'rate_limit': '10/s'}}
)

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_client.ping()
# Load environment variables
load_dotenv()

app = FastAPI(
    title="Boingo API Wrapper",
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

# Security scheme for Swagger UI
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# Configuration (hardcoded values)
BOINGO_API_URL = "https://api.boingo.ai"
BOINGO_EMAIL = "yohanistadese06@gmail.com"
BOINGO_PASSWORD = "Nodex_1@1_pass"

# Models
class LoginRequest(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    status: str
    type: str
    is_verified: bool
    has_subscribed: bool
    phone_number: Optional[str] = None
    role: Optional[Dict[str, Any]] = None

class LoginResponse(BaseModel):
    status: int
    message: str
    data: Dict[str, Any]

class ScrapingTargetCreate(BaseModel):
    website_url: str = Field(..., description="The URL of the website to scrape. Will automatically add https:// if missing.")
    location: str = Field(..., description="The location to search for properties (e.g., 'Mexico, USA')")
    schedule_time: datetime = Field(..., description="When to run the scraping job (in ISO 8601 format)")
    frequency: str = Field(
        ...,
        description="How often to run the scraping job. Must be one of: Daily, Weekly, Monthly (case-insensitive)"
    )
    search_range: int = Field(..., description="The search range in kilometers")
    max_properties: int = Field(..., description="Maximum number of properties to scrape")

    @validator('website_url')
    def validate_website_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        return v

    @validator('frequency')
    def validate_frequency(cls, v):
        v = v.capitalize()  # Convert to proper case (first letter uppercase)
        if v not in ["Daily", "Weekly", "Monthly"]:
            raise ValueError('frequency must be one of: Daily, Weekly, Monthly')
        return v

class ScrapingTargetUpdate(BaseModel):
    id: str
    website_url: str = Field(..., description="The URL of the website to scrape. Will automatically add https:// if missing.")
    location: str = Field(..., description="The location to search for properties (e.g., 'Mexico, USA')")
    schedule_time: datetime = Field(..., description="When to run the scraping job (in ISO 8601 format)")
    frequency: str = Field(
        ...,
        description="How often to run the scraping job. Must be one of: Daily, Weekly, Monthly (case-insensitive)"
    )
    search_range: int = Field(..., description="The search range in kilometers")
    max_properties: int = Field(..., description="Maximum number of properties to scrape")

    @validator('website_url')
    def validate_website_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            v = 'https://' + v
        return v

    @validator('frequency')
    def validate_frequency(cls, v):
        v = v.capitalize()  # Convert to proper case (first letter uppercase)
        if v not in ["Daily", "Weekly", "Monthly"]:
            raise ValueError('frequency must be one of: Daily, Weekly, Monthly')
        return v

class ScrapingTargetDelete(BaseModel):
    id: str
    force: bool

class ScrapingTargetPause(BaseModel):
    id: str

class AgentStatus(BaseModel):
    agent_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None

class ScrapingResultCreate(BaseModel):
    source_url: str
    listing_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    progress: float
    status: str
    scraped_at: datetime
    target_id: str
    agent_status: List[AgentStatus]

class ScrapingResultUpdate(BaseModel):
    id: str
    source_url: str
    listing_type: str
    data: Dict[str, Any]
    progress: float
    status: str
    scraped_at: datetime
    last_updated: datetime
    target_id: str

class ScrapingResultDelete(BaseModel):
    id: str
    force: bool

# Authentication
async def get_auth_token():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BOINGO_API_URL}/auth/login",
            json={"email": BOINGO_EMAIL, "password": BOINGO_PASSWORD}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Authentication failed")
        return response.json().get("data", {}).get("token")

# Endpoints
@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication API"])
async def login(credentials: LoginRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BOINGO_API_URL}/auth/login",
            json=credentials.dict()
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

@app.get("/scraping-target", tags=["Scraping Target API"])
async def get_all_targets(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-target",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting targets: {str(e)}")

@app.get("/scraping-target/{target_id}", tags=["Scraping Target API"])
async def get_target_by_id(target_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        print(f"\n=== Get Target Request ===")
        print(f"Target ID: {target_id}")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{BOINGO_API_URL}/scraping-target/{target_id}",
                    headers={"Authorization": f"Bearer {credentials.credentials}"}
                )
                
                print(f"\n=== Get Target Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                
                # Accept all 2xx status codes as success
                if response.status_code < 200 or response.status_code >= 300:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API Error: {error_detail}"
                    )
                return response.json()
            except httpx.RequestError as e:
                print(f"\n=== Network Error ===")
                print(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Network error: {str(e)}"
                )
    except Exception as e:
        print(f"\n=== General Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting target: {str(e)}"
        )

@app.post("/scraping-target", tags=["Scraping Target API"])
async def create_target(target: ScrapingTargetCreate, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        # Convert the target data to dict and ensure datetime is properly formatted
        target_data = target.dict()
        target_data['schedule_time'] = target_data['schedule_time'].isoformat()
        
        print("\n=== Create Target Requestsss ===")
        print(f"URL: {BOINGO_API_URL}/scraping-target")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(target_data, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{BOINGO_API_URL}/scraping-target",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=target_data
                )
                
                print("\n=== Create Target Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                
                # Accept both 200 and 201 as success codes
                if response.status_code not in [200, 201]:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API Error: {error_detail}"
                    )
                return response.json()
            except httpx.RequestError as e:
                print(f"\n=== Network Error ===")
                print(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Network error: {str(e)}"
                )
    except Exception as e:
        print(f"\n=== General Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating target: {str(e)}"
        )

@app.put("/scraping-target", tags=["Scraping Target API"])
async def update_target(target: ScrapingTargetUpdate, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        # Convert the target data to dict and ensure datetime is properly formatted
        target_data = target.dict()
        if isinstance(target_data.get('schedule_time'), datetime):
            target_data['schedule_time'] = target_data['schedule_time'].isoformat()
        
        print("\n=== Update Target Request ===")
        print(f"URL: {BOINGO_API_URL}/scraping-target")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(target_data, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{BOINGO_API_URL}/scraping-target",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=target_data
                )
                
                print("\n=== Update Target Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                
                # Accept all 2xx status codes as success
                if response.status_code < 200 or response.status_code >= 300:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API Error: {error_detail}"
                    )
                return response.json()
            except httpx.RequestError as e:
                print(f"\n=== Network Error ===")
                print(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Network error: {str(e)}"
                )
    except Exception as e:
        print(f"\n=== General Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating target: {str(e)}"
        )

@app.delete("/scraping-target", tags=["Scraping Target API"])
async def delete_target(target: ScrapingTargetDelete, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BOINGO_API_URL}/scraping-target",
                headers={"Authorization": f"Bearer {credentials.credentials}"},
                json=target.dict()
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting target: {str(e)}")

@app.post("/scraping-target/pause", tags=["Scraping Target API"])
async def pause_target(target: ScrapingTargetPause, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BOINGO_API_URL}/scraping-target/pause",
                headers={"Authorization": f"Bearer {credentials.credentials}"},
                json=target.dict()
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pausing target: {str(e)}")

@app.post("/scraping-target/unpause", tags=["Scraping Target API"])
async def unpause_target(target: ScrapingTargetPause, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BOINGO_API_URL}/scraping-target/unpause",
                headers={"Authorization": f"Bearer {credentials.credentials}"},
                json=target.dict()
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error unpausing target: {str(e)}")

@app.get("/scraping-results", tags=["Scraping Result API"])
async def get_all_results(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-results",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@app.get("/scraping-results/{result_id}", tags=["Scraping Result API"])
async def get_result_by_id(result_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        print(f"\n=== Get Result Request ===")
        print(f"Result ID: {result_id}")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{BOINGO_API_URL}/scraping-results/{result_id}",
                    headers={"Authorization": f"Bearer {credentials.credentials}"}
                )
                
                print(f"\n=== Get Result Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                
                # Accept all 2xx status codes as success
                if response.status_code < 200 or response.status_code >= 300:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API Error: {error_detail}"
                    )
                return response.json()
            except httpx.RequestError as e:
                print(f"\n=== Network Error ===")
                print(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Network error: {str(e)}"
                )
    except Exception as e:
        print(f"\n=== General Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting result: {str(e)}"
        )

@app.post("/scraping-results", tags=["Scraping Result API"])
async def create_result(result: ScrapingResultCreate, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        # Convert the result data to dict and ensure datetime is properly formatted
        result_data = result.dict()
        result_data['scraped_at'] = result_data['scraped_at'].isoformat()
        for agent in result_data['agent_status']:
            agent['start_time'] = agent['start_time'].isoformat()
            if agent.get('end_time'):
                agent['end_time'] = agent['end_time'].isoformat()
        
        print("\n=== Create Result Request ===")
        print(f"URL: {BOINGO_API_URL}/scraping-results")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(result_data, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{BOINGO_API_URL}/scraping-results",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=result_data
                )
                
                print("\n=== Create Result Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                
                # Accept both 200 and 201 as success codes
                if response.status_code not in [200, 201]:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API Error: {error_detail}"
                    )
                return response.json()
            except httpx.RequestError as e:
                print(f"\n=== Network Error ===")
                print(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Network error: {str(e)}"
                )
    except Exception as e:
        print(f"\n=== General Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating result: {str(e)}"
        )

@app.put("/scraping-results", tags=["Scraping Result API"])
async def update_result(result: ScrapingResultUpdate, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        # Convert the result data to dict and ensure datetime is properly formatted
        result_data = result.dict()
        result_data['scraped_at'] = result_data['scraped_at'].isoformat()
        result_data['last_updated'] = result_data['last_updated'].isoformat()
        
        print("\n=== Update Result Request ===")
        print(f"URL: {BOINGO_API_URL}/scraping-results")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(result_data, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{BOINGO_API_URL}/scraping-results",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=result_data
                )
                
                print("\n=== Update Result Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response body: {response.text}")
                
                # Accept all 2xx status codes as success
                if response.status_code < 200 or response.status_code >= 300:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"API Error: {error_detail}"
                    )
                return response.json()
            except httpx.RequestError as e:
                print(f"\n=== Network Error ===")
                print(f"Error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Network error: {str(e)}"
                )
    except Exception as e:
        print(f"\n=== General Error ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating result: {str(e)}"
        )

@app.delete("/scraping-results", tags=["Scraping Result API"])
async def delete_result(result: ScrapingResultDelete, credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BOINGO_API_URL}/scraping-results",
                headers={"Authorization": f"Bearer {credentials.credentials}"},
                json=result.dict()
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting result: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 