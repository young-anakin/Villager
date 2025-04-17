from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

class LoginRequest(BaseModel):
    email: str
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "password123"
            }
        }

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
    role: str

class LoginResponse(BaseModel):
    status: int
    message: str
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "status": 200,
                "message": "Login successful",
                "data": {
                    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "user": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "email": "user@example.com",
                        "first_name": "John",
                        "last_name": "Doe"
                    }
                }
            }
        }

class ScrapingTargetCreate(BaseModel):
    website_url: str = Field(..., description="URL of the website to scrape, must start with http:// or https://")
    location: str = Field(..., description="Location to target scraping")
    schedule_time: Optional[str] = Field(None, description="Time to schedule the scraping")
    frequency: str = Field(..., description="Frequency of scraping, must be one of: 'Daily', 'Weekly', 'Monthly' (case-insensitive)")
    search_range: Optional[int] = Field(None, description="Range to search in miles")
    max_properties: Optional[int] = Field(None, description="Maximum number of properties to scrape")
    
    @validator('website_url')
    def validate_website_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('website_url must start with http:// or https://')
        return v
    
    @validator('frequency')
    def validate_frequency(cls, v):
        valid_frequencies = ["Daily", "Weekly", "Monthly"]
        if v.capitalize() not in valid_frequencies:
            raise ValueError(f"frequency must be one of {valid_frequencies}")
        return v.capitalize()
    
    class Config:
        schema_extra = {
            "example": {
                "website_url": "https://www.example.com",
                "location": "Los Angeles, CA",
                "schedule_time": "09:00",
                "frequency": "Daily",
                "search_range": 10,
                "max_properties": 100
            }
        }

class ScrapingTargetUpdate(BaseModel):
    id: str = Field(..., description="ID of the scraping target to update")
    website_url: Optional[str] = Field(None, description="URL of the website to scrape, must start with http:// or https://")
    location: Optional[str] = Field(None, description="Location to target scraping")
    schedule_time: Optional[str] = Field(None, description="Time to schedule the scraping")
    frequency: Optional[str] = Field(None, description="Frequency of scraping, must be one of: 'Daily', 'Weekly', 'Monthly' (case-insensitive)")
    search_range: Optional[int] = Field(None, description="Range to search in miles")
    max_properties: Optional[int] = Field(None, description="Maximum number of properties to scrape")
    
    @validator('website_url')
    def validate_website_url(cls, v):
        if v is not None and not v.startswith(('http://', 'https://')):
            raise ValueError('website_url must start with http:// or https://')
        return v
    
    @validator('frequency')
    def validate_frequency(cls, v):
        if v is None:
            return v
        valid_frequencies = ["Daily", "Weekly", "Monthly"]
        if v.capitalize() not in valid_frequencies:
            raise ValueError(f"frequency must be one of {valid_frequencies}")
        return v.capitalize()
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "website_url": "https://www.example.com",
                "location": "Los Angeles, CA",
                "frequency": "Weekly"
            }
        }

class ScrapingTargetDelete(BaseModel):
    id: str = Field(..., description="ID of the scraping target to delete")
    force: bool = Field(False, description="Force delete the target")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "force": True
            }
        }

class ScrapingTargetPause(BaseModel):
    id: str = Field(..., description="ID of the scraping target to pause")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }

class AgentStatus(BaseModel):
    agent_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "agent_name": "crawler-agent",
                "status": "Success",
                "start_time": "2023-05-23T14:30:00",
                "end_time": "2023-05-23T14:45:00"
            }
        }

class ScrapingResultCreate(BaseModel):
    source_url: str
    listing_type: str
    data: Dict[str, Any]
    progress: int
    status: str
    scraped_at: datetime
    target_id: str
    agent_status: List[AgentStatus]
    
    class Config:
        schema_extra = {
            "example": {
                "source_url": "https://www.zillow.com/homes/for_sale/Chicago-IL/",
                "listing_type": "residential",
                "data": {
                    "property_type": "single_family",
                    "address": "123 Main Street, Chicago, IL 60601",
                    "price": 425000
                },
                "progress": 100,
                "status": "Success",
                "scraped_at": "2023-06-15T09:45:23.415Z",
                "target_id": "a7b9c3d4-e5f6-47a7-8c9d-0e1f2a3b4c5d",
                "agent_status": [
                    {
                        "agent_name": "crawler-agent",
                        "status": "Success",
                        "start_time": "2023-06-15T09:42:18.332Z",
                        "end_time": "2023-06-15T09:45:23.415Z"
                    }
                ]
            }
        }

class ScrapingResultUpdate(BaseModel):
    id: str
    source_url: Optional[str] = None
    listing_type: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    progress: Optional[int] = None
    status: Optional[str] = None
    scraped_at: datetime
    last_updated: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "source_url": "https://www.zillow.com/homes/for_sale/Chicago-IL/",
                "status": "Success",
                "progress": 100,
                "scraped_at": "2023-06-15T09:45:23.415Z",
                "last_updated": "2023-06-15T10:15:42.123Z"
            }
        }

class ScrapingResultDelete(BaseModel):
    id: str
    force: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "force": True
            }
        }

class AgentStatusCreate(BaseModel):
    agent_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    scraping_result_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "agent_name": "Scraping Agent",
                "status": "Queued",
                "start_time": "2023-05-23T14:30:00",
                "end_time": None,
                "scraping_result_id": None
            }
        }

class AgentStatusUpdate(BaseModel):
    id: str
    agent_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    scraping_result_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "d2441e12-92e6-4116-af01-3e83af7961e6",
                "agent_name": "Extracting Agent",
                "status": "Success",
                "start_time": "2023-05-23T14:30:00",
                "end_time": "2023-05-23T14:45:00",
                "scraping_result_id": "605f0ed8-7f7f-4254-b0f9-08fbdf72a872"
            }
        }

class AgentStatusDelete(BaseModel):
    id: str
    force: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "id": "e33fe5be-dd83-4216-befc-b9d029c7daa3",
                "force": True
            }
        }

# If you want to add Scriping Analytics models, you can add them here
# class ScripingAnalyticsCreate(BaseModel):
#     ... 