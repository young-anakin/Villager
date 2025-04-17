import uuid
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import logging
from datetime import datetime
from Crawler.queue_manager import add_to_queue  # Adjust path based on your structure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Imports from your project
from ..models.models import (
    ScrapingTargetCreate, ScrapingTargetUpdate,
    ScrapingTargetDelete, ScrapingTargetPause
)
from ..core.config import BOINGO_API_URL
from .auth import security

router = APIRouter(
    prefix="/scraping-target",
    tags=["Scraping Target API"],
    responses={404: {"description": "Not found"}},
)

# Simplified response model for API responses
class ScrapingResultResponse(BaseModel):
    id: str
    target_id: str
    status: str
    website_url: str
    num_listings: Optional[int] = None
    error: Optional[str] = None

@router.get("")
async def get_all_targets(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get all scraping targets."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-target",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        logger.error(f"Error getting targets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting targets: {str(e)}")

@router.get("/{target_id}")
async def get_target_by_id(target_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get scraping target by ID."""
    try:
        logger.info(f"Fetching target by ID: {target_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-target/{target_id}",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response body: {response.text}")
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
        logger.error(f"Network error while fetching target {target_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching target {target_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting target: {str(e)}")

@router.post("", response_model=ScrapingResultResponse)
async def create_target(target: ScrapingTargetCreate, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Create a new scraping target and queue it in queue.json."""
    try:
        logger.info(f"Creating new scraping target for URL: {target.website_url}")
        logger.debug(f"Request body received: {json.dumps(target.dict(), indent=2)}")
        logger.debug(f"Credentials token: {credentials.credentials[:10]}... (truncated for security)")

        # Notify Boingo API
        logger.info("Notifying Boingo API")
        logger.debug(f"Boingo API URL: {BOINGO_API_URL}/scraping-target")
        async with httpx.AsyncClient() as client:
            logger.debug("Preparing to send POST request to Boingo API")
            boingo_response = await client.post(
                f"{BOINGO_API_URL}/scraping-target",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=target.dict()
            )
            logger.info(f"Boingo API response status: {boingo_response.status_code}")
            logger.debug(f"Boingo API response headers: {dict(boingo_response.headers)}")
            logger.debug(f"Boingo API response text (raw): {boingo_response.text!r}")
            logger.debug(f"Response text length: {len(boingo_response.text)} characters")
            logger.debug(f"Response text first 60 chars: {boingo_response.text[:60]!r}")
            logger.debug(f"Response text last 60 chars: {boingo_response.text[-60:]!r}")

            # Handle Boingo API response
            if boingo_response.status_code not in [200, 201]:
                error_msg = boingo_response.text
                logger.warning(f"Boingo API returned non-success status: {boingo_response.status_code}")
                logger.debug(f"Error response content: {error_msg!r}")
                if boingo_response.status_code == 400 and "Already Registered" in error_msg:
                    logger.warning(f"Target {target.website_url} already registered")
                    raise HTTPException(status_code=400, detail="Scraping target already registered")
                raise HTTPException(status_code=boingo_response.status_code, detail=f"Failed to notify Boingo API: {error_msg}")

            # Parse target_id from response
            response_text = boingo_response.text.strip()
            logger.debug(f"Stripped response text: {response_text!r}")
            logger.debug(f"Stripped response length: {len(response_text)} characters")
            try:
                logger.debug("Attempting to parse JSON from response")
                response_data = json.loads(response_text)
                logger.debug(f"Parsed JSON: {json.dumps(response_data, indent=2)}")
                target_id = response_data.get("data", {}).get("id")
                logger.debug(f"Extracted target_id: {target_id}")
                if not target_id:
                    logger.error("Boingo API response does not contain data.id")
                    logger.debug(f"Full parsed response: {response_data}")
                    raise HTTPException(status_code=500, detail="Boingo API response does not contain a valid target ID")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.debug(f"Failed response text: {response_text!r}")
                logger.debug(f"Error position: line {e.lineno}, column {e.colno}, char {e.pos}")
                logger.debug(f"Text around error (char {e.pos-10}:{e.pos+10}): {response_text[max(0, e.pos-10):e.pos+10]!r}")
                raise HTTPException(status_code=500, detail=f"Invalid JSON response from Boingo API: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected parsing error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to parse Boingo API response: {str(e)}")

        # Queue task
        logger.debug("Generating task ID")
        task_id = str(uuid.uuid4())
        logger.debug(f"Task ID generated: {task_id}")
        logger.debug(f"Adding to queue: {{'website_url': {target.website_url}, 'target_id': {target_id}}}")
        add_to_queue("scraping", {"website_url": target.website_url, "target_id": target_id})
        logger.info(f"Task queued in queue.json - Task ID: {task_id}, Target ID: {target_id}")

        logger.debug("Preparing response")
        response = ScrapingResultResponse(
            id=task_id,
            target_id=target_id,
            status="queued",
            website_url=target.website_url
        )
        logger.debug(f"Returning response: {response.dict()}")
        return response

    except Exception as e:
        logger.error(f"Error creating target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating target: {str(e)}")


@router.get("/status/{task_id}", response_model=ScrapingResultResponse)
async def get_task_status(task_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Check the status of a scraping task (simplified without Redis/Celery)."""
    try:
        logger.info(f"Checking status for task ID: {task_id}")
        # Since we're not using Celery/Redis here, this is a placeholder
        # You could extend queue_manager.py to track status if needed
        return ScrapingResultResponse(
            id=task_id,
            target_id="unknown",  # Would need to track this in queue.json or elsewhere
            status="queued",      # Simplified; enhance if status tracking is added
            website_url="unknown"
        )
    except Exception as e:
        logger.error(f"Error retrieving task status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving task status: {str(e)}")

@router.put("")
async def update_target(target: ScrapingTargetUpdate, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Update an existing scraping target."""
    try:
        logger.info(f"Updating scraping target: {json.dumps(target.dict(), indent=2)}")
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{BOINGO_API_URL}/scraping-target",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=target.dict()
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response: {response.text}")
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
        logger.error(f"Network error while updating target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating target: {str(e)}")

@router.delete("")
async def delete_target(target: ScrapingTargetDelete, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Delete a scraping target."""
    try:
        logger.info(f"Deleting scraping target: {json.dumps(target.dict(), indent=2)}")
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BOINGO_API_URL}/scraping-target",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=target.dict()
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response: {response.text}")
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
        logger.error(f"Network error while deleting target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error deleting target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting target: {str(e)}")

@router.post("/pause")
async def pause_target(target: ScrapingTargetPause, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Pause a scraping target."""
    try:
        logger.info(f"Pausing scraping target: {json.dumps(target.dict(), indent=2)}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BOINGO_API_URL}/scraping-target/pause",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=target.dict()
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response: {response.text}")
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
        logger.error(f"Network error while pausing target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error pausing target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error pausing target: {str(e)}")

@router.post("/unpause")
async def unpause_target(target: ScrapingTargetPause, credentials: HTTPAuthorizationCredentials = Security(security)):
    """Unpause a scraping target."""
    try:
        logger.info(f"Unpausing scraping target: {json.dumps(target.dict(), indent=2)}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BOINGO_API_URL}/scraping-target/unpause",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=target.dict()
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response: {response.text}")
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
        logger.error(f"Network error while unpausing target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error unpausing target: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error unpausing target: {str(e)}")