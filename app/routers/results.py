# app/routers/results.py
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials
import httpx
import json
import logging
from datetime import datetime
from ..models.models import (
    ScrapingResultCreate, ScrapingResultUpdate,
    ScrapingResultDelete
)
from ..core.config import BOINGO_API_URL
from .auth import security

# Configure logging (consistent with targets.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/scraping-results",
    tags=["Scraping Result API"],
    responses={404: {"description": "Not found"}},
)

@router.get("")
async def get_all_results(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get all scraping results
    """
    try:
        logger.info("Fetching all scraping results")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-results",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response body: {response.text}")
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
        logger.error(f"Network error while fetching all results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@router.get("/{result_id}")
async def get_result_by_id(result_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get scraping result by ID
    """
    try:
        logger.info(f"Fetching scraping result by ID: {result_id}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-results/{result_id}",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response body: {response.text}")
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
        logger.error(f"Network error while fetching result {result_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting result {result_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting result: {str(e)}")

@router.post("")
async def create_result(result: ScrapingResultCreate, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Create a new scraping result (manual creation, not via Celery)
    """
    try:
        logger.info(f"Creating new scraping result for target ID: {result.target_id}")
        # Convert the result data to dict and ensure datetime is properly formatted
        result_data = result.dict()
        result_data['scraped_at'] = result_data['scraped_at'].isoformat()
        for agent in result_data['agent_status']:
            agent['start_time'] = agent['start_time'].isoformat()
            if agent.get('end_time'):
                agent['end_time'] = agent['end_time'].isoformat()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BOINGO_API_URL}/scraping-results",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=result_data
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response body: {response.text}")
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
        logger.error(f"Network error while creating result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error creating result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating result: {str(e)}")

@router.put("")
async def update_result(result: ScrapingResultUpdate, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Update an existing scraping result
    """
    try:
        logger.info(f"Updating scraping result: {result.id}")
        # Convert the result data to dict and ensure datetime is properly formatted
        result_data = result.dict()
        result_data['scraped_at'] = result_data['scraped_at'].isoformat()
        result_data['last_updated'] = result_data['last_updated'].isoformat()

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{BOINGO_API_URL}/scraping-results",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=result_data
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response body: {response.text}")
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
        logger.error(f"Network error while updating result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating result: {str(e)}")

@router.delete("")
async def delete_result(result: ScrapingResultDelete, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Delete a scraping result
    """
    try:
        logger.info(f"Deleting scraping result: {result.id}")
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BOINGO_API_URL}/scraping-results",
                headers={
                    "Authorization": f"Bearer {credentials.credentials}",
                    "Content-Type": "application/json"
                },
                json=result.dict()
            )
            logger.info(f"Boingo API response status: {response.status_code}")
            logger.debug(f"Boingo API response body: {response.text}")
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
        logger.error(f"Network error while deleting result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error deleting result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting result: {str(e)}")