from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials
import httpx
from ..core.config import BOINGO_API_URL
from .auth import security

router = APIRouter(
    prefix="/scraping-analytics",
    tags=["Scraping Analytics API"],
    responses={404: {"description": "Not found"}},
)

@router.get("")
async def get_analytics(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get scraping analytics data
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-analytics",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

@router.get("/summary")
async def get_analytics_summary(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get scraping analytics summary
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/scraping-analytics/summary",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics summary: {str(e)}") 