from fastapi import APIRouter, HTTPException, Security, Query
from fastapi.security import HTTPAuthorizationCredentials
import httpx
import json
from datetime import datetime
from typing import Optional
from ..models.models import AgentStatusCreate, AgentStatusUpdate, AgentStatusDelete
from ..core.config import BOINGO_API_URL
from .auth import security

router = APIRouter(
    prefix="/agent-status",
    tags=["Agent Status API"],
    responses={404: {"description": "Not found"}},
)

@router.get("")
async def get_all_agent_statuses(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get all agent statuses
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BOINGO_API_URL}/agent-status",
                headers={"Authorization": f"Bearer {credentials.credentials}"}
            )
            # Accept all 2xx status codes as success
            if response.status_code < 200 or response.status_code >= 300:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent statuses: {str(e)}")

@router.get("/queued")
async def get_queued_agent_statuses(
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    Get queued agent statuses with optional filtering by agent name
    """
    try:
        print(f"\n=== Get Queued Agent Statuses Request ===")
        print(f"Agent Name Filter: {agent_name}")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        
        url = f"{BOINGO_API_URL}/agent-status/queued"
        if agent_name:
            url += f"?agent_name={agent_name}"
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {credentials.credentials}"}
                )
                
                print(f"\n=== Get Queued Agent Statuses Response ===")
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
            detail=f"Error getting queued agent statuses: {str(e)}"
        )

@router.get("/{agent_id}")
async def get_agent_status_by_id(agent_id: str, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get agent status by ID
    """
    try:
        print(f"\n=== Get Agent Status Request ===")
        print(f"Agent ID: {agent_id}")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{BOINGO_API_URL}/agent-status/{agent_id}",
                    headers={"Authorization": f"Bearer {credentials.credentials}"}
                )
                
                print(f"\n=== Get Agent Status Response ===")
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
            detail=f"Error getting agent status: {str(e)}"
        )

@router.post("")
async def create_agent_status(agent_status: AgentStatusCreate, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Create a new agent status
    """
    try:
        # Convert the agent status data to dict and ensure datetime is properly formatted
        agent_data = agent_status.dict()
        agent_data['start_time'] = agent_data['start_time'].isoformat()
        if agent_data.get('end_time'):
            agent_data['end_time'] = agent_data['end_time'].isoformat()
        
        print("\n=== Create Agent Status Request ===")
        print(f"URL: {BOINGO_API_URL}/agent-status")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(agent_data, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{BOINGO_API_URL}/agent-status",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=agent_data
                )
                
                print("\n=== Create Agent Status Response ===")
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
            detail=f"Error creating agent status: {str(e)}"
        )

@router.put("")
async def update_agent_status(agent_status: AgentStatusUpdate, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Update an existing agent status
    """
    try:
        # Convert the agent status data to dict and ensure datetime is properly formatted
        agent_data = agent_status.dict()
        agent_data['start_time'] = agent_data['start_time'].isoformat()
        if agent_data.get('end_time'):
            agent_data['end_time'] = agent_data['end_time'].isoformat()
        
        print("\n=== Update Agent Status Request ===")
        print(f"URL: {BOINGO_API_URL}/agent-status")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(agent_data, indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(
                    f"{BOINGO_API_URL}/agent-status",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=agent_data
                )
                
                print("\n=== Update Agent Status Response ===")
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
            detail=f"Error updating agent status: {str(e)}"
        )

@router.delete("")
async def delete_agent_status(agent_status: AgentStatusDelete, credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Delete an agent status
    """
    try:
        print("\n=== Delete Agent Status Request ===")
        print(f"URL: {BOINGO_API_URL}/agent-status")
        print(f"Authorization: Bearer {credentials.credentials[:20]}...")
        print(f"Request body: {json.dumps(agent_status.dict(), indent=2)}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(
                    f"{BOINGO_API_URL}/agent-status",
                    headers={
                        "Authorization": f"Bearer {credentials.credentials}",
                        "Content-Type": "application/json"
                    },
                    json=agent_status.dict()
                )
                
                print("\n=== Delete Agent Status Response ===")
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
            detail=f"Error deleting agent status: {str(e)}"
        ) 