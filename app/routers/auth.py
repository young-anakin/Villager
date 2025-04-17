from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import json
import os
from ..models.models import LoginRequest, LoginResponse
from ..core.config import BOINGO_API_URL, BOINGO_EMAIL, BOINGO_PASSWORD

router = APIRouter(
    prefix="/auth",
    tags=["Authentication API"],
    responses={401: {"description": "Unauthorized"}},
)

security = HTTPBearer(auto_error=True)

async def get_auth_token():
    """
    Get authentication token from Boingo API
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BOINGO_API_URL}/auth/login",
                json={
                    "email": BOINGO_EMAIL,
                    "password": BOINGO_PASSWORD
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=401,
                    detail=f"Authentication failed: {response.text}"
                )
                
            data = response.json()
            return data.get("data", {}).get("token")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting authentication token: {str(e)}"
        )

@router.post("/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
    """
    Login to Boingo API
    """
    try:
        print("\n=== Login Request ===")
        print(f"URL: {BOINGO_API_URL}/auth/login")
        print(f"Email: {login_data.email}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{BOINGO_API_URL}/auth/login",
                    json=login_data.dict()
                )
                
                print("\n=== Login Response ===")
                print(f"Status code: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                
                # Print a masked version of the token for security
                response_data = response.json()
                if "data" in response_data and "token" in response_data["data"]:
                    token = response_data["data"]["token"]
                    masked_token = token[:10] + "..." + token[-10:] if len(token) > 20 else "***"
                    print(f"Token: {masked_token}")
                else:
                    print(f"Response body: {response.text}")
                
                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = json.dumps(error_json, indent=2)
                    except:
                        pass
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Login failed: {error_detail}"
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
            detail=f"Error during login: {str(e)}"
        ) 