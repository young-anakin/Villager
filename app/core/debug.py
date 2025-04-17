import functools
import traceback
import time
import json
from fastapi import Request, Response
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_debug.log')
    ]
)
logger = logging.getLogger("api_debugger")

def debug_request(func):
    """
    Decorator to add detailed debugging for API endpoint functions
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request from kwargs if it exists
        request = next((arg for arg in args if isinstance(arg, Request)), 
                      kwargs.get('request', None))
        
        # Generate a unique request ID
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        # Log request info
        start_time = time.time()
        logger.info(f"[{request_id}] Request started: {func.__name__}")
        
        # Log request details if available
        if request:
            client_host = request.client.host if request.client else "unknown"
            method = request.method
            url = str(request.url)
            headers = dict(request.headers)
            # Mask Authorization header
            if 'authorization' in headers:
                auth = headers['authorization']
                if auth.startswith('Bearer ') and len(auth) > 15:
                    headers['authorization'] = f"Bearer {auth[7:15]}...{auth[-4:]}"
                else:
                    headers['authorization'] = "Bearer [MASKED]"
                    
            logger.info(f"[{request_id}] {method} {url} from {client_host}")
            logger.debug(f"[{request_id}] Headers: {json.dumps(headers)}")
            
            # Try to log request body if present
            try:
                body = await request.json()
                # Mask password and sensitive fields if present
                if isinstance(body, dict):
                    masked_body = body.copy()
                    for k in masked_body:
                        if k.lower() in ('password', 'token', 'secret', 'key', 'credential'):
                            masked_body[k] = '[MASKED]'
                    logger.debug(f"[{request_id}] Request body: {json.dumps(masked_body)}")
            except:
                # Body might not be JSON or already consumed
                pass
        
        # Execute the function and capture result or exception
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Log successful response
            if isinstance(result, Response):
                status_code = result.status_code
                logger.info(f"[{request_id}] Response: {status_code} in {elapsed:.2f}s")
            elif isinstance(result, dict):
                result_copy = result.copy()
                # Mask sensitive data in response
                for k, v in result_copy.items():
                    if k.lower() in ('token', 'password', 'secret', 'key', 'credential'):
                        result_copy[k] = '[MASKED]'
                logger.info(f"[{request_id}] Response: 200 in {elapsed:.2f}s")
                logger.debug(f"[{request_id}] Response body: {json.dumps(result_copy)[:1000]}...")
            else:
                logger.info(f"[{request_id}] Response: 200 in {elapsed:.2f}s")
                
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            # Log detailed exception info
            logger.error(f"[{request_id}] Exception in {elapsed:.2f}s: {type(e).__name__}: {str(e)}")
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
            raise  # Re-raise the exception
            
    return wrapper 