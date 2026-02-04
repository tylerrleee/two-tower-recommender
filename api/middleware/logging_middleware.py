"""
Structured JSON Logging Middleware - AWS CloudWatch compatible

Logs every request with:
- Correlation ID
- User context
- Request/response metadata
- Execution time
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import json
import time
import uuid

logger = logging.getLogger(__name__)

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Add structured logging to every request"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timer
        start_time = time.time()
        
        # Extract user context (if authenticated)
        user_email = getattr(request.state, "user_email", None)
        org_id = getattr(request.state, "organization_id", None)
        
        # Log request
        log_data = {
            "event"         : "request_start",
            "correlation_id": correlation_id,
            "method"        : request.method,
            "path"          : request.url.path,
            "user_email"    : user_email,
            "organization_id": org_id,
            "client_ip"     : request.client.host if request.client else None,
            "user_agent"    : request.headers.get("user-agent")
        }
        
        logger.info(json.dumps(log_data))
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log response
            log_data.update({
                "event": "request_complete",
                "status_code": response.status_code,
                "execution_time_ms": round(execution_time * 1000, 2)
            })
            
            logger.info(json.dumps(log_data))
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            # Log error
            execution_time = time.time() - start_time
            
            log_data.update({
                "event": "request_error",
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": round(execution_time * 1000, 2)
            })
            
            logger.error(json.dumps(log_data))
            
            raise
