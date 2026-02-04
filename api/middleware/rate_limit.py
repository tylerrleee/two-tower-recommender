"""
Rate Limiting Middleware - Prevent API abuse

Limits: TO CHANGE
- 100 requests/hour per user
- 10 CSV uploads/day per user
- 5 matching jobs/day per user
- 5 concurrent matching jobs per organization
"""

# TODO use REDIS for dynamic limiting per container
# Store limit values in a central middleware config file

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit requests per user and organization"""
    
    def __init__(self, app):
        super().__init__(app)
        # In-memory storage 
        self.request_counts: Dict[str, list] = defaultdict(list)
        self.csv_uploads: Dict[str, list] = defaultdict(list)
        self.matching_jobs: Dict[str, list] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        # Extract user from JWT (set by auth middleware)
        user_email  = getattr(request.state, "user_email", None)
        org_id      = getattr(request.state, "organization_id", None)
        
        if user_email:
            # General API rate limit
            if not self._check_request_limit(user_email):
                logger.warning(f"Rate limit exceeded for user {user_email}")
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Max 100 requests per hour."
                )
            
            # CSV upload limit
            if "/applicants/upload" in request.url.path:
                if not self._check_csv_upload_limit(user_email):
                    raise HTTPException(
                        status_code=429,
                        detail="CSV upload limit exceeded. Max 10 uploads per day."
                    )
            
            # Matching job limit
            if "/matching/batch" in request.url.path:
                if not self._check_matching_job_limit(user_email):
                    raise HTTPException(
                        status_code=429,
                        detail="Matching job limit exceeded. Max 5 jobs per day."
                    )
        
        response = await call_next(request)
        return response

 
    def _check_request_limit(self, user_email: str) -> bool:
        """Check general request limit: 100/hour"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        # Clean old entries
        self.request_counts[user_email] = [
            ts for ts in self.request_counts[user_email] if ts > cutoff
        ]
        
        # Check limit
        if len(self.request_counts[user_email]) >= 100:
            return False
        
        # Record request
        self.request_counts[user_email].append(now)
        return True
    
    def _check_csv_upload_limit(self, user_email: str) -> bool:
        """Check CSV upload limit: 10/day"""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        self.csv_uploads[user_email] = [
            ts for ts in self.csv_uploads[user_email] if ts > cutoff
        ]
        
        if len(self.csv_uploads[user_email]) >= 10: # put 10 in a seperate api config file
            return False
        
        self.csv_uploads[user_email].append(now)
        return True
    
    def _check_matching_job_limit(self, user_email: str) -> bool:
        """Check matching job limit: 5/day"""
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        self.matching_jobs[user_email] = [
            ts for ts in self.matching_jobs[user_email] if ts > cutoff
        ]
        
        if len(self.matching_jobs[user_email]) >= 5:
            return False
        
        self.matching_jobs[user_email].append(now)
        return True