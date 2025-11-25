"""
Activity Logger Middleware
Tracks all significant actions for audit trail
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from models import ActivityLog, async_session_maker
import json

class ActivityLoggerMiddleware(BaseHTTPMiddleware):
    """Middleware to log all significant API actions"""
    
    # Actions to skip logging
    SKIP_PATHS = {'/health', '/docs', '/openapi.json', '/redoc', '/sqladmin'}
    
    async def dispatch(self, request: Request, call_next):
        # Skip logging for static files and health checks
        if any(request.url.path.startswith(path) for path in self.SKIP_PATHS):
            return await call_next(request)
        
        # Execute request
        response = await call_next(request)
        
        # Log significant actions (POST, PUT, DELETE with successful responses)
        if request.method in ['POST', 'PUT', 'DELETE'] and response.status_code < 400:
            try:
                await self._log_activity(request, response)
            except Exception as e:
                # Don't let logging errors break the request
                print(f"Activity logging error: {e}")
        
        return response
    
    async def _log_activity(self, request: Request, response):
        """Log the activity to database"""
        async with async_session_maker() as session:
            # Extract user ID from request state (set by auth middleware)
            user_id = getattr(request.state, 'user_id', None)
            
            # Create log entry
            log = ActivityLog(
                user_id=user_id,
                action=f"{request.method} {request.url.path}",
                resource_type=self._extract_resource(request.url.path),
                resource_id=self._extract_id(request.url.path),
                details={
                    'status_code': response.status_code,
                    'method': request.method,
                    'path': request.url.path
                },
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('user-agent')
            )
            
            session.add(log)
            await session.commit()
    
    def _extract_resource(self, path: str) -> str:
        """Extract resource type from path"""
        if '/jobs' in path:
            return 'job'
        elif '/users' in path or '/admin/users' in path:
            return 'user'
        elif '/pricing' in path:
            return 'pricing'
        elif '/tools' in path:
            return 'tool'
        return 'other'
    
    def _extract_id(self, path: str) -> str:
        """Extract resource ID from path (UUID or job ID)"""
        parts = path.split('/')
        for part in parts:
            # Check for UUID pattern or job ID pattern
            if len(part) >= 8 and ('-' in part or len(part) == 36):
                return part
        return None
