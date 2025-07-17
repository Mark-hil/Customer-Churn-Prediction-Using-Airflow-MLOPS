from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from .config import API_PREFIX, API_TITLE, API_DESCRIPTION, API_VERSION, DEBUG
from .api.endpoints import predict, management
import logging
import time
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix=API_PREFIX)
app.include_router(management.router)  # Management endpoints don't use API_PREFIX

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Churn Prediction API",
        "version": API_VERSION,
        "docs": "/api/v1/docs"
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests and responses."""
    # Log request
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000
    process_time = round(process_time, 2)
    
    # Log response
    logger.info(
        f"Response: {request.method} {request.url} - "
        f"Status: {response.status_code} - "
        f"Process Time: {process_time}ms"
    )
    
    # Add headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Add GZip compression for responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug"
    )
