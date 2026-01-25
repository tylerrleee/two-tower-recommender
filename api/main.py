from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import logging
from datetime import datetime
from contextlib import asynccontextmanager
import base64
from io import StringIO

from api.models import (
    BatchMatchingRequest, MatchingResponse, HealthCheckResponse,
    ErrorResponse, CSVUploadRequest
)
from api.inference import MatchingInference
from api.exceptions import (
    ModelNotLoadedException, InsufficientDataException,
    MatchingFailedException, InvalidCSVException
)

# LOGGING CONFIG
logging.basicConfig(
    level       = logging.INFO,
    format      = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers    =[
            logging.FileHandler('api.log'),
            logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GLOBAL STATE
inference_engine: MatchingInference = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Manage App lifespan (startup <-> shutdown)"""
    logger.info("Starting app...")
    global inference_engine

    try:
        inference_engine = MatchingInference(
            model_path="models/best_model.pt",
            sbert_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384
        )
        inference_engine.load_model()
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        inference_engine = None
    
    yield

    logger.info("Shutting down app...")

# FASTAPI APP
app = FastAPI(
    title = "Mentor-Mentee Matching API",
    description = "TESTING",
    version = "1.0.0",
    lifespan = lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins       = ["*"], # Restrict in prod
    allow_credentials   = True,
    allow_methods       =["*"],
    allow_headers       =["*"]
)

# Exception Handling

@app.exception_handler(Exception)
async def global_exception_handler( request, exc):
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info = True)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content = {
            "error" : "Internal server error",
            "detail" : str(exc),
            "timestamp" : datetime.utcnow().isoformat()
        }
    )

# Engpoints

@app.get("/", tags = ["Root"])
async def root():
    """ Root Endpoint"""
    return {
        "message" : "Mentor-Mentee Matching API",
        "version" : "1.0.0",
        "docs"    : "/docs"
    }
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post(
    "/match/batch",
    response_model=MatchingResponse,
    status_code=status.HTTP_200_OK,
    tags=["Matching"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)

async def batch_matching(request: BatchMatchingRequest):
    """
    Generate optimal mentor-mentee matches from batch applicant data
    
    - **applicants**: List of applicant data (mentors + mentees)
    - **use_faiss**: Use FAISS for faster approximate matching
    - **top_k**: Number of candidates for FAISS (5-50)
    """
    if inference_engine is None:
        raise ModelNotLoadedException()
    
    try:
        logger.info(f"Received batch matching request: {len(request.applicants)} applicants")
        
        # Convert to DataFrame
        df = pd.DataFrame([app.model_dump() for app in request.applicants])
        
        # Run matching
        results = inference_engine.match(
            df          = df,
            use_faiss   = request.use_faiss,
            top_k       = request.top_k
        )
        
        logger.info(f" Matching completed: {results['total_groups']} groups")
        return MatchingResponse(**results)
        
    except ValueError as e:
        logger.error(f" Validation error: {str(e)}")
        raise InsufficientDataException(str(e))
    except Exception as e:
        logger.error(f" Matching failed: {str(e)}", exc_info=True)
        raise MatchingFailedException(str(e))


async def csv_matching(file: UploadFile = File(...)):
    """
    Generate matches from uploaded CSV file
    
    Expected CSV columns: role, name, ufl_email, major, year, bio, interests, goals, etc.
    """
    if inference_engine is None:
        raise ModelNotLoadedException()
    
    try:
        logger.info(f"Received CSV upload: {file.filename}")
        
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = {'role', 'name', 'ufl_email', 'major', 'year'}
        missing = required_cols - set(df.columns)
        if missing:
            raise InvalidCSVException(f"Missing required columns: {missing}")
        
        # Run matching
        results = inference_engine.match(df=df, use_faiss=False)
        
        logger.info(f"✓ CSV matching completed: {results['total_groups']} groups")
        return MatchingResponse(**results)
        
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        raise InvalidCSVException(f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"CSV matching failed: {str(e)}", exc_info=True)
        raise MatchingFailedException(str(e))

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get loaded model information"""
    
    if inference_engine is None:
        raise ModelNotLoadedException()
    
    return {
        "model_loaded": True,
        "metadata": inference_engine.model_metadata,
        "device": inference_engine.device,
        "embedding_dim": inference_engine.embedding_dim
    }
