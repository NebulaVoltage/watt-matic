import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import detection, forecasting, models, health
from backend.services.sgcc_service import get_sgcc_predictor
from backend.services.forecasting_service import get_uci_forecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gridbalance.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load both frozen models ONCE at application startup.
    """
    logger.info("Initializing GridBalance ML Inference Backend...")
    try:
        sgcc = get_sgcc_predictor()
        logger.info(f"SGCC Classifier initialized. Loaded: {sgcc.is_loaded()}")
    except Exception as e:
        logger.error(f"Failed to load SGCC model: {e}")

    try:
        uci = get_uci_forecaster()
        logger.info(f"UCI Forecaster initialized. Loaded: {uci.is_loaded()}")
    except Exception as e:
        logger.error(f"Failed to load UCI forecasting model: {e}")

    yield

    logger.info("Shutting down GridBalance ML Inference Backend...")

app = FastAPI(
    title="GridBalance ML Inference API",
    description="Real-time Inference Backend for Meter Tampering Detection (SGCC) and Electricity Load Forecasting (UCI).",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(detection.router, prefix="/api/v1")
app.include_router(forecasting.router, prefix="/api/v1")

@app.get("/", summary="Root Status Endpoint")
def read_root():
    return {
        "title": "GridBalance ML Inference API",
        "status": "online",
        "docs_url": "/docs",
        "api_v1_prefix": "/api/v1"
    }
