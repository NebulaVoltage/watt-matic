from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import health, model_info, detection, forecasting, dashboard

app = FastAPI(
    title="GridBalance Smart Grid Intelligence API",
    description="REST API for Meter Tampering Classification (SGCC) and Next-Hour Load Forecasting (UCI).",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(detection.router)
app.include_router(forecasting.router)
app.include_router(dashboard.router)

@app.get("/")
def root():
    return {
        "platform": "GridBalance Smart Grid Intelligence Platform",
        "status": "online",
        "docs_url": "/docs"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
