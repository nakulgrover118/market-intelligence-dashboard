from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)

# Defaults to the Vite dev server's origin; MARKET_CORS_ORIGINS overrides
# this for a real deployment (see docs/deployment.md) — the deployed
# frontend's origin must be listed here or every request will be blocked
# by the browser, not this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
