"""quantlib-mm REST API.

Exposes every module of the quantlib_mm Python library over HTTP so the
Next.js showcase frontend can drive live computations from user input.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    correlation,
    montecarlo,
    options,
    portfolio,
    risk,
    sample_data,
    timeseries,
    yield_curve,
)

app = FastAPI(
    title="quantlib-mm API",
    description="REST surface for the quantlib-mm quantitative finance library.",
    version="0.1.0",
)


_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_origins_env = os.environ.get("ALLOWED_ORIGINS", _default_origins)
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


app.include_router(montecarlo.router, prefix="/api", tags=["montecarlo"])
app.include_router(options.router, prefix="/api", tags=["options"])
app.include_router(portfolio.router, prefix="/api", tags=["portfolio"])
app.include_router(risk.router, prefix="/api", tags=["risk"])
app.include_router(timeseries.router, prefix="/api", tags=["timeseries"])
app.include_router(correlation.router, prefix="/api", tags=["correlation"])
app.include_router(yield_curve.router, prefix="/api", tags=["yield-curve"])
app.include_router(sample_data.router, prefix="/api", tags=["sample-data"])
