"""Monte Carlo GBM simulation endpoint."""

import numpy as np
from fastapi import APIRouter

from quantlib_mm import GeometricBrownianMotion

from ..schemas import MonteCarloRequest, MonteCarloResponse

router = APIRouter()


@router.post("/montecarlo/simulate", response_model=MonteCarloResponse)
def simulate(req: MonteCarloRequest) -> MonteCarloResponse:
    gbm = GeometricBrownianMotion(
        s0=req.s0,
        mu=req.mu,
        sigma=req.sigma,
        T=req.T,
        n_steps=req.n_steps,
        n_paths=req.n_paths,
        antithetic=req.antithetic,
        seed=req.seed,
    )
    paths = gbm.simulate()
    stats = gbm.compute_statistics(paths)

    # Down-sample paths for the chart: up to 50 sample paths
    n_show = min(50, paths.shape[0])
    rng = np.random.default_rng(req.seed if req.seed is not None else 0)
    idx = rng.choice(paths.shape[0], size=n_show, replace=False)
    sample_paths = paths[idx].tolist()

    times = np.linspace(0.0, req.T, req.n_steps + 1).tolist()

    p5 = np.percentile(paths, 5, axis=0).tolist()
    p95 = np.percentile(paths, 95, axis=0).tolist()

    return MonteCarloResponse(
        times=times,
        sample_paths=sample_paths,
        mean_path=stats.mean_path.tolist(),
        p5_path=p5,
        p95_path=p95,
        terminal_mean=stats.mean,
        terminal_std=stats.std,
        terminal_min=stats.min,
        terminal_max=stats.max,
        percentiles=stats.percentiles,
        max_drawdown_mean=stats.max_drawdown_mean,
    )
