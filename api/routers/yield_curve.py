"""Yield curve endpoint."""

import numpy as np
from fastapi import APIRouter

from quantlib_mm import YieldCurve

from ..schemas import YieldCurveRequest, YieldCurveResponse

router = APIRouter()


@router.post("/yield-curve/build", response_model=YieldCurveResponse)
def build(req: YieldCurveRequest) -> YieldCurveResponse:
    curve = YieldCurve.bootstrap(par_rates=req.par_rates, maturities=req.maturities)

    maturities = np.asarray(req.maturities, dtype=float)
    zero_rates = curve.rates.tolist()
    discount_factors = [curve.discount_factor(float(t)) for t in maturities]

    # Forward rates between consecutive maturities
    forward_rates: list[float] = []
    forward_starts: list[float] = []
    forward_ends: list[float] = []
    for t1, t2 in zip(maturities[:-1], maturities[1:]):
        forward_rates.append(float(curve.forward_rate(float(t1), float(t2))))
        forward_starts.append(float(t1))
        forward_ends.append(float(t2))

    return YieldCurveResponse(
        maturities=maturities.tolist(),
        zero_rates=zero_rates,
        discount_factors=discount_factors,
        forward_rates=forward_rates,
        forward_starts=forward_starts,
        forward_ends=forward_ends,
    )
