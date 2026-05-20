"""Option pricing endpoints: Black-Scholes, binomial tree, Monte Carlo."""

from fastapi import APIRouter

from quantlib_mm import BinomialTree, BlackScholes, Greeks, MonteCarloOptionPricer

from ..schemas import (
    BinomialNode,
    BinomialRequest,
    BinomialResponse,
    BlackScholesRequest,
    BlackScholesResponse,
    GreeksPayload,
    MCPricingRequest,
    MCPricingResponse,
)

router = APIRouter()


def _greeks(S, K, T, r, sigma, option_type: str) -> GreeksPayload:
    g = Greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
    return GreeksPayload(**g.summary())


@router.post("/options/black-scholes", response_model=BlackScholesResponse)
def black_scholes(req: BlackScholesRequest) -> BlackScholesResponse:
    bs = BlackScholes(S=req.S, K=req.K, T=req.T, r=req.r, sigma=req.sigma)
    return BlackScholesResponse(
        call_price=bs.call_price(),
        put_price=bs.put_price(),
        call_greeks=_greeks(req.S, req.K, req.T, req.r, req.sigma, "call"),
        put_greeks=_greeks(req.S, req.K, req.T, req.r, req.sigma, "put"),
        parity_holds=bs.put_call_parity(),
    )


@router.post("/options/binomial", response_model=BinomialResponse)
def binomial(req: BinomialRequest) -> BinomialResponse:
    tree = BinomialTree(
        S=req.S,
        K=req.K,
        T=req.T,
        r=req.r,
        sigma=req.sigma,
        n_steps=req.n_steps,
        option_type=req.option_type,
        style=req.style,
    )
    price = tree.price()
    layers = tree.get_tree()

    nodes: list[BinomialNode] = []
    for step, layer in enumerate(layers):
        for j, value in enumerate(layer):
            spot = req.S * (tree.u ** (step - 2 * j))
            nodes.append(
                BinomialNode(step=step, index=j, spot=float(spot), value=float(value))
            )

    return BinomialResponse(
        price=price,
        u=float(tree.u),
        d=float(tree.d),
        p=float(tree.p),
        nodes=nodes,
    )


@router.post("/options/monte-carlo", response_model=MCPricingResponse)
def monte_carlo(req: MCPricingRequest) -> MCPricingResponse:
    pricer = MonteCarloOptionPricer(
        S=req.S,
        K=req.K,
        T=req.T,
        r=req.r,
        sigma=req.sigma,
        n_paths=req.n_paths,
        n_steps=req.n_steps,
    )

    if req.option_type == "call":
        price = pricer.price_european_call()
    elif req.option_type == "put":
        price = pricer.price_european_put()
    elif req.option_type == "asian-arithmetic":
        price = pricer.price_asian_call(averaging="arithmetic")
    else:
        price = pricer.price_asian_call(averaging="geometric")

    lo, hi = pricer.confidence_interval(0.95)
    return MCPricingResponse(
        price=price,
        std_error=pricer.last_std_error or 0.0,
        ci_lower=lo,
        ci_upper=hi,
    )
