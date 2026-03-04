"""Tests for the Black-Scholes option pricing module."""

import math

import numpy as np
import pytest

from quantlib_mm.black_scholes import BlackScholes


# ---------------------------------------------------------------------------
# Fixtures / shared parameters
# ---------------------------------------------------------------------------

# Classic textbook example:
#   S=100, K=100, T=1, r=0.05, sigma=0.20
# Expected (rounded):
#   call ~ 10.4506, put ~ 5.5735
SPOT = 100.0
STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
VOL = 0.20

EXPECTED_CALL = 10.4506
EXPECTED_PUT = 5.5735


@pytest.fixture
def bs():
    """Return a BlackScholes instance with standard textbook parameters."""
    return BlackScholes(S=SPOT, K=STRIKE, T=EXPIRY, r=RATE, sigma=VOL)


# ---------------------------------------------------------------------------
# Test 1: Call price matches known analytical value
# ---------------------------------------------------------------------------

def test_call_price_known_value(bs):
    """The call price should match the known analytical result within 1 cent."""
    price = bs.call_price()
    assert price == pytest.approx(EXPECTED_CALL, abs=0.01), (
        f"Call price {price:.4f} deviates from expected {EXPECTED_CALL}"
    )


# ---------------------------------------------------------------------------
# Test 2: Put price matches known analytical value
# ---------------------------------------------------------------------------

def test_put_price_known_value(bs):
    """The put price should match the known analytical result within 1 cent."""
    price = bs.put_price()
    assert price == pytest.approx(EXPECTED_PUT, abs=0.01), (
        f"Put price {price:.4f} deviates from expected {EXPECTED_PUT}"
    )


# ---------------------------------------------------------------------------
# Test 3: Put-call parity holds
# ---------------------------------------------------------------------------

def test_put_call_parity(bs):
    """Put-call parity C - P == S - K*exp(-rT) must hold."""
    assert bs.put_call_parity(), (
        "Put-call parity violated: "
        f"C - P = {bs.call_price() - bs.put_price():.8f}, "
        f"S - K*exp(-rT) = {SPOT - STRIKE * math.exp(-RATE * EXPIRY):.8f}"
    )


# ---------------------------------------------------------------------------
# Test 4: Call and put prices are non-negative and satisfy basic bounds
# ---------------------------------------------------------------------------

def test_price_bounds(bs):
    """Option prices must be non-negative and satisfy standard no-arb bounds."""
    call = bs.call_price()
    put = bs.put_price()

    # Prices must be non-negative
    assert call >= 0
    assert put >= 0

    # Call <= S
    assert call <= SPOT

    # Put <= K * exp(-rT)
    assert put <= STRIKE * math.exp(-RATE * EXPIRY)

    # Call >= max(0, S - K*exp(-rT))  (lower bound for European call)
    intrinsic_call = max(0.0, SPOT - STRIKE * math.exp(-RATE * EXPIRY))
    assert call >= intrinsic_call - 1e-10


# ---------------------------------------------------------------------------
# Test 5: Implied volatility recovers the original sigma
# ---------------------------------------------------------------------------

def test_implied_volatility_roundtrip(bs):
    """Computing IV from the BS price should recover the original sigma."""
    call_price = bs.call_price()
    put_price = bs.put_price()

    iv_call = BlackScholes.implied_volatility(
        market_price=call_price,
        S=SPOT,
        K=STRIKE,
        T=EXPIRY,
        r=RATE,
        option_type="call",
    )
    iv_put = BlackScholes.implied_volatility(
        market_price=put_price,
        S=SPOT,
        K=STRIKE,
        T=EXPIRY,
        r=RATE,
        option_type="put",
    )

    assert iv_call == pytest.approx(VOL, abs=1e-6), (
        f"Implied vol from call {iv_call:.6f} != original sigma {VOL}"
    )
    assert iv_put == pytest.approx(VOL, abs=1e-6), (
        f"Implied vol from put {iv_put:.6f} != original sigma {VOL}"
    )
