// Typed client for the quantlib-mm FastAPI backend.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8765";

async function post<TResp>(path: string, body: unknown): Promise<TResp> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} → ${res.status} ${await res.text()}`);
  return res.json() as Promise<TResp>;
}

async function get<TResp>(path: string): Promise<TResp> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path} → ${res.status} ${await res.text()}`);
  return res.json() as Promise<TResp>;
}

// ---------- Monte Carlo ----------
export interface MonteCarloRequest {
  s0: number;
  mu: number;
  sigma: number;
  T: number;
  n_steps: number;
  n_paths: number;
  antithetic?: boolean;
  seed?: number | null;
}
export interface MonteCarloResponse {
  times: number[];
  sample_paths: number[][];
  mean_path: number[];
  p5_path: number[];
  p95_path: number[];
  terminal_mean: number;
  terminal_std: number;
  terminal_min: number;
  terminal_max: number;
  percentiles: Record<number, number>;
  max_drawdown_mean: number;
}
export const simulateMC = (b: MonteCarloRequest) =>
  post<MonteCarloResponse>("/api/montecarlo/simulate", b);

// ---------- Black-Scholes ----------
export interface BlackScholesRequest {
  S: number;
  K: number;
  T: number;
  r: number;
  sigma: number;
}
export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}
export interface BlackScholesResponse {
  call_price: number;
  put_price: number;
  call_greeks: Greeks;
  put_greeks: Greeks;
  parity_holds: boolean;
}
export const priceBS = (b: BlackScholesRequest) =>
  post<BlackScholesResponse>("/api/options/black-scholes", b);

// ---------- Binomial ----------
export interface BinomialRequest {
  S: number;
  K: number;
  T: number;
  r: number;
  sigma: number;
  n_steps: number;
  option_type: "call" | "put";
  style: "european" | "american";
}
export interface BinomialNode {
  step: number;
  index: number;
  spot: number;
  value: number;
}
export interface BinomialResponse {
  price: number;
  u: number;
  d: number;
  p: number;
  nodes: BinomialNode[];
}
export const priceBinomial = (b: BinomialRequest) =>
  post<BinomialResponse>("/api/options/binomial", b);

// ---------- Efficient Frontier ----------
export interface EfficientFrontierRequest {
  tickers: string[];
  n_points?: number;
  risk_free_rate?: number;
}
export interface FrontierPoint {
  risk: number;
  ret: number;
}
export interface FrontierSummary {
  weights: Record<string, number>;
  return: number;
  volatility: number;
  sharpe: number;
}
export interface EfficientFrontierResponse {
  frontier: FrontierPoint[];
  min_variance: FrontierSummary;
  max_sharpe: FrontierSummary;
  expected_returns: Record<string, number>;
  annualised_vols: Record<string, number>;
}
export const computeFrontier = (b: EfficientFrontierRequest) =>
  post<EfficientFrontierResponse>("/api/portfolio/efficient-frontier", b);

// ---------- Risk ----------
export interface RiskRequest {
  ticker: string;
  confidence?: number;
  risk_free_rate?: number;
}
export interface RiskResponse {
  var_historical: number;
  var_parametric: number;
  cvar: number;
  max_drawdown: number;
  sortino: number;
  calmar: number;
  histogram: { counts: number[]; edges: number[] };
  drawdown_series: number[];
  cumulative_wealth: number[];
}
export const riskMetrics = (b: RiskRequest) =>
  post<RiskResponse>("/api/risk/metrics", b);

// ---------- Yield Curve ----------
export interface YieldCurveRequest {
  maturities: number[];
  par_rates: number[];
}
export interface YieldCurveResponse {
  maturities: number[];
  zero_rates: number[];
  discount_factors: number[];
  forward_rates: number[];
  forward_starts: number[];
  forward_ends: number[];
}
export const buildYieldCurve = (b: YieldCurveRequest) =>
  post<YieldCurveResponse>("/api/yield-curve/build", b);

// ---------- Sample data ----------
export const listTickers = () => get<string[]>("/api/sample-data/tickers");
