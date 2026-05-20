"use client";

import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { riskMetrics, type RiskResponse } from "@/lib/api";
import { SectionWrap } from "../primitives/SectionWrap";
import { Panel } from "../primitives/Panel";

const TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"];

export function RiskDemo() {
  const [ticker, setTicker] = useState("NVDA");
  const [confidence, setConfidence] = useState(0.95);
  const [data, setData] = useState<RiskResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    riskMetrics({ ticker, confidence, risk_free_rate: 0.02 })
      .then((d) => !cancelled && setData(d));
    return () => {
      cancelled = true;
    };
  }, [ticker, confidence]);

  const histData =
    data?.histogram.counts.map((c, i) => ({
      mid: (data.histogram.edges[i] + data.histogram.edges[i + 1]) / 2,
      count: c,
      ret: ((data.histogram.edges[i] + data.histogram.edges[i + 1]) / 2) * 100,
    })) ?? [];

  const drawdownData = data?.drawdown_series.map((dd, i) => ({ i, dd: dd * 100 })) ?? [];

  return (
    <SectionWrap
      variant="dark"
      eyebrow="05 — Risk"
      title="Tail behaviour, drawdowns, ratios."
      intro="Historical and parametric VaR, Conditional VaR, max drawdown, Sortino and Calmar — all in one call against a synthetic returns series."
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8">
        <Panel>
          <h4 className="font-mono text-xs uppercase tracking-widest text-lime mb-4">Asset</h4>
          <div className="flex flex-wrap gap-2 mb-6">
            {TICKERS.map((t) => (
              <button
                key={t}
                onClick={() => setTicker(t)}
                className={`px-3 py-2 rounded-full text-xs font-mono uppercase tracking-widest transition-colors ${
                  ticker === t ? "bg-lime text-navy-900" : "bg-white/5 text-ink-500 hover:bg-white/10"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <h4 className="font-mono text-xs uppercase tracking-widest text-lime mb-4">Confidence</h4>
          <div className="flex flex-wrap gap-2 mb-8">
            {[0.9, 0.95, 0.99].map((c) => (
              <button
                key={c}
                onClick={() => setConfidence(c)}
                className={`px-3 py-2 rounded-full text-xs font-mono transition-colors ${
                  confidence === c ? "bg-lime text-navy-900" : "bg-white/5 text-ink-500 hover:bg-white/10"
                }`}
              >
                {(c * 100).toFixed(0)}%
              </button>
            ))}
          </div>
          {data && (
            <div className="grid grid-cols-2 gap-4">
              <Stat label={`VaR hist`} value={pct(data.var_historical)} accent />
              <Stat label="VaR param" value={pct(data.var_parametric)} />
              <Stat label="CVaR" value={pct(data.cvar)} accent />
              <Stat label="Max DD" value={pct(data.max_drawdown)} />
              <Stat label="Sortino" value={data.sortino.toFixed(2)} />
              <Stat label="Calmar" value={data.calmar.toFixed(2)} />
            </div>
          )}
        </Panel>
        <div className="grid grid-cols-1 gap-6">
          <Panel className="h-64">
            <div className="font-mono text-xs uppercase tracking-widest text-lime mb-2">
              Return distribution with VaR
            </div>
            {data && (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={histData}>
                  <CartesianGrid stroke="#1A1F3A" />
                  <XAxis
                    dataKey="ret"
                    stroke="#7B82A8"
                    tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }}
                    tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                  />
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{
                      background: "#0F1437",
                      border: "1px solid #252A4D",
                      borderRadius: 8,
                      fontFamily: "JetBrains Mono",
                      fontSize: 12,
                    }}
                    formatter={(v: number, _name: string, p: { payload?: { ret?: number } }) => [v, `return ${p?.payload?.ret?.toFixed(2)}%`]}
                  />
                  <Bar dataKey="count">
                    {histData.map((row, i) => (
                      <Cell key={i} fill={row.mid <= data.var_historical ? "#D4F23F" : "#4949FF"} />
                    ))}
                  </Bar>
                  <ReferenceLine
                    x={data.var_historical * 100}
                    stroke="#D4F23F"
                    strokeDasharray="3 3"
                    label={{ value: "VaR", position: "top", fill: "#D4F23F", fontSize: 10, fontFamily: "JetBrains Mono" }}
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Panel>
          <Panel className="h-64">
            <div className="font-mono text-xs uppercase tracking-widest text-lime mb-2">Drawdown over time</div>
            {data && (
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={drawdownData}>
                  <defs>
                    <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#D4F23F" stopOpacity={0.5} />
                      <stop offset="100%" stopColor="#D4F23F" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#1A1F3A" />
                  <XAxis dataKey="i" stroke="#7B82A8" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} />
                  <YAxis stroke="#7B82A8" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{
                      background: "#0F1437",
                      border: "1px solid #252A4D",
                      borderRadius: 8,
                      fontFamily: "JetBrains Mono",
                      fontSize: 12,
                    }}
                    formatter={(v: number) => `${v.toFixed(2)}%`}
                    labelFormatter={(label: number) => `day ${label}`}
                  />
                  <Area type="monotone" dataKey="dd" stroke="#D4F23F" fill="url(#ddGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Panel>
        </div>
      </div>
    </SectionWrap>
  );
}

function pct(v: number) {
  return `${(v * 100).toFixed(2)}%`;
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-ink-500 font-mono">{label}</div>
      <div className={`mt-1 font-mono ${accent ? "text-lime" : "text-white"}`}>{value}</div>
    </div>
  );
}
