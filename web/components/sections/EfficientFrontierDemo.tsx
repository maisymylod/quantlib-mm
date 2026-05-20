"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Label,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { computeFrontier, type EfficientFrontierResponse } from "@/lib/api";
import { SectionWrap } from "../primitives/SectionWrap";

const ALL_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"];

export function EfficientFrontierDemo() {
  const [selected, setSelected] = useState<string[]>(["AAPL", "MSFT", "GOOG", "NVDA"]);
  const [data, setData] = useState<EfficientFrontierResponse | null>(null);

  useEffect(() => {
    if (selected.length < 2) return;
    let cancelled = false;
    computeFrontier({ tickers: selected, n_points: 40, risk_free_rate: 0.02 })
      .then((d) => !cancelled && setData(d));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const toggle = (t: string) => {
    setSelected((prev) =>
      prev.includes(t)
        ? prev.length > 2
          ? prev.filter((x) => x !== t)
          : prev
        : [...prev, t]
    );
  };

  const frontierData = data?.frontier.map((p) => ({ ...p, ret: p.ret * 100, risk: p.risk * 100 })) ?? [];
  const minVar = data
    ? [{
        risk: data.min_variance.volatility * 100,
        ret: data.min_variance.return * 100,
      }]
    : [];
  const maxSharpe = data
    ? [{
        risk: data.max_sharpe.volatility * 100,
        ret: data.max_sharpe.return * 100,
      }]
    : [];

  return (
    <SectionWrap
      variant="lavender"
      eyebrow="04 — Portfolio"
      title="The efficient frontier."
      intro="Pick assets. The library runs SLSQP to find the minimum-variance and maximum-Sharpe portfolios, then traces the frontier between them."
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8">
        <div className="bg-white/60 rounded-2xl p-6 md:p-8">
          <h4 className="font-mono text-xs uppercase tracking-widest text-electric mb-4">Universe</h4>
          <div className="flex flex-wrap gap-2 mb-6">
            {ALL_TICKERS.map((t) => (
              <button
                key={t}
                onClick={() => toggle(t)}
                className={`px-3 py-2 rounded-full text-xs font-mono transition-colors ${
                  selected.includes(t)
                    ? "bg-electric text-white"
                    : "bg-white/80 text-navy-800 hover:bg-white"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          {data && (
            <>
              <div className="border-t border-electric/20 pt-4 mb-4">
                <div className="font-mono text-xs uppercase tracking-widest text-electric mb-2">Max-Sharpe portfolio</div>
                <div className="num-display text-3xl text-navy-900">
                  Sharpe {data.max_sharpe.sharpe.toFixed(2)}
                </div>
                <div className="text-xs font-mono text-navy-800/70 mt-1">
                  return {(data.max_sharpe.return * 100).toFixed(1)}% · vol {(data.max_sharpe.volatility * 100).toFixed(1)}%
                </div>
                <div className="mt-3 space-y-1">
                  {Object.entries(data.max_sharpe.weights).map(([t, w]) => (
                    <WeightBar key={t} ticker={t} weight={w} />
                  ))}
                </div>
              </div>
              <div className="border-t border-electric/20 pt-4">
                <div className="font-mono text-xs uppercase tracking-widest text-electric mb-2">Min-variance portfolio</div>
                <div className="num-display text-3xl text-navy-900">
                  σ {(data.min_variance.volatility * 100).toFixed(1)}%
                </div>
                <div className="mt-3 space-y-1">
                  {Object.entries(data.min_variance.weights).map(([t, w]) => (
                    <WeightBar key={t} ticker={t} weight={w} />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
        <div className="bg-white/60 rounded-2xl p-6 md:p-8 min-h-[480px]">
          <ResponsiveContainer width="100%" height={460}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 20 }}>
              <CartesianGrid stroke="#C4C0F4" strokeOpacity={0.5} />
              <XAxis
                type="number"
                dataKey="risk"
                stroke="#1A1F3A"
                tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              >
                <Label value="Volatility (annualised)" position="bottom" offset={20} fontSize={11} />
              </XAxis>
              <YAxis
                type="number"
                dataKey="ret"
                stroke="#1A1F3A"
                tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              >
                <Label value="Expected return" angle={-90} position="insideLeft" offset={10} fontSize={11} />
              </YAxis>
              <ZAxis range={[60, 60]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  background: "#FFFFFF",
                  border: "1px solid #2727FF",
                  borderRadius: 8,
                  fontFamily: "JetBrains Mono",
                  fontSize: 12,
                }}
                formatter={(v: number) => `${v.toFixed(2)}%`}
              />
              <Scatter name="Frontier" data={frontierData} fill="#2727FF" line={{ stroke: "#2727FF", strokeWidth: 2 }} shape="circle" />
              <Scatter name="Max Sharpe" data={maxSharpe} fill="#D4F23F" shape="star" />
              <Scatter name="Min variance" data={minVar} fill="#1A1F3A" shape="diamond" />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="mt-2 flex flex-wrap gap-6 text-xs font-mono text-navy-800/80">
            <LegendDot color="#2727FF" label="Efficient frontier" />
            <LegendDot color="#D4F23F" label="Max Sharpe" />
            <LegendDot color="#1A1F3A" label="Min variance" />
          </div>
        </div>
      </div>
    </SectionWrap>
  );
}

function WeightBar({ ticker, weight }: { ticker: string; weight: number }) {
  return (
    <div className="flex items-center gap-3 text-xs font-mono">
      <span className="w-12 text-navy-800">{ticker}</span>
      <div className="flex-1 h-2 bg-white rounded-full overflow-hidden">
        <div className="h-full bg-electric" style={{ width: `${Math.max(weight * 100, 0).toFixed(1)}%` }} />
      </div>
      <span className="w-12 text-right text-navy-800">{(weight * 100).toFixed(1)}%</span>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-3 h-3 rounded-full" style={{ background: color }} />
      <span>{label}</span>
    </div>
  );
}
