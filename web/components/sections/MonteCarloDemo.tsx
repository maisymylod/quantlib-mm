"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { simulateMC, type MonteCarloResponse } from "@/lib/api";
import { SectionWrap } from "../primitives/SectionWrap";
import { Slider } from "../primitives/Slider";
import { Panel } from "../primitives/Panel";

export function MonteCarloDemo() {
  const [s0, setS0] = useState(100);
  const [mu, setMu] = useState(0.08);
  const [sigma, setSigma] = useState(0.25);
  const [T, setT] = useState(1.0);
  const [data, setData] = useState<MonteCarloResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    simulateMC({
      s0,
      mu,
      sigma,
      T,
      n_steps: 100,
      n_paths: 500,
      antithetic: true,
      seed: 42,
    })
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [s0, mu, sigma, T]);

  const chartData = useMemo(() => {
    if (!data) return [];
    return data.times.map((t, i) => {
      const row: Record<string, number> = {
        t,
        mean: data.mean_path[i],
        p5: data.p5_path[i],
        p95: data.p95_path[i],
      };
      // Sample 12 individual paths for context
      data.sample_paths.slice(0, 12).forEach((p, j) => {
        row[`p${j}`] = p[i];
      });
      return row;
    });
  }, [data]);

  return (
    <SectionWrap
      variant="dark"
      eyebrow="01 — Monte Carlo"
      title="Geometric Brownian Motion paths."
      intro="Drift, volatility, and horizon. The library simulates 500 paths with antithetic variates, then returns percentile bands and per-path statistics."
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8">
        <Panel variant="dark">
          <h4 className="font-mono text-xs uppercase tracking-widest text-lime mb-6">
            Inputs
          </h4>
          <div className="space-y-6">
            <Slider label="S₀ (initial price)" value={s0} onChange={setS0} min={10} max={500} step={10} />
            <Slider label="μ (drift)" value={mu} onChange={setMu} min={-0.1} max={0.4} step={0.01} format={(v) => `${(v * 100).toFixed(0)}%`} />
            <Slider label="σ (volatility)" value={sigma} onChange={setSigma} min={0.05} max={0.8} step={0.01} format={(v) => `${(v * 100).toFixed(0)}%`} />
            <Slider label="T (years)" value={T} onChange={setT} min={0.1} max={5} step={0.1} format={(v) => `${v.toFixed(1)} y`} />
          </div>
          {data && (
            <div className="mt-8 pt-6 border-t border-white/10 grid grid-cols-2 gap-4">
              <Stat label="Terminal mean" value={`$${data.terminal_mean.toFixed(2)}`} />
              <Stat label="Terminal σ" value={`$${data.terminal_std.toFixed(2)}`} />
              <Stat label="P5" value={`$${data.percentiles[5].toFixed(2)}`} />
              <Stat label="P95" value={`$${data.percentiles[95].toFixed(2)}`} />
              <Stat label="Mean max DD" value={`${(data.max_drawdown_mean * 100).toFixed(1)}%`} />
              <Stat label="Status" value={loading ? "running…" : "live"} accent />
            </div>
          )}
        </Panel>
        <Panel variant="dark" className="min-h-[480px]">
          {chartData.length > 0 && (
            <ResponsiveContainer width="100%" height={460}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="#1A1F3A" />
                <XAxis
                  dataKey="t"
                  tickFormatter={(v) => v.toFixed(1)}
                  stroke="#7B82A8"
                  tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
                />
                <YAxis stroke="#7B82A8" tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }} />
                <Tooltip
                  contentStyle={{
                    background: "#0F1437",
                    border: "1px solid #252A4D",
                    borderRadius: 8,
                    fontFamily: "JetBrains Mono",
                    fontSize: 12,
                  }}
                  formatter={(v: number) => v.toFixed(2)}
                />
                {Array.from({ length: 12 }, (_, j) => (
                  <Line
                    key={j}
                    type="monotone"
                    dataKey={`p${j}`}
                    stroke="#4949FF"
                    strokeWidth={1}
                    strokeOpacity={0.35}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
                <Line type="monotone" dataKey="p5" stroke="#A3A8C9" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="p95" stroke="#A3A8C9" strokeWidth={1.5} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="mean" stroke="#D4F23F" strokeWidth={3} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
          <div className="mt-4 flex flex-wrap gap-6 text-xs font-mono text-ink-500">
            <LegendDot color="#D4F23F" label="Mean path" />
            <LegendDot color="#A3A8C9" label="P5 / P95" />
            <LegendDot color="#4949FF" label="Sample paths" />
          </div>
        </Panel>
      </div>
    </SectionWrap>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-ink-500 font-mono">{label}</div>
      <div className={`mt-1 font-mono ${accent ? "text-lime" : "text-white"}`}>{value}</div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-3 h-1 rounded-full" style={{ background: color }} />
      <span>{label}</span>
    </div>
  );
}
