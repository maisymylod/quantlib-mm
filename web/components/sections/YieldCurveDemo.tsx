"use client";

import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { buildYieldCurve, type YieldCurveResponse } from "@/lib/api";
import { SectionWrap } from "../primitives/SectionWrap";
import { Panel } from "../primitives/Panel";

const PRESETS: Record<string, { maturities: number[]; par_rates: number[] }> = {
  "Upward sloping": {
    maturities: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    par_rates: [0.043, 0.044, 0.0445, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485],
  },
  Inverted: {
    maturities: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    par_rates: [0.0525, 0.052, 0.051, 0.0498, 0.049, 0.0482, 0.0475, 0.047, 0.0465, 0.046],
  },
  Flat: {
    maturities: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    par_rates: [0.045, 0.045, 0.0451, 0.045, 0.0449, 0.045, 0.045, 0.0451, 0.045, 0.045],
  },
};

export function YieldCurveDemo() {
  const [presetName, setPresetName] = useState<keyof typeof PRESETS>("Upward sloping");
  const [data, setData] = useState<YieldCurveResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    buildYieldCurve(PRESETS[presetName]).then((d) => !cancelled && setData(d));
    return () => {
      cancelled = true;
    };
  }, [presetName]);

  const chartData = data?.maturities.map((t, i) => ({
    t,
    zero: data.zero_rates[i] * 100,
    forward:
      i < data.forward_rates.length
        ? data.forward_rates[i] * 100
        : null,
  })) ?? [];

  return (
    <SectionWrap
      variant="electric"
      eyebrow="06 — Yield curve"
      title="Bootstrap, discount, forward."
      intro="Convert par-bond yields into zero rates with the classic recursive bootstrap, then back out the implied forward rates between consecutive maturities."
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8">
        <div className="rounded-2xl bg-white/10 border border-white/20 p-6 md:p-8 backdrop-blur">
          <h4 className="font-mono text-xs uppercase tracking-widest text-lime mb-4">Curve shape</h4>
          <div className="flex flex-col gap-2">
            {Object.keys(PRESETS).map((name) => (
              <button
                key={name}
                onClick={() => setPresetName(name as keyof typeof PRESETS)}
                className={`text-left px-4 py-3 rounded-xl text-sm font-mono transition-colors ${
                  presetName === name ? "bg-lime text-navy-900" : "bg-navy-900/30 text-white hover:bg-navy-900/50"
                }`}
              >
                {name}
              </button>
            ))}
          </div>
          {data && (
            <div className="mt-8 pt-6 border-t border-white/20">
              <div className="font-mono text-xs uppercase tracking-widest text-lime mb-3">
                Discount factors
              </div>
              <div className="space-y-1 text-xs font-mono">
                {data.maturities.map((m, i) => (
                  <div key={m} className="flex justify-between">
                    <span className="text-white/70">{m.toFixed(0)}y</span>
                    <span>{data.discount_factors[i].toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="rounded-2xl bg-navy-900 p-6 md:p-8 min-h-[480px]">
          {data && (
            <ResponsiveContainer width="100%" height={460}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="#1A1F3A" />
                <XAxis
                  dataKey="t"
                  stroke="#7B82A8"
                  tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
                  tickFormatter={(v: number) => `${v.toFixed(0)}y`}
                />
                <YAxis
                  stroke="#7B82A8"
                  tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }}
                  tickFormatter={(v: number) => `${v.toFixed(1)}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: "#0F1437",
                    border: "1px solid #252A4D",
                    borderRadius: 8,
                    fontFamily: "JetBrains Mono",
                    fontSize: 12,
                  }}
                  formatter={(v: number) => `${v.toFixed(3)}%`}
                />
                <Line type="monotone" dataKey="zero" stroke="#D4F23F" strokeWidth={3} dot={{ r: 4, fill: "#D4F23F" }} name="Zero rate" />
                <Line type="monotone" dataKey="forward" stroke="#FFFFFF" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} name="1y forward" />
              </LineChart>
            </ResponsiveContainer>
          )}
          <div className="mt-4 flex flex-wrap gap-6 text-xs font-mono text-white/80">
            <LegendDot color="#D4F23F" label="Zero rate" />
            <LegendDot color="#FFFFFF" label="1y forward (between maturities)" />
          </div>
        </div>
      </div>
    </SectionWrap>
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
