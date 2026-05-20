"use client";

import { useEffect, useState } from "react";
import { priceBS, type BlackScholesResponse } from "@/lib/api";
import { SectionWrap } from "../primitives/SectionWrap";
import { Slider } from "../primitives/Slider";

export function BlackScholesDemo() {
  const [S, setS] = useState(100);
  const [K, setK] = useState(100);
  const [T, setT] = useState(1.0);
  const [r, setR] = useState(0.05);
  const [sigma, setSigma] = useState(0.2);
  const [data, setData] = useState<BlackScholesResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    priceBS({ S, K, T, r, sigma }).then((d) => !cancelled && setData(d));
    return () => {
      cancelled = true;
    };
  }, [S, K, T, r, sigma]);

  return (
    <SectionWrap
      variant="electric"
      eyebrow="02 — Black-Scholes"
      title="European option pricing, closed form."
      intro="The Black-Scholes formula in one tap. Move the sliders to see the call, put, and all five Greeks update live. Put-call parity is verified server-side."
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8">
        <div className="rounded-2xl bg-white/10 border border-white/20 p-6 md:p-8 backdrop-blur">
          <h4 className="font-mono text-xs uppercase tracking-widest text-lime mb-6">
            Inputs
          </h4>
          <div className="space-y-6">
            <Slider label="S (spot)" value={S} onChange={setS} min={50} max={200} step={1} format={(v) => `$${v.toFixed(0)}`} />
            <Slider label="K (strike)" value={K} onChange={setK} min={50} max={200} step={1} format={(v) => `$${v.toFixed(0)}`} />
            <Slider label="T (years)" value={T} onChange={setT} min={0.05} max={3} step={0.05} format={(v) => `${v.toFixed(2)} y`} />
            <Slider label="r (risk-free)" value={r} onChange={setR} min={0.0} max={0.15} step={0.005} format={(v) => `${(v * 100).toFixed(2)}%`} />
            <Slider label="σ (volatility)" value={sigma} onChange={setSigma} min={0.05} max={0.8} step={0.01} format={(v) => `${(v * 100).toFixed(0)}%`} />
          </div>
        </div>
        {data && (
          <div className="grid grid-cols-2 gap-4">
            <PriceTile label="Call price" value={data.call_price} subtitle="max(S - K, 0)" />
            <PriceTile label="Put price" value={data.put_price} subtitle="max(K - S, 0)" />
            <div className="col-span-2 rounded-2xl bg-navy-900 p-6 md:p-8">
              <div className="font-mono text-xs uppercase tracking-widest text-lime mb-4">
                Call Greeks
              </div>
              <GreeksRow g={data.call_greeks} />
            </div>
            <div className="col-span-2 rounded-2xl bg-navy-900 p-6 md:p-8">
              <div className="font-mono text-xs uppercase tracking-widest text-lime mb-4">
                Put Greeks
              </div>
              <GreeksRow g={data.put_greeks} />
            </div>
            <div className="col-span-2 text-xs font-mono text-white/70">
              put-call parity:{" "}
              <span className={data.parity_holds ? "text-lime" : "text-red-400"}>
                {data.parity_holds ? "verified ✓" : "violated"}
              </span>
            </div>
          </div>
        )}
      </div>
    </SectionWrap>
  );
}

function PriceTile({ label, value, subtitle }: { label: string; value: number; subtitle: string }) {
  return (
    <div className="rounded-2xl bg-navy-900 p-6 md:p-8">
      <div className="font-mono text-xs uppercase tracking-widest text-lime mb-3">{label}</div>
      <div className="num-display text-5xl md:text-6xl text-white">
        ${value.toFixed(2)}
      </div>
      <div className="mt-3 text-xs font-mono text-ink-500">{subtitle}</div>
    </div>
  );
}

function GreeksRow({ g }: { g: { delta: number; gamma: number; theta: number; vega: number; rho: number } }) {
  return (
    <div className="grid grid-cols-5 gap-3 text-center">
      <GreekTile label="Δ" name="Delta" value={g.delta} />
      <GreekTile label="Γ" name="Gamma" value={g.gamma} />
      <GreekTile label="Θ" name="Theta" value={g.theta} />
      <GreekTile label="ν" name="Vega" value={g.vega} />
      <GreekTile label="ρ" name="Rho" value={g.rho} />
    </div>
  );
}

function GreekTile({ label, name, value }: { label: string; name: string; value: number }) {
  return (
    <div>
      <div className="text-2xl font-display text-lime">{label}</div>
      <div className="text-[10px] font-mono uppercase tracking-widest text-ink-500 mt-1">{name}</div>
      <div className="font-mono text-sm mt-1 text-white">{value.toFixed(4)}</div>
    </div>
  );
}
