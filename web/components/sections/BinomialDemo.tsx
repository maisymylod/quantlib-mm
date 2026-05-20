"use client";

import { useEffect, useMemo, useState } from "react";
import { priceBinomial, type BinomialResponse } from "@/lib/api";
import { SectionWrap } from "../primitives/SectionWrap";
import { Slider } from "../primitives/Slider";
import { Panel } from "../primitives/Panel";

export function BinomialDemo() {
  const [S, setS] = useState(100);
  const [K, setK] = useState(100);
  const [T, setT] = useState(1.0);
  const [sigma, setSigma] = useState(0.25);
  const [nSteps, setNSteps] = useState(6);
  const [style, setStyle] = useState<"european" | "american">("american");
  const [optType, setOptType] = useState<"call" | "put">("put");
  const [data, setData] = useState<BinomialResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    priceBinomial({ S, K, T, r: 0.05, sigma, n_steps: nSteps, option_type: optType, style })
      .then((d) => !cancelled && setData(d));
    return () => {
      cancelled = true;
    };
  }, [S, K, T, sigma, nSteps, style, optType]);

  return (
    <SectionWrap
      variant="dark"
      eyebrow="03 — Binomial tree"
      title="Cox-Ross-Rubinstein, with early exercise."
      intro="A discrete-time lattice with up/down probabilities calibrated to volatility. American options get the early-exercise check at every node."
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-8">
        <Panel>
          <h4 className="font-mono text-xs uppercase tracking-widest text-lime mb-6">Inputs</h4>
          <div className="space-y-6">
            <Slider label="S (spot)" value={S} onChange={setS} min={50} max={200} step={1} format={(v) => `$${v.toFixed(0)}`} />
            <Slider label="K (strike)" value={K} onChange={setK} min={50} max={200} step={1} format={(v) => `$${v.toFixed(0)}`} />
            <Slider label="T (years)" value={T} onChange={setT} min={0.1} max={3} step={0.1} format={(v) => `${v.toFixed(1)} y`} />
            <Slider label="σ (volatility)" value={sigma} onChange={setSigma} min={0.05} max={0.8} step={0.01} format={(v) => `${(v * 100).toFixed(0)}%`} />
            <Slider label="N (steps)" value={nSteps} onChange={(v) => setNSteps(Math.round(v))} min={2} max={12} step={1} format={(v) => `${v.toFixed(0)}`} />
            <div className="grid grid-cols-2 gap-3 pt-2">
              <PillToggle label="European" active={style === "european"} onClick={() => setStyle("european")} />
              <PillToggle label="American" active={style === "american"} onClick={() => setStyle("american")} />
              <PillToggle label="Call" active={optType === "call"} onClick={() => setOptType("call")} />
              <PillToggle label="Put" active={optType === "put"} onClick={() => setOptType("put")} />
            </div>
          </div>
          {data && (
            <div className="mt-8 pt-6 border-t border-white/10 grid grid-cols-2 gap-4">
              <Stat label="Price" value={`$${data.price.toFixed(3)}`} accent />
              <Stat label="u" value={data.u.toFixed(4)} />
              <Stat label="d" value={data.d.toFixed(4)} />
              <Stat label="p̃" value={data.p.toFixed(4)} />
            </div>
          )}
        </Panel>
        <Panel className="min-h-[480px] flex items-center justify-center">
          {data && <BinomialTreeSVG nodes={data.nodes} nSteps={nSteps} />}
        </Panel>
      </div>
    </SectionWrap>
  );
}

function BinomialTreeSVG({ nodes, nSteps }: { nodes: BinomialResponse["nodes"]; nSteps: number }) {
  const w = 800;
  const h = 460;
  const padX = 40;
  const padY = 20;
  const innerW = w - 2 * padX;
  const innerH = h - 2 * padY;

  const maxValue = useMemo(() => Math.max(...nodes.map((n) => n.value)), [nodes]);

  const xy = (step: number, idx: number) => {
    const x = padX + (step / nSteps) * innerW;
    const totalRows = nSteps + 1;
    const rowSpacing = innerH / totalRows;
    const y = padY + (idx + (totalRows - step - 1) / 2) * rowSpacing;
    return { x, y };
  };

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
      {/* Edges */}
      {nodes
        .filter((n) => n.step < nSteps)
        .map((n) => {
          const a = xy(n.step, n.index);
          const upChild = xy(n.step + 1, n.index);
          const downChild = xy(n.step + 1, n.index + 1);
          return (
            <g key={`e-${n.step}-${n.index}`}>
              <line x1={a.x} y1={a.y} x2={upChild.x} y2={upChild.y} stroke="#252A4D" strokeWidth={1} />
              <line x1={a.x} y1={a.y} x2={downChild.x} y2={downChild.y} stroke="#252A4D" strokeWidth={1} />
            </g>
          );
        })}
      {/* Nodes */}
      {nodes.map((n) => {
        const p = xy(n.step, n.index);
        const intensity = maxValue > 0 ? n.value / maxValue : 0;
        const fill =
          n.value > 0
            ? `rgba(212, 242, 63, ${0.25 + 0.7 * intensity})`
            : "#1A1F3A";
        return (
          <g key={`n-${n.step}-${n.index}`}>
            <circle cx={p.x} cy={p.y} r={18} fill={fill} stroke="#D4F23F" strokeWidth={n.value > 0 ? 1 : 0.3} />
            <text x={p.x} y={p.y + 3} textAnchor="middle" fontSize={9} fontFamily="JetBrains Mono" fill={n.value > 0 ? "#0F1437" : "#A3A8C9"}>
              {n.value.toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function PillToggle({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-2 rounded-full text-xs font-mono uppercase tracking-widest transition-colors ${
        active ? "bg-lime text-navy-900" : "bg-white/5 text-ink-500 hover:bg-white/10"
      }`}
    >
      {label}
    </button>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-ink-500 font-mono">{label}</div>
      <div className={`mt-1 font-mono ${accent ? "text-lime text-lg" : "text-white"}`}>{value}</div>
    </div>
  );
}
