import { VerticalBars } from "../primitives/VerticalBars";

export function Hero() {
  return (
    <section className="relative overflow-hidden bg-navy-950 min-h-[100vh] flex items-center">
      <VerticalBars
        className="absolute top-0 right-0 w-2/5 h-72 opacity-60"
        color="#D4F23F"
        count={40}
      />
      <VerticalBars
        className="absolute bottom-12 left-12 w-32 h-16 opacity-40"
        color="#2727FF"
      />
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-24 w-full grid md:grid-cols-2 gap-12 items-center relative z-10">
        <div>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-full border-2 border-lime flex items-center justify-center font-display font-bold text-lime">
              q
            </div>
            <span className="font-display text-xl tracking-tight">
              quantlib-mm
            </span>
          </div>
          <h1 className="font-display font-semibold text-5xl md:text-7xl leading-[0.95] tracking-tighter2 mb-6">
            Python quantitative finance,{" "}
            <span className="text-lime">built from first principles.</span>
          </h1>
          <p className="text-lg md:text-xl text-ink-500 max-w-xl mb-10">
            An interactive showcase of every model in the library: Monte Carlo,
            Black-Scholes, Greeks, mean-variance optimisation, VaR/CVaR, and
            yield curves. Move the sliders, the math runs live.
          </p>
          <div className="flex flex-wrap gap-4">
            <a
              href="#playground"
              className="inline-flex items-center gap-2 bg-lime text-navy-900 font-semibold px-6 py-3 rounded-full hover:bg-lime-400 transition-colors"
            >
              Open the playground →
            </a>
            <a
              href="https://github.com/maisymylod/quantlib-mm"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 border border-white/20 px-6 py-3 rounded-full hover:bg-white/5 transition-colors"
            >
              View on GitHub
            </a>
          </div>
        </div>
        <div className="relative aspect-square hidden md:block">
          <HeroGraphic />
        </div>
      </div>
      <div className="absolute bottom-4 left-6 right-6 flex justify-between text-[10px] uppercase tracking-widest text-ink-400 font-mono">
        <span>v0.1.0 · MIT license</span>
        <span>numpy · scipy · pure python</span>
      </div>
    </section>
  );
}

function HeroGraphic() {
  // Stylised animated path strip suggesting Monte Carlo trajectories.
  return (
    <svg viewBox="0 0 400 400" className="w-full h-full">
      <defs>
        <linearGradient id="pathGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#D4F23F" stopOpacity="0.0" />
          <stop offset="60%" stopColor="#D4F23F" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#D4F23F" stopOpacity="0.4" />
        </linearGradient>
      </defs>
      {/* Background grid */}
      <g stroke="#1A1F3A" strokeWidth="1">
        {Array.from({ length: 8 }, (_, i) => (
          <line key={i} x1={50 + i * 40} y1={40} x2={50 + i * 40} y2={360} />
        ))}
        {Array.from({ length: 7 }, (_, i) => (
          <line key={i} x1={50} y1={60 + i * 40} x2={370} y2={60 + i * 40} />
        ))}
      </g>
      {/* Paths */}
      {[
        "M50 250 Q 100 230, 140 220 T 220 180 T 300 140 T 370 90",
        "M50 250 Q 100 270, 140 260 T 220 230 T 300 240 T 370 200",
        "M50 250 Q 100 240, 140 270 T 220 280 T 300 250 T 370 230",
        "M50 250 Q 100 220, 140 200 T 220 170 T 300 200 T 370 160",
        "M50 250 Q 100 260, 140 240 T 220 250 T 300 220 T 370 250",
      ].map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke="url(#pathGrad)"
          strokeWidth="2"
          opacity={0.5 + i * 0.1}
        />
      ))}
      {/* Mean path */}
      <path
        d="M50 250 L 90 240 L 130 230 L 170 220 L 210 210 L 250 200 L 290 192 L 330 180 L 370 170"
        fill="none"
        stroke="#D4F23F"
        strokeWidth="3"
      />
      {/* Start/end dots */}
      <circle cx="50" cy="250" r="6" fill="#2727FF" />
      <circle cx="370" cy="170" r="6" fill="#D4F23F" />
      {/* Labels */}
      <text x="50" y="380" fontSize="10" fill="#A3A8C9" fontFamily="JetBrains Mono">
        t=0
      </text>
      <text x="340" y="380" fontSize="10" fill="#A3A8C9" fontFamily="JetBrains Mono">
        t=T
      </text>
      <text
        x="20"
        y="60"
        fontSize="10"
        fill="#A3A8C9"
        fontFamily="JetBrains Mono"
        transform="rotate(-90 20 60)"
      >
        log S(t)
      </text>
    </svg>
  );
}
