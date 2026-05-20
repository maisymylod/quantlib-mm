import { NumberedCard } from "../primitives/NumberedCard";
import { SectionWrap } from "../primitives/SectionWrap";

export function GlanceCards() {
  return (
    <SectionWrap
      variant="dark"
      eyebrow="The library at a glance"
      title="Three engines, one toolkit."
      intro="Every model is implemented in plain NumPy and SciPy. No surprises, no black boxes, no compiled blobs."
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <NumberedCard number="01" title="Monte Carlo">
          Geometric Brownian Motion simulation with antithetic variance reduction,
          plus risk-neutral pricing for European, Asian, and barrier options.
        </NumberedCard>
        <NumberedCard number="02" title="Option pricing">
          Closed-form Black-Scholes with all five Greeks, plus Cox-Ross-Rubinstein
          binomial trees with American early-exercise.
        </NumberedCard>
        <NumberedCard number="03" title="Risk &amp; portfolio">
          Historical and parametric VaR, CVaR, drawdown, Sortino, Calmar.
          Markowitz mean-variance frontier with min-variance and max-Sharpe
          optimisers. Yield curve bootstrap and PCA on correlation matrices.
        </NumberedCard>
      </div>
    </SectionWrap>
  );
}
