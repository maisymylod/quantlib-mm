import { NumberedCard } from "../primitives/NumberedCard";
import { SectionWrap } from "../primitives/SectionWrap";

export function HowItsBuilt() {
  return (
    <SectionWrap
      variant="lavender"
      eyebrow="How it's built"
      title="Boring tech, sharp math."
      intro="No magic. Each function is a small, readable implementation of a textbook formula. Then we test it."
    >
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <NumberedCard number="01" title="Pure NumPy &amp; SciPy" variant="lavender">
          Every model is implemented in vectorised NumPy with scipy.optimize and
          scipy.stats for the heavy lifting. No Cython, no C extensions, no
          surprises in the import graph.
        </NumberedCard>
        <NumberedCard number="02" title="Tested end to end" variant="lavender">
          102 pytest tests cover every public method, including parity checks
          (Black-Scholes ⇄ binomial in the limit), numerical-stability bounds,
          and statistical edge cases.
        </NumberedCard>
        <NumberedCard number="03" title="MIT licensed" variant="lavender">
          The library and this entire showcase are open source. Clone the repo,
          read the math, fork what you need. Improvements welcome.
        </NumberedCard>
      </div>
    </SectionWrap>
  );
}
