import { VerticalBars } from "../primitives/VerticalBars";

export function PlaygroundHeader() {
  return (
    <section
      id="playground"
      className="relative bg-navy-950 border-t border-white/5"
    >
      <VerticalBars
        className="absolute top-12 right-12 w-32 h-20 opacity-60"
        color="#D4F23F"
      />
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-24">
        <div className="uppercase tracking-widest text-xs font-mono mb-4 text-lime">
          Playground
        </div>
        <h2 className="font-display text-4xl md:text-6xl font-semibold tracking-tighter2 max-w-4xl">
          Where math meets your inputs.
        </h2>
        <p className="mt-6 text-ink-500 text-lg max-w-2xl">
          Six interactive demos, each wired straight to a function in the
          library. Move the sliders. The Python runs on the backend, results
          come straight back.
        </p>
      </div>
    </section>
  );
}
