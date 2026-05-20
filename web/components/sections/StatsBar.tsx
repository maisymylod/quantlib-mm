import { StatTile } from "../primitives/StatTile";

export function StatsBar() {
  return (
    <section className="bg-navy-900 border-y border-white/5">
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-16 grid grid-cols-2 md:grid-cols-4 gap-10">
        <StatTile value="11" label="modules" />
        <StatTile value="102" label="tests passing" />
        <StatTile value="0" label="C extensions" />
        <StatTile value="MIT" label="open source" />
      </div>
    </section>
  );
}
