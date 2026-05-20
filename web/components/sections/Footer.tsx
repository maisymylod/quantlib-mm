export function Footer() {
  return (
    <footer className="bg-navy-950 border-t border-white/10">
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-12 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full border-2 border-lime flex items-center justify-center font-display font-bold text-lime">
            q
          </div>
          <div>
            <div className="font-display font-semibold">quantlib-mm</div>
            <div className="text-xs font-mono text-ink-500 mt-1">
              MIT licensed · built with NumPy, SciPy, FastAPI &amp; Next.js
            </div>
          </div>
        </div>
        <div className="flex gap-6 text-sm font-mono text-ink-500">
          <a
            href="https://github.com/maisymylod/quantlib-mm"
            target="_blank"
            rel="noreferrer"
            className="hover:text-lime transition-colors"
          >
            GitHub
          </a>
          <a href="#playground" className="hover:text-lime transition-colors">
            Playground
          </a>
        </div>
      </div>
    </footer>
  );
}
