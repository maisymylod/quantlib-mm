type Props = {
  number: string;
  title: string;
  children: React.ReactNode;
  variant?: "lime-on-dark" | "blue-on-dark" | "lavender";
};

export function NumberedCard({
  number,
  title,
  children,
  variant = "lime-on-dark",
}: Props) {
  if (variant === "lavender") {
    return (
      <div className="flex flex-col rounded-2xl bg-lavender-200 text-navy-900 p-8 min-h-[300px]">
        <div className="num-display text-7xl text-electric/80 leading-none mb-6">
          {number}
        </div>
        <h3 className="text-2xl font-semibold mb-3">{title}</h3>
        <div className="text-navy-800/80 text-sm leading-relaxed">{children}</div>
      </div>
    );
  }
  if (variant === "blue-on-dark") {
    return (
      <div className="flex flex-col rounded-2xl bg-electric text-white p-8 min-h-[300px]">
        <div className="num-display text-7xl text-white/90 leading-none mb-6">
          {number}
        </div>
        <h3 className="text-2xl font-semibold mb-3">{title}</h3>
        <div className="text-white/80 text-sm leading-relaxed">{children}</div>
      </div>
    );
  }
  return (
    <div className="flex flex-col rounded-2xl bg-navy-800 border border-white/5 p-8 min-h-[300px]">
      <div className="num-display text-7xl text-lime leading-none mb-6">
        {number}
      </div>
      <h3 className="text-2xl font-semibold mb-3">{title}</h3>
      <div className="text-ink-500 text-sm leading-relaxed">{children}</div>
    </div>
  );
}
