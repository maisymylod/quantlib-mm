type Props = {
  value: string;
  label: string;
  accent?: "lime" | "lavender" | "white";
};

export function StatTile({ value, label, accent = "lime" }: Props) {
  const valueClass = {
    lime: "text-lime",
    lavender: "text-lavender",
    white: "text-white",
  }[accent];
  return (
    <div className="flex flex-col items-center">
      <div className={`num-display text-5xl md:text-6xl ${valueClass}`}>
        {value}
      </div>
      <div className="mt-2 text-sm text-ink-500 uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}
