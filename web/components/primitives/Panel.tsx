type Props = {
  variant?: "dark" | "electric" | "lavender";
  className?: string;
  children: React.ReactNode;
};

export function Panel({ variant = "dark", className = "", children }: Props) {
  const base = {
    dark: "bg-navy-800/80 border border-white/5 text-white",
    electric: "bg-white/10 border border-white/20 text-white backdrop-blur",
    lavender: "bg-lavender-200 border border-electric/20 text-navy-900",
  }[variant];
  return (
    <div className={`rounded-2xl p-6 md:p-8 ${base} ${className}`}>
      {children}
    </div>
  );
}
