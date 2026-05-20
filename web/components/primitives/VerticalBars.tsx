type Props = {
  className?: string;
  count?: number;
  color?: string;
  variant?: "tall" | "short";
};

export function VerticalBars({
  className = "",
  count = 18,
  color = "currentColor",
  variant = "tall",
}: Props) {
  const heights =
    variant === "tall"
      ? Array.from({ length: count }, (_, i) => 40 + ((i * 17) % 60))
      : Array.from({ length: count }, (_, i) => 8 + ((i * 13) % 24));

  return (
    <svg
      className={className}
      viewBox={`0 0 ${count * 6} 100`}
      preserveAspectRatio="none"
      aria-hidden
    >
      {heights.map((h, i) => (
        <rect
          key={i}
          x={i * 6}
          y={50 - h / 2}
          width="2"
          height={h}
          fill={color}
          opacity={0.6 + (i % 4) * 0.1}
        />
      ))}
    </svg>
  );
}
