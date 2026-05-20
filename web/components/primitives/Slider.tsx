type Props = {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  format?: (v: number) => string;
  onLavender?: boolean;
};

export function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  format,
  onLavender = false,
}: Props) {
  const display = format ? format(value) : value.toFixed(step >= 1 ? 0 : 2);
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <label
          className={`text-xs uppercase tracking-widest font-mono ${
            onLavender ? "text-navy-700" : "text-ink-500"
          }`}
        >
          {label}
        </label>
        <span
          className={`font-mono text-sm ${
            onLavender ? "text-navy-900" : "text-white"
          }`}
        >
          {display}
        </span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={onLavender ? "on-lavender" : ""}
      />
    </div>
  );
}
