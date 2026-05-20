import { VerticalBars } from "./VerticalBars";

type Props = {
  id?: string;
  variant?: "dark" | "electric" | "lavender";
  eyebrow?: string;
  title: string;
  intro?: string;
  showBars?: boolean;
  children: React.ReactNode;
};

const variantClasses: Record<NonNullable<Props["variant"]>, string> = {
  dark: "bg-navy-900 text-white",
  electric: "bg-electric text-white",
  lavender: "bg-lavender-300 text-navy-900",
};

const barColor: Record<NonNullable<Props["variant"]>, string> = {
  dark: "#D4F23F",
  electric: "#FFFFFF",
  lavender: "#2727FF",
};

const eyebrowClasses: Record<NonNullable<Props["variant"]>, string> = {
  dark: "text-lime",
  electric: "text-lime",
  lavender: "text-electric",
};

export function SectionWrap({
  id,
  variant = "dark",
  eyebrow,
  title,
  intro,
  showBars = true,
  children,
}: Props) {
  return (
    <section
      id={id}
      className={`relative overflow-hidden ${variantClasses[variant]}`}
    >
      {showBars && (
        <VerticalBars
          className="absolute top-12 right-12 w-32 h-20 opacity-70"
          color={barColor[variant]}
        />
      )}
      <div className="max-w-7xl mx-auto px-6 md:px-12 py-20 md:py-28">
        <div className="max-w-3xl mb-12">
          {eyebrow && (
            <div
              className={`uppercase tracking-widest text-xs font-mono mb-4 ${eyebrowClasses[variant]}`}
            >
              {eyebrow}
            </div>
          )}
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tighter2 mb-4">
            {title}
          </h2>
          {intro && (
            <p
              className={`text-base md:text-lg ${
                variant === "lavender" ? "text-navy-800/80" : "text-ink-500"
              }`}
            >
              {intro}
            </p>
          )}
        </div>
        {children}
      </div>
    </section>
  );
}
