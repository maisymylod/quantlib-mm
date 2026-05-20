import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: "#0F1437",
          950: "#080B26",
          900: "#0F1437",
          800: "#1A1F3A",
          700: "#252A4D",
        },
        lime: {
          DEFAULT: "#D4F23F",
          400: "#E0F76C",
          500: "#D4F23F",
          600: "#B8D02A",
        },
        electric: {
          DEFAULT: "#2727FF",
          500: "#2727FF",
          400: "#4949FF",
          600: "#1A1AE6",
        },
        lavender: {
          DEFAULT: "#C4C0F4",
          200: "#E0DEFA",
          300: "#C4C0F4",
          400: "#A8A2EE",
        },
        ink: {
          DEFAULT: "#A3A8C9",
          500: "#A3A8C9",
          400: "#7B82A8",
        },
      },
      fontFamily: {
        display: ['"Space Grotesk"', "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      letterSpacing: {
        tightish: "-0.02em",
        tighter2: "-0.04em",
      },
    },
  },
  plugins: [],
};
export default config;
