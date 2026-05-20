import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "quantlib-mm — Python quantitative finance, built from first principles",
  description:
    "Interactive showcase: Monte Carlo, Black-Scholes, Greeks, efficient frontier, VaR/CVaR, and yield curves, powered by the open-source quantlib-mm Python library.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
