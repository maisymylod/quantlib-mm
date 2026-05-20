import { BinomialDemo } from "@/components/sections/BinomialDemo";
import { BlackScholesDemo } from "@/components/sections/BlackScholesDemo";
import { EfficientFrontierDemo } from "@/components/sections/EfficientFrontierDemo";
import { Footer } from "@/components/sections/Footer";
import { GlanceCards } from "@/components/sections/GlanceCards";
import { Hero } from "@/components/sections/Hero";
import { HowItsBuilt } from "@/components/sections/HowItsBuilt";
import { MonteCarloDemo } from "@/components/sections/MonteCarloDemo";
import { PlaygroundHeader } from "@/components/sections/PlaygroundHeader";
import { RiskDemo } from "@/components/sections/RiskDemo";
import { StatsBar } from "@/components/sections/StatsBar";
import { YieldCurveDemo } from "@/components/sections/YieldCurveDemo";

export default function Page() {
  return (
    <main>
      <Hero />
      <StatsBar />
      <GlanceCards />
      <PlaygroundHeader />
      <MonteCarloDemo />
      <BlackScholesDemo />
      <BinomialDemo />
      <EfficientFrontierDemo />
      <RiskDemo />
      <YieldCurveDemo />
      <HowItsBuilt />
      <Footer />
    </main>
  );
}
