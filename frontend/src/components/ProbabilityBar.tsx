// A magnitude indicator, not a full chart: sequential-blue fill scaled to
// the probability. Empirically (see docs/roadmap.md's reliability
// diagrams) predicted probabilities cluster between ~3% and ~20% — a
// 0-100% scale would make every bar look nearly empty, so the visual
// scale caps at DISPLAY_MAX, not 100%. The numeric percentage is always
// shown alongside the bar (color never carries the value alone).
const DISPLAY_MAX = 0.3;

function fillColor(probability: number): string {
  if (probability >= 0.2) return "var(--seq-700)";
  if (probability >= 0.15) return "var(--seq-600)";
  if (probability >= 0.1) return "var(--seq-500)";
  if (probability >= 0.05) return "var(--seq-400)";
  return "var(--seq-300)";
}

export function ProbabilityBar({ probability }: { probability: number }) {
  const widthPct = Math.min(100, (probability / DISPLAY_MAX) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          position: "relative",
          width: 80,
          height: 8,
          borderRadius: 4,
          background: "var(--gridline)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            width: `${widthPct}%`,
            background: fillColor(probability),
            borderRadius: 4,
          }}
        />
      </div>
      <span style={{ fontVariantNumeric: "tabular-nums", color: "var(--text-primary)", fontSize: 13 }}>
        {(probability * 100).toFixed(1)}%
      </span>
    </div>
  );
}
