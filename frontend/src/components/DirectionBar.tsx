// Direction (bullish vs bearish) is a POLARITY, not a magnitude — the
// diverging blue/red pair is the correct color job here (same pair
// ShapChart uses for the same reason), not a sequential single-hue ramp.
//
// Why this exists instead of a single number: up_probability and
// down_probability are independent model outputs and are NOT
// complementary — both can be simultaneously elevated (a low-volatility
// regime lowers the bar for a big move in *either* direction — see
// docs/roadmap.md's Phase 7 finding, which holds symmetrically for both
// models). Showing only one number, or only "whichever is bigger", would
// hide that. The bar always shows both real values; the lean label is a
// simple, clearly-labeled derived view on top, not a separate prediction.
// DISPLAY_MAX caps the visual scale at 30%, not 100%: predicted
// probabilities empirically cluster between ~3% and ~20% (see the
// reliability diagrams in docs/roadmap.md), so a 0-100% scale would
// render every bar as nearly empty.
const DISPLAY_MAX = 0.3;

const LEAN_THRESHOLD = 0.03; // 3 percentage points — a plain UI heuristic, not a statistical cutoff

function leanLabel(up: number, down: number): { text: string; color: string } {
  const spread = up - down;
  if (spread > LEAN_THRESHOLD) return { text: "Bullish lean", color: "var(--series-blue)" };
  if (spread < -LEAN_THRESHOLD) return { text: "Bearish lean", color: "var(--series-red)" };
  return { text: "Mixed / neutral", color: "var(--text-muted)" };
}

export function DirectionBar({ upProbability, downProbability }: { upProbability: number; downProbability: number }) {
  const upWidthPct = Math.min(50, (upProbability / DISPLAY_MAX) * 50);
  const downWidthPct = Math.min(50, (downProbability / DISPLAY_MAX) * 50);
  const lean = leanLabel(upProbability, downProbability);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 220 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 56,
            textAlign: "right",
            fontSize: 12,
            fontVariantNumeric: "tabular-nums",
            color: "var(--series-red)",
          }}
        >
          {(downProbability * 100).toFixed(1)}%
        </span>
        <div style={{ position: "relative", flex: 1, height: 10, background: "var(--gridline)", borderRadius: 5 }}>
          {/* Center zero-line */}
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: -2,
              bottom: -2,
              width: 1,
              background: "var(--text-muted)",
            }}
          />
          <div
            style={{
              position: "absolute",
              right: "50%",
              top: 0,
              bottom: 0,
              width: `${downWidthPct}%`,
              background: "var(--series-red)",
              borderRadius: "5px 0 0 5px",
            }}
          />
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: 0,
              bottom: 0,
              width: `${upWidthPct}%`,
              background: "var(--series-blue)",
              borderRadius: "0 5px 5px 0",
            }}
          />
        </div>
        <span
          style={{
            width: 56,
            textAlign: "left",
            fontSize: 12,
            fontVariantNumeric: "tabular-nums",
            color: "var(--series-blue)",
          }}
        >
          {(upProbability * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{ display: "flex", justifyContent: "center", fontSize: 11, color: lean.color }}>{lean.text}</div>
    </div>
  );
}
