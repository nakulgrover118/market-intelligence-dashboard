import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { FeatureContribution } from "../api/types";

// SHAP contribution is a polarity measure (pushes the probability up vs
// down), not a plain magnitude — the diverging blue<->red pair is the
// correct color job here, not a single sequential hue.
const POSITIVE_COLOR = "var(--series-blue)";
const NEGATIVE_COLOR = "var(--series-red)";

function formatFeatureName(raw: string): string {
  if (raw.startsWith("sector__sector_")) {
    return `Sector: ${raw.replace("sector__sector_", "")}`;
  }
  return raw.replace("numeric__", "").replace(/_/g, " ");
}

interface TooltipPayload {
  active?: boolean;
  payload?: { payload: FeatureContribution & { label: string } }[];
}

function ShapTooltip({ active, payload }: TooltipPayload) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div
      style={{
        background: "var(--surface-card)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "8px 12px",
        fontSize: 13,
        color: "var(--text-primary)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{row.label}</div>
      <div style={{ color: "var(--text-secondary)" }}>Value: {row.feature_value.toFixed(3)}</div>
      <div style={{ color: row.shap_value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR }}>
        {row.shap_value >= 0 ? "Increases" : "Decreases"} probability by {Math.abs(row.shap_value).toFixed(3)}
      </div>
    </div>
  );
}

export function ShapChart({ features }: { features: FeatureContribution[] }) {
  const data = [...features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .reverse() // Recharts vertical bar charts plot top-to-bottom in array order
    .map((f) => ({ ...f, label: formatFeatureName(f.feature) }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={Math.max(280, data.length * 32)}>
        <BarChart data={data} layout="vertical" margin={{ left: 24, right: 24, top: 8, bottom: 8 }}>
          <XAxis
            type="number"
            stroke="var(--text-muted)"
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={160}
            stroke="var(--text-muted)"
            tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine x={0} stroke="var(--baseline, var(--gridline))" />
          <Tooltip content={<ShapTooltip />} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
          <Bar dataKey="shap_value" radius={2}>
            {data.map((entry) => (
              <Cell key={entry.feature} fill={entry.shap_value >= 0 ? POSITIVE_COLOR : NEGATIVE_COLOR} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ display: "flex", gap: 16, fontSize: 12, color: "var(--text-secondary)", marginTop: 4 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: POSITIVE_COLOR, display: "inline-block" }} />
          Increases probability
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: NEGATIVE_COLOR, display: "inline-block" }} />
          Decreases probability
        </span>
      </div>
    </div>
  );
}
