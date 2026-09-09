import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getLatestPredictions } from "../api/client";
import type { Horizon, Prediction } from "../api/types";
import { DirectionBar } from "../components/DirectionBar";

const HORIZONS: Horizon[] = [5, 20];

export function Dashboard() {
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;
    setPredictions(null);
    setError(null);
    getLatestPredictions(horizon)
      .then((data) => {
        if (!cancelled) setPredictions(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [horizon]);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 24px" }}>
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, marginBottom: 4 }}>Market Intelligence Dashboard</h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 14, margin: 0 }}>
          Two independent probabilities — a large ({">"}1.5&sigma;-scaled) move up, and a large move down —
          over the next {horizon} trading days, for NSE stocks, indices, and gold/silver ETF proxies. Both
          numbers can be elevated at once (a calm market makes a big move easier in <em>either</em> direction
          — see the methodology page). These are probabilities, not BUY/SELL signals.
        </p>
      </header>

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {HORIZONS.map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              border: "1px solid var(--border)",
              background: h === horizon ? "var(--series-blue)" : "var(--surface-card)",
              color: h === horizon ? "#fff" : "var(--text-primary)",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            {h}-day
          </button>
        ))}
      </div>

      {error && <p style={{ color: "var(--status-critical)" }}>Failed to load: {error}</p>}
      {!error && !predictions && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}

      {predictions && (
        <table>
          <thead>
            <tr style={{ textAlign: "left", borderBottom: "1px solid var(--gridline)" }}>
              <th style={headerStyle}>Instrument</th>
              <th style={headerStyle}>Sector</th>
              <th style={headerStyle}>Down / Up probability</th>
              <th style={headerStyle}>As of</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((row) => (
              <tr
                key={row.ticker}
                onClick={() => navigate(`/instrument/${encodeURIComponent(row.ticker)}?horizon=${horizon}`)}
                style={{ borderBottom: "1px solid var(--gridline)", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--surface-card)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <td style={cellStyle}>
                  <div style={{ fontWeight: 600 }}>{row.name}</div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{row.ticker}</div>
                </td>
                <td style={{ ...cellStyle, color: "var(--text-secondary)" }}>{row.sector ?? "—"}</td>
                <td style={cellStyle}>
                  <DirectionBar upProbability={row.up_probability} downProbability={row.down_probability} />
                </td>
                <td style={{ ...cellStyle, color: "var(--text-muted)", fontSize: 12 }}>
                  {row.as_of_date.slice(0, 10)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const headerStyle: React.CSSProperties = {
  padding: "8px 12px",
  fontSize: 12,
  color: "var(--text-muted)",
  fontWeight: 500,
  textTransform: "uppercase",
  letterSpacing: 0.4,
};

const cellStyle: React.CSSProperties = {
  padding: "10px 12px",
  fontSize: 14,
};
