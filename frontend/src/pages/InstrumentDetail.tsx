import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { getPredictionDetail } from "../api/client";
import type { Horizon, PredictionDetail } from "../api/types";
import { ShapChart } from "../components/ShapChart";

const HORIZONS: Horizon[] = [5, 20];

export function InstrumentDetail() {
  const { ticker = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const horizon = (Number(searchParams.get("horizon")) || 5) as Horizon;

  const [detail, setDetail] = useState<PredictionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    getPredictionDetail(ticker, horizon)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker, horizon]);

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: "32px 24px" }}>
      <Link to="/" style={{ fontSize: 13, color: "var(--text-secondary)" }}>
        ← Back to dashboard
      </Link>

      <div style={{ display: "flex", gap: 8, margin: "16px 0" }}>
        {HORIZONS.map((h) => (
          <button
            key={h}
            onClick={() => setSearchParams({ horizon: String(h) })}
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
      {!error && !detail && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}

      {detail && (
        <>
          <header style={{ marginBottom: 24 }}>
            <h1 style={{ fontSize: 22, marginBottom: 4 }}>{detail.name}</h1>
            <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0 }}>
              {detail.ticker} · {detail.sector ?? "—"}
            </p>
          </header>

          <div style={{ display: "flex", gap: 16, marginBottom: 8, flexWrap: "wrap" }}>
            <div
              style={{
                flex: "1 1 220px",
                background: "var(--surface-card)",
                border: "1px solid var(--border)",
                borderLeft: "3px solid var(--series-red)",
                borderRadius: 10,
                padding: 20,
              }}
            >
              <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                P(move down {">"}1.5&sigma;, next {detail.horizon}d)
              </div>
              <div
                style={{
                  fontSize: 36,
                  fontWeight: 700,
                  fontVariantNumeric: "tabular-nums",
                  color: "var(--series-red)",
                }}
              >
                {(detail.down_probability * 100).toFixed(1)}%
              </div>
            </div>
            <div
              style={{
                flex: "1 1 220px",
                background: "var(--surface-card)",
                border: "1px solid var(--border)",
                borderLeft: "3px solid var(--series-blue)",
                borderRadius: 10,
                padding: 20,
              }}
            >
              <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                P(move up {">"}1.5&sigma;, next {detail.horizon}d)
              </div>
              <div
                style={{
                  fontSize: 36,
                  fontWeight: 700,
                  fontVariantNumeric: "tabular-nums",
                  color: "var(--series-blue)",
                }}
              >
                {(detail.up_probability * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 24 }}>
            As of {detail.as_of_date.slice(0, 10)}. These are two independent models — both can be elevated
            at once (a calm market lowers the bar for a big move in either direction; see the methodology
            page), so a wide gap between them is a more meaningful "lean" than either number alone.
          </div>

          <h2 style={{ fontSize: 16, marginBottom: 4, color: "var(--series-blue)" }}>Why the up-move probability</h2>
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 0 }}>
            The features that pushed this prediction up or down the most, per{" "}
            <a href="https://shap.readthedocs.io/" target="_blank" rel="noreferrer">
              SHAP
            </a>{" "}
            attribution. See the methodology page for an important caveat: a meaningful part of this
            model's edge reflects current-volatility regime detection, not pure directional signal.
          </p>
          <ShapChart features={detail.up_top_features} />

          <h2 style={{ fontSize: 16, marginBottom: 4, marginTop: 32, color: "var(--series-red)" }}>
            Why the down-move probability
          </h2>
          <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 0 }}>
            The independent downside model's own SHAP attribution — not derived from the up-model above.
          </p>
          <ShapChart features={detail.down_top_features} />
        </>
      )}
    </div>
  );
}
