import type { Horizon, Instrument, Prediction, PredictionDetail } from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getInstruments(): Promise<Instrument[]> {
  return fetchJson<Instrument[]>("/instruments");
}

export function getLatestPredictions(horizon: Horizon): Promise<Prediction[]> {
  return fetchJson<Prediction[]>(`/predictions/latest?horizon=${horizon}`);
}

export function getPredictionDetail(ticker: string, horizon: Horizon): Promise<PredictionDetail> {
  return fetchJson<PredictionDetail>(`/predictions/${encodeURIComponent(ticker)}?horizon=${horizon}`);
}
