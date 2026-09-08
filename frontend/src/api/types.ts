export interface Instrument {
  ticker: string;
  name: string;
  sector: string | null;
  asset_class: string;
}

export interface Prediction {
  ticker: string;
  name: string;
  sector: string | null;
  horizon: number;
  probability: number;
  as_of_date: string;
}

export interface FeatureContribution {
  feature: string;
  feature_value: number;
  shap_value: number;
}

export interface PredictionDetail extends Prediction {
  top_features: FeatureContribution[];
}

export type Horizon = 5 | 20;
