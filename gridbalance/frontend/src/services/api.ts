const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

export interface SingleMeterResult {
  meter_id: string;
  prediction: 'potential_tampering' | 'normal';
  probability: number;
  threshold: number;
  risk_level: 'Low' | 'Moderate' | 'High' | 'Very High';
  data_quality: {
    observation_count: number;
    missing_count: number;
    missing_ratio: number;
    longest_missing_streak: number;
    missing_streak_count: number;
  };
  top_features: { feature: string; value: number; importance: number }[];
  consumption_history: { date: string; consumption: number }[];
}

export interface BatchMeterResult {
  total_meters: number;
  normal_count: number;
  potential_tampering_count: number;
  high_risk_count: number;
  results: {
    meter_id: string;
    prediction: string;
    probability: number;
    risk_level: string;
    observation_count: number;
    missing_count: number;
    missing_ratio: number;
  }[];
}

export interface ForecastResult {
  forecast_horizon: string;
  predicted_load_kwh: number;
  current_load_kwh: number;
  change_from_current_kwh: number;
  change_percentage: number;
  model: string;
  historical_chart: { timestamp: string; load_kwh: number; type: string }[];
  top_features: { feature: string; importance: number }[];
}

export interface HealthStatus {
  status: string;
  sgcc_model: string;
  forecast_model: string;
  timestamp: string;
}

export interface ModelInfoResponse {
  sgcc: {
    model: string;
    features: number;
    threshold: number;
    test_f1: number;
    test_pr_auc: number;
    test_roc_auc: number;
    test_recall: number;
    test_precision: number;
  };
  forecasting: {
    model: string;
    features: number;
    forecast_horizon: string;
    test_mae: number;
    test_rmse: number;
    test_r2: number;
    test_smape: number;
  };
  disclaimer: string;
}

export const fetchHealth = async (): Promise<HealthStatus> => {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error('API server unavailable');
  return res.json();
};

export const fetchModelInfo = async (): Promise<ModelInfoResponse> => {
  const res = await fetch(`${API_BASE_URL}/models`);
  if (!res.ok) throw new Error('Failed to fetch model metadata');
  return res.json();
};

export const fetchDashboardSummary = async () => {
  const res = await fetch(`${API_BASE_URL}/dashboard/summary`);
  if (!res.ok) throw new Error('Failed to fetch dashboard summary');
  return res.json();
};

export const analyzeSingleMeter = async (meter_id: string, readings?: { date: string; consumption: number }[]): Promise<SingleMeterResult> => {
  const res = await fetch(`${API_BASE_URL}/detection/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ meter_id, readings })
  });
  if (!res.ok) throw new Error('Meter analysis failed');
  return res.json();
};

export const analyzeBatchMeters = async (file: File): Promise<BatchMeterResult> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/detection/batch`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error('Batch processing failed');
  return res.json();
};

export const forecastNextHour = async (history?: { timestamp: string; load_kwh: number }[]): Promise<ForecastResult> => {
  const res = await fetch(`${API_BASE_URL}/forecast/next-hour`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ history })
  });
  if (!res.ok) throw new Error('Load forecasting failed');
  return res.json();
};

export const forecastFromCSV = async (file: File): Promise<ForecastResult> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE_URL}/forecast/upload-csv`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) throw new Error('Forecast CSV processing failed');
  return res.json();
};
