export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'offline';
  sgcc_model: string;
  forecast_model: string;
  timestamp: string;
}

export interface ModelMetrics {
  model_name: string;
  feature_count: number;
  decision_threshold?: number;
  test_f1?: number;
  test_pr_auc?: number;
  test_roc_auc?: number;
  test_recall?: number;
  test_precision?: number;
  forecast_horizon?: string;
  test_mae_kwh?: number;
  test_rmse_kwh?: number;
  test_r2?: number;
  test_smape_pct?: number;
  sealed_test_size?: number;
  disclaimer: string;
}

export interface ModelInfoResponse {
  sgcc: ModelMetrics;
  forecasting: ModelMetrics;
  disclaimer: string;
}

export interface TopFeature {
  feature: string;
  value: number;
  importance: number;
  contribution: number;
}

export interface DataQuality {
  observation_count: number;
  missing_count: number;
  missing_ratio: number;
  longest_missing_streak: number;
  missing_streak_count: number;
}

export interface ConsumptionReading {
  date: string;
  consumption: number;
}

export interface SingleMeterResponse {
  meter_id: string;
  prediction: 'potential_tampering' | 'normal' | string;
  probability: number;
  threshold: number;
  risk_level: 'High' | 'Medium' | 'Low' | string;
  data_quality: DataQuality;
  top_features: TopFeature[];
  consumption_history: ConsumptionReading[];
}

export interface BatchMeterResponse {
  total_meters: number;
  flagged_meters: number;
  normal_meters: number;
  results: SingleMeterResponse[];
}

export interface LoadPointInput {
  timestamp: string;
  load_kwh: number;
}

export interface ForecastInput {
  history: LoadPointInput[];
}

export interface ForecastResponse {
  forecast_horizon: string;
  target_timestamp: string;
  predicted_load_kwh: number;
  model_info: Record<string, any>;
}

const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        if (typeof errJson.detail === 'string') {
          errorDetail = errJson.detail;
        } else if (Array.isArray(errJson.detail)) {
          errorDetail = errJson.detail.map((e: any) => e.msg || JSON.stringify(e)).join('; ');
        } else {
          errorDetail = JSON.stringify(errJson.detail);
        }
      }
    } catch {
      // Ignore JSON parse failure on error body
    }
    throw new ApiError(errorDetail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async checkHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, {
        headers: { 'Accept': 'application/json' }
      });
      return await handleResponse<HealthResponse>(res);
    } catch (err: any) {
      return {
        status: 'offline',
        sgcc_model: 'not_loaded',
        forecast_model: 'not_loaded',
        timestamp: new Date().toISOString()
      };
    }
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetch(`${API_BASE_URL}/models`, {
      headers: { 'Accept': 'application/json' }
    });
    return handleResponse<ModelInfoResponse>(res);
  },

  async analyzeMeter(payload: { meter_id: string; dates: string[]; consumption: number[] }): Promise<SingleMeterResponse> {
    const res = await fetch(`${API_BASE_URL}/detection/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload)
    });
    return handleResponse<SingleMeterResponse>(res);
  },

  async analyzeBatchCsv(file: File): Promise<BatchMeterResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE_URL}/detection/batch`, {
      method: 'POST',
      body: formData
    });
    return handleResponse<BatchMeterResponse>(res);
  },

  async forecastNextHour(history: LoadPointInput[]): Promise<ForecastResponse> {
    const res = await fetch(`${API_BASE_URL}/forecast/next-hour`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ history })
    });
    return handleResponse<ForecastResponse>(res);
  }
};
