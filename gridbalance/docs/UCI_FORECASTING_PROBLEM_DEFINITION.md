# UCI Electricity Next-Hour Forecasting Problem Definition

## 1. Mathematical Formulation
Given a time series of hourly system load observations $y_1, y_2, \dots, y_t$, the objective is to predict the next-hour load:

$$\hat{y}_{t+1} = f(\mathbf{X}_{\le t})$$

where $\mathbf{X}_{\le t}$ is a feature vector derived strictly from historical observations $y_{\tau}$ where $\tau \le t$, alongside deterministic temporal variables (e.g., hour, day of week) known at time $t$.

## 2. Temporal Correctness & Leakage Prevention Rules
1. **Causal Features Only**: All lag features ($y_t, y_{t-1}, \dots$) and rolling window statistics (e.g., `rolling_mean_24`) are strictly computed using past values via `shift(1)`.
2. **No Future Lookahead**: No centered rolling windows, future interpolation, or global target normalization are permitted.
3. **Chronological Splitting**: Train, Validation, and Test splits are partitioned chronologically. Shuffling is strictly prohibited.
