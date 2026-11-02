# UCI Electricity Load Forecasting: EMA Feature Inference & Stateless Serving Audit
**Document ID:** `GB-AUDIT-P2-UCI-EMA`  
**Research Branch:** Branch 2 — UCI Electricity Load Diagrams Next-Hour Forecasting  
**Status:** Completed & Peer-Reviewed  
**Author:** Antigravity Senior ML Research Auditor  
**Date:** October 31, 2026  

---

## 1. Executive Summary

During Phase 2 candidate evaluation for the UCI load forecasting branch, an expanded 33-feature LightGBM model was proposed as a potential replacement for the frozen 29-feature XGBoost model. The 4 newly added features were: `load_ema_6`, `load_ema_24`, `load_rolling_min_24`, and `load_rolling_max_24`. 

While offline cross-validation reported an incremental improvement with these features, the Phase 1 Research Audit flagged an acute architectural risk: **Exponential Moving Average (EMA) is a recursive Infinite Impulse Response (IIR) filter. Can an EMA feature be computed correctly within a stateless REST API request without introducing severe initialization warmup distortion?**

Phase 2 Experiment 6 performed an exhaustive mathematical and empirical audit of EMA warmup distortion across varying finite-history windows ($24\text{h}$ to $336\text{h}$) against the exact ground-truth training feature matrix.

### Core Audit Findings
- **Catastrophic 24-Hour Warmup Bias:** When an inference client submits the standard 24-hour historical payload, dynamically computing `load_ema_24` produces an average feature error of **9,802.85 kWh** and a maximum error of **22,286.33 kWh**. Given that the overall model forecast error is only $\sim 4,400\text{ kWh}$, an input distortion of $9,800\text{ kWh}$ completely destabilizes the model at inference time.
- **Warmup Convergence Requires 168 Hours (7 Days):** The warmup error decays exponentially. It requires a historical window of at least **168 consecutive hours (7 full days)** to suppress `load_ema_24` error below $0.15\text{ kWh}$ ($0.06\text{ kWh}$ mean). Zero distortion ($<0.001\text{ kWh}$) requires 336 hours (14 days).
- **Stateless REST Contract Breach:** The existing FastAPI endpoint contract (`/api/v1/predict/forecast`) is designed as a lightweight, stateless service expecting the last 24 hourly readings. Serving the 33-feature EMA model on this 24-hour contract violates feature consistency and invalidates model accuracy.
- **Engineering Verdict:** **DO NOT DEPLOY the 33-feature EMA candidate on the 24-hour stateless API.** The frozen 29-feature model (or a drop-in 29-feature LightGBM direct model) contains exclusively Finite Impulse Response (FIR) features with a strict maximum lag of 24 hours, guaranteeing $100\%$ mathematical fidelity and zero warmup distortion under stateless serving.

---

## 2. Mathematical Etiology of EMA Warmup Distortion

The Exponential Moving Average of a time series $y_t$ with span parameter $S$ is defined recursively:

$$S_t = \alpha y_t + (1 - \alpha) S_{t-1}$$

where the smoothing coefficient is $\alpha = \frac{2}{S + 1}$. 

Unrolling the recursion backwards to time $t - K$:

$$S_t = \alpha \sum_{k=0}^{K-1} (1 - \alpha)^k y_{t-k} + (1 - \alpha)^K S_{t-K}$$

In training, $S_t$ is computed over the contiguous 3-year history ($K \to \infty$), so the initial condition term $(1 - \alpha)^K S_{t-K} \to 0$.

However, in a stateless REST API call, the client supplies only a finite window of $H$ historical observations $\{y_{t-H+1}, \dots, y_t\}$. The API must initialize the recursion at $k = t - H + 1$ using an arbitrary initial value (typically the first observation $S_0 = y_{t-H+1}$).

The resulting truncation error $\epsilon_t(H) = |S_t^{\text{stateless}} - S_t^{\text{ground\_truth}}|$ is governed by:

$$\epsilon_t(H) \approx (1 - \alpha)^H \cdot |y_{t-H+1} - S_{t-H+1}^{\text{true}}|$$

For `load_ema_24` (span $S=24$):
$$\alpha = \frac{2}{24 + 1} = 0.08 \implies (1 - \alpha) = 0.92$$

When $H = 24$ hours:
$$(1 - \alpha)^{24} = (0.92)^{24} \approx 0.1352 \quad (13.52\%\text{ residual bias})$$

Because aggregate grid load fluctuates between $100,000$ and $400,000\text{ kWh}$, an initial step discrepancy $|y_0 - S_0|$ easily reaches $150,000\text{ kWh}$, injecting a persistent distortion of $150,000 \times 0.1352 \approx 20,280\text{ kWh}$ into the model's feature vector!

---

## 3. Empirical Warmup Audit Across History Windows

To measure the real-world error, we evaluated 4,416 hourly test timestamps in H2 2014, computing `load_ema_6` and `load_ema_24` from finite history windows $H \in \{24, 48, 72, 120, 168, 336\}$ hours and comparing them against the authoritative ground-truth training feature matrix:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    EMA INFERENCE WARMUP ERROR AS A FUNCTION OF HISTORY WINDOW                     │
├──────────────┬─────────────────────┬──────────────────────┬────────────────────┬───────────────────┤
│ Window (Hrs) │ EMA-6 Max Error kWh │ EMA-6 Mean Error kWh │EMA-24 Max Error kWh│EMA-24 Mean Error  │
├──────────────┼─────────────────────┼──────────────────────┼────────────────────┼───────────────────┤
│   24 hours   │      43.12 kWh      │      15.09 kWh       │    22,286.33 kWh   │    9,802.85 kWh   │
│   48 hours   │       0.01 kWh      │       0.005 kWh      │     3,244.22 kWh   │    1,339.21 kWh   │
│   72 hours   │       0.00 kWh      │       0.000 kWh      │       430.40 kWh   │      180.19 kWh   │
│  120 hours   │       0.00 kWh      │       0.000 kWh      │         7.39 kWh   │        3.30 kWh   │
│  168 hours   │       0.00 kWh      │       0.000 kWh      │         0.15 kWh   │        0.06 kWh   │
│  336 hours   │       0.00 kWh      │       0.000 kWh      │         0.00 kWh   │        0.00 kWh   │
└──────────────┴─────────────────────┴──────────────────────┴────────────────────┴───────────────────┘
```

![EMA Warmup Convergence](file:///D:/watt-matic/gridbalance/graphs/optimization_v2/phase2/uci_ema_warmup_convergence.png)

### Key Observations:
1. **EMA-6 Converges Fast:** Because $\alpha = \frac{2}{7} = 0.2857$, $(1 - \alpha)^{24} = (0.7143)^{24} \approx 0.0003$. EMA-6 warmup distortion is virtually gone within 48 hours.
2. **EMA-24 Requires 7 Days:** Because of its long memory ($1 - \alpha = 0.92$), EMA-24 retains noticeable bias even after 72 hours ($430\text{ kWh}$ max error). It requires a full **168 hours (7 days)** to drop below $0.15\text{ kWh}$ maximum error.

---

## 4. Serving Architecture Comparison

To operationalize the 33-feature model, three architectural alternatives were investigated:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      SERVING ARCHITECTURES FOR RECURSIVE EMA FEATURES                           │
├──────────────────────┬──────────────────────┬─────────────────────────┬─────────────────────────┤
│ Architectural Design │ Implementation Flow  │ Advantages              │ Disadvantages           │
├──────────────────────┼──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 1. 24h Finite Window │ Client sends 24h     │ Maintains small payload │ CATASTROPHIC BIAS       │
│    (Current Setup)   │ API initializes S_0  │ Low network latency     │ 9,800 kWh feature error │
├──────────────────────┼──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 2. 168h Finite Window│ Client sends 168h    │ Fully stateless         │ 7x larger payload       │
│    (7-Day History)   │ Warmup error < 0.1kWh│ No DB state required    │ Breaks 24h API contract │
├──────────────────────┼──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 3. Explicit State In │ Client sends y_t +   │ Tiny payload            │ Shifts state burden to  │
│    Payload           │ ema_6 + ema_24 state │ Exact math precision    │ client; brittle API     │
├──────────────────────┼──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 4. Server-Side State │ Redis / PostgreSQL   │ Transparent to client   │ Statefulness destroys   │
│    (Cached EMA)      │ updates S_t per hour │ Exact math precision    │ horizontal autoscaling  │
├──────────────────────┼──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 5. Strict FIR Only   │ 29 Features          │ 100% Stateless & exact  │ Rejects candidate EMA   │
│    (FROZEN BASELINE) │ Lags 1..24, Roll 24h │ Exactly 24h history     │ features                │
└──────────────────────┴──────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 5. Architectural Verdict & Deployment Decision

1. **Rejection of 33-Feature Model for Standard 24h Endpoint:**
   - Under the existing stateless REST contract where clients submit the past 24 hours of load data, deploying the 33-feature EMA model would introduce a **9,802 kWh average systematic error** into `load_ema_24`. This invalidates the model in production.
2. **Preservation of the 29-Feature Contract:**
   - The 29-feature feature specification (composed strictly of instantaneous lags $1..24$, rolling mean $6, 12, 24$, and rolling std $6, 24$) has a **strict finite impulse response (FIR)** memory of exactly 24 steps.
   - For any 24-element vector $[y_{t-23}, \dots, y_t]$, all 29 features can be reconstructed with **zero mathematical distortion, zero warmup bias, and complete causal determinism**.
3. **Drop-in LightGBM 29-Feature Model as Alternative:**
   - If the engineering team desires the faster inference speed of LightGBM ($\sim 2.8\text{ ms}$ vs $\sim 6.5\text{ ms}$ for XGBoost), a **29-feature LightGBM direct model** can be deployed as an exact drop-in replacement without altering the API contract or risking warmup bias.

---
*Report Certified by Antigravity Machine Learning Audit Suite.*
