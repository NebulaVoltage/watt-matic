# GridBalance — Phase 4 Workstream B: Test Partition Disclosure & Methodological Limitations
**Document ID:** `GB-P4-WB-TEST-DISCLOSURE`  
**Research Stage:** Phase 4 — Research Hardening, Reproducibility & Scientific Verification  
**Author:** Antigravity Senior ML Research Scientist, Independent Benchmark Auditor  
**Date:** October 31, 2026  
**Status:** Methodological Disclosure Complete — Governance Standards Enforced  

---

## 1. Formal Scientific Disclosure

In accordance with rigorous empirical standards, this document issues a formal methodological disclosure regarding the status of the held-out test partitions utilized throughout the GridBalance project:

### **FORMAL CLASSIFICATION: HISTORICALLY EXAMINED HELD-OUT BENCHMARKS**

> [!WARNING]
> The historical SGCC test partition ($N = 6,356$ customers) and the UCI test partition ($N = 4,416$ hours, H2 2014) **have been repeatedly evaluated during earlier exploratory research phases (Phases 1 through 3)**.
> 
> Therefore, within all GridBalance documentation, research reports, academic manuscripts, and model cards:
> * They **MUST NOT** be described as "virgin," "untouched," "unseen," or "independently sealed" test sets.
> * They **MUST BE REFERRED TO STRICTLY** as **Historically Examined Held-Out Benchmarks**.

---

## 2. Research Limitations Created by Historical Inspection

When a held-out test partition is evaluated multiple times across experimental iterations, the statistical guarantee of test set independence decays:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TEST SET CONTAMINATION RISK MATRIX                                   │
├──────────────────────────────┬──────────────────┬──────────────────────────────────────────────────────┤
│ Methodological Risk          │ Severity Level   │ Impact on Research Claims                            │
├──────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────┤
│ **Information Leakage**      │ **LOW (Zero)**   │ Feature transformers and imputers were fitted        │
│                              │                  │ strictly on training data; no direct data leakage.   │
├──────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────┤
│ **Implicit Hyperparameter**  │ **MEDIUM**       │ Model selection (e.g., stopping rounds, feature      │
│ **Selection Bias**           │                  │ counts) was indirectly guided by knowing which model │
│                              │                  │ scored well on the test partition.                   │
├──────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────┤
│ **Threshold Snooping**       │ **LOW**          │ Optimal operational thresholds (e.g., t=0.50) were   │
│                              │                  │ derived strictly on Validation data (N=6,355).       │
├──────────────────────────────┼──────────────────┼──────────────────────────────────────────────────────┤
│ **Generalization Inflation** │ **MEDIUM**       │ Test scores (e.g., F1=0.4479, MAE=4,440.05 kWh) may  │
│                              │                  │ reflect optimistic bias relative to truly unseen     │
│                              │                  │ future distribution shifts.                          │
└──────────────────────────────┴──────────────────┴──────────────────────────────────────────────────────┘
```

---

## 3. Strict Candidate Selection Protocol

To mitigate the risk of test set snooping in Phase 4 and all future phases:
1. **Decision Rules Based on CV & Validation Only:** Model candidates (e.g., Candidate B vs Candidate C vs Candidate D) must be selected **strictly based on 5-fold Cross-Validation and the frozen Validation partition ($N=6,355$ for SGCC; H1 2014 for UCI)**.
2. **No Test Set Optimization:** Test partition metrics are reported purely for historical continuity and backward reconciliation. They are never used as objective functions or stopping criteria.
3. **Disclosure in Model Cards:** All public model documentation must explicitly state that test metrics reflect historically examined test partitions.

---

## 4. Requirements for External Validation

To validate the research claims of GridBalance under true out-of-distribution and prospective operating conditions, the following external datasets are required:

1. **SGCC Meter Tampering Branch:**
   * **Prospective Temporal Validation:** An unreleased post-2016 partition from the State Grid Corporation of China across the same meter topology.
   * **Geographic Generalization:** Smart meter consumption telemetry from alternative distribution utilities (e.g., UK Low Carbon London dataset, Irish CER Smart Metering project) to assess whether tampering signatures transfer across climate zones and consumption cultures.
2. **UCI Load Forecasting Branch:**
   * **Prospective Chronological Validation:** Telemetry from the Portuguese transmission grid or ENTSO-E covering 2015–2018 to evaluate long-term macroeconomic load shifts and renewable penetration effects.
   * **Adversarial / Extreme Weather Evaluation:** Grid load data during unprecedented climate events (e.g., extreme heatwaves, polar vortexes) where decision tree extrapolation is challenged.

---
*Report certified by Antigravity Machine Learning Quality Assurance Suite.*
