# AI Prompts & Critique Log

## Week 08 · Monday — Time Series Analysis

---

### Prompt 1 — Overall Structure & Approach

**Prompt used:**
> "Build a comprehensive time series analysis covering all 7 sub-steps for the Week 08 Monday assignment. Use the Brazilian E-Commerce (Olist) dataset for e-commerce sales forecasting and the Pump Sensor dataset for equipment failure prediction. Create modular, well-documented code with named constants, error handling, and business-friendly explanations."

**Critique:**
- The AI provided a well-structured single-file script covering all sub-steps.
- I modified the data loading paths to use relative paths instead of hardcoded absolute paths.
- The AI correctly identified that sensor_15 is 100% null and should be dropped.
- I verified the ADF test interpretation is correct: p-value < 0.05 → stationary.
- I confirmed the temporal train/test split is used (not random) — critical for time-series.

---

### Prompt 2 — ARIMA Parameter Selection

**Prompt used:**
> "How should I select ARIMA(p,d,q) parameters based on ACF and PACF analysis for the e-commerce daily revenue series?"

**Critique:**
- AI suggested d=1 based on ADF test showing non-stationarity, which is correct.
- PACF and ACF analysis was used for p and q — standard Box-Jenkins methodology.
- I added a fallback mechanism: if ARIMA(2,1,2) fails to converge, it falls back to ARIMA(1,1,1).
- The MAPE metric choice explanation was accurate — inventory teams do think in percentages.

---

### Prompt 3 — Sensor Failure Labeling Strategy

**Prompt used:**
> "With only 7 BROKEN events and 14,477 RECOVERING events in the sensor data, how should I create a binary target for 24-hour failure prediction?"

**Critique:**
- AI suggested grouping consecutive non-NORMAL rows into failure episodes, which is the correct approach.
- Labeling the 1440 minutes (24h) before each episode onset as positive is sound.
- Only NORMAL rows are retained for training, which prevents data leakage from using active-failure data.
- I added the `class_weight='balanced'` parameter to handle remaining class imbalance — the AI initially suggested this but I verified it's appropriate for this ratio.
- One concern: if all failure episodes fall in the training period, the test set has no positives. The code handles this gracefully.

---

### Prompt 4 — Cost Matrix Analysis

**Prompt used:**
> "Compare a simple threshold rule against the ML model using a cost matrix where missed failures cost $50,000 and false alarms cost $500."

**Critique:**
- The cost ratio (100:1) could vary in practice — I documented this as a named constant so it's easy to adjust.
- The AI correctly tested both "above" and "below" threshold directions, since failures could manifest as sensor readings going up or down.
- The analysis of when rules outperform ML vs. fail is well-reasoned and matches my understanding from the curriculum.
- For the fleet-wide analysis, the daily cost calculation correctly scaled test-set rates to 100K sensors.

---

### Prompt 5 — F1 vs Cost-Optimal Threshold Discussion

**Prompt used:**
> "Explain why the F1-optimal threshold might differ from the cost-optimal threshold in production."

**Critique:**
- AI explanation was correct: F1 treats FP and FN symmetrically, but real costs are asymmetric.
- The conclusion that F1 should not be the production optimization target when costs are asymmetric is textbook-accurate.
- I kept this explanation in the script output as it directly answers the assignment question.
- No modifications needed — the reasoning is sound.

---

### General Critique

**What the AI got right:**
1. Temporal train/test split throughout (never random)
2. Named constants instead of magic numbers
3. Modular functions (2+ per sub-step)
4. Defensive error handling (try/except, fallbacks)
5. Business-friendly metric interpretations

**What I changed:**
1. Added relative path configuration instead of hardcoded paths
2. Added explicit documentation for the forward-fill strategy
3. Increased Random Forest `n_estimators` from 100 to 200 for better stability
4. Added the maintenance-team traffic-light presentation
5. Refined the Prophet `changepoint_prior_scale` to 0.05 for regularization
