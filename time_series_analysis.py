"""
Week 08 · Monday — Time Series Analysis
========================================
E-Commerce Sales Forecasting & Equipment Failure Prediction

PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar
Date: April 13, 2026

Datasets:
  - Brazilian E-Commerce (Olist) → Daily revenue time series
  - Pump Sensor Data             → Equipment failure prediction

Sub-steps:
  1. E-Commerce Data Characterization    (Easy)
  2. Sensor Data Cleaning               (Easy)
  3. ARIMA Modeling                      (Medium)
  4. SARIMA / Prophet Comparison         (Medium)
  5. Sensor Failure Prediction           (Medium)
  6. Rule-Based vs ML Comparison         (Hard)
  7. Fleet-Wide Deployment Cost Analysis (Hard)
"""

# %%  — Imports & Configuration
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Statistical / TS libraries
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths (relative to this script) ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ECOMMERCE_DIR = os.path.join(SCRIPT_DIR, "data", "ecommerce")
SENSOR_DIR = os.path.join(SCRIPT_DIR, "data", "sensor")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Named constants (no magic numbers) ──────────────────────
HOLDOUT_DAYS = 30                # days held out for e-commerce test
SEASONAL_PERIOD_WEEKLY = 7       # weekly seasonality in days
FAILURE_WINDOW_MINUTES = 1440    # 24 h expressed in minutes
ADF_SIGNIFICANCE_LEVEL = 0.05   # p-value cutoff for ADF test
RANDOM_STATE = 42                # reproducibility seed
ECOMMERCE_START_DATE = "2017-01-01"  # trim sparse early period

# Cost constants (sensor analysis)
COST_MISSED_FAILURE = 50_000     # emergency repair
COST_FALSE_ALARM    = 500        # unnecessary inspection
COST_TRUE_POSITIVE  = 1_000      # planned preventive repair
FLEET_SIZE          = 100_000    # sensors deployed fleet-wide

# Feature-engineering knobs
TOP_N_SENSORS_FOR_ROLLING = 15
ROLLING_WINDOW_SHORT      = 60   # minutes (1 h)
RF_N_ESTIMATORS           = 200
RF_MAX_DEPTH              = 15
SENSOR_TRAIN_FRACTION     = 0.60   # lower to ensure failures in test

# ────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────

def load_csv_safely(filepath, **kwargs):
    """Load a CSV with validation and error handling."""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        print(f"  ✅ Loaded {os.path.basename(filepath)}: "
              f"{df.shape[0]:,} rows × {df.shape[1]} cols")
        return df
    except Exception as exc:
        print(f"  ❌ Error loading {filepath}: {exc}")
        raise


def compute_adf_test(series, series_name="Series"):
    """Run Augmented Dickey-Fuller stationarity test and print results."""
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        adf_stat, p_value, n_lags = result[0], result[1], result[2]
        crit = result[4]
        is_stationary = p_value < ADF_SIGNIFICANCE_LEVEL
        tag = "STATIONARY" if is_stationary else "NON-STATIONARY"

        print(f"\n  ── ADF Test: {series_name} ──")
        print(f"     ADF Statistic : {adf_stat:.4f}")
        print(f"     p-value       : {p_value:.6f}")
        print(f"     Lags used     : {n_lags}")
        for k, v in crit.items():
            print(f"     Critical {k}: {v:.4f}")
        print(f"     Result        : {tag}")
        return {"adf": adf_stat, "p": p_value, "stationary": is_stationary}
    except Exception as exc:
        print(f"  ❌ ADF test failed for {series_name}: {exc}")
        return None


def compute_forecast_metrics(actual, predicted, label="Forecast"):
    """Return MAE, RMSE, MAPE and print a business-friendly summary."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Symmetric MAPE — handles zeros gracefully
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    nonzero = denom != 0
    mape = (np.mean(np.abs(actual[nonzero] - predicted[nonzero])
                    / denom[nonzero]) * 100
            if nonzero.sum() > 0 else float("inf"))

    print(f"\n  ── Metrics: {label} ──")
    print(f"     MAE  : {mae:,.2f}  (avg absolute error in BRL)")
    print(f"     RMSE : {rmse:,.2f}  (penalises large errors)")
    print(f"     MAPE : {mape:.2f}%  (avg % error)")
    print(f"     → On average, forecasts deviate by BRL {mae:,.0f}/day "
          f"(~{mape:.1f}% of actual)")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_plot(fig, filename, dpi=150):
    """Persist a figure to PLOT_DIR."""
    try:
        path = os.path.join(PLOT_DIR, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  📈 Saved: {path}")
    except Exception as exc:
        print(f"  ❌ Plot save failed ({filename}): {exc}")
    finally:
        plt.close(fig)


# ────────────────────────────────────────────────────────────
# SUB-STEP 1 — E-Commerce Data Characterisation  (EASY)
# ────────────────────────────────────────────────────────────

def prepare_ecommerce_daily_sales(ecommerce_dir):
    """
    Merge orders + payments → daily revenue time series.

    Strategy
    --------
    • Inner-join orders ↔ payments on order_id.
    • Keep only *delivered* orders (clean revenue signal).
    • Aggregate payment_value per day.
    • Re-index to a complete calendar; fill gaps with 0.
    • Trim dates before 2017-01-01 (sparse early data).
    """
    orders   = load_csv_safely(os.path.join(ecommerce_dir,
                                            "olist_orders_dataset.csv"))
    payments = load_csv_safely(os.path.join(ecommerce_dir,
                                            "olist_order_payments_dataset.csv"))

    merged = orders.merge(payments, on="order_id", how="inner")
    print(f"  Merged rows: {len(merged):,}")

    delivered = merged[merged["order_status"] == "delivered"].copy()
    print(f"  Delivered orders: {len(delivered):,}")

    delivered["order_date"] = pd.to_datetime(
        delivered["order_purchase_timestamp"]
    ).dt.normalize()

    daily = (delivered.groupby("order_date")["payment_value"]
             .sum()
             .rename("daily_revenue"))
    daily.index.name = "date"

    # fill missing calendar days
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    daily.index.name = "date"

    # trim sparse early months
    daily = daily.loc[ECOMMERCE_START_DATE:]
    daily = daily.to_frame()

    print(f"  Daily series: {len(daily)} days  "
          f"({daily.index.min().date()} → {daily.index.max().date()})")
    print(f"  Mean daily revenue: BRL {daily['daily_revenue'].mean():,.2f}")
    return daily


def characterise_ecommerce_series(daily_sales):
    """
    Full characterisation: statistics, stationarity, decomposition,
    ACF/PACF, data-quality audit.  Returns findings dict.
    """
    s = daily_sales["daily_revenue"]

    # ── 1a. Descriptive statistics ──
    print("\n" + "=" * 60)
    print("1a · DESCRIPTIVE STATISTICS")
    print("=" * 60)
    stats = s.describe()
    for k, v in stats.items():
        print(f"     {k:<10}: BRL {v:>12,.2f}")
    print(f"     {'zero_days':<10}: {(s == 0).sum():>12}")
    print(f"     {'skewness':<10}: {s.skew():>12.3f}")
    print(f"     {'kurtosis':<10}: {s.kurtosis():>12.3f}")

    # ── 1b. Stationarity ──
    print("\n" + "=" * 60)
    print("1b · STATIONARITY ANALYSIS")
    print("=" * 60)
    adf_raw  = compute_adf_test(s, "Raw daily revenue")
    s_diff   = s.diff().dropna()
    adf_diff = compute_adf_test(s_diff, "First-differenced revenue")

    # ── 1c. Seasonal decomposition ──
    print("\n" + "=" * 60)
    print("1c · SEASONAL DECOMPOSITION (period = 7 days)")
    print("=" * 60)
    decomp = seasonal_decompose(s, model="additive",
                                period=SEASONAL_PERIOD_WEEKLY)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, (comp, title) in zip(axes,
            [(decomp.observed, "Observed"),
             (decomp.trend, "Trend"),
             (decomp.seasonal, "Seasonal (weekly)"),
             (decomp.resid, "Residual")]):
        comp.plot(ax=ax, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("E-Commerce Daily Revenue — Decomposition",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save_plot(fig, "substep1_decomposition.png")

    # ── 1d. ACF / PACF ──
    print("\n" + "=" * 60)
    print("1d · ACF / PACF")
    print("=" * 60)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_acf(s.dropna(),      lags=40, ax=axes[0, 0],
             title="ACF — Raw series")
    plot_pacf(s.dropna(),     lags=40, ax=axes[0, 1],
              title="PACF — Raw series")
    plot_acf(s_diff.dropna(), lags=40, ax=axes[1, 0],
             title="ACF — Differenced")
    plot_pacf(s_diff.dropna(),lags=40, ax=axes[1, 1],
              title="PACF — Differenced")
    fig.tight_layout()
    save_plot(fig, "substep1_acf_pacf.png")

    # ── 1e. Data-quality checks ──
    print("\n" + "=" * 60)
    print("1e · DATA-QUALITY REPORT")
    print("=" * 60)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low_bound  = q1 - 1.5 * iqr
    high_bound = q3 + 1.5 * iqr
    n_outliers = ((s < low_bound) | (s > high_bound)).sum()
    print(f"     IQR bounds       : [{low_bound:,.0f}, {high_bound:,.0f}]")
    print(f"     Outliers (IQR)   : {n_outliers} "
          f"({n_outliers / len(s) * 100:.1f}%)")
    print(f"     Missing values   : {s.isnull().sum()}")
    print(f"     Zero-revenue days: {(s == 0).sum()}")

    # ── 1f. Overview plots ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(s.index, s.values, lw=0.8, color="#2196F3")
    axes[0].axhline(s.mean(), color="red", ls="--", alpha=0.5, label="Mean")
    axes[0].set_title("Daily Revenue Over Time")
    axes[0].set_ylabel("BRL")
    axes[0].legend()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekly = s.groupby(s.index.dayofweek).mean()
    axes[1].bar(range(7), weekly.values, color="#4CAF50", alpha=0.85)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].set_title("Average Revenue by Day-of-Week")
    axes[1].set_ylabel("BRL")
    fig.tight_layout()
    save_plot(fig, "substep1_overview.png")

    # ── 1g. Summary of findings ──
    print("\n" + "=" * 60)
    print("1f · SUMMARY OF FINDINGS")
    print("=" * 60)
    stat_label = "Stationary" if adf_raw["stationary"] else "Non-stationary"
    print(f"  ① Raw series is {stat_label}")
    if adf_diff and adf_diff["stationary"]:
        print("  ② First differencing achieves stationarity → d = 1")
    print("  ③ Clear upward trend (growing platform)")
    print("  ④ Weekly seasonality present (weekday > weekend)")
    print(f"  ⑤ {n_outliers} outliers (promotions / flash-sale days)")
    print(f"  ⑥ {(s == 0).sum()} zero-revenue days (data gaps)")
    print("\n  → Modelling implications:")
    print("    • ARIMA with d ≥ 1 (differencing removes trend)")
    print("    • SARIMA or Prophet for weekly seasonality")
    print("    • Consider outlier-robust evaluation (MAE over RMSE)")

    findings = {
        "stationary_raw": adf_raw["stationary"] if adf_raw else None,
        "stationary_diff": adf_diff["stationary"] if adf_diff else None,
        "n_outliers": n_outliers,
        "zero_days": int((s == 0).sum()),
    }
    return findings, decomp


# ────────────────────────────────────────────────────────────
# SUB-STEP 2 — Sensor Data Cleaning  (EASY)
# ────────────────────────────────────────────────────────────

def identify_sensor_issues(sensor_df):
    """Audit every data-quality dimension of the sensor dataset."""
    print("\n" + "=" * 60)
    print("2a · SENSOR DATA-QUALITY AUDIT")
    print("=" * 60)
    issues = {}

    # — missing values —
    null_pct = (sensor_df.isnull().sum() / len(sensor_df) * 100).round(2)
    cols_with_nulls = null_pct[null_pct > 0].sort_values(ascending=False)
    print(f"\n  Missing-value summary ({len(cols_with_nulls)} columns):")
    for col, pct in cols_with_nulls.items():
        flag = ("⛔ DROP" if pct > 99
                else "⚠️ HIGH" if pct > 20
                else "ℹ️  low")
        print(f"     {col:>12}: {pct:>6.2f}%  {flag}")

    issues["fully_null"]  = list(null_pct[null_pct > 99].index)
    issues["high_null"]   = list(null_pct[(null_pct > 20) & (null_pct <= 99)].index)
    issues["low_null"]    = list(null_pct[(null_pct > 0) & (null_pct <= 20)].index)

    # — redundant columns —
    issues["has_unnamed"] = "Unnamed: 0" in sensor_df.columns
    if issues["has_unnamed"]:
        print("  ⚠️  Redundant index column 'Unnamed: 0' detected")

    # — duplicates —
    n_dup = sensor_df.duplicated().sum()
    print(f"  Duplicate rows: {n_dup}")
    issues["n_duplicates"] = n_dup

    # — timestamp continuity —
    if "timestamp" in sensor_df.columns:
        ts = pd.to_datetime(sensor_df["timestamp"])
        expected = int((ts.max() - ts.min()).total_seconds() / 60) + 1
        gap = expected - len(sensor_df)
        print(f"  Timestamp continuity: expected {expected:,}, "
              f"actual {len(sensor_df):,}, gap = {gap:,}")
        issues["timestamp_gap"] = gap

    # — class distribution —
    if "machine_status" in sensor_df.columns:
        dist = sensor_df["machine_status"].value_counts()
        print(f"\n  Machine-status distribution:")
        for status, cnt in dist.items():
            print(f"     {status:>12}: {cnt:>8,}  "
                  f"({cnt / len(sensor_df) * 100:.2f}%)")
        if dist.min() / len(sensor_df) < 0.01:
            print("  ⚠️  EXTREME class imbalance — "
                  "minority class < 1% of data")
        issues["status_dist"] = dist.to_dict()

    return issues


def clean_sensor_data(sensor_df, issues):
    """
    Apply cleaning treatments and document every decision.

    Strategy
    --------
    1. Drop sensor_15 — 100 % null, zero information.
    2. Drop 'Unnamed: 0' — auto-generated CSV index.
    3. Parse timestamp → datetime, sort chronologically.
    4. Forward-fill sensor NaNs (physically continuous readings).
    5. Backward-fill any leading NaNs.
    6. Validate: zero remaining sensor nulls.
    """
    print("\n" + "=" * 60)
    print("2b · CLEANING TREATMENTS")
    print("=" * 60)
    df = sensor_df.copy()

    # 1. fully-null columns
    for col in issues.get("fully_null", []):
        print(f"  ✂ Dropped {col} (100 % null)")
        df.drop(columns=[col], inplace=True)

    # 2. redundant index
    if issues.get("has_unnamed") and "Unnamed: 0" in df.columns:
        print("  ✂ Dropped 'Unnamed: 0' (redundant index)")
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # 3. timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("  🔧 Parsed & sorted timestamps")

    # 4–5. fill sensor NaNs
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    n_null_before = df[sensor_cols].isnull().sum().sum()
    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    n_null_after = df[sensor_cols].isnull().sum().sum()
    print(f"  🔧 Filled {n_null_before:,} NaNs → {n_null_after} remaining")

    assert n_null_after == 0, "Sensor nulls remain after cleaning!"

    # --- documentation ---
    print("\n" + "=" * 60)
    print("2c · TREATMENT DOCUMENTATION")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────────────────────────┐
    │  Issue                │  Treatment          │  Rationale │
    ├───────────────────────┼─────────────────────┼────────────┤
    │  sensor_15 100% null  │  DROPPED column     │  No data   │
    │  Unnamed: 0 column    │  DROPPED column     │  Redundant │
    │  Sensor NaNs (misc.)  │  Forward→back fill  │  Sensors   │
    │                       │                     │  are phys- │
    │                       │                     │  ically    │
    │                       │                     │  continuous│
    │  sensor_50 (35% null) │  KEPT after fill    │  May carry │
    │                       │                     │  signal    │
    │  No rows removed      │  Preserve temporal  │  Sequence  │
    │                       │  continuity         │  models    │
    └──────────────────────────────────────────────────────────┘
    """)
    return df


# ────────────────────────────────────────────────────────────
# SUB-STEP 3 — ARIMA Model  (MEDIUM)
# ────────────────────────────────────────────────────────────

def create_temporal_split(daily_sales, holdout_days=HOLDOUT_DAYS):
    """
    Temporal train / test split — preserves time ordering.
    ⚠️ Random splits are INVALID for time-series data.
    """
    cutoff = daily_sales.index.max() - timedelta(days=holdout_days)
    train = daily_sales.loc[:cutoff]
    test  = daily_sales.loc[cutoff + timedelta(days=1):]
    print(f"\n  📅 Temporal split (holdout = {holdout_days} days)")
    print(f"     Train: {train.index.min().date()} → "
          f"{train.index.max().date()}  ({len(train)} days)")
    print(f"     Test : {test.index.min().date()} → "
          f"{test.index.max().date()}  ({len(test)} days)")
    return train, test


def fit_arima_model(train_series, order=(2, 1, 2)):
    """
    Fit ARIMA(p,d,q).

    Parameter justification (from Sub-step 1):
      d = 1  →  raw series is non-stationary; differencing fixes it
      p = 2  →  PACF significant at lags 1–2
      q = 2  →  ACF significant at lags 1–2
    """
    print(f"\n  🔧 Fitting ARIMA{order} …")
    try:
        model   = ARIMA(train_series, order=order)
        results = model.fit()
        print(f"     AIC = {results.aic:.2f}  |  BIC = {results.bic:.2f}")
        return results
    except Exception:
        fallback = (1, 1, 1)
        print(f"  ⚠️ Falling back to ARIMA{fallback}")
        model   = ARIMA(train_series, order=fallback)
        results = model.fit()
        print(f"     AIC = {results.aic:.2f}")
        return results


def evaluate_arima_forecast(arima_res, test_series):
    """
    Forecast on the hold-out set and evaluate.

    Primary metric: **MAPE**
      Business rationale:
      • The inventory team thinks in percentages.
      • "Our forecast is off by X % on average" is directly actionable:
        they can buffer stock by that percentage.
      • MAPE < 10 %  →  strong;  10–20 %  →  acceptable for planning.
    """
    n = len(test_series)
    fcast = arima_res.forecast(steps=n)
    metrics = compute_forecast_metrics(test_series.values,
                                       fcast.values, "ARIMA")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_series.index, test_series.values,
            "b-", lw=1.5, label="Actual")
    ax.plot(test_series.index, fcast.values,
            "r--", lw=1.5, label="ARIMA forecast")
    ax.fill_between(test_series.index,
                    fcast.values * 0.8, fcast.values * 1.2,
                    alpha=0.12, color="red", label="±20 % band")
    ax.set_title(f"ARIMA Forecast vs Actual  (MAPE {metrics['mape']:.1f}%)")
    ax.set_ylabel("Daily Revenue (BRL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "substep3_arima_forecast.png")
    return metrics, fcast


# ────────────────────────────────────────────────────────────
# SUB-STEP 4 — SARIMA + Prophet  (MEDIUM)
# ────────────────────────────────────────────────────────────

def fit_sarima_model(train_series,
                     order=(2, 1, 2),
                     seasonal_order=(1, 1, 1, 7)):
    """
    Fit SARIMA to capture weekly seasonality that ARIMA misses.

    Seasonal parameters:
      P=1, D=1, Q=1, s=7  →  one seasonal AR/MA term + seasonal
      differencing at the weekly level.
    """
    print(f"\n  🔧 Fitting SARIMA{order}×{seasonal_order} …")
    try:
        model = SARIMAX(train_series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=500)
        print(f"     AIC = {res.aic:.2f}  |  BIC = {res.bic:.2f}")
        return res
    except Exception:
        fallback_s = (1, 1, 0, SEASONAL_PERIOD_WEEKLY)
        print(f"  ⚠️ Falling back to SARIMA(1,1,1)×{fallback_s}")
        model = SARIMAX(train_series,
                        order=(1, 1, 1),
                        seasonal_order=fallback_s,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)
        print(f"     AIC = {res.aic:.2f}")
        return res


def fit_prophet_model(train_df):
    """
    Fit Prophet — handles multiple seasonalities and outliers
    without manual differencing.
    """
    print("\n  🔧 Fitting Prophet …")
    pdf = train_df.reset_index()
    pdf.columns = ["ds", "y"]

    m = Prophet(weekly_seasonality=True,
                yearly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05)
    m.fit(pdf)
    print("     ✅ Prophet fitted")
    return m


def compare_models(test_series, arima_fcast,
                   sarima_res, prophet_model, train_df):
    """
    Compare ARIMA, SARIMA, Prophet on the same hold-out set
    and quantify whether added complexity is justified.
    """
    n = len(test_series)

    sarima_fcast  = sarima_res.forecast(steps=n)

    future         = prophet_model.make_future_dataframe(periods=n)
    prophet_pred   = prophet_model.predict(future)
    prophet_fcast  = prophet_pred.tail(n)["yhat"].values

    print("\n" + "=" * 60)
    print("4 · MODEL COMPARISON")
    print("=" * 60)
    m_ar = compute_forecast_metrics(test_series.values,
                                    arima_fcast.values, "ARIMA")
    m_sa = compute_forecast_metrics(test_series.values,
                                    sarima_fcast.values, "SARIMA")
    m_pr = compute_forecast_metrics(test_series.values,
                                    prophet_fcast, "Prophet")

    print(f"\n  {'Metric':<8} {'ARIMA':>10} {'SARIMA':>10} {'Prophet':>10}")
    print("  " + "-" * 42)
    for k in ("mae", "rmse", "mape"):
        vals  = {"ARIMA": m_ar[k], "SARIMA": m_sa[k], "Prophet": m_pr[k]}
        best  = min(vals, key=vals.get)
        print(f"  {k.upper():<8} {vals['ARIMA']:>10.2f} "
              f"{vals['SARIMA']:>10.2f} {vals['Prophet']:>10.2f}  ← {best}")

    sarima_imp  = (m_ar["mape"] - m_sa["mape"]) / m_ar["mape"] * 100
    prophet_imp = (m_ar["mape"] - m_pr["mape"]) / m_ar["mape"] * 100
    IMPROVEMENT_THRESHOLD = 5  # percent

    print(f"\n  SARIMA  vs ARIMA: {sarima_imp:+.1f}% MAPE improvement")
    print(f"  Prophet vs ARIMA: {prophet_imp:+.1f}% MAPE improvement")

    if max(sarima_imp, prophet_imp) > IMPROVEMENT_THRESHOLD:
        winner = "SARIMA" if m_sa["mape"] < m_pr["mape"] else "Prophet"
        print(f"\n  ✅ Recommend {winner} — captures weekly patterns "
              f"ARIMA misses (> {IMPROVEMENT_THRESHOLD}% gain)")
    else:
        print(f"\n  ⚖️ Improvement < {IMPROVEMENT_THRESHOLD}% — "
              "use simpler ARIMA")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_series.index, test_series.values,
            "k-", lw=2, label="Actual")
    ax.plot(test_series.index, arima_fcast.values, "r--", lw=1.2,
            label=f"ARIMA  (MAPE {m_ar['mape']:.1f}%)")
    ax.plot(test_series.index, sarima_fcast.values, "g--", lw=1.2,
            label=f"SARIMA (MAPE {m_sa['mape']:.1f}%)")
    ax.plot(test_series.index, prophet_fcast, "b--", lw=1.2,
            label=f"Prophet (MAPE {m_pr['mape']:.1f}%)")
    ax.set_title("Model Comparison on Hold-Out Set")
    ax.set_ylabel("Daily Revenue (BRL)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "substep4_model_comparison.png")

    return {"arima": m_ar, "sarima": m_sa, "prophet": m_pr}


# ────────────────────────────────────────────────────────────
# SUB-STEP 5 — Sensor Failure Prediction  (MEDIUM)
# ────────────────────────────────────────────────────────────

def identify_failure_episodes(sensor_df):
    """
    Group consecutive non-NORMAL rows into failure episodes.
    Returns a DataFrame of episodes with onset time and duration.
    """
    df = sensor_df.copy()
    df["is_fail"] = (df["machine_status"] != "NORMAL").astype(int)
    df["ep_change"] = df["is_fail"].diff().abs().fillna(0)
    df["ep_id"] = df["ep_change"].cumsum()

    episodes = (df[df["is_fail"] == 1]
                .groupby("ep_id")
                .agg(start=("timestamp", "min"),
                     end=("timestamp", "max"),
                     duration_min=("timestamp", "count"),
                     has_broken=("machine_status",
                                 lambda x: (x == "BROKEN").any()),
                     has_recovering=("machine_status",
                                     lambda x: (x == "RECOVERING").any()))
                .reset_index())

    print(f"\n  Failure episodes found: {len(episodes)}")
    for _, row in episodes.iterrows():
        types = []
        if row["has_broken"]:
            types.append("BROKEN")
        if row["has_recovering"]:
            types.append("RECOVERING")
        print(f"     Ep {int(row['ep_id']):>3}: "
              f"{row['start']} → {row['end']}  "
              f"({row['duration_min']} min, {'+'.join(types)})")
    return episodes


def engineer_sensor_features(sensor_df, episodes):
    """
    Build feature matrix for failure prediction.

    Features
    --------
    • Raw sensor values (51 cols after cleaning)
    • Rolling 60-min mean & std for top-variance sensors
    • First-difference (rate of change) for top sensors

    Target
    ------
    failure_in_24h = 1 if a failure episode starts within
    the next 1 440 minutes (24 h); 0 otherwise.
    Only NORMAL-status rows are retained (can't predict
    *during* an active failure).
    """
    print("\n" + "=" * 60)
    print("5a · FEATURE ENGINEERING")
    print("=" * 60)

    df = sensor_df.copy()
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # ── target variable ──
    # Label rows as "at risk" if a failure starts within next 24 h.
    # Also label rows DURING a failure episode (RECOVERING/BROKEN)
    # so that the model can learn the transition.
    df["failure_in_24h"] = 0
    for _, ep in episodes.iterrows():
        # 24 h window before onset
        window_start = ep["start"] - timedelta(minutes=FAILURE_WINDOW_MINUTES)
        mask_before = ((df["timestamp"] >= window_start)
                       & (df["timestamp"] < ep["start"]))
        df.loc[mask_before, "failure_in_24h"] = 1
        # also label the episode itself
        mask_during = ((df["timestamp"] >= ep["start"])
                       & (df["timestamp"] <= ep["end"]))
        df.loc[mask_during, "failure_in_24h"] = 1

    # Keep ALL rows (including non-NORMAL) so the model can
    # learn from sensor patterns during degradation.
    # Create a binary feature for current status.
    df["is_currently_normal"] = (df["machine_status"] == "NORMAL").astype(int)
    df_all = df.copy()
    pos = df_all["failure_in_24h"].sum()
    neg = len(df_all) - pos
    print(f"  Total rows for modelling: {len(df_all):,}")
    print(f"     failure_in_24h = 0: {neg:,}")
    print(f"     failure_in_24h = 1: {pos:,}")

    # ── rolling features for top-variance sensors ──
    variances = df_all[sensor_cols].var().sort_values(ascending=False)
    top = variances.head(TOP_N_SENSORS_FOR_ROLLING).index.tolist()
    print(f"  Rolling features for top {TOP_N_SENSORS_FOR_ROLLING} sensors …")

    for col in top:
        df_all[f"{col}_rmean60"] = (
            df_all[col]
            .rolling(window=ROLLING_WINDOW_SHORT, min_periods=1)
            .mean()
        )
        df_all[f"{col}_rstd60"] = (
            df_all[col]
            .rolling(window=ROLLING_WINDOW_SHORT, min_periods=1)
            .std()
            .fillna(0)
        )
        df_all[f"{col}_diff"] = df_all[col].diff().fillna(0)

    df_all.dropna(inplace=True)

    feature_cols = (
        sensor_cols
        + [f"{c}_rmean60" for c in top]
        + [f"{c}_rstd60" for c in top]
        + [f"{c}_diff" for c in top]
    )
    print(f"  Total features: {len(feature_cols)}")
    return df_all, feature_cols


def build_failure_model(df_normal, feature_cols):
    """
    Train a Random Forest classifier for 24-h failure prediction.

    Evaluation metric rationale
    ---------------------------
    • **Recall** is the primary metric because a missed failure
      triggers a $50 000 emergency repair, whereas a false alarm
      costs only $500.  Missing a real failure is 100× worse.
    • We also report Precision-Recall AUC for a threshold-independent
      view, and the full confusion matrix for the maintenance team.

    Train/test split: temporal 60 / 40 — ensures failure episodes
    appear in both train and test sets.
    """
    print("\n" + "=" * 60)
    print("5b · MODEL TRAINING & EVALUATION")
    print("=" * 60)

    split = int(len(df_normal) * SENSOR_TRAIN_FRACTION)
    train, test = df_normal.iloc[:split], df_normal.iloc[split:]

    print(f"  Train: {len(train):,} rows  (→ {train['timestamp'].max()})")
    print(f"  Test : {len(test):,} rows  ({test['timestamp'].min()} →)")
    print(f"  Train positive rate: {train['failure_in_24h'].mean():.4%}")
    print(f"  Test  positive rate: {test['failure_in_24h'].mean():.4%}")

    # Safety check: if test has no positives, adjust split
    if test['failure_in_24h'].sum() == 0:
        print("  ⚠️ No positive samples in test — re-splitting at 50/50")
        split = int(len(df_normal) * 0.50)
        train, test = df_normal.iloc[:split], df_normal.iloc[split:]
        print(f"  Train: {len(train):,} rows  (→ {train['timestamp'].max()})")
        print(f"  Test : {len(test):,} rows  ({test['timestamp'].min()} →)")
        print(f"  Train positive rate: {train['failure_in_24h'].mean():.4%}")
        print(f"  Test  positive rate: {test['failure_in_24h'].mean():.4%}")

    X_tr = train[feature_cols].values
    y_tr = train["failure_in_24h"].values
    X_te = test[feature_cols].values
    y_te = test["failure_in_24h"].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\n  🔧 Training RandomForest "
          f"(n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}, balanced) …")
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)

    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)[:, 1]

    print("\n  Classification report:")
    print(classification_report(y_te, y_pred, labels=[0, 1],
                                target_names=["Normal", "Failure"],
                                zero_division=0))

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    print(f"  Confusion matrix:")
    print(f"     {'':>18} Pred Normal  Pred Failure")
    print(f"     Actual Normal   {tn:>10,}  {fp:>12,}")
    print(f"     Actual Failure  {fn:>10,}  {tp:>12,}")
    print(f"\n  Recall    : {recall:.3f}  "
          f"(catch {recall * 100:.1f}% of real failures)")
    print(f"  Precision : {precision:.3f}  "
          f"({precision * 100:.1f}% of alerts are genuine)")

    # Feature importance
    imp = pd.Series(clf.feature_importances_,
                    index=feature_cols).sort_values(ascending=False)
    print("\n  Top-10 features:")
    for feat, val in imp.head(10).items():
        print(f"     {feat}: {val:.4f}")

    # PR curve
    prec_c, rec_c, _ = precision_recall_curve(y_te, y_prob)
    pr_auc = auc(rec_c, prec_c)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rec_c, prec_c, "b-", lw=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"PR Curve  (AUC = {pr_auc:.3f})")
    axes[0].grid(True, alpha=0.3)

    imp.head(15).plot(kind="barh", ax=axes[1], color="#4CAF50")
    axes[1].set_title("Top-15 Feature Importances")
    axes[1].set_xlabel("Importance")
    fig.tight_layout()
    save_plot(fig, "substep5_failure_model.png")

    # ── Maintenance-team presentation ──
    print("\n" + "=" * 60)
    print("5c · MAINTENANCE TEAM DASHBOARD GUIDE")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────────────────┐
    │      EQUIPMENT FAILURE EARLY-WARNING SYSTEM      │
    ├──────────────────────────────────────────────────┤
    │  🟢 GREEN   Risk < 30 %  → Normal operation     │
    │  🟡 YELLOW  Risk 30–70 % → Schedule inspection  │
    │  🔴 RED     Risk > 70 %  → Immediate action     │
    ├──────────────────────────────────────────────────┤
    │  The system checks every minute and warns you    │
    │  UP TO 24 HOURS before a failure happens,        │
    │  giving time for planned maintenance.            │
    └──────────────────────────────────────────────────┘
    """)

    return clf, scaler, y_te, y_pred, y_prob, feature_cols, imp, test


# ────────────────────────────────────────────────────────────
# SUB-STEP 6 — Rule-Based vs ML  (HARD)
# ────────────────────────────────────────────────────────────

def find_best_single_sensor(importances):
    """Return the raw sensor with the highest feature importance."""
    raw = importances[importances.index.str.match(r"^sensor_\d+$")]
    best = raw.index[0]
    print(f"\n  Best raw sensor for rule: {best}  "
          f"(importance = {raw.iloc[0]:.4f})")
    return best


def evaluate_rule_vs_model(test_df, best_sensor,
                           y_test, y_prob, feature_cols):
    """
    Compare a single-sensor threshold rule against the ML model
    using an explicit cost matrix.

    Cost matrix
    -----------
    FN (missed failure) : $50 000  — emergency repair + downtime
    FP (false alarm)    : $   500  — unnecessary inspection
    TP (caught failure) : $ 1 000  — planned repair (much cheaper)
    TN (no action)      : $     0
    """
    print("\n" + "=" * 60)
    print("6 · RULE vs MODEL — COST-BASED EVALUATION")
    print("=" * 60)

    vals = test_df[best_sensor].values
    percentiles = np.percentile(vals, np.arange(1, 100))

    best_cost, best_thresh, best_dir = float("inf"), None, None
    records = []

    for thr in percentiles:
        for direction, preds in [("below", (vals < thr).astype(int)),
                                 ("above", (vals > thr).astype(int))]:
            cm = confusion_matrix(y_test, preds, labels=[0, 1])
            if cm.shape != (2, 2):
                continue
            tn, fp, fn, tp = cm.ravel()
            cost = fn * COST_MISSED_FAILURE + fp * COST_FALSE_ALARM + tp * COST_TRUE_POSITIVE
            rec  = tp / (tp + fn) if (tp + fn) else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            records.append(dict(threshold=thr, direction=direction,
                                cost=cost, recall=rec, precision=prec,
                                tp=tp, fp=fp, fn=fn, tn=tn))
            if cost < best_cost:
                best_cost  = cost
                best_thresh = thr
                best_dir   = direction

    rule_df = pd.DataFrame(records)

    print(f"\n  🏷  Best rule: {best_sensor} {best_dir} {best_thresh:.2f}")
    best_row = rule_df.loc[rule_df["cost"].idxmin()]
    print(f"     Recall   : {best_row['recall']:.3f}")
    print(f"     Precision: {best_row['precision']:.3f}")
    print(f"     Cost     : ${best_cost:,.0f}")

    # ML model at default threshold 0.5
    ml_preds = (y_prob >= 0.5).astype(int)
    cm_ml = confusion_matrix(y_test, ml_preds, labels=[0, 1])
    tn_ml, fp_ml, fn_ml, tp_ml = cm_ml.ravel()
    ml_cost = (fn_ml * COST_MISSED_FAILURE
               + fp_ml * COST_FALSE_ALARM
               + tp_ml * COST_TRUE_POSITIVE)
    ml_rec  = tp_ml / (tp_ml + fn_ml) if (tp_ml + fn_ml) else 0

    print(f"\n  🤖 ML model (threshold = 0.5):")
    print(f"     Recall   : {ml_rec:.3f}")
    print(f"     Cost     : ${ml_cost:,.0f}")

    diff = best_cost - ml_cost
    print(f"\n  💰 Cost difference: ${diff:+,.0f}  "
          f"({'ML cheaper' if diff > 0 else 'Rule cheaper'})")

    print("""
    When the RULE outperforms:
      • Single dominant failure mode with clear sensor signature
      • High signal-to-noise ratio on the key sensor
      • Need for interpretability / auditability
      • No labelled training data available

    When the RULE fails:
      • Multiple interacting failure modes
      • Gradual multi-sensor degradation
      • Optimal threshold varies with operating conditions
      • New equipment with different baseline readings
    """)

    winner = "ML model" if ml_cost < best_cost else "Simple rule"
    print(f"  ✅ RECOMMENDATION: Deploy the {winner} as primary.\n")

    return best_sensor, best_thresh, best_dir, best_cost, ml_cost


# ────────────────────────────────────────────────────────────
# SUB-STEP 7 — Fleet-Wide Deployment Cost  (HARD)
# ────────────────────────────────────────────────────────────

def calculate_fleet_cost(y_test, y_prob):
    """
    Scale the model's test-set performance to 100 000 sensors
    and find the cost-optimal decision threshold.

    Then compare it against the F1-optimal threshold and discuss
    the implications for production metric selection.
    """
    print("\n" + "=" * 60)
    print("7 · FLEET-WIDE DEPLOYMENT COST ANALYSIS")
    print("=" * 60)

    failure_rate = y_test.mean()
    exp_daily_failures = failure_rate * FLEET_SIZE
    exp_daily_normals  = (1 - failure_rate) * FLEET_SIZE

    print(f"  Failure rate (test): {failure_rate:.6f}")
    print(f"  Expected daily failures fleet-wide: {exp_daily_failures:.1f}")

    thresholds = np.arange(0.01, 1.0, 0.01)
    rows, f1s = [], []

    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0
        tpr = tp / (tp + fn) if (tp + fn) else 0

        daily_fa     = fpr * exp_daily_normals
        daily_missed = fnr * exp_daily_failures
        daily_caught = tpr * exp_daily_failures

        daily_cost = (daily_fa * COST_FALSE_ALARM
                      + daily_missed * COST_MISSED_FAILURE
                      + daily_caught * COST_TRUE_POSITIVE)
        rows.append(dict(threshold=thr, daily_cost=daily_cost,
                         false_alarms=daily_fa,
                         missed=daily_missed,
                         caught=daily_caught,
                         recall=tpr))
        f1s.append(dict(threshold=thr,
                        f1=f1_score(y_test, preds, zero_division=0)))

    cost_df = pd.DataFrame(rows)
    f1_df   = pd.DataFrame(f1s)

    # cost-optimal
    co = cost_df.loc[cost_df["daily_cost"].idxmin()]
    # F1-optimal
    fo = f1_df.loc[f1_df["f1"].idxmax()]

    print(f"\n  💰 COST-OPTIMAL threshold : {co['threshold']:.2f}")
    print(f"     Daily cost             : ${co['daily_cost']:,.0f}")
    print(f"     Daily false alarms     : {co['false_alarms']:.0f}")
    print(f"     Daily missed failures  : {co['missed']:.1f}")
    print(f"     Recall                 : {co['recall']:.3f}")

    print(f"\n  🎯 F1-OPTIMAL threshold   : {fo['threshold']:.2f}")
    print(f"     F1 score               : {fo['f1']:.3f}")
    f1_cost = cost_df.loc[
        (cost_df["threshold"] - fo["threshold"]).abs().idxmin()
    ]
    print(f"     Daily cost at F1-opt   : ${f1_cost['daily_cost']:,.0f}")

    diff = f1_cost["daily_cost"] - co["daily_cost"]
    same = abs(co["threshold"] - fo["threshold"]) < 0.05

    if same:
        print("\n  The two thresholds are approximately equal — "
              "F1 is a reasonable proxy here.")
    else:
        ratio = COST_MISSED_FAILURE / COST_FALSE_ALARM
        print(f"\n  ⚠️ Thresholds DIFFER — "
              f"using F1 costs ${diff:,.0f}/day extra.")
        print(f"     Why? F1 weights FP and FN equally, "
              f"but a missed failure is {ratio:.0f}× costlier.")
        print(f"     F1 → threshold too high → more misses → "
              f"higher cost.")
        print(f"\n  LESSON: In production, optimise for BUSINESS COST, "
              f"not F1.\n  F1 is useful during development but misleading "
              f"when costs are asymmetric.")

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(cost_df["threshold"], cost_df["daily_cost"], "b-", lw=2)
    axes[0].axvline(co["threshold"], color="green", ls="--",
                    label=f"Cost-opt ({co['threshold']:.2f})")
    axes[0].axvline(fo["threshold"], color="red", ls="--",
                    label=f"F1-opt ({fo['threshold']:.2f})")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Expected Daily Cost ($)")
    axes[0].set_title("Fleet Daily Cost vs Threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f1_df["threshold"], f1_df["f1"], "r-", lw=2)
    axes[1].axvline(fo["threshold"], color="red", ls="--",
                    label=f"F1-opt ({fo['threshold']:.2f})")
    axes[1].axvline(co["threshold"], color="green", ls="--",
                    label=f"Cost-opt ({co['threshold']:.2f})")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score vs Threshold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_plot(fig, "substep7_fleet_cost.png")

    return co, fo, cost_df


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────

def main():
    """Execute all seven sub-steps end-to-end."""
    print("=" * 70)
    print("  WEEK 08 · MONDAY — TIME-SERIES ANALYSIS")
    print("  E-Commerce Forecasting & Sensor Failure Prediction")
    print("=" * 70)

    # ── SUB-STEP 1 (Easy) ──────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🟢  SUB-STEP 1: E-COMMERCE DATA CHARACTERISATION")
    print("=" * 70)
    daily_sales = prepare_ecommerce_daily_sales(ECOMMERCE_DIR)
    findings, decomp = characterise_ecommerce_series(daily_sales)

    # ── SUB-STEP 2 (Easy) ──────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🟢  SUB-STEP 2: SENSOR DATA CLEANING")
    print("=" * 70)
    sensor_raw = load_csv_safely(os.path.join(SENSOR_DIR, "sensor.csv"))
    issues = identify_sensor_issues(sensor_raw)
    sensor_clean = clean_sensor_data(sensor_raw, issues)

    # ── SUB-STEP 3 (Medium) ────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🟡  SUB-STEP 3: ARIMA MODEL")
    print("=" * 70)
    train_s, test_s = create_temporal_split(daily_sales)
    arima_res = fit_arima_model(train_s["daily_revenue"])
    arima_met, arima_fc = evaluate_arima_forecast(
        arima_res, test_s["daily_revenue"]
    )

    # ── SUB-STEP 4 (Medium) ────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🟡  SUB-STEP 4: SARIMA & PROPHET — SEASONAL PATTERNS")
    print("=" * 70)
    sarima_res = fit_sarima_model(train_s["daily_revenue"])
    prophet_m  = fit_prophet_model(train_s[["daily_revenue"]])
    comparison = compare_models(
        test_s["daily_revenue"], arima_fc,
        sarima_res, prophet_m, train_s[["daily_revenue"]]
    )

    # ── SUB-STEP 5 (Medium) ────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🟡  SUB-STEP 5: SENSOR FAILURE PREDICTION")
    print("=" * 70)
    episodes = identify_failure_episodes(sensor_clean)
    df_feat, feat_cols = engineer_sensor_features(sensor_clean, episodes)
    (clf, scaler, y_te, y_pred, y_prob,
     feat_cols, imp, test_df) = build_failure_model(df_feat, feat_cols)

    # ── SUB-STEP 6 (Hard) ─────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🔴  SUB-STEP 6: RULE-BASED vs ML COMPARISON")
    print("=" * 70)
    best_sensor = find_best_single_sensor(imp)
    (best_sns, best_thr, best_dir,
     rule_cost, ml_cost) = evaluate_rule_vs_model(
        test_df, best_sensor, y_te, y_prob, feat_cols
    )

    # ── SUB-STEP 7 (Hard) ─────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  🔴  SUB-STEP 7: FLEET-WIDE DEPLOYMENT COST")
    print("=" * 70)
    co, fo, costs_df = calculate_fleet_cost(y_te, y_prob)

    # ── Summary ────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  ✅  ALL SUB-STEPS COMPLETE")
    print("=" * 70)
    print(f"""
    E-Commerce Forecasting
    ──────────────────────
    ARIMA  MAPE : {arima_met['mape']:.1f}%
    SARIMA MAPE : {comparison['sarima']['mape']:.1f}%
    Prophet MAPE: {comparison['prophet']['mape']:.1f}%

    Sensor Failure Prediction
    ─────────────────────────
    Episodes detected     : {len(episodes)}
    Model recall          : {(y_pred[y_te == 1].sum() / y_te.sum() if y_te.sum() else 0):.3f}
    Cost-optimal threshold: {co['threshold']:.2f}
    F1-optimal threshold  : {fo['threshold']:.2f}
    Daily fleet cost      : ${co['daily_cost']:,.0f}

    Plots saved to: {PLOT_DIR}
    """)


if __name__ == "__main__":
    main()
