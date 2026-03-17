#!/usr/bin/env python3
# Generates the dual-model USDPEN notebook
import json

cid = [0]
def _id():
    cid[0] += 1
    return f"c{cid[0]:04d}"

def cc(src):
    lines = src.strip().split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "id": _id(), "metadata": {}, "outputs": [], "execution_count": None, "source": source}

def mc(src):
    lines = src.strip().split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": source}

C = []

# ===================== TITLE =====================
C.append(mc("# USDPEN Conditional Variance Model\n## Dual Model: Garman-Klass vs Parkinson"))

# ===================== SECTION 1 =====================
C.append(mc("""## 1. Setup, Constants & Data Loading

We load OHLC + fixing data for 5 LatAm currencies and configure model parameters. The model separates USDPEN variance into:
- **Systematic** (regional co-movement): where PEN "should" be given what COP/CLP/MXN/BRL have done
- **Idiosyncratic** (Peru-specific): how uncertain that estimate is

Two variance estimators compete for the idiosyncratic piece: **Garman-Klass** (uses OHLC) and **Parkinson** (uses high-low range only). The backtest picks the winner."""))

C.append(cc("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.optimize import minimize_scalar
from IPython.display import display, HTML
import ipywidgets as widgets
import warnings
warnings.filterwarnings("ignore")

INTERVENTION_START = pd.Timestamp("2025-11-01")
INTERVENTION_END = pd.Timestamp("2026-02-28")
POST_SHOCK_START = pd.Timestamp("2026-03-01")

PERCENTILES = {
    1: -2.3263, 5: -1.6449, 10: -1.2816, 25: -0.6745,
    50: 0.0, 75: 0.6745, 90: 1.2816, 95: 1.6449, 99: 2.3263
}
CI_LEVELS = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
CURRENCIES = ["pen", "cop", "clp", "mxn", "brl"]
REGRESSORS = ["cop", "clp", "mxn", "brl"]
PIP = 0.0001
SEED_WINDOW = 60

COLORS = {"pen": "#1f77b4", "cop": "#ff7f0e", "clp": "#2ca02c", "mxn": "#d62728", "brl": "#9467bd"}
COLOR_SYS = "#3B82F6"
COLOR_IDIO = "#F97316"
COLOR_TOTAL = "#111827"
plt.rcParams.update({"figure.figsize": (14, 5), "axes.grid": True, "grid.alpha": 0.3, "font.size": 11, "figure.dpi": 100})

def shade_regimes(ax, alpha=0.15):
    ylim = ax.get_ylim()
    ax.axvspan(INTERVENTION_START, INTERVENTION_END, alpha=alpha, color="#D1D5DB", label="BCRP Intervention", zorder=0)
    ax.axvspan(POST_SHOCK_START, pd.Timestamp("2026-12-31"), alpha=alpha, color="#FCA5A5", label="Iran Conflict", zorder=0)
    ax.set_ylim(ylim)

def label_regime(dt):
    if dt >= POST_SHOCK_START: return "post_shock"
    elif dt >= INTERVENTION_START: return "intervention"
    return "normal"

print("Setup complete.")"""))

# Cell 1.2 — Load data
C.append(mc("### 1.2 — Load Data"))

C.append(cc("""raw = pd.read_excel("model_levels.xlsx", sheet_name=0)
print("Original columns:", raw.columns.tolist())
print("Shape:", raw.shape)

# Parse date
date_col = [c for c in raw.columns if "date" in c.lower()][0]
raw["date"] = pd.to_datetime(raw[date_col], format="%d/%m/%Y", errors="coerce")
if raw["date"].isna().sum() > 0:
    # Try other formats
    raw["date"] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")

# Standardize column names: find ccy + field patterns
col_map = {}
for col in raw.columns:
    cl = col.strip().lower().replace(" ", "_")
    if "date" in cl:
        col_map[col] = "date"
        continue
    for ccy in CURRENCIES:
        if cl.startswith(ccy) or cl.startswith("usd" + ccy):
            for field in ["open", "high", "low", "close", "fixing", "fix"]:
                if field in cl:
                    fname = "fixing" if field == "fix" else field
                    col_map[col] = f"{ccy}_{fname}"
                    break
            break

raw = raw.rename(columns=col_map)

# Replace #N/D with NaN
for col in raw.columns:
    if col != "date":
        raw[col] = pd.to_numeric(raw[col].replace({"#N/D": np.nan, "#N/A": np.nan}), errors="coerce")

raw = raw.set_index("date").sort_index()

print(f"\\nRenamed columns: {raw.columns.tolist()}")
print(f"Date range: {raw.index[0].date()} to {raw.index[-1].date()}")
print(f"\\nNon-null fixing counts:")
for ccy in CURRENCIES:
    fc = f"{ccy}_fixing"
    if fc in raw.columns:
        print(f"  {ccy.upper()}: {raw[fc].notna().sum()}")

display(raw.head())
display(raw.tail())"""))

# ===================== SECTION 2 =====================
C.append(mc("""## 2. Data Cleaning & Return Computation

Three key concepts:

1. **Fixing-to-fixing returns**: Fixings are NDF settlement references. Returns between consecutive fixings are the correct unit.
2. **Calendar-day scaling**: A Friday-to-Monday return spans 3 calendar days. Raw r² overstates daily variance by 3x. Correction: divide r² by calendar days elapsed before feeding into EWMA.
3. **Holiday handling**: Each currency has different holidays. Returns computed only between consecutive non-null fixings per currency. When PEN traded but a regional currency didn't, regional return = 0 (stale-price convention)."""))

C.append(mc("### 2.1 — Compute Returns"))

C.append(cc("""# Remove rows where ALL fixings are null (pure weekends)
fix_cols = [f"{ccy}_fixing" for ccy in CURRENCIES if f"{ccy}_fixing" in raw.columns]
df = raw.copy()
all_null = df[fix_cols].isna().all(axis=1)
df = df[~all_null].copy()

# Compute fixing-to-fixing returns and calendar days per currency
for ccy in CURRENCIES:
    fc = f"{ccy}_fixing"
    if fc not in df.columns:
        continue

    # Forward-fill index of last valid fixing
    valid = df[fc].notna()
    df[f"{ccy}_fixing_return"] = np.nan
    df[f"{ccy}_calendar_days"] = np.nan

    valid_dates = df.index[valid]
    valid_vals = df.loc[valid, fc].values

    for i in range(1, len(valid_dates)):
        dt = valid_dates[i]
        dt_prev = valid_dates[i - 1]
        cal_days = (dt - dt_prev).days
        ret = np.log(valid_vals[i] / valid_vals[i - 1])
        df.loc[dt, f"{ccy}_fixing_return"] = ret
        df.loc[dt, f"{ccy}_calendar_days"] = cal_days

    # OHLC variance estimators (only on days with valid OHLC)
    h = f"{ccy}_high"
    l = f"{ccy}_low"
    o = f"{ccy}_open"
    c = f"{ccy}_close"

    if all(x in df.columns for x in [h, l, o, c]):
        ohlc_valid = df[h].notna() & df[l].notna() & df[o].notna() & df[c].notna()
        caldays = df[f"{ccy}_calendar_days"]

        # Parkinson: (1/(4*ln2)) * (ln(H/L))^2 / cal_days
        log_hl = np.log(df[h] / df[l])
        df[f"{ccy}_parkinson_var"] = np.where(
            ohlc_valid & (caldays > 0),
            (1 / (4 * np.log(2))) * log_hl**2 / caldays,
            np.nan
        )

        # Garman-Klass: [0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2] / cal_days
        log_co = np.log(df[c] / df[o])
        df[f"{ccy}_gk_var"] = np.where(
            ohlc_valid & (caldays > 0),
            (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2) / caldays,
            np.nan
        )

# Stale-price convention: fill NaN regional returns with 0 on PEN trading days
pen_valid = df["pen_fixing_return"].notna()
for reg in REGRESSORS:
    rc = f"{reg}_fixing_return"
    if rc in df.columns:
        df.loc[pen_valid, rc] = df.loc[pen_valid, rc].fillna(0.0)

# Keep only rows where PEN has a return
df_model = df[df["pen_fixing_return"].notna()].copy()
print(f"Model rows: {len(df_model)} (PEN fixing-to-fixing observations)")
print(f"Date range: {df_model.index[0].date()} to {df_model.index[-1].date()}")"""))

C.append(mc("### 2.2 — Summary Statistics"))

C.append(cc("""ret_cols = [f"{ccy}_fixing_return" for ccy in CURRENCIES]
ret_display = df_model[ret_cols].copy()
ret_display.columns = [c.upper() for c in CURRENCIES]

summary = ret_display.describe().T
summary["skew"] = ret_display.skew()
summary["kurtosis"] = ret_display.kurtosis()
summary["mean_bps"] = summary["mean"] * 10000
summary["std_bps"] = summary["std"] * 10000
print("Fixing Return Summary Statistics:")
display(summary[["count", "mean_bps", "std_bps", "min", "max", "skew", "kurtosis"]].round(4))

print("\\nAvg calendar days between fixings:")
for ccy in CURRENCIES:
    cd = f"{ccy}_calendar_days"
    if cd in df_model.columns:
        print(f"  {ccy.upper()}: {df_model[cd].mean():.1f} days")"""))

C.append(mc("### 2.3 — Diagnostic Plots"))

C.append(cc("""# Fixing level plots
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
for i, ccy in enumerate(CURRENCIES):
    fc = f"{ccy}_fixing"
    ax = axes[i]
    valid = df[fc].notna()
    ax.plot(df.index[valid], df.loc[valid, fc], color=COLORS[ccy], linewidth=1)
    last_val = df.loc[valid, fc].iloc[-1]
    ax.set_title(f"USD{ccy.upper()} Fixing  (last: {last_val:.4f})", loc="left", fontsize=11)
    shade_regimes(ax)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()

# Correlation heatmap
ret_df = df_model[[f"{c}_fixing_return" for c in CURRENCIES]].copy()
ret_df.columns = [c.upper() for c in CURRENCIES]
corr = ret_df.corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(corr.columns)
for ii in range(5):
    for jj in range(5):
        ax.text(jj, ii, f"{corr.values[ii,jj]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(corr.values[ii,jj]) > 0.6 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Fixing Return Correlations (Full Sample)")
plt.tight_layout()
plt.show()

# Rolling 60-day correlation with PEN
fig, ax = plt.subplots(figsize=(14, 5))
for reg in REGRESSORS:
    rc = df_model[f"pen_fixing_return"].rolling(60).corr(df_model[f"{reg}_fixing_return"])
    ax.plot(df_model.index, rc, label=reg.upper(), color=COLORS[reg], linewidth=1)
shade_regimes(ax)
ax.set_title("Rolling 60-Day Correlation with USDPEN")
ax.set_ylabel("Correlation")
ax.legend(loc="lower left")
ax.axhline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.show()"""))

# ===================== SECTION 3 =====================
C.append(mc("""## 3. Volatility Estimators (GK & Parkinson)

**Parkinson (1980)**: σ²_P = (1/(4·ln2)) · (ln(H/L))². Estimates variance from high-low range. ~5x more efficient than close-to-close. Only needs H, L — most robust OHLC fields.

**Garman-Klass (1980)**: σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))². Adds drift correction via open and close. ~7-8x more efficient. Needs reliable open/close.

**Why test both**: PEN open can be noisy (BCRP intervention). Parkinson avoids this. GK uses more information but is more data-sensitive. The backtest determines which produces better-calibrated probability statements."""))

C.append(mc("### 3.1 — Compare Estimators"))

C.append(cc("""fig, ax = plt.subplots(figsize=(14, 5))
pen_gk = df_model["pen_gk_var"].dropna()
pen_pk = df_model["pen_parkinson_var"].dropna()
pen_cc = df_model["pen_fixing_return"].dropna() ** 2 / df_model.loc[df_model["pen_fixing_return"].notna(), "pen_calendar_days"]

ax.plot(pen_gk.rolling(20).mean().index, pen_gk.rolling(20).mean() * 1e8, label="GK (20d avg)", color=COLOR_SYS, linewidth=1.2)
ax.plot(pen_pk.rolling(20).mean().index, pen_pk.rolling(20).mean() * 1e8, label="Parkinson (20d avg)", color=COLOR_IDIO, linewidth=1.2)
ax.plot(pen_cc.rolling(20).mean().index, pen_cc.rolling(20).mean() * 1e8, label="Close-to-Close (20d avg)", color=COLOR_TOTAL, linestyle="--", linewidth=1)
shade_regimes(ax)
ax.set_title("USDPEN Daily Variance Estimates: GK vs Parkinson vs Close-to-Close")
ax.set_ylabel("Variance (x1e-8)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()"""))

# ===================== SECTION 4 =====================
C.append(mc("""## 4. Model Functions

Four components:
1. **EWMA covariance** with calendar-day scaling — Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1}·r'_{t-1}/caldays
2. **Betas from covariance** — β_i = Cov(PEN,i)/Var(i), updated daily
3. **Variance decomposition** — systematic + idiosyncratic
4. **Conditional range** — expected move ± z·√(idio_var)·spot·10000 in pips"""))

C.append(mc("### 4.1 — All Functions"))

C.append(cc("""def compute_ewma_covariance(df_m, ret_cols, calday_cols, lam_cov, seed_window=SEED_WINDOW):
    # EWMA covariance with calendar-day scaling
    n = len(df_m)
    k = len(ret_cols)
    dates = df_m.index
    data = df_m[ret_cols].values
    caldays = df_m[calday_cols].values

    # Seed with sample cov of first seed_window valid rows
    valid_mask = ~np.isnan(data).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    seed_end = valid_idx[min(seed_window, len(valid_idx)) - 1] + 1
    seed_data = data[:seed_end]
    seed_data = seed_data[~np.isnan(seed_data).any(axis=1)]
    sigma = np.cov(seed_data, rowvar=False)

    cov_dict = {}
    var_rows = []
    corr_rows = []

    for t in range(seed_end, n):
        r_prev = data[t - 1].copy()
        cd = calday_cols[0]  # use PEN calendar days for scaling
        cd_val = caldays[t - 1, 0] if not np.isnan(caldays[t - 1, 0]) else 1.0
        cd_val = max(cd_val, 1.0)

        # Replace NaN returns with 0 for outer product
        r_clean = np.nan_to_num(r_prev, nan=0.0).reshape(-1, 1)
        sigma = lam_cov * sigma + (1 - lam_cov) * (r_clean @ r_clean.T) / cd_val

        dt = dates[t]
        cov_dict[dt] = sigma.copy()

        vr = {"date": dt}
        for j in range(k):
            vr[ret_cols[j]] = sigma[j, j]
        var_rows.append(vr)

        stds = np.sqrt(np.diag(sigma))
        stds[stds == 0] = 1e-12
        corr_mat = sigma / np.outer(stds, stds)
        cr = {"date": dt}
        for j in range(k):
            for l in range(j + 1, k):
                cr[f"{ret_cols[j]}_{ret_cols[l]}"] = corr_mat[j, l]
        corr_rows.append(cr)

    df_var = pd.DataFrame(var_rows).set_index("date")
    df_corr = pd.DataFrame(corr_rows).set_index("date")
    return cov_dict, df_corr, df_var


def compute_betas(cov_dict, df_m, ret_cols, target_idx=0, reg_indices=[1, 2, 3, 4]):
    # Beta_i = Cov(PEN,i)/Var(i) from EWMA covariance
    records = []
    for dt, sigma in cov_dict.items():
        betas = {"date": dt}
        for ri in reg_indices:
            var_i = sigma[ri, ri]
            cov_pi = sigma[target_idx, ri]
            cname = ret_cols[ri].replace("_fixing_return", "")
            betas[f"beta_{cname}"] = cov_pi / var_i if var_i > 1e-20 else 0.0
        records.append(betas)

    betas_df = pd.DataFrame(records).set_index("date")

    # Fitted and residual
    common = betas_df.index.intersection(df_m.index)
    fitted = pd.Series(0.0, index=common)
    for ri in reg_indices:
        cname = ret_cols[ri].replace("_fixing_return", "")
        bcol = f"beta_{cname}"
        fitted += betas_df.loc[common, bcol].values * df_m.loc[common, ret_cols[ri]].values

    betas_df["fitted"] = np.nan
    betas_df["residual"] = np.nan
    target_col = ret_cols[target_idx]
    betas_df.loc[common, "fitted"] = fitted.values
    betas_df.loc[common, "residual"] = (df_m.loc[common, target_col] - fitted).values

    # Rolling 60-day R-squared
    actual = df_m.loc[common, target_col]
    resid = actual.values - fitted.values
    resid_s = pd.Series(resid, index=common)
    roll_ss_res = (resid_s**2).rolling(60).sum()
    roll_ss_tot = ((actual - actual.rolling(60).mean())**2).rolling(60).sum()
    r2 = 1 - roll_ss_res / roll_ss_tot
    betas_df["r_squared"] = np.nan
    betas_df.loc[common, "r_squared"] = r2.values

    return betas_df


def compute_idio_var(betas_df, cov_dict, df_m, lam_idio, method="gk",
                     ret_cols=None, target_idx=0, reg_indices=[1, 2, 3, 4]):
    # EWMA idiosyncratic variance using residuals and optionally OHLC
    residuals = betas_df["residual"].dropna()
    resid_vals = residuals.values
    dates = residuals.index
    ccy = "pen"

    # Seed
    seed_n = min(SEED_WINDOW, len(resid_vals))
    idio_ewma = np.var(resid_vals[:seed_n])

    # OHLC variance series for blending
    ohlc_col = f"{ccy}_{method}_var" if method in ("gk", "parkinson") else None

    records = []
    for i, dt in enumerate(dates):
        if dt not in cov_dict:
            continue

        sigma = cov_dict[dt]
        beta_cols = [f"beta_{ret_cols[ri].replace('_fixing_return', '')}" for ri in reg_indices]
        beta_vec = betas_df.loc[dt, beta_cols].values.astype(float)

        # Systematic variance = sum(beta_i * Cov(PEN, i))
        sys_var = 0.0
        for j, ri in enumerate(reg_indices):
            sys_var += beta_vec[j] * sigma[target_idx, ri]
        sys_var = max(sys_var, 0)

        # Update EWMA of residual^2 with calendar-day scaling
        if i > 0:
            cd_col = f"{ccy}_calendar_days"
            cd = df_m.loc[dt, cd_col] if dt in df_m.index and cd_col in df_m.columns else 1.0
            cd = max(cd if not np.isnan(cd) else 1.0, 1.0)
            idio_ewma = lam_idio * idio_ewma + (1 - lam_idio) * resid_vals[i - 1]**2 / cd

        # Blend with OHLC if available
        if ohlc_col and dt in df_m.index and ohlc_col in df_m.columns:
            ohlc_val = df_m.loc[dt, ohlc_col]
            if not np.isnan(ohlc_val):
                # OHLC idio estimate = OHLC total var - systematic var
                ohlc_idio = ohlc_val - sys_var
                if ohlc_idio > 0:
                    # Blend: 70% EWMA residual, 30% OHLC-derived
                    idio_ewma = 0.7 * idio_ewma + 0.3 * ohlc_idio

        # Realized PEN EWMA var for comparison
        total = sys_var + idio_ewma
        records.append({
            "date": dt,
            "systematic_var": sys_var,
            "idiosyncratic_var": max(idio_ewma, 1e-12),
            "total_var": total,
            "systematic_pct": sys_var / total if total > 0 else 0,
        })

    return pd.DataFrame(records).set_index("date")


def build_monitor(df_m, betas_df, var_df, ewma_corr_df, model_name=""):
    # Full daily monitoring DataFrame
    common = betas_df.index.intersection(var_df.index).intersection(df_m.index)
    records = []

    for dt in common:
        row = {"date": dt}
        row["usdpen_fixing"] = df_m.loc[dt, "pen_fixing"] if "pen_fixing" in df_m.columns else np.nan
        row["usdpen_return"] = df_m.loc[dt, "pen_fixing_return"]
        row["expected_return"] = betas_df.loc[dt, "fitted"]
        row["residual"] = betas_df.loc[dt, "residual"]
        row["systematic_var"] = var_df.loc[dt, "systematic_var"]
        row["idiosyncratic_var"] = var_df.loc[dt, "idiosyncratic_var"]
        row["total_var"] = var_df.loc[dt, "total_var"]
        row["systematic_pct"] = var_df.loc[dt, "systematic_pct"]

        if dt in ewma_corr_df.index:
            row["avg_correlation"] = ewma_corr_df.loc[dt].mean()
        else:
            row["avg_correlation"] = np.nan

        bcols = [c for c in betas_df.columns if c.startswith("beta_")]
        row["beta_sum"] = sum(betas_df.loc[dt, bc] for bc in bcols)

        spot = row["usdpen_fixing"]
        idio_v = row["idiosyncratic_var"]

        # Percentile levels
        idio_std = np.sqrt(idio_v) if idio_v > 0 else 0
        exp_ret = row["expected_return"]
        if not np.isnan(spot) and idio_std > 0:
            for pct, z in PERCENTILES.items():
                row[f"p{pct}_level"] = spot * np.exp(exp_ret + z * idio_std)
            row["range_95_pips"] = 2 * 1.96 * idio_std * spot / PIP
            row["idio_vol_bps"] = idio_std * 10000
        else:
            row["range_95_pips"] = np.nan
            row["idio_vol_bps"] = np.nan

        records.append(row)

    monitor = pd.DataFrame(records).set_index("date").sort_index()
    monitor["cumul_residual_20d"] = monitor["residual"].rolling(20).sum()
    roll_std = monitor["cumul_residual_20d"].rolling(20).std()
    monitor["residual_zscore"] = monitor["cumul_residual_20d"] / roll_std
    monitor["model"] = model_name
    return monitor


def conditional_range(regional_returns, betas, idio_var, prev_fixing, pen_open=None, pen_current=None):
    # Full percentile table + contributions
    exp_ret = sum(betas.get(f"beta_{ccy}", 0) * regional_returns.get(ccy, 0) for ccy in REGRESSORS)
    exp_pips = exp_ret * prev_fixing / PIP
    exp_level = prev_fixing * np.exp(exp_ret)
    idio_std = np.sqrt(idio_var)

    result = {
        "expected_return": exp_ret,
        "expected_pips": exp_pips,
        "expected_level": exp_level,
        "idio_std": idio_std,
        "idio_vol_bps": idio_std * 10000,
    }

    # Percentile table
    pct_table = {}
    for pct, z in PERCENTILES.items():
        level = prev_fixing * np.exp(exp_ret + z * idio_std)
        pips = (level - prev_fixing) / PIP
        pct_chg = (np.exp(exp_ret + z * idio_std) - 1) * 100
        pct_table[pct] = {"level": level, "pips": pips, "pct": pct_chg}
    result["percentiles"] = pct_table

    # CI bands
    for ci, z in CI_LEVELS.items():
        lo = prev_fixing * np.exp(exp_ret - z * idio_std)
        hi = prev_fixing * np.exp(exp_ret + z * idio_std)
        result[f"ci_{int(ci*100)}_lo"] = lo
        result[f"ci_{int(ci*100)}_hi"] = hi
        result[f"ci_{int(ci*100)}_half_pips"] = z * idio_std * prev_fixing / PIP

    # Contributions
    contribs = {}
    for ccy in REGRESSORS:
        b = betas.get(f"beta_{ccy}", 0)
        r = regional_returns.get(ccy, 0)
        contribs[ccy] = {"pips": b * r * prev_fixing / PIP, "beta": b, "return": r}
    result["contributions"] = contribs

    # Current position assessment
    if pen_current is not None:
        current_ret = np.log(pen_current / prev_fixing)
        current_z = (current_ret - exp_ret) / idio_std if idio_std > 0 else 0
        current_pct = stats.norm.cdf(current_z) * 100
        result["current_z"] = current_z
        result["current_percentile"] = current_pct
        result["current_vs_expected_pips"] = (pen_current - exp_level) / PIP

    if pen_open is not None:
        open_ret = np.log(pen_open / prev_fixing)
        open_z = (open_ret - exp_ret) / idio_std if idio_std > 0 else 0
        result["open_z"] = open_z
        result["open_percentile"] = stats.norm.cdf(open_z) * 100

    return result

print("All model functions defined.")"""))

# ===================== SECTION 5 =====================
C.append(mc("""## 5. Lambda Optimization

Two lambdas optimized separately via grid search:
- **λ_cov** (0.95-0.99): slow decay for stable betas/correlation
- **λ_idio** (0.88-0.97): faster decay for responsive confidence bands

Criterion: out-of-sample log-likelihood. LL = Σ[-0.5·ln(2π·σ²) - 0.5·(r-μ)²/σ²]. Higher = better calibrated."""))

C.append(mc("### 5.1 — Optimization"))

C.append(cc("""def optimize_lambdas(df_m, ret_cols, calday_cols, vol_method, train_window=120,
                     lcov_grid=np.arange(0.95, 0.995, 0.01),
                     lidio_grid=np.arange(0.88, 0.975, 0.01)):
    # Grid search for optimal lambda pair
    target_idx = 0
    reg_indices = [1, 2, 3, 4]
    best_ll = -np.inf
    best_pair = (0.97, 0.94)
    results = []

    for lc in lcov_grid:
        cov_dict, _, _ = compute_ewma_covariance(df_m, ret_cols, calday_cols, lc)
        betas = compute_betas(cov_dict, df_m, ret_cols, target_idx, reg_indices)

        for li in lidio_grid:
            var_df = compute_idio_var(betas, cov_dict, df_m, li, method=vol_method,
                                      ret_cols=ret_cols, target_idx=target_idx, reg_indices=reg_indices)
            # Out-of-sample log-likelihood
            common = var_df.index.intersection(df_m.index)
            if len(common) <= train_window:
                continue
            test = common[train_window:]
            actual = df_m.loc[test, ret_cols[0]].values
            expected = betas.loc[test, "fitted"].values
            idio_var = var_df.loc[test, "idiosyncratic_var"].values

            valid = ~np.isnan(actual) & ~np.isnan(expected) & (idio_var > 0)
            if valid.sum() < 20:
                continue

            a = actual[valid]
            e = expected[valid]
            v = idio_var[valid]
            ll = np.sum(-0.5 * np.log(2 * np.pi * v) - 0.5 * (a - e)**2 / v)
            ll_avg = ll / valid.sum()

            results.append({"lam_cov": round(lc, 3), "lam_idio": round(li, 3), "ll": ll_avg})

            if ll_avg > best_ll:
                best_ll = ll_avg
                best_pair = (round(lc, 3), round(li, 3))

    return pd.DataFrame(results), best_pair

# Build return and calday column lists
ret_cols = [f"{ccy}_fixing_return" for ccy in CURRENCIES]
calday_cols = [f"{ccy}_calendar_days" for ccy in CURRENCIES]

print("Optimizing lambdas for GK model...")
results_gk, best_gk = optimize_lambdas(df_model, ret_cols, calday_cols, "gk")
print(f"  GK optimal: lambda_cov={best_gk[0]}, lambda_idio={best_gk[1]}")

print("Optimizing lambdas for Parkinson model...")
results_pk, best_pk = optimize_lambdas(df_model, ret_cols, calday_cols, "parkinson")
print(f"  Parkinson optimal: lambda_cov={best_pk[0]}, lambda_idio={best_pk[1]}")"""))

C.append(mc("### 5.2 — Lambda Heatmaps"))

C.append(cc("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (res_df, name, best) in enumerate([(results_gk, "GK", best_gk), (results_pk, "Parkinson", best_pk)]):
    ax = axes[i]
    pivot = res_df.pivot(index="lam_idio", columns="lam_cov", values="ll")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.2f}" for x in pivot.index], fontsize=8)
    ax.set_xlabel("lambda_cov")
    ax.set_ylabel("lambda_idio")
    ax.set_title(f"{name} — Log-Likelihood\\nBest: cov={best[0]}, idio={best[1]}")
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()"""))

C.append(mc("### 5.3 — Lambda Override (uncomment to override)"))

C.append(cc("""# Uncomment and set manually if desired:
# best_gk = (0.97, 0.94)
# best_pk = (0.97, 0.94)
print(f"Using lambdas — GK: cov={best_gk[0]}, idio={best_gk[1]} | Parkinson: cov={best_pk[0]}, idio={best_pk[1]}")"""))

# ===================== SECTION 6 =====================
C.append(mc("""## 6. Model Execution

Run both models end-to-end. They share the EWMA covariance structure but differ in idiosyncratic variance estimation."""))

C.append(mc("### 6.1 — Shared: EWMA Covariance & Betas"))

C.append(cc("""# Use GK lambda_cov (shared for both models)
lam_cov_shared = best_gk[0]
target_idx = 0
reg_indices = [1, 2, 3, 4]

cov_dict, df_corr, df_ewma_var = compute_ewma_covariance(df_model, ret_cols, calday_cols, lam_cov_shared)
betas_df = compute_betas(cov_dict, df_model, ret_cols, target_idx, reg_indices)

print(f"Covariance computed: {len(cov_dict)} dates")
print("\\nCurrent betas:")
last_b = betas_df.iloc[-1]
for reg in REGRESSORS:
    print(f"  {reg.upper()}: {last_b[f'beta_{reg}']:.3f}")
print(f"  R2: {last_b['r_squared']:.1%}")

display(betas_df[[f"beta_{r}" for r in REGRESSORS] + ["r_squared"]].tail(10).round(4))"""))

C.append(mc("### 6.2 — Model 1: Garman-Klass"))

C.append(cc("""var_gk = compute_idio_var(betas_df, cov_dict, df_model, best_gk[1], method="gk",
                         ret_cols=ret_cols, target_idx=target_idx, reg_indices=reg_indices)
monitor_gk = build_monitor(df_model, betas_df, var_gk, df_corr, "GK")
monitor_gk.to_csv("monitor_gk.csv")

last_gk = var_gk.iloc[-1]
print(f"Model 1 (GK) — lambda_idio={best_gk[1]}")
print(f"  Systematic: {last_gk['systematic_pct']:.1%}, Idio vol: {np.sqrt(last_gk['idiosyncratic_var'])*10000:.1f} bps")
print(f"  Total vol: {np.sqrt(last_gk['total_var'])*10000:.1f} bps")
print(f"  95% range: {monitor_gk['range_95_pips'].iloc[-1]:.0f} pips")"""))

C.append(mc("### 6.3 — Model 2: Parkinson"))

C.append(cc("""var_pk = compute_idio_var(betas_df, cov_dict, df_model, best_pk[1], method="parkinson",
                         ret_cols=ret_cols, target_idx=target_idx, reg_indices=reg_indices)
monitor_pk = build_monitor(df_model, betas_df, var_pk, df_corr, "Parkinson")
monitor_pk.to_csv("monitor_parkinson.csv")

last_pk = var_pk.iloc[-1]
print(f"Model 2 (Parkinson) — lambda_idio={best_pk[1]}")
print(f"  Systematic: {last_pk['systematic_pct']:.1%}, Idio vol: {np.sqrt(last_pk['idiosyncratic_var'])*10000:.1f} bps")
print(f"  Total vol: {np.sqrt(last_pk['total_var'])*10000:.1f} bps")
print(f"  95% range: {monitor_pk['range_95_pips'].iloc[-1]:.0f} pips")"""))

C.append(mc("### 6.4 — Side-by-Side Comparison (Last 10 Days)"))

C.append(cc("""compare_cols = ["usdpen_fixing", "expected_return", "residual", "idio_vol_bps", "range_95_pips", "systematic_pct"]
gk_last = monitor_gk[compare_cols].tail(10).round(4).copy()
pk_last = monitor_pk[compare_cols].tail(10).round(4).copy()
gk_last.columns = [f"GK_{c}" for c in compare_cols]
pk_last.columns = [f"PK_{c}" for c in compare_cols]
combined = pd.concat([gk_last, pk_last], axis=1)
display(combined)"""))

# ===================== SECTION 7 =====================
C.append(mc("""## 7. Live Intraday Widget

Two use cases:
1. **Pre-open**: Set regional levels observed in the market, see where PEN fixing "should" print
2. **Intraday repricing**: As PEN trades, see where it sits vs the conditional distribution

Parameters (betas, idio_var) are fixed at last estimates — correct because updating requires a completed fixing cycle."""))

C.append(mc("### 7.1 — Widget"))

C.append(cc("""from IPython.display import clear_output as _clear

last_fix = df_model.iloc[-1]
next_date = (df_model.index[-1] + pd.tseries.offsets.BDay(1)).date()

w_date = widgets.Text(value=str(next_date), description="Date:", layout=widgets.Layout(width="180px"))
w_pen_open = widgets.FloatText(value=round(float(last_fix.get("pen_open", last_fix["pen_fixing"])), 4),
                                description="PEN Open:", step=0.0001, layout=widgets.Layout(width="180px"))
w_pen_curr = widgets.FloatText(value=round(float(last_fix["pen_fixing"]), 4),
                                description="PEN Current:", step=0.0001, layout=widgets.Layout(width="180px"))
w_cop = widgets.FloatText(value=round(float(last_fix.get("cop_fixing", 0)), 2),
                           description="COP:", step=0.01, layout=widgets.Layout(width="180px"))
w_clp = widgets.FloatText(value=round(float(last_fix.get("clp_fixing", 0)), 2),
                           description="CLP:", step=0.01, layout=widgets.Layout(width="180px"))
w_mxn = widgets.FloatText(value=round(float(last_fix.get("mxn_fixing", 0)), 4),
                           description="MXN:", step=0.0001, layout=widgets.Layout(width="180px"))
w_brl = widgets.FloatText(value=round(float(last_fix.get("brl_fixing", 0)), 4),
                           description="BRL:", step=0.0001, layout=widgets.Layout(width="180px"))
w_model = widgets.Dropdown(options=["GK", "Parkinson", "Both"], value="Both", description="Model:")
btn = widgets.Button(description="Compute Range", button_style="primary", icon="calculator")
out = widgets.Output()

def on_compute(b):
    with out:
        _clear(wait=True)

        prev_fix = {ccy: float(last_fix[f"{ccy}_fixing"]) for ccy in CURRENCIES}
        new_levels = {"cop": w_cop.value, "clp": w_clp.value, "mxn": w_mxn.value, "brl": w_brl.value}
        reg_rets = {ccy: np.log(new_levels[ccy] / prev_fix[ccy]) for ccy in REGRESSORS}

        betas = {f"beta_{ccy}": betas_df.iloc[-1][f"beta_{ccy}"] for ccy in REGRESSORS}
        models = []
        if w_model.value in ("GK", "Both"):
            models.append(("GK", var_gk["idiosyncratic_var"].iloc[-1]))
        if w_model.value in ("Parkinson", "Both"):
            models.append(("Parkinson", var_pk["idiosyncratic_var"].iloc[-1]))

        for mname, idio_v in models:
            cr = conditional_range(reg_rets, betas, idio_v, prev_fix["pen"],
                                   pen_open=w_pen_open.value, pen_current=w_pen_curr.value)

            L = []
            L.append("=" * 66)
            L.append(f"  USDPEN CONDITIONAL RANGE — {w_date.value}  [{mname}]")
            L.append("=" * 66)

            L.append("  REGIONAL MOVES TODAY")
            for ccy in REGRESSORS:
                pct = reg_rets[ccy] * 100
                L.append(f"    USD{ccy.upper()}: {new_levels[ccy]:.4f}  ({pct:+.2f}%)")

            L.append("")
            L.append("  EXPECTED PEN FIXING DISTRIBUTION")
            L.append(f"    Prev fixing: {prev_fix['pen']:.4f}")
            L.append(f"    {'Pctl':>6}  {'Level':>10}  {'Pips':>8}  {'%Chg':>8}")
            L.append(f"    {'-'*40}")
            for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                p = cr["percentiles"][pct]
                L.append(f"    {pct:>5}%  {p['level']:>10.4f}  {p['pips']:>+8.0f}  {p['pct']:>+7.2f}%")
            L.append(f"    Idio vol: {cr['idio_vol_bps']:.1f} bps")

            if "current_z" in cr:
                L.append("")
                L.append("  PEN CURRENT POSITION")
                L.append(f"    Current: {w_pen_curr.value:.4f}  (z={cr['current_z']:+.2f}, pctl={cr['current_percentile']:.0f}%)")
                L.append(f"    vs Expected: {cr['current_vs_expected_pips']:+.0f} pips")
                sig = "RICH" if cr["current_z"] < -1.5 else ("CHEAP" if cr["current_z"] > 1.5 else "FAIR")
                L.append(f"    Signal: {sig}")

            L.append("")
            L.append("  CONTRIBUTION BREAKDOWN")
            for ccy in REGRESSORS:
                c = cr["contributions"][ccy]
                L.append(f"    {ccy.upper()}: {c['pips']:+.1f} pips  (beta={c['beta']:.3f}, ret={c['return']*100:+.2f}%)")

            L.append("=" * 66)
            print("\\n".join(L))

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 2.5))
            labels = [c.upper() for c in REGRESSORS]
            vals = [cr["contributions"][c]["pips"] for c in REGRESSORS]
            ax.barh(labels, vals, color=[COLORS[c] for c in REGRESSORS])
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Contribution (pips)")
            ax.set_title(f"Contribution to Expected PEN Move [{mname}]")
            plt.tight_layout()
            plt.show()

btn.on_click(on_compute)
row1 = widgets.HBox([w_date, w_pen_open, w_pen_curr, w_model])
row2 = widgets.HBox([w_cop, w_clp, w_mxn, w_brl])
display(widgets.VBox([row1, row2, btn, out]))"""))

# ===================== SECTION 8 =====================
C.append(mc("""## 8. Charts & Diagnostics

Visual diagnostics for both models. Regime shading: light gray = BCRP intervention (Nov25–Feb26), light red = Iran conflict (Mar26+)."""))

C.append(mc("### 8.1 — Betas Over Time\nLook for: stability of betas across regimes, any sign changes, beta sum evolution."))

C.append(cc("""fig, ax = plt.subplots(figsize=(14, 5))
for reg in REGRESSORS:
    ax.plot(betas_df.index, betas_df[f"beta_{reg}"], label=reg.upper(), color=COLORS[reg], linewidth=1.2)
shade_regimes(ax)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("USDPEN Betas on Regional Currencies")
ax.set_ylabel("Beta")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()"""))

C.append(mc("### 8.2 — Variance Decomposition\nLook for: regime shifts in systematic vs idiosyncratic, intervention period compression."))

C.append(cc("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for i, (vdf, name) in enumerate([(var_gk, "GK"), (var_pk, "Parkinson")]):
    ax = axes[i]
    ax.fill_between(vdf.index, 0, vdf["systematic_var"]*1e8, alpha=0.7, color=COLOR_SYS, label="Systematic")
    ax.fill_between(vdf.index, vdf["systematic_var"]*1e8, vdf["total_var"]*1e8, alpha=0.7, color=COLOR_IDIO, label="Idiosyncratic")
    shade_regimes(ax)
    ax.set_title(f"Variance Decomposition — {name}")
    ax.set_ylabel("Var (x1e-8)")
    ax.legend(loc="upper left")
plt.tight_layout()
plt.show()"""))

C.append(mc("### 8.3 — GK vs Parkinson Idio Vol\nLook for: divergence between estimators, which reacts faster to regime shifts."))

C.append(cc("""fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(var_gk.index, np.sqrt(var_gk["idiosyncratic_var"])*10000, label="GK", color=COLOR_SYS, linewidth=1.2)
ax.plot(var_pk.index, np.sqrt(var_pk["idiosyncratic_var"])*10000, label="Parkinson", color=COLOR_IDIO, linewidth=1.2)
shade_regimes(ax)
ax.set_title("Idiosyncratic Vol Comparison: GK vs Parkinson")
ax.set_ylabel("Vol (bps)")
ax.legend()
plt.tight_layout()
plt.show()"""))

C.append(mc("### 8.4 — Average Pairwise Correlation\nLook for: correlation spikes during stress, drops during intervention."))

C.append(cc("""fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df_corr.index, df_corr.mean(axis=1), color=COLOR_SYS, linewidth=1.2)
shade_regimes(ax)
ax.set_title("Average Pairwise LatAm FX Correlation")
ax.set_ylabel("Avg Correlation")
plt.tight_layout()
plt.show()"""))

C.append(mc("### 8.5 — Conditional Range: Last 3 Months — GK\nLook for: actual moves inside bands, breach clustering."))

C.append(cc("""def plot_cond_range(mon, title, n_days=None):
    df_p = mon.dropna(subset=["range_95_pips", "usdpen_return"]).copy()
    if n_days:
        df_p = df_p.tail(n_days)
    spot = df_p["usdpen_fixing"]
    actual_pips = df_p["usdpen_return"] * spot / PIP
    exp_pips = df_p["expected_return"] * spot / PIP
    idio_std = np.sqrt(df_p["idiosyncratic_var"])

    fig, ax = plt.subplots(figsize=(14, 5))
    for ci, z, a, lbl in [(0.99, 2.5758, 0.12, "99%"), (0.95, 1.96, 0.25, "95%")]:
        hi = exp_pips + z * idio_std * spot / PIP
        lo = exp_pips - z * idio_std * spot / PIP
        ax.fill_between(df_p.index, lo, hi, alpha=a, color=COLOR_SYS, label=f"{lbl} CI")
    ax.plot(df_p.index, exp_pips, color=COLOR_TOTAL, linewidth=1, label="Expected")
    ax.scatter(df_p.index, actual_pips, s=10, color=COLOR_TOTAL, alpha=0.5, label="Actual", zorder=5)
    shade_regimes(ax)
    ax.set_title(title)
    ax.set_ylabel("Pips")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

plot_cond_range(monitor_gk, "USDPEN Daily Moves vs Model Bands — GK (Last 3 Months)", n_days=63)"""))

C.append(mc("### 8.6 — Conditional Range: Last 3 Months — Parkinson"))

C.append(cc("""plot_cond_range(monitor_pk, "USDPEN Daily Moves vs Model Bands — Parkinson (Last 3 Months)", n_days=63)"""))

C.append(mc("### 8.7 — Actual Percentile Histogram\nLook for: uniform distribution = well-calibrated model. Peaks at 0 or 100 = tails too narrow."))

C.append(cc("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, (mon, name) in enumerate([(monitor_gk, "GK"), (monitor_pk, "Parkinson")]):
    ax = axes[i]
    # Compute actual percentile for each day
    valid = mon.dropna(subset=["usdpen_return", "expected_return", "idiosyncratic_var"])
    z_actual = (valid["usdpen_return"] - valid["expected_return"]) / np.sqrt(valid["idiosyncratic_var"])
    pct_actual = stats.norm.cdf(z_actual) * 100
    ax.hist(pct_actual, bins=20, range=(0, 100), density=True, alpha=0.7, color=COLOR_SYS, edgecolor="white")
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1, label="Uniform")
    ax.set_xlabel("Actual Percentile")
    ax.set_ylabel("Density")
    ax.set_title(f"Actual Percentile Distribution — {name}")
    ax.legend()
plt.tight_layout()
plt.show()"""))

C.append(mc("### 8.8 — Residual Z-Score\nLook for: sustained z > 1.5 = PEN cheap vs region, z < -1.5 = PEN rich."))

C.append(cc("""fig, ax = plt.subplots(figsize=(14, 4))
z_vals = monitor_gk["residual_zscore"].dropna()
colors_z = ["#2563EB" if v < -1.5 else "#DC2626" if v > 1.5 else "#9CA3AF" for v in z_vals]
ax.scatter(z_vals.index, z_vals.values, c=colors_z, s=10, alpha=0.7)
ax.axhline(1.5, color="#DC2626", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(-1.5, color="#2563EB", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(2.0, color="#DC2626", linestyle=":", linewidth=0.8, alpha=0.3)
ax.axhline(-2.0, color="#2563EB", linestyle=":", linewidth=0.8, alpha=0.3)
shade_regimes(ax)
ax.set_title("USDPEN Idiosyncratic Residual Z-Score")
ax.set_ylabel("Z-Score")
plt.tight_layout()
plt.show()"""))

# ===================== SECTION 9 =====================
C.append(mc("""## 9. Backtesting Framework

Testing probability statement accuracy:
- **Coverage**: H0 = breach rate equals theoretical → Christoffersen LR test
- **Independence**: H0 = no breach clustering → 2×2 transition LR test
- **Calibration**: H0 = conditional CDF values ~ Uniform(0,1) → KS test
- **Basel traffic light**: rolling 99% breach count (Green ≤4, Yellow 5-9, Red ≥10)
- **Market-making simulation**: fade strategy on breach days"""))

C.append(mc("### 9.1 — Backtest Functions"))

C.append(cc("""def backtest_coverage(mon_df, train_window=120):
    df = mon_df.dropna(subset=["usdpen_return", "expected_return", "idiosyncratic_var"]).copy()
    if len(df) <= train_window:
        return None
    test = df.iloc[train_window:]
    actual = test["usdpen_return"].values
    expected = test["expected_return"].values
    idio_std = np.sqrt(test["idiosyncratic_var"].values)
    results = {"n_test": len(test), "test_dates": test.index}

    for ci, z in CI_LEVELS.items():
        upper = expected + z * idio_std
        lower = expected - z * idio_std
        breaches = (actual > upper) | (actual < lower)
        n1 = int(breaches.sum())
        n0 = len(breaches) - n1
        n_total = len(breaches)
        emp_rate = n1 / n_total
        theo_rate = 1 - ci

        # Unconditional coverage LR
        p0 = theo_rate
        p1 = max(min(emp_rate, 1 - 1e-10), 1e-10)
        log_l0 = n0 * np.log(1 - p0) + n1 * np.log(p0)
        log_l1 = n0 * np.log(1 - p1) + n1 * np.log(p1)
        lr_uc = max(-2 * (log_l0 - log_l1), 0)
        pval_uc = 1 - stats.chi2.cdf(lr_uc, 1)

        # Independence LR
        b = breaches.astype(int)
        t00 = t01 = t10 = t11 = 0
        for k in range(1, len(b)):
            if b[k-1]==0 and b[k]==0: t00 += 1
            elif b[k-1]==0 and b[k]==1: t01 += 1
            elif b[k-1]==1 and b[k]==0: t10 += 1
            else: t11 += 1

        eps = 1e-10
        pi01 = max(t01/(t00+t01), eps) if (t00+t01)>0 else eps
        pi11 = max(t11/(t10+t11), eps) if (t10+t11)>0 else eps
        pi = max((t01+t11)/n_total, eps)
        log_l1_ind = t00*np.log(1-pi01+eps) + t01*np.log(pi01+eps)
        if (t10+t11)>0:
            log_l1_ind += t10*np.log(1-pi11+eps) + t11*np.log(pi11+eps)
        log_l0_ind = (t00+t10)*np.log(1-pi+eps) + (t01+t11)*np.log(pi+eps)
        lr_ind = max(2*(log_l1_ind - log_l0_ind), 0)
        pval_ind = 1 - stats.chi2.cdf(lr_ind, 1)
        lr_cc = lr_uc + lr_ind
        pval_cc = 1 - stats.chi2.cdf(lr_cc, 2)

        results[ci] = {
            "theo_rate": theo_rate, "emp_rate": emp_rate,
            "n_breaches": n1, "n_total": n_total,
            "lr_uc": lr_uc, "pval_uc": pval_uc,
            "lr_ind": lr_ind, "pval_ind": pval_ind,
            "lr_cc": lr_cc, "pval_cc": pval_cc,
            "breach_dates": test.index[breaches],
        }
    return results


def pit_test(mon_df, train_window=120):
    df = mon_df.dropna(subset=["usdpen_return", "expected_return", "idiosyncratic_var"]).copy()
    test = df.iloc[train_window:]
    z = (test["usdpen_return"].values - test["expected_return"].values) / np.sqrt(test["idiosyncratic_var"].values)
    pit = stats.norm.cdf(z)
    ks_stat, ks_pval = stats.kstest(pit, "uniform")
    return {"ks_stat": ks_stat, "ks_pval": ks_pval, "pit_values": pit}


def basel_traffic_light(mon_df, window=250, train_window=120):
    df = mon_df.dropna(subset=["usdpen_return", "expected_return", "idiosyncratic_var"]).copy()
    test = df.iloc[train_window:]
    actual = test["usdpen_return"].values
    exp = test["expected_return"].values
    idio_std = np.sqrt(test["idiosyncratic_var"].values)
    z = CI_LEVELS[0.99]
    breaches = ((actual > exp + z*idio_std) | (actual < exp - z*idio_std)).astype(int)
    bs = pd.Series(breaches, index=test.index)
    ew = min(window, len(bs))
    rc = bs.rolling(ew, min_periods=1).sum()
    cl = rc.apply(lambda x: "Green" if x<=4 else ("Yellow" if x<=9 else "Red"))
    return rc, cl, ew


def mm_backtest(mon_df, ci_level=0.95, holding_days=1, cost_pips=3, train_window=120):
    df = mon_df.dropna(subset=["usdpen_return", "expected_return", "idiosyncratic_var", "usdpen_fixing"]).copy()
    test = df.iloc[train_window:]
    z = CI_LEVELS[ci_level]
    trades = []
    for i in range(len(test)):
        row = test.iloc[i]
        actual = row["usdpen_return"]
        exp = row["expected_return"]
        idio_std = np.sqrt(row["idiosyncratic_var"])
        spot = row["usdpen_fixing"]
        if actual > exp + z*idio_std or actual < exp - z*idio_std:
            direction = "sell" if actual > exp + z*idio_std else "buy"
            if i + holding_days < len(test):
                next_ret = test.iloc[i+holding_days]["usdpen_return"]
                pnl = (-next_ret if direction=="sell" else next_ret) * spot / PIP - cost_pips
                trades.append({"date": test.index[i], "direction": direction,
                              "pnl_pips": pnl, "regime": label_regime(test.index[i])})
    if not trades:
        return {"total_trades": 0, "trade_df": pd.DataFrame()}
    tdf = pd.DataFrame(trades).set_index("date")
    tdf["cumul_pnl"] = tdf["pnl_pips"].cumsum()
    n = len(tdf)
    wr = (tdf["pnl_pips"]>0).mean()
    avg = tdf["pnl_pips"].mean()
    vol = tdf["pnl_pips"].std()
    sharpe = avg / vol * np.sqrt(252) if vol > 0 else 0
    mdd = (tdf["cumul_pnl"] - tdf["cumul_pnl"].cummax()).min()
    return {"total_trades": n, "win_rate": wr, "avg_pnl": avg,
            "gross_mean": avg + cost_pips, "cost_drag": cost_pips,
            "daily_vol": vol, "sharpe": sharpe, "max_dd": mdd, "trade_df": tdf}

print("Backtest functions defined.")"""))

C.append(mc("### 9.2 — Run Backtests"))

C.append(cc("""for mon, name in [(monitor_gk, "GK"), (monitor_pk, "Parkinson")]:
    print(f"\\n{'='*70}")
    print(f"  BACKTEST RESULTS — {name}")
    print(f"{'='*70}")

    cov_res = backtest_coverage(mon)
    if cov_res:
        n = cov_res["n_test"]
        print(f"\\n  Coverage Test ({n} OOS days):")
        print(f"  {'CI':>6} | {'Theo':>8} | {'Empirical':>10} | {'Breaches':>8} | {'LR_uc':>8} | {'p_uc':>8} | {'Reject?':>8}")
        print(f"  {'-'*70}")
        for ci in [0.90, 0.95, 0.99]:
            r = cov_res[ci]
            reject = "YES" if r["pval_uc"] < 0.05 else "no"
            print(f"  {ci:6.0%} | {r['theo_rate']:8.1%} | {r['emp_rate']:10.1%} | "
                  f"{r['n_breaches']:8d} | {r['lr_uc']:8.3f} | {r['pval_uc']:8.3f} | {reject:>8}")

        r95 = cov_res[0.95]
        print(f"\\n  Independence (95%): LR={r95['lr_ind']:.3f}, p={r95['pval_ind']:.3f} — H0 {'rejected' if r95['pval_ind']<0.05 else 'not rejected'}")
        print(f"  Cond Coverage (95%): LR={r95['lr_cc']:.3f}, p={r95['pval_cc']:.3f} — H0 {'rejected' if r95['pval_cc']<0.05 else 'not rejected'}")

    pit = pit_test(mon)
    print(f"\\n  PIT Test: KS={pit['ks_stat']:.4f}, p={pit['ks_pval']:.4f} — H0 (uniform) {'rejected' if pit['ks_pval']<0.05 else 'not rejected'}")

    mm = mm_backtest(mon)
    if mm["total_trades"] > 0:
        print(f"\\n  Fade Strategy: {mm['total_trades']} trades, WR={mm['win_rate']:.1%}, avg={mm['avg_pnl']:.1f} pips, Sharpe={mm['sharpe']:.2f}")"""))

C.append(mc("### 9.3 — Backtest Charts"))

C.append(cc("""for mon, name in [(monitor_gk, "GK"), (monitor_pk, "Parkinson")]:
    cov_res = backtest_coverage(mon)
    pit = pit_test(mon)
    mm = mm_backtest(mon)
    rc, cl, ew = basel_traffic_light(mon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Coverage bars
    ax = axes[0, 0]
    if cov_res:
        ci_labels = ["90%", "95%", "99%"]
        theo = [cov_res[ci]["theo_rate"]*100 for ci in [0.90, 0.95, 0.99]]
        emp = [cov_res[ci]["emp_rate"]*100 for ci in [0.90, 0.95, 0.99]]
        x = np.arange(3); w = 0.35
        ax.bar(x-w/2, theo, w, label="Theoretical", color=COLOR_SYS, alpha=0.8)
        ax.bar(x+w/2, emp, w, label="Empirical", color=COLOR_IDIO, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(ci_labels)
        ax.set_ylabel("Breach %"); ax.set_title(f"Coverage — {name}"); ax.legend()

    # PIT histogram
    ax = axes[0, 1]
    ax.hist(pit["pit_values"], bins=20, range=(0,1), density=True, alpha=0.7, color=COLOR_SYS, edgecolor="white")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"PIT Histogram — {name}")

    # Basel traffic light
    ax = axes[1, 0]
    ax.plot(rc.index, rc.values, color=COLOR_TOTAL, linewidth=1.2)
    cmap = {"Green": "#10B981", "Yellow": "#F59E0B", "Red": "#EF4444"}
    for i in range(1, len(rc)):
        ax.axvspan(rc.index[i-1], rc.index[i], alpha=0.12, color=cmap.get(cl.iloc[i], "gray"))
    ax.axhline(4, color="#F59E0B", linestyle="--", linewidth=0.8)
    ax.axhline(9, color="#EF4444", linestyle="--", linewidth=0.8)
    ax.set_title(f"Basel Traffic Light — {name}")
    ax.set_ylabel("99% Breaches")

    # Cumulative P&L
    ax = axes[1, 1]
    if mm["total_trades"] > 0:
        tdf = mm["trade_df"]
        ax.plot(tdf.index, tdf["cumul_pnl"], color=COLOR_TOTAL, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Fade Strategy P&L — {name}")
    ax.set_ylabel("Cumul P&L (pips)")

    plt.suptitle(f"Backtest Dashboard — {name}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()"""))

# ===================== SECTION 10 =====================
C.append(mc("""## 10. Model Comparison & Winner Selection

Score both models on 5 criteria:
1. Log-likelihood (higher = better calibrated)
2. PIT KS p-value (higher = more uniform → better calibration)
3. |95% breach rate - 5%| (lower = better)
4. |99% breach rate - 1%| (lower = better)
5. Conditional coverage p-value (higher = better)

Winner = best on 3+ criteria."""))

C.append(mc("### 10.1 — Comparison Table"))

C.append(cc("""scores = {}
for mon, name, lambdas in [(monitor_gk, "GK", best_gk), (monitor_pk, "Parkinson", best_pk)]:
    cov_res = backtest_coverage(mon)
    pit = pit_test(mon)
    mm = mm_backtest(mon)

    # Compute LL
    df_t = mon.dropna(subset=["usdpen_return", "expected_return", "idiosyncratic_var"]).iloc[120:]
    a = df_t["usdpen_return"].values
    e = df_t["expected_return"].values
    v = df_t["idiosyncratic_var"].values
    valid = (v > 0)
    ll = np.mean(-0.5 * np.log(2*np.pi*v[valid]) - 0.5*(a[valid]-e[valid])**2/v[valid])

    scores[name] = {
        "lam_cov": lambdas[0], "lam_idio": lambdas[1],
        "ll": ll, "ks_pval": pit["ks_pval"],
        "breach_95_err": abs(cov_res[0.95]["emp_rate"] - 0.05) if cov_res else 1,
        "breach_99_err": abs(cov_res[0.99]["emp_rate"] - 0.01) if cov_res else 1,
        "cc_pval": cov_res[0.95]["pval_cc"] if cov_res else 0,
        "band_95_pips": mon["range_95_pips"].iloc[-1],
        "sharpe": mm["sharpe"] if mm["total_trades"] > 0 else 0,
        "max_dd": mm["max_dd"] if mm["total_trades"] > 0 else 0,
    }

comp = pd.DataFrame(scores).T
display(comp.round(4))"""))

C.append(mc("### 10.2 — Winner"))

C.append(cc("""# Score: higher is better for ll, ks_pval, cc_pval; lower is better for breach errors
gk = scores["GK"]
pk = scores["Parkinson"]
gk_wins = 0
pk_wins = 0

criteria = [
    ("Log-Likelihood", gk["ll"] > pk["ll"]),
    ("PIT KS p-value", gk["ks_pval"] > pk["ks_pval"]),
    ("95% Breach Error", gk["breach_95_err"] < pk["breach_95_err"]),
    ("99% Breach Error", gk["breach_99_err"] < pk["breach_99_err"]),
    ("Cond Coverage p", gk["cc_pval"] > pk["cc_pval"]),
]

print("Criterion-by-criterion comparison:")
for name, gk_better in criteria:
    winner = "GK" if gk_better else "Parkinson"
    if gk_better:
        gk_wins += 1
    else:
        pk_wins += 1
    print(f"  {name:>25}: {winner}")

print(f"\\nScore: GK {gk_wins} — Parkinson {pk_wins}")
winner = "GK" if gk_wins >= 3 else "Parkinson"
winner_params = best_gk if winner == "GK" else best_pk
print(f"\\n{'='*50}")
print(f"  WINNER: {winner}")
print(f"  Recommended parameters: lambda_cov={winner_params[0]}, lambda_idio={winner_params[1]}")
print(f"{'='*50}")"""))

# ===================== BUILD =====================
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": C,
}

with open("/home/user/RepoTRD/USDPEN_Dual_Model.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook generated: {len(C)} cells.")
