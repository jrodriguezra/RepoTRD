import numpy as np
import pandas as pd
from scipy import stats


# ── B1: Rolling beta estimation ─────────────────────────────────────────────

def estimate_betas(returns_df, target="r_pen", regressors=None, window=120, method="rolling", halflife=60):
    # Estimate USDPEN betas on regional currencies via rolling OLS or EWMA WLS
    if regressors is None:
        regressors = ["r_cop", "r_clp", "r_mxn", "r_brl"]

    dates = returns_df["date"].values
    y = returns_df[target].values
    X = returns_df[regressors].values
    n = len(y)

    beta_names = [f"beta_{r.replace('r_', '')}" for r in regressors]
    results = []

    if method == "rolling":
        for i in range(window, n):
            y_win = y[i - window:i]
            X_win = X[i - window:i]
            betas, r_sq = _ols(y_win, X_win)
            row = {"date": dates[i]}
            for j, name in enumerate(beta_names):
                row[name] = betas[j]
            row["r_squared"] = r_sq
            results.append(row)

    elif method == "ewma":
        for i in range(window, n):
            # Use all data up to i, with exponential weights
            y_win = y[:i]
            X_win = X[:i]
            lags = np.arange(len(y_win) - 1, -1, -1, dtype=float)
            weights = np.exp(-np.log(2) * lags / halflife)
            betas, r_sq = _wls(y_win, X_win, weights)
            row = {"date": dates[i]}
            for j, name in enumerate(beta_names):
                row[name] = betas[j]
            row["r_squared"] = r_sq
            results.append(row)

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _ols(y, X):
    # Simple OLS returning coefficients and R-squared
    X_c = np.column_stack([np.ones(len(y)), X])
    try:
        betas = np.linalg.lstsq(X_c, y, rcond=None)[0]
        fitted = X_c @ betas
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return betas[1:], r_sq  # skip intercept
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.nan


def _wls(y, X, weights):
    # Weighted least squares returning coefficients and R-squared
    w = np.sqrt(weights)
    y_w = y * w
    X_c = np.column_stack([np.ones(len(y)), X])
    X_w = X_c * w[:, np.newaxis]
    try:
        betas = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
        fitted = X_c @ betas
        resid = y - fitted
        ss_res = np.sum(weights * resid ** 2)
        y_mean = np.average(y, weights=weights)
        ss_tot = np.sum(weights * (y - y_mean) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return betas[1:], r_sq
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan), np.nan


# ── B2: EWMA covariance matrix ──────────────────────────────────────────────

def ewma_covariance(returns_df, columns=None, decay=0.94):
    # Compute EWMA covariance matrix for each date using RiskMetrics convention
    if columns is None:
        columns = ["r_pen", "r_cop", "r_clp", "r_mxn", "r_brl"]

    data = returns_df[columns].values
    dates = returns_df["date"].values
    n, k = data.shape

    # Initialize with sample covariance of first 20 observations
    init_window = min(20, n)
    cov = np.cov(data[:init_window].T)

    cov_matrices = {}
    var_df_rows = []
    corr_df_rows = []

    for i in range(1, n):
        r = data[i - 1].reshape(-1, 1)
        cov = decay * cov + (1 - decay) * (r @ r.T)
        cov_matrices[dates[i]] = cov.copy()

        # Store variances
        row = {"date": dates[i]}
        for j, col in enumerate(columns):
            row[f"var_{col}"] = cov[j, j]
        var_df_rows.append(row)

        # Store correlations
        std = np.sqrt(np.diag(cov))
        std_outer = np.outer(std, std)
        std_outer[std_outer == 0] = 1.0
        corr = cov / std_outer
        crow = {"date": dates[i]}
        for j in range(k):
            for l in range(j + 1, k):
                crow[f"corr_{columns[j]}_{columns[l]}"] = corr[j, l]
        corr_df_rows.append(crow)

    var_df = pd.DataFrame(var_df_rows)
    corr_df = pd.DataFrame(corr_df_rows)

    return cov_matrices, var_df, corr_df


# ── B3: Variance decomposition ──────────────────────────────────────────────

def variance_decomposition(betas_df, cov_matrices, returns_df,
                           target="r_pen", regressors=None, decay=0.94):
    # Decompose USDPEN variance into systematic and idiosyncratic components
    if regressors is None:
        regressors = ["r_cop", "r_clp", "r_mxn", "r_brl"]

    beta_cols = [f"beta_{r.replace('r_', '')}" for r in regressors]

    # Compute residuals from betas
    merged = betas_df.merge(returns_df[["date", target] + regressors], on="date")
    merged["expected"] = sum(merged[bc] * merged[reg] for bc, reg in zip(beta_cols, regressors))
    merged["residual"] = merged[target] - merged["expected"]

    # EWMA variance of residuals
    resid = merged["residual"].values
    idio_var = np.zeros(len(resid))
    idio_var[0] = resid[0] ** 2
    for i in range(1, len(resid)):
        idio_var[i] = decay * idio_var[i - 1] + (1 - decay) * resid[i - 1] ** 2

    # EWMA variance of USDPEN returns for comparison
    pen_ret = merged[target].values
    realized_var = np.zeros(len(pen_ret))
    realized_var[0] = pen_ret[0] ** 2
    for i in range(1, len(pen_ret)):
        realized_var[i] = decay * realized_var[i - 1] + (1 - decay) * pen_ret[i - 1] ** 2

    results = []
    for i, row in merged.iterrows():
        dt = row["date"]
        dt_key = pd.Timestamp(dt)

        if dt_key not in cov_matrices:
            continue

        beta_vec = np.array([row[bc] for bc in beta_cols])
        # Extract regional sub-matrix from the full covariance matrix
        # The cov_matrices are keyed by date with columns in order of
        # ["r_pen", "r_cop", "r_clp", "r_mxn", "r_brl"]
        full_cov = cov_matrices[dt_key]
        # Regional currencies are indices 1:5 in the full cov matrix
        reg_cov = full_cov[1:, 1:]

        sys_var = float(beta_vec @ reg_cov @ beta_vec)
        idio = idio_var[i]
        total = sys_var + idio
        sys_pct = sys_var / total if total > 0 else 0.0

        results.append({
            "date": dt,
            "systematic_var": sys_var,
            "idiosyncratic_var": idio,
            "total_var": total,
            "systematic_pct": sys_pct,
            "realized_var": realized_var[i],
            "residual": row["residual"],
            "expected_return": row["expected"]
        })

    return pd.DataFrame(results)


# ── B4: Conditional range calculator ─────────────────────────────────────────

def conditional_range(date_val, regional_returns_today, betas, idio_var, spot_level):
    # Compute conditional expected move and confidence intervals
    # regional_returns_today: dict or array of regional returns
    # betas: dict or array matching regional returns
    if isinstance(regional_returns_today, dict):
        regional_returns_today = np.array(list(regional_returns_today.values()))
    if isinstance(betas, dict):
        betas = np.array(list(betas.values()))

    expected_return = float(np.dot(betas, regional_returns_today))
    expected_move_pips = expected_return * spot_level * 10000

    idio_std = np.sqrt(idio_var)

    confidence_levels = [0.90, 0.95, 0.99]
    result = {
        "date": date_val,
        "spot_level": spot_level,
        "expected_return": expected_return,
        "expected_move_pips": expected_move_pips,
    }

    for cl in confidence_levels:
        z = stats.norm.ppf(1 - (1 - cl) / 2)
        lower_ret = expected_return - z * idio_std
        upper_ret = expected_return + z * idio_std

        lower_pips = lower_ret * spot_level * 10000
        upper_pips = upper_ret * spot_level * 10000
        lower_level = spot_level * np.exp(lower_ret)
        upper_level = spot_level * np.exp(upper_ret)

        cl_str = f"{int(cl * 100)}"
        result[f"lower_pip_{cl_str}"] = lower_pips
        result[f"upper_pip_{cl_str}"] = upper_pips
        result[f"lower_level_{cl_str}"] = lower_level
        result[f"upper_level_{cl_str}"] = upper_level
        result[f"band_width_pips_{cl_str}"] = upper_pips - lower_pips

    return result


# ── B5: Daily monitoring dashboard data ─────────────────────────────────────

def build_daily_monitor(returns_df, betas_df, var_decomp_df, corr_df, levels_df,
                        output_path="output"):
    # Build comprehensive daily monitoring DataFrame
    regressors = ["r_cop", "r_clp", "r_mxn", "r_brl"]
    beta_cols = [f"beta_{r.replace('r_', '')}" for r in regressors]

    # Start from var_decomp which has systematic/idiosyncratic vars
    monitor = var_decomp_df[["date", "systematic_var", "idiosyncratic_var",
                             "total_var", "systematic_pct", "residual",
                             "expected_return"]].copy()

    # Add levels and returns
    monitor = monitor.merge(
        returns_df[["date", "r_pen"]], on="date", how="left"
    )
    monitor = monitor.merge(
        levels_df[["date", "usdpen_mid"]], on="date", how="left"
    )

    monitor = monitor.rename(columns={"r_pen": "usdpen_return", "usdpen_mid": "usdpen_level"})

    # Rolling 20-day cumulative residual
    monitor["cumul_residual_20d"] = monitor["residual"].rolling(20).sum()

    # Residual z-score
    rolling_std = monitor["cumul_residual_20d"].rolling(60).std()
    monitor["residual_zscore"] = monitor["cumul_residual_20d"] / rolling_std

    # Average regional correlation
    corr_cols = [c for c in corr_df.columns if c.startswith("corr_")]
    monitor = monitor.merge(corr_df[["date"]], on="date", how="left")
    corr_merged = corr_df.copy()
    corr_merged["avg_regional_correlation"] = corr_merged[corr_cols].mean(axis=1)
    monitor = monitor.merge(
        corr_merged[["date", "avg_regional_correlation"]], on="date", how="left"
    )

    # Beta sum
    monitor = monitor.merge(betas_df[["date"] + beta_cols], on="date", how="left")
    monitor["pen_beta_sum"] = monitor[beta_cols].sum(axis=1)

    # 95% CI width in pips
    z_95 = stats.norm.ppf(0.975)
    monitor["range_95_pips"] = (
        2 * z_95 * np.sqrt(monitor["idiosyncratic_var"]) * monitor["usdpen_level"] * 10000
    )

    # Clean up
    keep_cols = [
        "date", "usdpen_return", "usdpen_level", "expected_return", "residual",
        "cumul_residual_20d", "residual_zscore", "systematic_var",
        "idiosyncratic_var", "total_var", "systematic_pct",
        "avg_regional_correlation", "pen_beta_sum", "range_95_pips"
    ]
    monitor = monitor[[c for c in keep_cols if c in monitor.columns]]
    monitor = monitor.sort_values("date").reset_index(drop=True)

    # Save
    import os
    os.makedirs(output_path, exist_ok=True)
    monitor.to_csv(os.path.join(output_path, "usdpen_daily_monitor.csv"), index=False)
    print(f"\nDaily monitor saved: {len(monitor)} rows")

    return monitor
