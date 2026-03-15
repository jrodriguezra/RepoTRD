import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# ── Regime definitions ───────────────────────────────────────────────────────

REGIME_DATES = {
    "normal": (None, pd.Timestamp("2025-10-31")),
    "intervention": (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-02-28")),
    "post_shock": (pd.Timestamp("2026-03-01"), None),
}


def _assign_regime(date_val):
    dt = pd.Timestamp(date_val)
    if dt <= REGIME_DATES["normal"][1]:
        return "normal"
    elif REGIME_DATES["intervention"][0] <= dt <= REGIME_DATES["intervention"][1]:
        return "intervention"
    else:
        return "post_shock"


# ── D1: Walk-forward coverage test ──────────────────────────────────────────

def backtest_coverage(returns_df, train_window=252, confidence_levels=None, decay=0.94):
    # Walk-forward backtest with Christoffersen coverage tests
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    target = "r_pen"
    regressors = ["r_cop", "r_clp", "r_mxn", "r_brl"]

    y = returns_df[target].values
    X = returns_df[regressors].values
    dates = returns_df["date"].values
    n = len(y)

    # Initialize EWMA variance of residuals
    # First pass: run OLS on training window to get initial residual variance
    y_train = y[:train_window]
    X_train = X[:train_window]
    X_c = np.column_stack([np.ones(train_window), X_train])
    betas = np.linalg.lstsq(X_c, y_train, rcond=None)[0]
    resid_train = y_train - X_c @ betas
    idio_var = np.var(resid_train)

    # Initialize EWMA covariance of regressors
    reg_cov = np.cov(X_train.T)

    records = []

    for i in range(train_window, n):
        # Use rolling window for betas
        start = max(0, i - train_window)
        y_win = y[start:i]
        X_win = X[start:i]
        X_c = np.column_stack([np.ones(len(y_win)), X_win])
        betas = np.linalg.lstsq(X_c, y_win, rcond=None)[0]
        beta_vec = betas[1:]

        # Compute expected return
        expected = float(np.dot(beta_vec, X[i]))
        actual = y[i]
        residual = actual - expected

        # Update EWMA idiosyncratic variance (using prior residual)
        if i > train_window:
            prev_resid = records[-1]["residual"]
            idio_var = decay * idio_var + (1 - decay) * prev_resid ** 2

        # Update EWMA covariance of regressors
        if i > train_window:
            r_prev = X[i - 1].reshape(-1, 1)
            reg_cov = decay * reg_cov + (1 - decay) * (r_prev @ r_prev.T)

        # Systematic variance
        sys_var = float(beta_vec @ reg_cov @ beta_vec)

        idio_std = np.sqrt(idio_var)

        # Check breaches for each confidence level
        breach_flags = {}
        for cl in confidence_levels:
            z = stats.norm.ppf(1 - (1 - cl) / 2)
            lower = expected - z * idio_std
            upper = expected + z * idio_std
            breach = (actual < lower) or (actual > upper)
            cl_str = f"{int(cl * 100)}"
            breach_flags[f"breach_{cl_str}"] = int(breach)
            breach_flags[f"lower_{cl_str}"] = lower
            breach_flags[f"upper_{cl_str}"] = upper

        records.append({
            "date": dates[i],
            "actual": actual,
            "expected": expected,
            "residual": residual,
            "idio_var": idio_var,
            "sys_var": sys_var,
            **breach_flags
        })

    bt_df = pd.DataFrame(records)

    # Compute test statistics
    results = {"backtest_df": bt_df}

    for cl in confidence_levels:
        cl_str = f"{int(cl * 100)}"
        breaches = bt_df[f"breach_{cl_str}"].values
        n_obs = len(breaches)
        n_breach = int(breaches.sum())
        empirical_rate = n_breach / n_obs
        theoretical_rate = 1 - cl

        # Christoffersen unconditional coverage LR test
        lr_uc = _lr_unconditional(n_obs, n_breach, theoretical_rate)

        # Christoffersen independence test
        lr_ind = _lr_independence(breaches)

        # Conditional coverage (joint)
        lr_cc = lr_uc["statistic"] + lr_ind["statistic"]
        pval_cc = 1 - stats.chi2.cdf(lr_cc, 2)

        results[cl_str] = {
            "n_obs": n_obs,
            "n_breaches": n_breach,
            "empirical_rate": empirical_rate,
            "theoretical_rate": theoretical_rate,
            "unconditional_coverage": lr_uc,
            "independence": lr_ind,
            "conditional_coverage": {"statistic": lr_cc, "p_value": pval_cc}
        }

    return results


def _lr_unconditional(n, n1, p0):
    # Likelihood ratio test for unconditional coverage
    n0 = n - n1
    p1 = n1 / n if n > 0 else 0

    if p1 == 0 or p1 == 1:
        return {"statistic": 0.0, "p_value": 1.0}

    lr = -2 * (n0 * np.log(1 - p0) + n1 * np.log(p0)
               - n0 * np.log(1 - p1) - n1 * np.log(p1))
    pval = 1 - stats.chi2.cdf(lr, 1)
    return {"statistic": float(lr), "p_value": float(pval)}


def _lr_independence(breaches):
    # Likelihood ratio test for independence of breaches
    # Build 2x2 transition matrix
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(breaches)):
        prev, curr = int(breaches[i - 1]), int(breaches[i])
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Transition probabilities
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

    if p == 0 or p == 1 or p01 == 0 or p01 == 1:
        return {"statistic": 0.0, "p_value": 1.0, "transition_matrix": [[n00, n01], [n10, n11]]}

    # Handle edge case where p11 is 0 or 1
    if p11 == 0 or p11 == 1:
        return {"statistic": 0.0, "p_value": 1.0, "transition_matrix": [[n00, n01], [n10, n11]]}

    lr_0 = n00 * np.log(1 - p) + n01 * np.log(p) + n10 * np.log(1 - p) + n11 * np.log(p)
    lr_1 = (n00 * np.log(1 - p01) + n01 * np.log(p01) +
            n10 * np.log(1 - p11) + n11 * np.log(p11))
    lr = -2 * (lr_0 - lr_1)
    pval = 1 - stats.chi2.cdf(lr, 1)

    return {
        "statistic": float(lr), "p_value": float(pval),
        "transition_matrix": [[n00, n01], [n10, n11]]
    }


# ── D2: PIT test ─────────────────────────────────────────────────────────────

def pit_test(actual_returns, expected_returns, conditional_variances):
    # Probability Integral Transform test
    pit_values = stats.norm.cdf(
        actual_returns,
        loc=expected_returns,
        scale=np.sqrt(conditional_variances)
    )

    # Remove NaN/inf
    pit_values = pit_values[np.isfinite(pit_values)]

    # KS test against uniform
    ks_stat, ks_pval = stats.kstest(pit_values, "uniform")

    # Histogram data
    hist_counts, hist_edges = np.histogram(pit_values, bins=20, range=(0, 1))
    hist_freq = hist_counts / len(pit_values)

    return {
        "pit_values": pit_values,
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_pval),
        "hist_counts": hist_counts,
        "hist_edges": hist_edges,
        "hist_freq": hist_freq,
        "n_obs": len(pit_values)
    }


# ── D3: Regime-segmented diagnostics ────────────────────────────────────────

def regime_diagnostics(backtest_df, confidence_levels=None):
    # Compute diagnostics segmented by regime
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    bt = backtest_df.copy()
    bt["regime"] = bt["date"].apply(_assign_regime)

    from scipy import stats as sp_stats
    results = []
    for regime in ["normal", "intervention", "post_shock"]:
        subset = bt[bt["regime"] == regime]
        if len(subset) == 0:
            continue

        row = {"regime": regime, "n_obs": len(subset)}

        for cl in confidence_levels:
            cl_str = f"{int(cl * 100)}"
            breach_col = f"breach_{cl_str}"
            if breach_col in subset.columns:
                row[f"breach_rate_{cl_str}"] = subset[breach_col].mean()

        # Average band width (95%)
        if "upper_95" in subset.columns and "lower_95" in subset.columns:
            band_width = (subset["upper_95"] - subset["lower_95"])
            # Convert to pips: assume average level ~ 3.7
            row["avg_band_width_pips"] = band_width.mean() * 37000  # approximate

        # Average absolute residual
        row["avg_abs_residual"] = subset["residual"].abs().mean()

        # Mean reversion speed: average days for z-score to cross back through zero
        # after exceeding ±1.5 (simplified: use residual z-score from cumulative residual)
        resid = subset["residual"].values
        cumul = pd.Series(resid).rolling(20).sum()
        rolling_std = cumul.rolling(60).std()
        zscore = cumul / rolling_std

        exceedances = []
        in_exceedance = False
        days_count = 0
        for z in zscore:
            if np.isnan(z):
                continue
            if not in_exceedance and abs(z) > 1.5:
                in_exceedance = True
                days_count = 0
            elif in_exceedance:
                days_count += 1
                if abs(z) <= 0:
                    exceedances.append(days_count)
                    in_exceedance = False

        row["mean_reversion_days"] = np.mean(exceedances) if exceedances else np.nan

        results.append(row)

    return pd.DataFrame(results)


# ── D4: Basel traffic light test ─────────────────────────────────────────────

def basel_traffic_light(breach_series_99, window=250):
    # Rolling count of 99% breaches with Basel traffic light classification
    breach_count = breach_series_99.rolling(window, min_periods=1).sum()

    def _classify(count):
        if count <= 4:
            return "Green"
        elif count <= 9:
            return "Yellow"
        else:
            return "Red"

    classification = breach_count.apply(_classify)

    return pd.DataFrame({
        "breach_count": breach_count,
        "traffic_light": classification
    })


# ── D5: Market-making simulation ─────────────────────────────────────────────

def mm_simulation(daily_monitor_df, entry_threshold=0.95,
                  holding_period=1, transaction_cost_pips=3):
    # Simulate market-making fade strategy
    from scipy import stats as sp_stats

    z = sp_stats.norm.ppf(1 - (1 - entry_threshold) / 2)
    monitor = daily_monitor_df.copy().reset_index(drop=True)

    actual_pips = monitor["usdpen_return"] * monitor["usdpen_level"] * 10000
    expected_pips = monitor["expected_return"] * monitor["usdpen_level"] * 10000
    idio_std_pips = np.sqrt(monitor["idiosyncratic_var"]) * monitor["usdpen_level"] * 10000

    upper_band = expected_pips + z * idio_std_pips
    lower_band = expected_pips - z * idio_std_pips

    trades = []
    for i in range(len(monitor) - holding_period):
        if actual_pips.iloc[i] > upper_band.iloc[i]:
            # Short USDPEN (fade the move)
            entry_excess = actual_pips.iloc[i] - expected_pips.iloc[i]
            # P&L = entry_excess minus next day's move continuation
            exit_return = monitor["usdpen_return"].iloc[i + 1:i + 1 + holding_period].sum()
            exit_pips = exit_return * monitor["usdpen_level"].iloc[i] * 10000
            pnl = entry_excess - exit_pips - transaction_cost_pips
            trades.append({
                "date": monitor["date"].iloc[i],
                "direction": "short",
                "entry_excess_pips": entry_excess,
                "pnl_pips": pnl,
                "regime": _assign_regime(monitor["date"].iloc[i])
            })
        elif actual_pips.iloc[i] < lower_band.iloc[i]:
            # Long USDPEN
            entry_excess = expected_pips.iloc[i] - actual_pips.iloc[i]
            exit_return = monitor["usdpen_return"].iloc[i + 1:i + 1 + holding_period].sum()
            exit_pips = exit_return * monitor["usdpen_level"].iloc[i] * 10000
            pnl = entry_excess + exit_pips - transaction_cost_pips
            trades.append({
                "date": monitor["date"].iloc[i],
                "direction": "long",
                "entry_excess_pips": entry_excess,
                "pnl_pips": pnl,
                "regime": _assign_regime(monitor["date"].iloc[i])
            })

    if not trades:
        return {"total_trades": 0, "message": "No trades triggered"}

    trades_df = pd.DataFrame(trades)
    cumul_pnl = trades_df["pnl_pips"].cumsum()
    max_dd = (cumul_pnl - cumul_pnl.cummax()).min()

    # Annualized Sharpe
    avg_pnl = trades_df["pnl_pips"].mean()
    std_pnl = trades_df["pnl_pips"].std()
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0

    # P&L by regime
    pnl_by_regime = trades_df.groupby("regime")["pnl_pips"].agg(["count", "sum", "mean"]).to_dict("index")

    return {
        "trades_df": trades_df,
        "total_trades": len(trades_df),
        "win_rate": (trades_df["pnl_pips"] > 0).mean(),
        "avg_pnl_pips": avg_pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown_pips": max_dd,
        "cumul_pnl": cumul_pnl,
        "pnl_by_regime": pnl_by_regime
    }


# ── D6: Backtest charts ─────────────────────────────────────────────────────

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })


def chart_coverage_summary(backtest_results, output_path="output/charts"):
    # D1 chart: grouped bar comparing theoretical vs empirical breach rates
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = ["90", "95", "99"]
    theoretical = []
    empirical = []
    for cl_str in levels:
        if cl_str in backtest_results:
            theoretical.append(backtest_results[cl_str]["theoretical_rate"])
            empirical.append(backtest_results[cl_str]["empirical_rate"])
        else:
            theoretical.append(0)
            empirical.append(0)

    x = np.arange(len(levels))
    width = 0.35
    ax.bar(x - width / 2, theoretical, width, label="Theoretical", color="#4c72b0")
    ax.bar(x + width / 2, empirical, width, label="Empirical", color="#dd8452")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cl}%" for cl in levels])
    ax.set_ylabel("Breach Rate")
    ax.set_title("Model Coverage: Theoretical vs Empirical Breach Rates")
    ax.legend()

    for i, (t, e) in enumerate(zip(theoretical, empirical)):
        ax.text(i - width / 2, t + 0.002, f"{t:.1%}", ha="center", fontsize=8)
        ax.text(i + width / 2, e + 0.002, f"{e:.1%}", ha="center", fontsize=8)

    _save_chart(fig, output_path, "D1_coverage_summary.png")


def chart_pit_histogram(pit_result, output_path="output/charts"):
    # D2 chart: PIT histogram
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    edges = pit_result["hist_edges"]
    freq = pit_result["hist_freq"]
    centers = (edges[:-1] + edges[1:]) / 2

    ax.bar(centers, freq, width=0.045, color="#4c72b0", edgecolor="white")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.5, label="Uniform reference (5%)")

    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Probability Integral Transform \u2014 Model Calibration")
    ax.set_xlim(0, 1)
    ax.legend()
    _save_chart(fig, output_path, "D2_pit_histogram.png")


def chart_breach_timeline(backtest_df, output_path="output/charts"):
    # D3 chart: breach timeline
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(backtest_df["date"])
    ax.plot(dates, backtest_df["actual"], color="gray", linewidth=0.5, alpha=0.7, label="USDPEN return")

    breaches = backtest_df[backtest_df["breach_99"] == 1]
    for regime, color in [("normal", "blue"), ("intervention", "gray"), ("post_shock", "red")]:
        mask = breaches["date"].apply(_assign_regime) == regime
        subset = breaches[mask]
        if len(subset) > 0:
            ax.scatter(pd.to_datetime(subset["date"]), subset["actual"],
                       s=30, color=color, zorder=5, label=f"99% breach ({regime})")

    ax.set_title("99% Confidence Interval Breaches Over Time")
    ax.set_ylabel("Return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "D3_breach_timeline.png")


def chart_basel_traffic_light(backtest_df, output_path="output/charts"):
    # D4 chart: Basel traffic light
    _setup_style()

    tl = basel_traffic_light(backtest_df["breach_99"])

    fig, ax = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(backtest_df["date"])

    # Background shading
    for i in range(len(dates) - 1):
        color_map = {"Green": "#90EE90", "Yellow": "#FFFF99", "Red": "#FFB6C1"}
        tl_color = color_map.get(tl["traffic_light"].iloc[i], "white")
        ax.axvspan(dates.iloc[i], dates.iloc[i + 1], alpha=0.3, color=tl_color)

    ax.plot(dates, tl["breach_count"], color="black", linewidth=1.5)
    ax.axhline(4, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(9, color="orange", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_title("Basel Traffic Light \u2014 Rolling 250-Day 99% Breaches")
    ax.set_ylabel("Breach Count")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "D4_basel_traffic_light.png")


def chart_mm_cumul_pnl(mm_result, output_path="output/charts"):
    # D5 chart: market-making cumulative P&L
    _setup_style()

    if mm_result.get("total_trades", 0) == 0:
        print("  No trades for MM P&L chart")
        return

    trades_df = mm_result["trades_df"]
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(trades_df["date"])
    cumul = mm_result["cumul_pnl"]

    # Color by regime
    for regime, color in [("normal", "#4c72b0"), ("intervention", "gray"), ("post_shock", "#d62728")]:
        mask = trades_df["regime"] == regime
        if mask.any():
            ax.fill_between(dates[mask], 0, cumul[mask], alpha=0.3, color=color, label=regime)

    ax.plot(dates, cumul, color="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_title("Market-Making Fade Strategy \u2014 Cumulative P&L")
    ax.set_ylabel("Cumulative P&L (pips)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "D5_mm_cumul_pnl.png")


def _save_chart(fig, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def generate_all_backtest_charts(backtest_results, pit_result, mm_result,
                                 output_path="output/charts"):
    # Generate all Part D charts
    print("\nGenerating backtest charts...")
    bt_df = backtest_results["backtest_df"]
    chart_coverage_summary(backtest_results, output_path)
    chart_pit_histogram(pit_result, output_path)
    chart_breach_timeline(bt_df, output_path)
    chart_basel_traffic_light(bt_df, output_path)
    chart_mm_cumul_pnl(mm_result, output_path)
    print("All Part D charts generated.")
