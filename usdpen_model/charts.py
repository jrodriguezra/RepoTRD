import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# ── Chart styling ────────────────────────────────────────────────────────────

COLORS = {
    "cop": "#1f77b4",
    "clp": "#ff7f0e",
    "mxn": "#2ca02c",
    "brl": "#d62728",
    "pen": "#9467bd",
    "systematic": "#4c72b0",
    "idiosyncratic": "#dd8452",
    "realized": "#000000",
}

INTERVENTION_START = pd.Timestamp("2025-11-01")
INTERVENTION_END = pd.Timestamp("2026-02-28")
POST_SHOCK_START = pd.Timestamp("2026-03-01")


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


def _add_regime_shading(ax, dates):
    # Add intervention and post-shock regime shading
    xlim = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if INTERVENTION_START <= pd.Timestamp(mdates.num2date(xlim[1])):
        start = max(INTERVENTION_START, pd.Timestamp(mdates.num2date(xlim[0])))
        end = min(INTERVENTION_END, pd.Timestamp(mdates.num2date(xlim[1])))
        if start < end:
            ax.axvspan(start, end, alpha=0.15, color="gray", label="BCRP intervention")

    if POST_SHOCK_START <= pd.Timestamp(mdates.num2date(xlim[1])):
        start = max(POST_SHOCK_START, pd.Timestamp(mdates.num2date(xlim[0])))
        end = pd.Timestamp(mdates.num2date(xlim[1]))
        if start < end:
            ax.axvspan(start, end, alpha=0.15, color="red", label="Iran conflict")


def _save_chart(fig, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ── C1: Rolling Betas Over Time ─────────────────────────────────────────────

def chart_rolling_betas(betas_df, output_path="output/charts"):
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    beta_cols = [("beta_cop", "COP", COLORS["cop"]),
                 ("beta_clp", "CLP", COLORS["clp"]),
                 ("beta_mxn", "MXN", COLORS["mxn"]),
                 ("beta_brl", "BRL", COLORS["brl"])]

    for col, label, color in beta_cols:
        ax.plot(betas_df["date"], betas_df[col], label=label, color=color, linewidth=1.2)

    ax.set_title("USDPEN Rolling Betas on Regional Currencies")
    ax.set_ylabel("Beta")
    ax.set_xlabel("Date")
    _add_regime_shading(ax, betas_df["date"])
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "C1_rolling_betas.png")


# ── C2: Variance Decomposition Over Time ────────────────────────────────────

def chart_variance_decomposition(var_decomp_df, output_path="output/charts"):
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = var_decomp_df["date"]
    ax.fill_between(dates, 0, var_decomp_df["systematic_var"],
                    alpha=0.7, color=COLORS["systematic"], label="Systematic")
    ax.fill_between(dates, var_decomp_df["systematic_var"],
                    var_decomp_df["systematic_var"] + var_decomp_df["idiosyncratic_var"],
                    alpha=0.7, color=COLORS["idiosyncratic"], label="Idiosyncratic")
    ax.plot(dates, var_decomp_df["realized_var"],
            color=COLORS["realized"], linestyle="--", linewidth=1.5, label="Realized USDPEN var")

    ax.set_title("USDPEN Variance Decomposition: Systematic vs Idiosyncratic")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Date")
    _add_regime_shading(ax, dates)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "C2_variance_decomposition.png")


# ── C3: Systematic Percentage Over Time ─────────────────────────────────────

def chart_systematic_pct(var_decomp_df, output_path="output/charts"):
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(var_decomp_df["date"], var_decomp_df["systematic_pct"] * 100,
            color=COLORS["systematic"], linewidth=1.2)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(75, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(var_decomp_df["date"].iloc[0], 51, "50%", fontsize=8, color="gray")
    ax.text(var_decomp_df["date"].iloc[0], 76, "75%", fontsize=8, color="gray")

    ax.set_title("Share of USDPEN Variance Explained by Regional FX")
    ax.set_ylabel("Systematic %")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    _add_regime_shading(ax, var_decomp_df["date"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "C3_systematic_pct.png")


# ── C4: Average Regional Correlation Over Time ──────────────────────────────

def chart_avg_correlation(monitor_df, output_path="output/charts"):
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(monitor_df["date"], monitor_df["avg_regional_correlation"],
            color=COLORS["pen"], linewidth=1.2)

    ax.set_title("Average Pairwise LatAm FX Correlation (EWMA \u03bb=0.94)")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    _add_regime_shading(ax, monitor_df["date"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "C4_avg_correlation.png")


# ── C5: Conditional Range Backtest Visualization ─────────────────────────────

def chart_conditional_range(monitor_df, output_path="output/charts"):
    _setup_style()
    # Actual move in pips
    actual_pips = monitor_df["usdpen_return"] * monitor_df["usdpen_level"] * 10000
    expected_pips = monitor_df["expected_return"] * monitor_df["usdpen_level"] * 10000

    from scipy import stats as sp_stats
    z_95 = sp_stats.norm.ppf(0.975)
    z_99 = sp_stats.norm.ppf(0.995)
    idio_std = np.sqrt(monitor_df["idiosyncratic_var"])
    band_95 = z_95 * idio_std * monitor_df["usdpen_level"] * 10000
    band_99 = z_99 * idio_std * monitor_df["usdpen_level"] * 10000

    for suffix, n_months in [("full", None), ("6m", 6)]:
        fig, ax = plt.subplots(figsize=(14, 7))
        df = monitor_df.copy()
        a_pips = actual_pips.copy()
        e_pips = expected_pips.copy()
        b95 = band_95.copy()
        b99 = band_99.copy()

        if n_months:
            cutoff = df["date"].max() - pd.DateOffset(months=n_months)
            mask = df["date"] >= cutoff
            df = df[mask]
            a_pips = a_pips[mask]
            e_pips = e_pips[mask]
            b95 = b95[mask]
            b99 = b99[mask]

        dates = df["date"]
        ax.fill_between(dates, e_pips - b99, e_pips + b99,
                        alpha=0.15, color="red", label="99% CI")
        ax.fill_between(dates, e_pips - b95, e_pips + b95,
                        alpha=0.2, color="blue", label="95% CI")
        ax.plot(dates, e_pips, color="blue", linewidth=1, label="Expected move")
        ax.scatter(dates, a_pips, s=8, color="black", alpha=0.5, label="Actual move", zorder=5)

        title = "USDPEN Conditional Range Model: Actual Moves vs Predicted Bands"
        if n_months:
            title += f" (Last {n_months}M)"
        ax.set_title(title)
        ax.set_ylabel("Move (pips)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        _save_chart(fig, output_path, f"C5_conditional_range_{suffix}.png")


# ── C6: Residual Z-Score ────────────────────────────────────────────────────

def chart_residual_zscore(monitor_df, output_path="output/charts"):
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    dates = monitor_df["date"]
    zscore = monitor_df["residual_zscore"]

    # Color segments
    for i in range(len(dates) - 1):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue
        color = "blue" if z < -1.5 else ("red" if z > 1.5 else "gray")
        ax.plot([dates.iloc[i], dates.iloc[i + 1]],
                [zscore.iloc[i], zscore.iloc[i + 1]],
                color=color, linewidth=1.2)

    ax.axhspan(-2.0, -1.5, alpha=0.1, color="blue")
    ax.axhspan(1.5, 2.0, alpha=0.1, color="red")
    ax.axhline(1.5, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(-1.5, color="blue", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(2.0, color="red", linestyle="-", alpha=0.4, linewidth=0.8)
    ax.axhline(-2.0, color="blue", linestyle="-", alpha=0.4, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax.set_title("USDPEN Idiosyncratic Residual Z-Score")
    ax.set_ylabel("Z-Score")
    ax.set_xlabel("Date")
    _add_regime_shading(ax, dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    _save_chart(fig, output_path, "C6_residual_zscore.png")


# ── C7: Beta Stability Heatmap ──────────────────────────────────────────────

def chart_beta_heatmap(betas_df, output_path="output/charts"):
    _setup_style()

    # Resample to monthly
    df = betas_df.copy()
    df = df.set_index("date")
    monthly = df[["beta_cop", "beta_clp", "beta_mxn", "beta_brl"]].resample("ME").mean()
    monthly = monthly.dropna()

    fig, ax = plt.subplots(figsize=(16, 4))
    data = monthly.values.T
    labels = ["COP", "CLP", "MXN", "BRL"]
    date_labels = [d.strftime("%b %y") for d in monthly.index]

    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Show every 3rd month label
    step = max(1, len(date_labels) // 20)
    ax.set_xticks(range(0, len(date_labels), step))
    ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), step)],
                       rotation=45, ha="right")

    ax.set_title("USDPEN Beta Evolution Heatmap")
    fig.colorbar(im, ax=ax, label="Beta", shrink=0.8)
    _save_chart(fig, output_path, "C7_beta_heatmap.png")


# ── Generate all charts ─────────────────────────────────────────────────────

def generate_all_charts(betas_df, var_decomp_df, monitor_df, output_path="output/charts"):
    # Generate all Part C charts
    print("\nGenerating charts...")
    chart_rolling_betas(betas_df, output_path)
    chart_variance_decomposition(var_decomp_df, output_path)
    chart_systematic_pct(var_decomp_df, output_path)
    chart_avg_correlation(monitor_df, output_path)
    chart_conditional_range(monitor_df, output_path)
    chart_residual_zscore(monitor_df, output_path)
    chart_beta_heatmap(betas_df, output_path)
    print("All Part C charts generated.")
