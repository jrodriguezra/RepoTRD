#!/usr/bin/env python3
# USDPEN Conditional Variance Model & Backtest
# Main runner script

import os
import sys
import pickle
import numpy as np
import pandas as pd

from usdpen_model.data_acquisition import (
    fetch_usdpen, fetch_usdcop, fetch_usdclp, fetch_usdbrl, fetch_usdmxn,
    merge_fx_data
)
from usdpen_model.variance_model import (
    estimate_betas, ewma_covariance, variance_decomposition,
    conditional_range, build_daily_monitor
)
from usdpen_model.charts import generate_all_charts
from usdpen_model.backtesting import (
    backtest_coverage, pit_test, regime_diagnostics,
    basel_traffic_light, mm_simulation, generate_all_backtest_charts
)


OUTPUT_PATH = "output"
CHARTS_PATH = os.path.join(OUTPUT_PATH, "charts")


def run_data_acquisition(banxico_token=None):
    # Part A: download and merge all FX data
    print("=" * 60)
    print("PART A: Data Acquisition")
    print("=" * 60)

    df_pen = fetch_usdpen()
    df_cop = fetch_usdcop()
    df_clp = fetch_usdclp()
    df_brl = fetch_usdbrl()
    df_mxn = fetch_usdmxn(banxico_token=banxico_token)

    levels_df, returns_df = merge_fx_data(df_pen, df_cop, df_clp, df_brl, df_mxn,
                                          output_path=OUTPUT_PATH)
    return levels_df, returns_df


def run_variance_model(levels_df, returns_df):
    # Part B: estimate betas, covariance, decompose variance, build monitor
    print("\n" + "=" * 60)
    print("PART B: Conditional Variance Model")
    print("=" * 60)

    # B1: Rolling betas
    print("\nEstimating rolling betas (EWMA method, halflife=60)...")
    betas_df = estimate_betas(returns_df, method="ewma", halflife=60, window=120)
    print(f"  Betas: {len(betas_df)} rows")

    # B2: EWMA covariance
    print("\nComputing EWMA covariance matrix (lambda=0.94)...")
    cov_matrices, var_df, corr_df = ewma_covariance(returns_df, decay=0.94)
    print(f"  Covariance matrices: {len(cov_matrices)} dates")

    # B3: Variance decomposition
    print("\nDecomposing variance...")
    var_decomp_df = variance_decomposition(betas_df, cov_matrices, returns_df)
    print(f"  Decomposition: {len(var_decomp_df)} rows")

    # B5: Daily monitor
    print("\nBuilding daily monitor...")
    monitor_df = build_daily_monitor(returns_df, betas_df, var_decomp_df,
                                     corr_df, levels_df, output_path=OUTPUT_PATH)

    # B4: Print today's conditional range
    print("\n" + "-" * 40)
    print("Latest conditional range:")
    latest = monitor_df.iloc[-1]
    regressors = ["r_cop", "r_clp", "r_mxn", "r_brl"]
    beta_cols = ["beta_cop", "beta_clp", "beta_mxn", "beta_brl"]

    latest_betas_row = betas_df.iloc[-1]
    latest_returns_row = returns_df[returns_df["date"] == latest["date"]]
    if len(latest_returns_row) > 0:
        reg_returns = {r: latest_returns_row[r].values[0] for r in regressors}
        betas_dict = {r: latest_betas_row[b] for r, b in zip(regressors, beta_cols)}
        cr = conditional_range(
            latest["date"],
            reg_returns,
            betas_dict,
            latest["idiosyncratic_var"],
            latest["usdpen_level"]
        )
        print(f"  Date: {cr['date']}")
        print(f"  Spot: {cr['spot_level']:.4f}")
        print(f"  Expected move: {cr['expected_move_pips']:.1f} pips")
        print(f"  95% CI: [{cr['lower_level_95']:.4f}, {cr['upper_level_95']:.4f}]")
        print(f"  99% CI: [{cr['lower_level_99']:.4f}, {cr['upper_level_99']:.4f}]")

    return betas_df, var_decomp_df, monitor_df, corr_df, cov_matrices


def run_charts(betas_df, var_decomp_df, monitor_df):
    # Part C: generate all charts
    print("\n" + "=" * 60)
    print("PART C: Charts")
    print("=" * 60)
    generate_all_charts(betas_df, var_decomp_df, monitor_df, output_path=CHARTS_PATH)


def run_backtesting(returns_df, monitor_df):
    # Part D: backtesting framework
    print("\n" + "=" * 60)
    print("PART D: Backtesting")
    print("=" * 60)

    # D1: Walk-forward coverage test
    print("\nRunning walk-forward coverage backtest...")
    bt_results = backtest_coverage(returns_df, train_window=252)
    bt_df = bt_results["backtest_df"]

    for cl_str in ["90", "95", "99"]:
        if cl_str in bt_results:
            r = bt_results[cl_str]
            print(f"\n  {cl_str}% level:")
            print(f"    Breaches: {r['n_breaches']}/{r['n_obs']} ({r['empirical_rate']:.1%} vs {r['theoretical_rate']:.1%})")
            print(f"    Unconditional coverage LR: {r['unconditional_coverage']['statistic']:.2f} (p={r['unconditional_coverage']['p_value']:.4f})")
            print(f"    Independence LR: {r['independence']['statistic']:.2f} (p={r['independence']['p_value']:.4f})")
            print(f"    Conditional coverage LR: {r['conditional_coverage']['statistic']:.2f} (p={r['conditional_coverage']['p_value']:.4f})")

    # D2: PIT test
    print("\nRunning PIT test...")
    pit_result = pit_test(
        bt_df["actual"].values,
        bt_df["expected"].values,
        bt_df["idio_var"].values
    )
    print(f"  KS statistic: {pit_result['ks_statistic']:.4f} (p={pit_result['ks_p_value']:.4f})")

    # D3: Regime diagnostics
    print("\nRegime diagnostics:")
    regime_diag = regime_diagnostics(bt_df)
    print(regime_diag.to_string(index=False))

    # D4: Basel traffic light
    print("\nBasel traffic light (latest):")
    tl = basel_traffic_light(bt_df["breach_99"])
    latest_tl = tl.iloc[-1]
    print(f"  Rolling 250-day 99% breaches: {int(latest_tl['breach_count'])}")
    print(f"  Classification: {latest_tl['traffic_light']}")

    # D5: Market-making simulation
    print("\nRunning market-making simulation...")
    mm_result = mm_simulation(monitor_df)
    if mm_result.get("total_trades", 0) > 0:
        print(f"  Total trades: {mm_result['total_trades']}")
        print(f"  Win rate: {mm_result['win_rate']:.1%}")
        print(f"  Avg P&L per trade: {mm_result['avg_pnl_pips']:.1f} pips")
        print(f"  Sharpe ratio: {mm_result['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {mm_result['max_drawdown_pips']:.1f} pips")
        print(f"  P&L by regime: {mm_result['pnl_by_regime']}")
    else:
        print("  No trades triggered")

    # D6: Backtest charts
    generate_all_backtest_charts(bt_results, pit_result, mm_result, output_path=CHARTS_PATH)

    return bt_results, pit_result, mm_result


def main():
    # Prompt for Banxico token
    banxico_token = os.environ.get("BANXICO_TOKEN", None)
    if banxico_token is None and sys.stdin.isatty():
        token_input = input("Enter Banxico API token (or press Enter to use yfinance fallback): ").strip()
        if token_input:
            banxico_token = token_input

    # Check for cached data
    pkl_path = os.path.join(OUTPUT_PATH, "fx_latam_merged.pkl")
    use_cache = False
    if os.path.exists(pkl_path):
        if sys.stdin.isatty():
            use_cache_input = input("Cached data found. Use cache? (y/n): ").strip().lower()
            use_cache = use_cache_input == "y"

    if use_cache:
        print("Loading cached data...")
        with open(pkl_path, "rb") as f:
            cached = pickle.load(f)
        levels_df = cached["levels"]
        returns_df = cached["returns"]
    else:
        levels_df, returns_df = run_data_acquisition(banxico_token)

    # Part B
    betas_df, var_decomp_df, monitor_df, corr_df, cov_matrices = run_variance_model(
        levels_df, returns_df
    )

    # Part C
    run_charts(betas_df, var_decomp_df, monitor_df)

    # Part D
    run_backtesting(returns_df, monitor_df)

    print("\n" + "=" * 60)
    print("DONE. All outputs saved to:", OUTPUT_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()
