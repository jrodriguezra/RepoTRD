# %% Cell 1 - Imports & Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

FILE_PATH = 'Data_historica.xlsx'

# grid search resolution
TAU1_GRID = np.arange(5, 601, 5)   # 5, 10, 15, ..., 600
TAU2_GRID = np.arange(5, 601, 5)   # 5, 10, 15, ..., 600

# %% Cell 2 - Functions

def svensson_basis(m, t1, t2):
    # returns the 4 basis functions evaluated at tenors m
    # for fixed t1, t2: r(m) = b0*Z0 + b1*Z1 + b2*Z2 + b3*Z3
    # Z0 = 1
    # Z1 = (1-exp(-m/t1)) / (m/t1)
    # Z2 = Z1 - exp(-m/t1)
    # Z3 = (1-exp(-m/t2)) / (m/t2) - exp(-m/t2)
    m = np.asarray(m, dtype=float)
    m = np.maximum(m, 0.01)
    mt1 = m / t1
    mt2 = m / t2
    Z0 = np.ones_like(m)
    Z1 = np.where(mt1 > 1e-8, (1 - np.exp(-mt1)) / mt1, 1.0)
    Z2 = np.where(mt1 > 1e-8, Z1 - np.exp(-mt1), 0.0)
    Z3 = np.where(mt2 > 1e-8, (1 - np.exp(-mt2)) / mt2 - np.exp(-mt2), 0.0)
    return np.column_stack([Z0, Z1, Z2, Z3])


def svensson(m, params):
    b0, b1, b2, b3, t1, t2 = params
    Z = svensson_basis(m, t1, t2)
    betas = np.array([b0, b1, b2, b3])
    return Z @ betas


def yield_to_price(y, tenor_days):
    y = np.asarray(y, dtype=float)
    tenor_days = np.asarray(tenor_days, dtype=float)
    return 100.0 / (1 + y) ** (tenor_days / 360.0)


def zc_duration(tenor_days):
    return np.asarray(tenor_days, dtype=float) / 360.0


def fit_svensson_grid_ols(tenors, yields, tau1_grid, tau2_grid):
    # for each (t1, t2) pair, solve betas by weighted least squares
    # in SBS price-space: error = (P - P(b)) / (P * D)
    #
    # since P(b) = 100 / (1+r(b))^(m/360) is nonlinear in betas,
    # we linearize: for small errors, the price error is approximately
    # dP/dr * (r - r_actual) = -D*P*(r - r_actual)
    # so (P-P(b))/(P*D) ≈ -(r(b) - y) = yield error
    #
    # this means minimizing the SBS objective is approximately
    # equivalent to weighted yield OLS with weight = 1
    # (the P*D normalization cancels)
    #
    # for exactness, we do iteratively reweighted least squares (IRLS):
    # 1. solve OLS in yield space
    # 2. compute exact SBS objective
    # 3. pick the best (t1, t2)

    tenors = np.asarray(tenors, dtype=float)
    yields = np.asarray(yields, dtype=float)
    valid_prices = yield_to_price(yields, tenors)
    durations = zc_duration(tenors)

    best_cost = np.inf
    best_params = None

    for t1 in tau1_grid:
        for t2 in tau2_grid:
            # build basis matrix
            Z = svensson_basis(tenors, t1, t2)

            # OLS: minimize ||y - Z*beta||^2
            # using numpy lstsq
            try:
                betas, residuals, rank, sv = np.linalg.lstsq(Z, yields, rcond=None)
            except np.linalg.LinAlgError:
                continue

            b0, b1, b2, b3 = betas

            # r(0) >= 0 constraint
            if b0 + b1 < 0:
                continue

            # compute exact SBS price-space objective
            model_yields = Z @ betas
            if np.any(model_yields <= -1):
                continue
            model_prices = yield_to_price(model_yields, tenors)
            errors = (valid_prices - model_prices) / (valid_prices * durations)
            cost = np.sum(errors ** 2)

            if cost < best_cost:
                best_cost = cost
                best_params = [b0, b1, b2, b3, t1, t2]

    if best_params is None:
        return np.array([0.04, 0.0, 0.0, 0.0, 90.0, 30.0]), np.inf

    return np.array(best_params), best_cost


def get_tibo(df, date):
    tibo_rows = df[(df['Fecha'] == date) & (df['ISIN'] == 'TIBO')]
    if len(tibo_rows) > 0:
        return tibo_rows['Yield'].values[0]
    return np.nan


# %% Cell 3 - Load Data

df = pd.read_excel(FILE_PATH)
df.columns = ['Fecha', 'ISIN', 'Yield', 'Dias']
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df = df.dropna(subset=['Yield', 'Dias'])
df = df[df['Dias'] > 0]
df['Yield'] = df['Yield'] / 100

all_dates = sorted(df['Fecha'].unique())
print(f'Total dates in file: {len(all_dates)}')
print(f'Date range: {all_dates[0].date()} to {all_dates[-1].date()}')
print(f'Grid: {len(TAU1_GRID)} x {len(TAU2_GRID)} = {len(TAU1_GRID)*len(TAU2_GRID)} (t1, t2) pairs')

daily_instruments = {}
for fecha, grp in df.groupby('Fecha'):
    g = grp[grp['ISIN'] != 'TIBO'].copy()
    daily_instruments[fecha] = g[['ISIN', 'Yield', 'Dias']].reset_index(drop=True)

# %% Cell 4 - Backtest Loop

backtest_dates = all_dates[1:]
print(f'Backtest dates: {len(backtest_dates)} ({backtest_dates[0].date()} to {backtest_dates[-1].date()})')

isin_results = []
daily_summaries = []

for d_idx, current_date in enumerate(backtest_dates):
    current_pos = all_dates.index(current_date)
    prev_date = all_dates[current_pos - 1]
    day_gap = (current_date - prev_date).days

    tibo = get_tibo(df, current_date)
    if np.isnan(tibo):
        continue

    if prev_date not in daily_instruments or current_date not in daily_instruments:
        continue
    prev_inst = daily_instruments[prev_date]
    curr_inst = daily_instruments[current_date]

    prev_isins_set = set(prev_inst['ISIN'].values)
    curr_isins_set = set(curr_inst['ISIN'].values)
    new_isins = curr_isins_set - prev_isins_set
    has_new_issues = len(new_isins) > 0

    # build fitting set
    fit_tenors = [1.0]
    fit_yields = [tibo]

    for _, row in prev_inst.iterrows():
        new_tenor = row['Dias'] - day_gap
        if new_tenor > 1:
            fit_tenors.append(float(new_tenor))
            fit_yields.append(float(row['Yield']))

    if has_new_issues:
        new_inst = curr_inst[curr_inst['ISIN'].isin(new_isins)]
        for _, row in new_inst.iterrows():
            fit_tenors.append(float(row['Dias']))
            fit_yields.append(float(row['Yield']))

    fit_tenors = np.array(fit_tenors)
    fit_yields = np.array(fit_yields)

    sort_idx = np.argsort(fit_tenors)
    fit_tenors = fit_tenors[sort_idx]
    fit_yields = fit_yields[sort_idx]

    if len(fit_tenors) < 4:
        continue

    # fit Svensson via grid search + OLS
    max_tenor = fit_tenors.max()
    t1_grid = TAU1_GRID[TAU1_GRID <= max_tenor]
    t2_grid = TAU2_GRID[TAU2_GRID <= max_tenor]

    params, cost = fit_svensson_grid_ols(fit_tenors, fit_yields, t1_grid, t2_grid)

    # compare vs actuals at ISIN level
    curr_isin_map = dict(zip(curr_inst['ISIN'].values, zip(curr_inst['Yield'].values, curr_inst['Dias'].values)))
    prev_isin_yield_map = dict(zip(prev_inst['ISIN'].values, prev_inst['Yield'].values))

    day_errors = []

    for isin, (actual_yield, actual_dias) in curr_isin_map.items():
        sv_yield = float(svensson(np.array([actual_dias]), params)[0])
        err_bps = (sv_yield - actual_yield) * 10000
        prev_yield = prev_isin_yield_map.get(isin, np.nan)

        isin_results.append({
            'Date': current_date,
            'ISIN': isin,
            'Tenor': int(actual_dias),
            'Svensson': sv_yield,
            'Actual': actual_yield,
            'Err_bps': err_bps,
            'Prev_Yield': prev_yield,
            'Has_New_Issue': has_new_issues,
            'Is_New_Issue': isin in new_isins,
        })
        day_errors.append(err_bps)

    day_errors = np.array(day_errors)
    daily_summaries.append({
        'Date': current_date,
        'Has_New_Issue': has_new_issues,
        'N_New_Issues': len(new_isins),
        'N_Instruments': len(day_errors),
        'MAE_bps': np.abs(day_errors).mean(),
        'RMSE_bps': np.sqrt((day_errors**2).mean()),
        'Max_bps': np.abs(day_errors).max(),
        'Median_bps': np.median(np.abs(day_errors)),
        'Fit_Cost': cost,
        'Day_Gap': day_gap,
        'Tau1': params[4],
        'Tau2': params[5],
    })

    if (d_idx + 1) % 5 == 0:
        print(f'  Processed {d_idx + 1}/{len(backtest_dates)} dates... (last t1={params[4]:.0f}, t2={params[5]:.0f})')

print(f'\nBacktest complete: {len(daily_summaries)} dates processed')

isin_df = pd.DataFrame(isin_results)
daily_df = pd.DataFrame(daily_summaries)

bins = [0, 30, 90, 180, 360, 9999]
labels_bucket = ['<30d', '30-90d', '90-180d', '180-360d', '>360d']
isin_df['Bucket'] = pd.cut(isin_df['Tenor'], bins=bins, labels=labels_bucket)

# %% Cell 5 - Summary Statistics

print('\n' + '='*80)
print('AGGREGATE SVENSSON (GRID+OLS) BACKTEST RESULTS')
print('='*80)

print(f'\nDates processed: {len(daily_df)}')
print(f'  New issue days: {daily_df["Has_New_Issue"].sum()}')
print(f'  No new issue days: {(~daily_df["Has_New_Issue"]).sum()}')
print(f'Total ISIN-date observations: {len(isin_df)}')

print(f'\n--- Overall ISIN-Level Yield Error (bps) ---')
print(f'  MAE:    {np.abs(isin_df["Err_bps"]).mean():.2f}')
print(f'  RMSE:   {np.sqrt((isin_df["Err_bps"]**2).mean()):.2f}')
print(f'  Median: {np.abs(isin_df["Err_bps"]).median():.2f}')
print(f'  p75:    {np.abs(isin_df["Err_bps"]).quantile(0.75):.2f}')
print(f'  p95:    {np.abs(isin_df["Err_bps"]).quantile(0.95):.2f}')
print(f'  Max:    {np.abs(isin_df["Err_bps"]).max():.2f}')

print(f'\n--- Daily MAE (bps) ---')
print(f'  median={daily_df["MAE_bps"].median():.2f}  mean={daily_df["MAE_bps"].mean():.2f}  p75={daily_df["MAE_bps"].quantile(0.75):.2f}  p95={daily_df["MAE_bps"].quantile(0.95):.2f}  max={daily_df["MAE_bps"].max():.2f}')

print(f'\n--- Daily Max Abs Error (bps) ---')
print(f'  median={daily_df["Max_bps"].median():.2f}  mean={daily_df["Max_bps"].mean():.2f}  p95={daily_df["Max_bps"].quantile(0.95):.2f}  max={daily_df["Max_bps"].max():.2f}')

ni_days = daily_df[daily_df['Has_New_Issue']]
no_ni_days = daily_df[~daily_df['Has_New_Issue']]
print(f'\n--- New Issue Days ({len(ni_days)}) ---')
if len(ni_days) > 0:
    print(f'  MAE: median={ni_days["MAE_bps"].median():.2f}  mean={ni_days["MAE_bps"].mean():.2f}  p95={ni_days["MAE_bps"].quantile(0.95):.2f}')
print(f'\n--- No New Issue Days ({len(no_ni_days)}) ---')
if len(no_ni_days) > 0:
    print(f'  MAE: median={no_ni_days["MAE_bps"].median():.2f}  mean={no_ni_days["MAE_bps"].mean():.2f}  p95={no_ni_days["MAE_bps"].quantile(0.95):.2f}')

# %% Cell 6 - Error by Tenor Bucket

print('\n' + '='*80)
print('YIELD ERROR BY TENOR BUCKET (bps)')
print('='*80)

bucket_stats = isin_df.groupby('Bucket', observed=True).agg(
    N=('Err_bps', 'count'),
    MAE=('Err_bps', lambda x: np.abs(x).mean()),
    RMSE=('Err_bps', lambda x: np.sqrt((x**2).mean())),
    Median=('Err_bps', lambda x: np.abs(x).median()),
    P95=('Err_bps', lambda x: np.abs(x).quantile(0.95)),
    Max=('Err_bps', lambda x: np.abs(x).max()),
    Mean_Signed=('Err_bps', 'mean'),
).round(2)
print(bucket_stats.to_string())

print('\n--- By 30d Tenor Bins ---')
fine_bins = list(range(0, 601, 30))
fine_labels = [f'{fine_bins[i]}-{fine_bins[i+1]}d' for i in range(len(fine_bins)-1)]
isin_df['Fine_Bucket'] = pd.cut(isin_df['Tenor'], bins=fine_bins, labels=fine_labels)

fine_stats = isin_df.groupby('Fine_Bucket', observed=True).agg(
    N=('Err_bps', 'count'),
    MAE=('Err_bps', lambda x: np.abs(x).mean()),
    RMSE=('Err_bps', lambda x: np.sqrt((x**2).mean())),
    Mean_Signed=('Err_bps', 'mean'),
).round(2)
print(fine_stats.to_string())

# %% Cell 7 - Tau Distribution

print('\n' + '='*80)
print('TAU PARAMETER DISTRIBUTION')
print('='*80)
print(f'\nTau1: mean={daily_df["Tau1"].mean():.0f}d  median={daily_df["Tau1"].median():.0f}d  std={daily_df["Tau1"].std():.0f}d  min={daily_df["Tau1"].min():.0f}d  max={daily_df["Tau1"].max():.0f}d')
print(f'Tau2: mean={daily_df["Tau2"].mean():.0f}d  median={daily_df["Tau2"].median():.0f}d  std={daily_df["Tau2"].std():.0f}d  min={daily_df["Tau2"].min():.0f}d  max={daily_df["Tau2"].max():.0f}d')

# %% Cell 8 - Plots

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# plot 1: boxplot by tenor bucket
ax = axes[0, 0]
bucket_order = [b for b in labels_bucket if b in isin_df['Bucket'].unique()]
box_data = [isin_df[isin_df['Bucket'] == b]['Err_bps'].values for b in bucket_order]
bp = ax.boxplot(box_data, positions=range(len(bucket_order)), widths=0.6,
                patch_artist=True, showfliers=True, flierprops={'markersize': 2})
for patch in bp['boxes']:
    patch.set_facecolor('#E8A838')
    patch.set_alpha(0.6)
ax.set_xticks(range(len(bucket_order)))
ax.set_xticklabels(bucket_order, fontsize=10)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Error (bps)')
ax.set_title('Grid+OLS — Error by Tenor Bucket')
ax.grid(True, alpha=0.3)

# plot 2: daily MAE time series
ax = axes[0, 1]
daily_sorted = daily_df.sort_values('Date')
ax.plot(daily_sorted['Date'], daily_sorted['MAE_bps'], color='#E8A838', linewidth=0.8, alpha=0.8)
ni_dates = daily_sorted[daily_sorted['Has_New_Issue']]
ax.scatter(ni_dates['Date'], ni_dates['MAE_bps'], color='#D4553A', s=15, zorder=5, alpha=0.6, label='New Issue')
ax.axhline(y=daily_sorted['MAE_bps'].median(), color='gray', linestyle='--', linewidth=0.5)
ax.set_ylabel('MAE (bps)')
ax.set_title('Daily MAE Over Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 3: tau1/tau2 over time
ax = axes[1, 0]
ax.plot(daily_sorted['Date'], daily_sorted['Tau1'], color='#2C5F8A', linewidth=0.8, alpha=0.8, label='τ₁')
ax.plot(daily_sorted['Date'], daily_sorted['Tau2'], color='#D4553A', linewidth=0.8, alpha=0.8, label='τ₂')
ax.set_ylabel('Days')
ax.set_title('Optimal τ₁ and τ₂ Over Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 4: bell curve overall
ax = axes[1, 1]
all_errs = isin_df['Err_bps'].dropna().values
ax.hist(all_errs, bins=50, density=True, color='#E8A838', alpha=0.5, edgecolor='white', linewidth=0.5)
mu_all, std_all = all_errs.mean(), all_errs.std()
x_range = np.linspace(all_errs.min() - 1, all_errs.max() + 1, 300)
ax.plot(x_range, stats.norm.pdf(x_range, mu_all, std_all), color='black', linewidth=1.5)
skew_all = stats.skew(all_errs)
kurt_all = stats.kurtosis(all_errs)
ax.text(0.95, 0.95, f'N={len(all_errs)}\nmu={mu_all:.2f}\nsd={std_all:.2f}\nskew={skew_all:.2f}\nkurt={kurt_all:.2f}',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('Error (bps)')
ax.set_title('All ISIN-Level Errors')
ax.grid(True, alpha=0.3)

plt.suptitle('Svensson (Grid+OLS) Backtest Results', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('svensson_grid_ols_backtest.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 9 - Z-Score Directional Test

print('\n' + '='*80)
print('DIRECTIONAL PREDICTIVE POWER - Z-SCORE TESTS')
print('='*80)

dir_df = isin_df.dropna(subset=['Prev_Yield']).copy()
dir_df['Sv_Direction'] = np.sign(dir_df['Svensson'] - dir_df['Prev_Yield'])
dir_df['Actual_Direction'] = np.sign(dir_df['Actual'] - dir_df['Prev_Yield'])
dir_df['Correct'] = (dir_df['Sv_Direction'] == dir_df['Actual_Direction']) & (dir_df['Sv_Direction'] != 0)
dir_df['Nonzero'] = (dir_df['Sv_Direction'] != 0) & (dir_df['Actual_Direction'] != 0)

print(f'\n{"Bucket":>12s}  {"N":>5s}  {"Correct":>8s}  {"Hit%":>6s}  {"Z-stat":>7s}  {"p-value":>8s}  {"Sig":>4s}')
print('-' * 60)

ztest_results = []
for bucket in bucket_order:
    subset = dir_df[dir_df['Bucket'] == bucket]
    nonzero = subset[subset['Nonzero']]
    n = len(nonzero)
    if n < 10:
        continue
    correct = nonzero['Correct'].sum()
    hit_rate = correct / n
    z = (hit_rate - 0.5) / np.sqrt(0.25 / n)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    print(f'{bucket:>12s}  {n:>5d}  {correct:>8d}  {hit_rate*100:>5.1f}%  {z:>7.2f}  {p:>8.4f}  {sig:>4s}')
    ztest_results.append({'Bucket': bucket, 'N': n, 'Correct': correct, 'HitRate': hit_rate, 'Z': z, 'P': p})

ztest_df = pd.DataFrame(ztest_results)
if len(ztest_df) > 0:
    total_n = ztest_df['N'].sum()
    total_correct = ztest_df['Correct'].sum()
    total_hit = total_correct / total_n
    total_z = (total_hit - 0.5) / np.sqrt(0.25 / total_n)
    total_p = 2 * (1 - stats.norm.cdf(abs(total_z)))
    print(f'\nAggregate: N={total_n}, Correct={total_correct}, Hit={total_hit*100:.1f}%, Z={total_z:.2f}, p={total_p:.4f}')

print(f'\n--- New Issue Days ---')
ni_dir = dir_df[dir_df['Has_New_Issue']]
ni_nz = ni_dir[ni_dir['Nonzero']]
if len(ni_nz) >= 10:
    n = len(ni_nz); c = ni_nz['Correct'].sum(); h = c/n
    z = (h-0.5)/np.sqrt(0.25/n); p = 2*(1-stats.norm.cdf(abs(z)))
    print(f'  N={n}, Correct={c}, Hit={h*100:.1f}%, Z={z:.2f}, p={p:.4f}')

print(f'\n--- No New Issue Days ---')
no_ni_dir = dir_df[~dir_df['Has_New_Issue']]
no_ni_nz = no_ni_dir[no_ni_dir['Nonzero']]
if len(no_ni_nz) >= 10:
    n = len(no_ni_nz); c = no_ni_nz['Correct'].sum(); h = c/n
    z = (h-0.5)/np.sqrt(0.25/n); p = 2*(1-stats.norm.cdf(abs(z)))
    print(f'  N={n}, Correct={c}, Hit={h*100:.1f}%, Z={z:.2f}, p={p:.4f}')

# %% Cell 10 - New Issue Impact

print('\n' + '='*80)
print('NEW ISSUE IMPACT ANALYSIS')
print('='*80)

new_issue_obs = isin_df[isin_df['Is_New_Issue']]
existing_ni = isin_df[(isin_df['Has_New_Issue']) & (~isin_df['Is_New_Issue'])]
existing_no_ni = isin_df[~isin_df['Has_New_Issue']]

if len(new_issue_obs) > 0:
    print(f'\nNew issue instruments: N={len(new_issue_obs)}, MAE={np.abs(new_issue_obs["Err_bps"]).mean():.2f} bps')
if len(existing_ni) > 0:
    print(f'Existing on new issue days: N={len(existing_ni)}, MAE={np.abs(existing_ni["Err_bps"]).mean():.2f} bps')
if len(existing_no_ni) > 0:
    print(f'Existing on no-new-issue days: N={len(existing_no_ni)}, MAE={np.abs(existing_no_ni["Err_bps"]).mean():.2f} bps')

print('\n' + '='*80)
print('SVENSSON (GRID+OLS) BACKTEST COMPLETE')
print('='*80)
