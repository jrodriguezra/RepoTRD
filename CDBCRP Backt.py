# %% Cell 1 - Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings(‘ignore’)

FILE_PATH = ‘Data_historica.xlsx’

# %% Cell 2 - Functions

def svensson(m, params):
# m: tenor in days, params: [b0, b1, b2, b3, t1, t2]
# returns: yield as decimal (effective annual rate)
b0, b1, b2, b3, t1, t2 = params
m = np.asarray(m, dtype=float)
m = np.maximum(m, 0.01)
mt1 = m / t1
mt2 = m / t2
term1 = np.where(mt1 > 1e-8, (1 - np.exp(-mt1)) / mt1, 1.0)
term2 = np.where(mt1 > 1e-8, term1 - np.exp(-mt1), 0.0)
term3 = np.where(mt2 > 1e-8, (1 - np.exp(-mt2)) / mt2 - np.exp(-mt2), 0.0)
return b0 + b1 * term1 + b2 * term2 + b3 * term3

def yield_to_price(y, tenor_days):
# zero-coupon price: P = 100 / (1 + y)^(T/360)
y = np.asarray(y, dtype=float)
tenor_days = np.asarray(tenor_days, dtype=float)
return 100.0 / (1 + y) ** (tenor_days / 360.0)

def price_to_yield(p, tenor_days):
# inverse: y = (100/P)^(360/T) - 1
p = np.asarray(p, dtype=float)
tenor_days = np.asarray(tenor_days, dtype=float)
return (100.0 / p) ** (360.0 / tenor_days) - 1

def zc_duration(tenor_days):
# zero-coupon modified duration = T/360 / (1+y)
# but SBS uses duration Dj which for ZC is simply T/360
# (the price weighting in the denominator already handles the (1+y) factor)
return tenor_days / 360.0

def sbs_objective(params, tenors, valid_prices, durations):
# SBS Annex 3: Min_b Σ [(Pj - Pj(b)) / (Pj * Dj)]²
# where Pj(b) = 100 / (1 + r_svensson(tenor_j))^(tenor_j/360)
b0, b1, b2, b3, t1, t2 = params
if t1 <= 0 or t2 <= 0:
return 1e10
# r(0) >= 0 constraint: b0 + b1 >= 0
if b0 + b1 < 0:
return 1e10

```
sv_yields = svensson(tenors, params)
# guard against negative yields producing infinite prices
if np.any(sv_yields <= -1):
    return 1e10

estimated_prices = yield_to_price(sv_yields, tenors)

# SBS error term: (Pj - Pj(b)) / (Pj * Dj)
errors = (valid_prices - estimated_prices) / (valid_prices * durations)

return np.sum(errors ** 2)
```

def fit_svensson_sbs(tenors, yields, initial_params=None, max_tenor=600.0):
tenors = np.asarray(tenors, dtype=float)
yields = np.asarray(yields, dtype=float)

```
# convert yields to valid prices
valid_prices = yield_to_price(yields, tenors)
durations = zc_duration(tenors)

if initial_params is None:
    long_idx = np.argmax(tenors)
    short_idx = np.argmin(tenors)
    b0_init = float(yields[long_idx])
    b1_init = float(yields[short_idx]) - b0_init
    initial_params = [b0_init, b1_init, 0.001, 0.0, max_tenor * 0.15, max_tenor * 0.30]

tau_min = 1.0
tau_max = max_tenor
bounds = [
    (None, None),         # b0
    (None, None),         # b1
    (None, None),         # b2
    (None, None),         # b3
    (tau_min, tau_max),   # t1
    (tau_min, tau_max),   # t2
]

best_result = None
best_cost = np.inf

# L-BFGS-B
try:
    result = minimize(
        sbs_objective, initial_params,
        args=(tenors, valid_prices, durations),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-15}
    )
    if result.fun < best_cost:
        best_cost = result.fun
        best_result = result
except Exception:
    pass

# Nelder-Mead with penalty
try:
    def penalized_obj(p):
        if p[4] <= 0 or p[5] <= 0 or p[4] > tau_max or p[5] > tau_max:
            return 1e10
        return sbs_objective(p, tenors, valid_prices, durations)

    result = minimize(
        penalized_obj, initial_params,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-15}
    )
    if result.x[4] > 0 and result.x[5] > 0 and result.fun < best_cost:
        best_cost = result.fun
        best_result = result
except Exception:
    pass

# Powell with penalty
try:
    result = minimize(
        penalized_obj, initial_params,
        method='Powell',
        options={'maxiter': 10000, 'ftol': 1e-15}
    )
    if result.x[4] > 0 and result.x[5] > 0 and result.fun < best_cost:
        best_cost = result.fun
        best_result = result
except Exception:
    pass

if best_result is None:
    return np.array(initial_params), np.inf

return best_result.x, best_result.fun
```

def get_tibo(df, date):
tibo_rows = df[(df[‘Fecha’] == date) & (df[‘ISIN’] == ‘TIBO’)]
if len(tibo_rows) > 0:
return tibo_rows[‘Yield’].values[0]
return np.nan

# %% Cell 3 - Load Data

df = pd.read_excel(FILE_PATH)
df.columns = [‘Fecha’, ‘ISIN’, ‘Yield’, ‘Dias’]
df[‘Fecha’] = pd.to_datetime(df[‘Fecha’], dayfirst=True)
df = df.dropna(subset=[‘Yield’, ‘Dias’])
df = df[df[‘Dias’] > 0]
df[‘Yield’] = df[‘Yield’] / 100

all_dates = sorted(df[‘Fecha’].unique())
print(f’Total dates in file: {len(all_dates)}’)
print(f’Date range: {all_dates[0].date()} to {all_dates[-1].date()}’)

# build instrument dicts

daily_instruments = {}
for fecha, grp in df.groupby(‘Fecha’):
g = grp[grp[‘ISIN’] != ‘TIBO’].copy()
daily_instruments[fecha] = g[[‘ISIN’, ‘Yield’, ‘Dias’]].reset_index(drop=True)

# %% Cell 4 - Backtest Loop

backtest_dates = all_dates[1:]
print(f’Backtest dates: {len(backtest_dates)} ({backtest_dates[0].date()} to {backtest_dates[-1].date()})’)

isin_results = []
daily_summaries = []
prev_params = None

for d_idx, current_date in enumerate(backtest_dates):
current_pos = all_dates.index(current_date)
prev_date = all_dates[current_pos - 1]
day_gap = (current_date - prev_date).days

```
tibo = get_tibo(df, current_date)
if np.isnan(tibo):
    continue

if prev_date not in daily_instruments or current_date not in daily_instruments:
    continue
prev_inst = daily_instruments[prev_date]
curr_inst = daily_instruments[current_date]

# identify new issues
prev_isins_set = set(prev_inst['ISIN'].values)
curr_isins_set = set(curr_inst['ISIN'].values)
new_isins = curr_isins_set - prev_isins_set
has_new_issues = len(new_isins) > 0

# build fitting set
fit_tenors = [1.0]
fit_yields = [tibo]

# shifted t-1 instruments
for _, row in prev_inst.iterrows():
    new_tenor = row['Dias'] - day_gap
    if new_tenor > 1:
        fit_tenors.append(float(new_tenor))
        fit_yields.append(float(row['Yield']))

# new issues
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

# fit Svensson with SBS price-space objective
max_tenor_fit = fit_tenors.max()
params, cost = fit_svensson_sbs(fit_tenors, fit_yields, initial_params=prev_params, max_tenor=max_tenor_fit)
prev_params = list(params)

# compare forecast vs actual at each ISIN
curr_isin_map = dict(zip(curr_inst['ISIN'].values, zip(curr_inst['Yield'].values, curr_inst['Dias'].values)))
prev_isin_yield_map = dict(zip(prev_inst['ISIN'].values, prev_inst['Yield'].values))

day_errors = []

for isin, (actual_yield, actual_dias) in curr_isin_map.items():
    sv_yield = float(svensson(np.array([actual_dias]), params)[0])

    # also compute price-level error
    sv_price = float(yield_to_price(np.array([sv_yield]), np.array([actual_dias]))[0])
    actual_price = float(yield_to_price(np.array([actual_yield]), np.array([actual_dias]))[0])

    err_yield_bps = (sv_yield - actual_yield) * 10000
    err_price = sv_price - actual_price

    prev_yield = prev_isin_yield_map.get(isin, np.nan)

    isin_results.append({
        'Date': current_date,
        'ISIN': isin,
        'Tenor': int(actual_dias),
        'Svensson_Yield': sv_yield,
        'Actual_Yield': actual_yield,
        'Svensson_Price': sv_price,
        'Actual_Price': actual_price,
        'Err_Yield_bps': err_yield_bps,
        'Err_Price': err_price,
        'Prev_Yield': prev_yield,
        'Has_New_Issue': has_new_issues,
        'Is_New_Issue': isin in new_isins,
    })
    day_errors.append(err_yield_bps)

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
})

if (d_idx + 1) % 10 == 0:
    print(f'  Processed {d_idx + 1}/{len(backtest_dates)} dates...')
```

print(f’\nBacktest complete: {len(daily_summaries)} dates processed’)

isin_df = pd.DataFrame(isin_results)
daily_df = pd.DataFrame(daily_summaries)

bins = [0, 30, 90, 180, 360, 9999]
labels_bucket = [’<30d’, ‘30-90d’, ‘90-180d’, ‘180-360d’, ‘>360d’]
isin_df[‘Bucket’] = pd.cut(isin_df[‘Tenor’], bins=bins, labels=labels_bucket)

# %% Cell 5 - Summary Statistics

print(’\n’ + ‘=’*80)
print(‘AGGREGATE SVENSSON (SBS PRICE-SPACE) BACKTEST RESULTS’)
print(’=’*80)

print(f’\nDates processed: {len(daily_df)}’)
print(f’  New issue days: {daily_df[“Has_New_Issue”].sum()}’)
print(f’  No new issue days: {(~daily_df[“Has_New_Issue”]).sum()}’)
print(f’Total ISIN-date observations: {len(isin_df)}’)

print(f’\n— Overall ISIN-Level Yield Error (bps) —’)
print(f’  MAE:    {np.abs(isin_df[“Err_Yield_bps”]).mean():.2f}’)
print(f’  RMSE:   {np.sqrt((isin_df[“Err_Yield_bps”]**2).mean()):.2f}’)
print(f’  Median: {np.abs(isin_df[“Err_Yield_bps”]).median():.2f}’)
print(f’  p75:    {np.abs(isin_df[“Err_Yield_bps”]).quantile(0.75):.2f}’)
print(f’  p95:    {np.abs(isin_df[“Err_Yield_bps”]).quantile(0.95):.2f}’)
print(f’  Max:    {np.abs(isin_df[“Err_Yield_bps”]).max():.2f}’)

print(f’\n— Overall ISIN-Level Price Error —’)
print(f’  MAE:    {np.abs(isin_df[“Err_Price”]).mean():.4f}’)
print(f’  RMSE:   {np.sqrt((isin_df[“Err_Price”]**2).mean()):.4f}’)
print(f’  Max:    {np.abs(isin_df[“Err_Price”]).max():.4f}’)

print(f’\n— Daily MAE (bps) —’)
print(f’  median={daily_df[“MAE_bps”].median():.2f}  mean={daily_df[“MAE_bps”].mean():.2f}  p75={daily_df[“MAE_bps”].quantile(0.75):.2f}  p95={daily_df[“MAE_bps”].quantile(0.95):.2f}  max={daily_df[“MAE_bps”].max():.2f}’)

print(f’\n— Daily Max Abs Error (bps) —’)
print(f’  median={daily_df[“Max_bps”].median():.2f}  mean={daily_df[“Max_bps”].mean():.2f}  p95={daily_df[“Max_bps”].quantile(0.95):.2f}  max={daily_df[“Max_bps”].max():.2f}’)

ni_days = daily_df[daily_df[‘Has_New_Issue’]]
no_ni_days = daily_df[~daily_df[‘Has_New_Issue’]]
print(f’\n— New Issue Days ({len(ni_days)}) —’)
if len(ni_days) > 0:
print(f’  MAE: median={ni_days[“MAE_bps”].median():.2f}  mean={ni_days[“MAE_bps”].mean():.2f}  p95={ni_days[“MAE_bps”].quantile(0.95):.2f}’)
print(f’\n— No New Issue Days ({len(no_ni_days)}) —’)
if len(no_ni_days) > 0:
print(f’  MAE: median={no_ni_days[“MAE_bps”].median():.2f}  mean={no_ni_days[“MAE_bps”].mean():.2f}  p95={no_ni_days[“MAE_bps”].quantile(0.95):.2f}’)

# %% Cell 6 - Error by Tenor Bucket

print(’\n’ + ‘=’*80)
print(‘YIELD ERROR BY TENOR BUCKET (bps)’)
print(’=’*80)

bucket_stats = isin_df.groupby(‘Bucket’, observed=True).agg(
N=(‘Err_Yield_bps’, ‘count’),
MAE=(‘Err_Yield_bps’, lambda x: np.abs(x).mean()),
RMSE=(‘Err_Yield_bps’, lambda x: np.sqrt((x**2).mean())),
Median=(‘Err_Yield_bps’, lambda x: np.abs(x).median()),
P95=(‘Err_Yield_bps’, lambda x: np.abs(x).quantile(0.95)),
Max=(‘Err_Yield_bps’, lambda x: np.abs(x).max()),
Mean_Signed=(‘Err_Yield_bps’, ‘mean’),
).round(2)
print(bucket_stats.to_string())

print(’\n— Price Error by Tenor Bucket —’)
price_bucket = isin_df.groupby(‘Bucket’, observed=True).agg(
N=(‘Err_Price’, ‘count’),
Price_MAE=(‘Err_Price’, lambda x: np.abs(x).mean()),
Price_RMSE=(‘Err_Price’, lambda x: np.sqrt((x**2).mean())),
Price_Max=(‘Err_Price’, lambda x: np.abs(x).max()),
).round(4)
print(price_bucket.to_string())

# fine bins

print(’\n— Yield Error by 30d Tenor Bins —’)
fine_bins = list(range(0, 601, 30))
fine_labels = [f’{fine_bins[i]}-{fine_bins[i+1]}d’ for i in range(len(fine_bins)-1)]
isin_df[‘Fine_Bucket’] = pd.cut(isin_df[‘Tenor’], bins=fine_bins, labels=fine_labels)

fine_stats = isin_df.groupby(‘Fine_Bucket’, observed=True).agg(
N=(‘Err_Yield_bps’, ‘count’),
MAE=(‘Err_Yield_bps’, lambda x: np.abs(x).mean()),
RMSE=(‘Err_Yield_bps’, lambda x: np.sqrt((x**2).mean())),
Mean_Signed=(‘Err_Yield_bps’, ‘mean’),
).round(2)
print(fine_stats.to_string())

# %% Cell 7 - Plot 1: Boxplot by Tenor Bucket

fig, ax = plt.subplots(figsize=(14, 6))
bucket_order = [b for b in labels_bucket if b in isin_df[‘Bucket’].unique()]
box_data = [isin_df[isin_df[‘Bucket’] == b][‘Err_Yield_bps’].values for b in bucket_order]
bp = ax.boxplot(box_data, positions=range(len(bucket_order)), widths=0.6,
patch_artist=True, showfliers=True, flierprops={‘markersize’: 2})
for patch in bp[‘boxes’]:
patch.set_facecolor(’#4A9B6E’)
patch.set_alpha(0.6)
ax.set_xticks(range(len(bucket_order)))
ax.set_xticklabels(bucket_order, fontsize=10)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor Bucket’)
ax.set_ylabel(‘Yield Error (bps)’)
ax.set_title(‘Svensson (SBS Price-Space) — ISIN-Level Error by Tenor Bucket’)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(‘svensson_sbs_bt_boxplot_bucket.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 8 - Plot 2: Boxplot by Fine Tenor Bins

fig, ax = plt.subplots(figsize=(18, 6))
fine_order = [b for b in fine_labels if b in isin_df[‘Fine_Bucket’].unique()]
box_data_fine = [isin_df[isin_df[‘Fine_Bucket’] == b][‘Err_Yield_bps’].values for b in fine_order]
non_empty = [(b, d) for b, d in zip(fine_order, box_data_fine) if len(d) > 0]
if len(non_empty) > 0:
fine_labels_plot, fine_data_plot = zip(*non_empty)
bp2 = ax.boxplot(fine_data_plot, positions=range(len(fine_labels_plot)), widths=0.6,
patch_artist=True, showfliers=True, flierprops={‘markersize’: 2})
for patch in bp2[‘boxes’]:
patch.set_facecolor(’#4A9B6E’)
patch.set_alpha(0.6)
ax.set_xticks(range(len(fine_labels_plot)))
ax.set_xticklabels(fine_labels_plot, rotation=45, fontsize=8)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor Bin’)
ax.set_ylabel(‘Yield Error (bps)’)
ax.set_title(‘Svensson (SBS Price-Space) — ISIN-Level Error by 30d Bins’)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(‘svensson_sbs_bt_boxplot_fine.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 9 - Plot 3: Daily MAE Time Series

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

daily_sorted = daily_df.sort_values(‘Date’)

ax = axes[0]
ax.plot(daily_sorted[‘Date’], daily_sorted[‘MAE_bps’], color=’#4A9B6E’, linewidth=0.8, alpha=0.8)
ni_dates = daily_sorted[daily_sorted[‘Has_New_Issue’]]
ax.scatter(ni_dates[‘Date’], ni_dates[‘MAE_bps’], color=’#D4553A’, s=15, zorder=5, alpha=0.6, label=‘New Issue Day’)
ax.axhline(y=daily_sorted[‘MAE_bps’].median(), color=‘gray’, linestyle=’–’, linewidth=0.5, label=f’Median={daily_sorted[“MAE_bps”].median():.1f}’)
ax.set_ylabel(‘MAE (bps)’)
ax.set_title(‘Daily MAE Over Time’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(daily_sorted[‘Date’], daily_sorted[‘Max_bps’], color=’#2C5F8A’, linewidth=0.8, alpha=0.8)
ax.scatter(ni_dates[‘Date’], ni_dates[‘Max_bps’], color=’#D4553A’, s=15, zorder=5, alpha=0.6, label=‘New Issue Day’)
ax.axhline(y=daily_sorted[‘Max_bps’].median(), color=‘gray’, linestyle=’–’, linewidth=0.5, label=f’Median={daily_sorted[“Max_bps”].median():.1f}’)
ax.set_ylabel(‘Max Abs Error (bps)’)
ax.set_title(‘Daily Max Error Over Time’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.xlabel(‘Date’)
plt.suptitle(‘Svensson (SBS Price-Space) Backtest — Daily Error Metrics’, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘svensson_sbs_bt_daily_timeseries.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 10 - Plot 4: Bell Curves by Tenor Bucket + Overall

fig, axes = plt.subplots(1, len(bucket_order), figsize=(4 * len(bucket_order), 5))
if len(bucket_order) == 1:
axes = [axes]

for i, bucket in enumerate(bucket_order):
ax = axes[i]
errs = isin_df[isin_df[‘Bucket’] == bucket][‘Err_Yield_bps’].dropna().values
if len(errs) < 5:
ax.set_visible(False)
continue
ax.hist(errs, bins=30, density=True, color=’#4A9B6E’, alpha=0.5, edgecolor=‘white’, linewidth=0.5)
mu_fit, std_fit = errs.mean(), errs.std()
x_range = np.linspace(errs.min() - 1, errs.max() + 1, 200)
ax.plot(x_range, stats.norm.pdf(x_range, mu_fit, std_fit), color=‘black’, linewidth=1.2)
skew = stats.skew(errs)
kurt = stats.kurtosis(errs)
ax.set_title(f’{bucket}’, fontsize=11)
ax.text(0.95, 0.95, f’N={len(errs)}\nmu={mu_fit:.1f}\nsd={std_fit:.1f}\nskew={skew:.2f}\nkurt={kurt:.2f}’,
transform=ax.transAxes, fontsize=7, va=‘top’, ha=‘right’,
bbox=dict(boxstyle=‘round’, facecolor=‘white’, alpha=0.8))
ax.axvline(x=0, color=‘gray’, linestyle=’–’, linewidth=0.5)
ax.set_xlabel(‘Error (bps)’, fontsize=8)

plt.suptitle(‘Svensson (SBS Price-Space) Error Distributions by Tenor Bucket’, fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(‘svensson_sbs_bt_bell_curves.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# overall

fig, ax = plt.subplots(figsize=(10, 5))
all_errs = isin_df[‘Err_Yield_bps’].dropna().values
ax.hist(all_errs, bins=50, density=True, color=’#4A9B6E’, alpha=0.5, edgecolor=‘white’, linewidth=0.5)
mu_all, std_all = all_errs.mean(), all_errs.std()
x_range = np.linspace(all_errs.min() - 1, all_errs.max() + 1, 300)
ax.plot(x_range, stats.norm.pdf(x_range, mu_all, std_all), color=‘black’, linewidth=1.5)
skew_all = stats.skew(all_errs)
kurt_all = stats.kurtosis(all_errs)
ax.text(0.95, 0.95, f’N={len(all_errs)}\nmu={mu_all:.2f}\nsd={std_all:.2f}\nskew={skew_all:.2f}\nkurt={kurt_all:.2f}’,
transform=ax.transAxes, fontsize=9, va=‘top’, ha=‘right’,
bbox=dict(boxstyle=‘round’, facecolor=‘white’, alpha=0.8))
ax.axvline(x=0, color=‘gray’, linestyle=’–’, linewidth=0.5)
ax.set_xlabel(‘Yield Error (bps)’)
ax.set_ylabel(‘Density’)
ax.set_title(‘Svensson (SBS Price-Space) — All ISIN-Level Yield Errors’)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(‘svensson_sbs_bt_bell_all.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 11 - Z-Score Directional Test

print(’\n’ + ‘=’*80)
print(‘DIRECTIONAL PREDICTIVE POWER - Z-SCORE TESTS’)
print(’=’*80)
print(’\nMetric: sign(Svensson(t) - Yield(t-1)) == sign(Yield(t) - Yield(t-1))\n’)

dir_df = isin_df.dropna(subset=[‘Prev_Yield’]).copy()
dir_df[‘Sv_Direction’] = np.sign(dir_df[‘Svensson_Yield’] - dir_df[‘Prev_Yield’])
dir_df[‘Actual_Direction’] = np.sign(dir_df[‘Actual_Yield’] - dir_df[‘Prev_Yield’])
dir_df[‘Correct’] = (dir_df[‘Sv_Direction’] == dir_df[‘Actual_Direction’]) & (dir_df[‘Sv_Direction’] != 0)
dir_df[‘Nonzero’] = (dir_df[‘Sv_Direction’] != 0) & (dir_df[‘Actual_Direction’] != 0)

print(f’{“Bucket”:>12s}  {“N”:>5s}  {“Correct”:>8s}  {“Hit%”:>6s}  {“Z-stat”:>7s}  {“p-value”:>8s}  {“Sig”:>4s}’)
print(’-’ * 60)

ztest_results = []
for bucket in bucket_order:
subset = dir_df[dir_df[‘Bucket’] == bucket]
nonzero = subset[subset[‘Nonzero’]]
n = len(nonzero)
if n < 10:
continue
correct = nonzero[‘Correct’].sum()
hit_rate = correct / n
z = (hit_rate - 0.5) / np.sqrt(0.25 / n)
p = 2 * (1 - stats.norm.cdf(abs(z)))
sig = ‘***’ if p < 0.01 else ’**’ if p < 0.05 else ’*’ if p < 0.10 else ‘’
print(f’{bucket:>12s}  {n:>5d}  {correct:>8d}  {hit_rate*100:>5.1f}%  {z:>7.2f}  {p:>8.4f}  {sig:>4s}’)
ztest_results.append({‘Bucket’: bucket, ‘N’: n, ‘Correct’: correct, ‘HitRate’: hit_rate, ‘Z’: z, ‘P’: p})

ztest_df = pd.DataFrame(ztest_results)
if len(ztest_df) > 0:
total_n = ztest_df[‘N’].sum()
total_correct = ztest_df[‘Correct’].sum()
total_hit = total_correct / total_n
total_z = (total_hit - 0.5) / np.sqrt(0.25 / total_n)
total_p = 2 * (1 - stats.norm.cdf(abs(total_z)))
print(f’\nAggregate: N={total_n}, Correct={total_correct}, Hit={total_hit*100:.1f}%, Z={total_z:.2f}, p={total_p:.4f}’)

print(f’\n— New Issue Days —’)
ni_dir = dir_df[dir_df[‘Has_New_Issue’]]
ni_nz = ni_dir[ni_dir[‘Nonzero’]]
if len(ni_nz) >= 10:
n = len(ni_nz); c = ni_nz[‘Correct’].sum(); h = c/n
z = (h-0.5)/np.sqrt(0.25/n); p = 2*(1-stats.norm.cdf(abs(z)))
print(f’  N={n}, Correct={c}, Hit={h*100:.1f}%, Z={z:.2f}, p={p:.4f}’)

print(f’\n— No New Issue Days —’)
no_ni_dir = dir_df[~dir_df[‘Has_New_Issue’]]
no_ni_nz = no_ni_dir[no_ni_dir[‘Nonzero’]]
if len(no_ni_nz) >= 10:
n = len(no_ni_nz); c = no_ni_nz[‘Correct’].sum(); h = c/n
z = (h-0.5)/np.sqrt(0.25/n); p = 2*(1-stats.norm.cdf(abs(z)))
print(f’  N={n}, Correct={c}, Hit={h*100:.1f}%, Z={z:.2f}, p={p:.4f}’)

# %% Cell 12 - New Issue Impact + Comparison vs Previous

print(’\n’ + ‘=’*80)
print(‘NEW ISSUE IMPACT ANALYSIS’)
print(’=’*80)

new_issue_obs = isin_df[isin_df[‘Is_New_Issue’]]
existing_ni = isin_df[(isin_df[‘Has_New_Issue’]) & (~isin_df[‘Is_New_Issue’])]
existing_no_ni = isin_df[~isin_df[‘Has_New_Issue’]]

if len(new_issue_obs) > 0:
print(f’\nNew issue instruments: N={len(new_issue_obs)}, MAE={np.abs(new_issue_obs[“Err_Yield_bps”]).mean():.2f} bps’)
if len(existing_ni) > 0:
print(f’Existing on new issue days: N={len(existing_ni)}, MAE={np.abs(existing_ni[“Err_Yield_bps”]).mean():.2f} bps’)
if len(existing_no_ni) > 0:
print(f’Existing on no-new-issue days: N={len(existing_no_ni)}, MAE={np.abs(existing_no_ni[“Err_Yield_bps”]).mean():.2f} bps’)

print(’\n’ + ‘=’*80)
print(‘SVENSSON (SBS PRICE-SPACE) BACKTEST COMPLETE’)
print(’=’*80)