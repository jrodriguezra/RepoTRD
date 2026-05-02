# %% Cell 1 - Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings(‘ignore’)

FILE_PATH = ‘Data_historica.xlsx’

# — Config —

CURVE_STEP = 10
CURVE_MAX = 600

# %% Cell 2 - Functions

def svensson(m, params):
b0, b1, b2, b3, t1, t2 = params
m = np.asarray(m, dtype=float)
m = np.maximum(m, 0.01)
mt1 = m / t1
mt2 = m / t2
term1 = np.where(mt1 > 1e-8, (1 - np.exp(-mt1)) / mt1, 1.0)
term2 = np.where(mt1 > 1e-8, term1 - np.exp(-mt1), 0.0)
term3 = np.where(mt2 > 1e-8, (1 - np.exp(-mt2)) / mt2 - np.exp(-mt2), 0.0)
return b0 + b1 * term1 + b2 * term2 + b3 * term3

def svensson_objective(params, tenors, yields, durations):
b0, b1, b2, b3, t1, t2 = params
if t1 <= 0 or t2 <= 0:
return 1e10
fitted = svensson(tenors, params)
errors = yields - fitted
weights = 1.0 / np.maximum(durations, 1.0)
return np.sum(weights * errors**2)

def fit_svensson(tenors, yields, initial_params=None, max_tenor=600.0):
tenors = np.asarray(tenors, dtype=float)
yields = np.asarray(yields, dtype=float)
durations = tenors.copy()

```
if initial_params is None:
    long_idx = np.argmax(tenors)
    short_idx = np.argmin(tenors)
    b0_init = float(yields[long_idx])
    b1_init = float(yields[short_idx]) - b0_init
    initial_params = [b0_init, b1_init, 0.001, 0.0, max_tenor * 0.15, max_tenor * 0.30]

tau_min = 1.0
tau_max = max_tenor
bounds = [
    (None, None),
    (None, None),
    (None, None),
    (None, None),
    (tau_min, tau_max),
    (tau_min, tau_max),
]

best_result = None
best_cost = np.inf

# L-BFGS-B
try:
    result = minimize(
        svensson_objective, initial_params,
        args=(tenors, yields, durations),
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
        return svensson_objective(p, tenors, yields, durations)

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

def build_fitting_set(prev_instruments, tibo, new_issues_df, day_gap):
# prev_instruments: df with ISIN, Yield, Dias
# new_issues_df: df with ISIN, Yield, Dias (new issues for today)
# returns tenors, yields, labels arrays
tenors = []
yields = []
labels = []

```
# TIBO at 1d
tenors.append(1.0)
yields.append(tibo)
labels.append('TIBO')

# shifted t-1 instruments (exclude those that mature)
for _, row in prev_instruments.iterrows():
    new_tenor = row['Dias'] - day_gap
    if new_tenor > 1:  # exclude matured or 1d instruments
        tenors.append(float(new_tenor))
        yields.append(float(row['Yield']))
        labels.append(row['ISIN'])

# new issues
if new_issues_df is not None and len(new_issues_df) > 0:
    for _, row in new_issues_df.iterrows():
        tenors.append(float(row['Dias']))
        yields.append(float(row['Yield']))
        labels.append(row['ISIN'])

tenors = np.array(tenors)
yields = np.array(yields)

sort_idx = np.argsort(tenors)
tenors = tenors[sort_idx]
yields = yields[sort_idx]
labels = [labels[i] for i in sort_idx]

return tenors, yields, labels
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

# start from date index 1 (need at least 1 previous date)

backtest_dates = all_dates[1:]
print(f’Backtest dates: {len(backtest_dates)} ({backtest_dates[0].date()} to {backtest_dates[-1].date()})’)

# storage

isin_results = []
grid_results = []
daily_summaries = []
day_by_day_curves = {}

# warm start params

prev_params = None

for d_idx, current_date in enumerate(backtest_dates):
# find previous available date
current_pos = all_dates.index(current_date)
prev_date = all_dates[current_pos - 1]
day_gap = (current_date - prev_date).days

```
# get TIBO for current date
tibo = get_tibo(df, current_date)
if np.isnan(tibo):
    continue

# get previous date instruments
if prev_date not in daily_instruments:
    continue
prev_inst = daily_instruments[prev_date]

# get current date instruments (actuals)
if current_date not in daily_instruments:
    continue
curr_inst = daily_instruments[current_date]

# identify new issues: ISINs in current but not in previous
prev_isins_set = set(prev_inst['ISIN'].values)
curr_isins_set = set(curr_inst['ISIN'].values)
new_isins = curr_isins_set - prev_isins_set
has_new_issues = len(new_isins) > 0

new_issues_df = None
if has_new_issues:
    new_issues_df = curr_inst[curr_inst['ISIN'].isin(new_isins)]

# build fitting set
fit_tenors, fit_yields, fit_labels = build_fitting_set(prev_inst, tibo, new_issues_df, day_gap)

if len(fit_tenors) < 4:
    continue

# fit Svensson with warm start
max_tenor_fit = fit_tenors.max()
params, cost = fit_svensson(fit_tenors, fit_yields, initial_params=prev_params, max_tenor=max_tenor_fit)
prev_params = list(params)  # warm start for next day

# build display grid
grid_max = min(CURVE_MAX, int(np.ceil(max_tenor_fit / CURVE_STEP) * CURVE_STEP))
display_grid = np.arange(CURVE_STEP, grid_max + 1, CURVE_STEP)
display_grid = display_grid[display_grid <= max_tenor_fit]

if len(display_grid) == 0:
    continue

# svensson curve on grid
sv_curve = svensson(display_grid, params)

# actual curve on same grid (monotone convex from current instruments)
curr_sorted = curr_inst.sort_values('Dias')
curr_tenors = curr_sorted['Dias'].values.astype(float)
curr_yields = curr_sorted['Yield'].values.astype(float)

if len(curr_tenors) < 2:
    continue

# monotone convex for actual curve on grid
from cdbcrp_backtest import monotone_convex_interpolate  # reuse function
# inline monotone convex to avoid import dependency
def mc_interp(tenors_in, yields_in, targets):
    t = np.array(tenors_in, dtype=float)
    y = np.array(yields_in, dtype=float)
    n = len(t)
    if n < 2:
        return np.full(len(targets), np.nan)
    ty = t * y
    f = np.zeros(n)
    f[0] = y[0]
    for i in range(1, n):
        f[i] = (ty[i] - ty[i - 1]) / (t[i] - t[i - 1])
    f_inst = np.zeros(n)
    f_inst[0] = f[0]
    f_inst[-1] = f[-1]
    for i in range(1, n - 1):
        dt_l = t[i] - t[i - 1]
        dt_r = t[i + 1] - t[i]
        f_inst[i] = (dt_r * f[i] + dt_l * f[i + 1]) / (dt_l + dt_r)
    for i in range(n):
        if i == 0:
            f_inst[i] = max(f_inst[i], 0)
            if n > 1:
                f_inst[i] = min(f_inst[i], 2 * f[0])
        elif i == n - 1:
            f_inst[i] = max(f_inst[i], 0)
            f_inst[i] = min(f_inst[i], 2 * f[-1])
        else:
            f_inst[i] = max(f_inst[i], 0)
            f_inst[i] = min(f_inst[i], 2 * min(f[i], f[i + 1]))
    result = np.zeros(len(targets))
    for k, tt in enumerate(targets):
        if tt <= t[0]:
            result[k] = y[0]
        elif tt >= t[-1]:
            result[k] = y[-1]
        else:
            idx = np.searchsorted(t, tt, side='right') - 1
            idx = min(idx, n - 2)
            t0, t1_ = t[idx], t[idx + 1]
            y0, y1 = y[idx], y[idx + 1]
            dt = t1_ - t0
            x = (tt - t0) / dt
            g = (t1_ * y1 - t0 * y0) / dt
            a = f_inst[idx] - g
            b = f_inst[idx + 1] - g
            integral_norm = g * x + a * (x - 2*x**2 + x**3) + b * (-x**2 + x**3)
            tty = t0 * y0 + dt * integral_norm
            result[k] = tty / tt
    return result

actual_curve = mc_interp(curr_tenors, curr_yields, display_grid)

# store grid-level results
for i, tenor in enumerate(display_grid):
    grid_results.append({
        'Date': current_date,
        'Tenor': int(tenor),
        'Svensson': sv_curve[i],
        'Actual': actual_curve[i],
        'Err_Svensson_bps': (sv_curve[i] - actual_curve[i]) * 10000,
        'Has_New_Issue': has_new_issues,
    })

# ISIN-level results: match by ISIN
curr_isin_map = dict(zip(curr_inst['ISIN'].values, curr_inst['Yield'].values))
curr_dias_map = dict(zip(curr_inst['ISIN'].values, curr_inst['Dias'].values))

# shifted prev instruments
for _, row in prev_inst.iterrows():
    isin = row['ISIN']
    new_tenor = row['Dias'] - day_gap
    if new_tenor > 1 and isin in curr_isin_map:
        sv_yield = float(svensson(np.array([new_tenor]), params)[0])
        actual_yield = curr_isin_map[isin]
        isin_results.append({
            'Date': current_date,
            'ISIN': isin,
            'Tenor': int(new_tenor),
            'Svensson': sv_yield,
            'Actual': actual_yield,
            'Err_Svensson_bps': (sv_yield - actual_yield) * 10000,
            'Has_New_Issue': has_new_issues,
        })

# daily summary
grid_errs = (sv_curve - actual_curve) * 10000
daily_summaries.append({
    'Date': current_date,
    'Has_New_Issue': has_new_issues,
    'N_New_Issues': len(new_isins) if has_new_issues else 0,
    'Svensson_MAE_bps': np.abs(grid_errs).mean(),
    'Svensson_Max_bps': np.abs(grid_errs).max(),
    'Svensson_RMSE_bps': np.sqrt((grid_errs**2).mean()),
    'Fit_Cost': cost,
    'Day_Gap': day_gap,
})

# store curves
day_by_day_curves[current_date] = {
    'grid': display_grid,
    'svensson': sv_curve,
    'actual': actual_curve,
}

if (d_idx + 1) % 10 == 0:
    print(f'  Processed {d_idx + 1}/{len(backtest_dates)} dates...')
```

print(f’\nBacktest complete: {len(daily_summaries)} dates processed’)

isin_df = pd.DataFrame(isin_results)
grid_df = pd.DataFrame(grid_results)
daily_df = pd.DataFrame(daily_summaries)

# %% Cell 5 - Summary Statistics

print(’\n’ + ‘=’*80)
print(‘AGGREGATE SVENSSON BACKTEST RESULTS’)
print(’=’*80)

print(f’\nDates processed: {len(daily_df)}’)
print(f’  New issue days: {daily_df[“Has_New_Issue”].sum()}’)
print(f’  No new issue days: {(~daily_df[“Has_New_Issue”]).sum()}’)

print(f’\n— Daily MAE (bps) —’)
print(f’  Svensson:  median={daily_df[“Svensson_MAE_bps”].median():.2f}  mean={daily_df[“Svensson_MAE_bps”].mean():.2f}  p75={daily_df[“Svensson_MAE_bps”].quantile(0.75):.2f}  p95={daily_df[“Svensson_MAE_bps”].quantile(0.95):.2f}  max={daily_df[“Svensson_MAE_bps”].max():.2f}’)

print(f’\n— Daily Max Abs Error (bps) —’)
print(f’  Svensson:  median={daily_df[“Svensson_Max_bps”].median():.2f}  mean={daily_df[“Svensson_Max_bps”].mean():.2f}  p95={daily_df[“Svensson_Max_bps”].quantile(0.95):.2f}  max={daily_df[“Svensson_Max_bps”].max():.2f}’)

# split by new issue

ni_days = daily_df[daily_df[‘Has_New_Issue’]]
no_ni_days = daily_df[~daily_df[‘Has_New_Issue’]]
print(f’\n— New Issue Days ({len(ni_days)}) —’)
if len(ni_days) > 0:
print(f’  Svensson MAE: median={ni_days[“Svensson_MAE_bps”].median():.2f}  mean={ni_days[“Svensson_MAE_bps”].mean():.2f}’)
print(f’\n— No New Issue Days ({len(no_ni_days)}) —’)
if len(no_ni_days) > 0:
print(f’  Svensson MAE: median={no_ni_days[“Svensson_MAE_bps”].median():.2f}  mean={no_ni_days[“Svensson_MAE_bps”].mean():.2f}’)

# %% Cell 6 - MAE and RMSE per Grid Node

print(’\n’ + ‘=’*80)
print(‘MAE AND RMSE PER GRID NODE (bps)’)
print(’=’*80)

tenor_stats = grid_df.groupby(‘Tenor’).agg(
Svensson_MAE=(‘Err_Svensson_bps’, lambda x: np.abs(x).mean()),
Svensson_RMSE=(‘Err_Svensson_bps’, lambda x: np.sqrt((x**2).mean())),
Svensson_Median=(‘Err_Svensson_bps’, lambda x: np.abs(x).median()),
N_Obs=(‘Err_Svensson_bps’, ‘count’),
).round(2)

print(tenor_stats.to_string())

# %% Cell 7 - Plot 1: Max Deviation per Tenor

fig, ax = plt.subplots(figsize=(14, 6))
max_dev = grid_df.groupby(‘Tenor’)[‘Err_Svensson_bps’].apply(lambda x: np.abs(x).max())
tenors_plot = max_dev.index.values
ax.bar(tenors_plot, max_dev.values, width=CURVE_STEP * 0.7, color=’#4A9B6E’, alpha=0.8, label=‘Svensson’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Max Absolute Error (bps)’)
ax.set_title(‘Svensson - Max Deviation per Grid Tenor’)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(‘svensson_backtest_max_deviation.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 8 - Plot 2: Boxplot per Node

all_tenors_in_data = sorted(grid_df[‘Tenor’].unique())
if len(all_tenors_in_data) > 20:
plot_tenors = all_tenors_in_data[::3]
else:
plot_tenors = all_tenors_in_data

fig, ax = plt.subplots(figsize=(16, 6))
box_data = [grid_df[grid_df[‘Tenor’] == t][‘Err_Svensson_bps’].values for t in plot_tenors]
bp = ax.boxplot(box_data, positions=range(len(plot_tenors)), widths=0.6,
patch_artist=True, showfliers=True, flierprops={‘markersize’: 2})
for patch in bp[‘boxes’]:
patch.set_facecolor(’#4A9B6E’)
patch.set_alpha(0.6)
ax.set_xticks(range(len(plot_tenors)))
ax.set_xticklabels([str(t) for t in plot_tenors], rotation=45, fontsize=8)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Error (bps)’)
ax.set_title(‘Svensson Error Distribution per Node’)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(‘svensson_backtest_boxplots.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 9 - Plot 3: Error Time Series by Tenor

repr_tenors = [30, 60, 90, 120, 180, 270, 360]
repr_tenors = [t for t in repr_tenors if t in grid_df[‘Tenor’].unique()]

fig, axes = plt.subplots(len(repr_tenors), 1, figsize=(16, 3 * len(repr_tenors)), sharex=True)
if len(repr_tenors) == 1:
axes = [axes]

for i, tenor in enumerate(repr_tenors):
ax = axes[i]
subset = grid_df[grid_df[‘Tenor’] == tenor].sort_values(‘Date’)
ax.plot(subset[‘Date’], subset[‘Err_Svensson_bps’], color=’#4A9B6E’, alpha=0.8, linewidth=0.8, label=‘Svensson’)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_ylabel(‘Error (bps)’)
ax.set_title(f’{tenor}d’, fontsize=10)
ax.legend(fontsize=7, loc=‘upper right’)
ax.grid(True, alpha=0.3)

plt.xlabel(‘Date’)
plt.suptitle(‘Svensson Error Time Series by Tenor’, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘svensson_backtest_error_timeseries.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 10 - Plot 4: Bell Curve Grid

n_cols = 4
n_rows = int(np.ceil(len(plot_tenors) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
axes_flat = axes.flatten()

for i, tenor in enumerate(plot_tenors):
ax = axes_flat[i]
errs = grid_df[grid_df[‘Tenor’] == tenor][‘Err_Svensson_bps’].dropna().values
if len(errs) < 5:
ax.set_visible(False)
continue

```
ax.hist(errs, bins=25, density=True, color='#4A9B6E', alpha=0.5, edgecolor='white', linewidth=0.5)

mu_fit, std_fit = errs.mean(), errs.std()
x_range = np.linspace(errs.min() - 1, errs.max() + 1, 200)
ax.plot(x_range, stats.norm.pdf(x_range, mu_fit, std_fit), color='black', linewidth=1.2)

skew = stats.skew(errs)
kurt = stats.kurtosis(errs)
ax.set_title(f'{tenor}d', fontsize=9)
ax.text(0.95, 0.95, f'mu={mu_fit:.1f}\nsd={std_fit:.1f}\nskew={skew:.2f}\nkurt={kurt:.2f}',
        transform=ax.transAxes, fontsize=6, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('bps', fontsize=7)
ax.tick_params(labelsize=6)
```

for j in range(len(plot_tenors), len(axes_flat)):
axes_flat[j].set_visible(False)

plt.suptitle(‘Svensson Error Distributions by Tenor’, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘svensson_backtest_error_distributions.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 11 - Z-Score Directional Test

# For Svensson: does the direction of (Svensson - yesterday actual) predict (Actual - yesterday actual)?

print(’\n’ + ‘=’*80)
print(‘DIRECTIONAL PREDICTIVE POWER - Z-SCORE TESTS’)
print(’=’*80)
print(’\nTest: does the Svensson curve move predict actual direction?’)
print(‘H0: direction prediction is random (50/50)’)
print(‘Metric: sign(Svensson(t) - Actual(t-1)) == sign(Actual(t) - Actual(t-1))\n’)

# we need yesterday’s actual for each tenor

# build a lookup: for each (date, tenor), store actual

grid_df_sorted = grid_df.sort_values([‘Tenor’, ‘Date’])
grid_df_sorted[‘Prev_Actual’] = grid_df_sorted.groupby(‘Tenor’)[‘Actual’].shift(1)
grid_df_sorted[‘Sv_Direction’] = np.sign(grid_df_sorted[‘Svensson’] - grid_df_sorted[‘Prev_Actual’])
grid_df_sorted[‘Actual_Direction’] = np.sign(grid_df_sorted[‘Actual’] - grid_df_sorted[‘Prev_Actual’])
grid_df_sorted[‘Correct’] = (grid_df_sorted[‘Sv_Direction’] == grid_df_sorted[‘Actual_Direction’]) & (grid_df_sorted[‘Sv_Direction’] != 0)
grid_df_sorted[‘Nonzero’] = (grid_df_sorted[‘Sv_Direction’] != 0) & (grid_df_sorted[‘Actual_Direction’] != 0)

# drop nans (first date per tenor has no prev)

dir_df = grid_df_sorted.dropna(subset=[‘Prev_Actual’])

print(f’{“Tenor”:>8s}  {“N”:>5s}  {“Correct”:>8s}  {“Hit%”:>6s}  {“Z-stat”:>7s}  {“p-value”:>8s}  {“Sig”:>4s}’)
print(’-’ * 55)

ztest_results = []
for tenor in sorted(dir_df[‘Tenor’].unique()):
subset = dir_df[dir_df[‘Tenor’] == tenor]
nonzero = subset[subset[‘Nonzero’]]
n = len(nonzero)
if n < 10:
continue
correct = nonzero[‘Correct’].sum()
hit_rate = correct / n
z = (hit_rate - 0.5) / np.sqrt(0.25 / n)
p = 2 * (1 - stats.norm.cdf(abs(z)))
sig = ‘***’ if p < 0.01 else ’**’ if p < 0.05 else ’*’ if p < 0.10 else ‘’
print(f’{tenor:>8d}  {n:>5d}  {correct:>8d}  {hit_rate*100:>5.1f}%  {z:>7.2f}  {p:>8.4f}  {sig:>4s}’)
ztest_results.append({‘Tenor’: tenor, ‘N’: n, ‘Correct’: correct, ‘HitRate’: hit_rate, ‘Z’: z, ‘P’: p})

ztest_df = pd.DataFrame(ztest_results)
if len(ztest_df) > 0:
print(f’\nAggregate across all tenors:’)
total_n = ztest_df[‘N’].sum()
total_correct = ztest_df[‘Correct’].sum()
total_hit = total_correct / total_n
total_z = (total_hit - 0.5) / np.sqrt(0.25 / total_n)
total_p = 2 * (1 - stats.norm.cdf(abs(total_z)))
print(f’  N={total_n}, Correct={total_correct}, Hit={total_hit*100:.1f}%, Z={total_z:.2f}, p={total_p:.4f}’)

# %% Cell 12 - ISIN-Level Summary

print(’\n’ + ‘=’*80)
print(‘ISIN-LEVEL BACKTEST RESULTS’)
print(’=’*80)

if len(isin_df) > 0:
print(f’\nTotal ISIN-date observations: {len(isin_df)}’)
print(f’\n— ISIN-Level Error (bps) —’)
print(f’  Svensson MAE:  {np.abs(isin_df[“Err_Svensson_bps”]).mean():.2f}’)
print(f’  Svensson RMSE: {np.sqrt((isin_df[“Err_Svensson_bps”]**2).mean()):.2f}’)

```
bins = [0, 90, 180, 360, 9999]
labels = ['<90d', '90-180d', '180-360d', '>360d']
isin_df['Bucket'] = pd.cut(isin_df['Tenor'], bins=bins, labels=labels)

print(f'\n--- By Tenor Bucket ---')
bucket_stats = isin_df.groupby('Bucket', observed=True).agg(
    N=('Err_Svensson_bps', 'count'),
    Svensson_MAE=('Err_Svensson_bps', lambda x: np.abs(x).mean()),
    Svensson_RMSE=('Err_Svensson_bps', lambda x: np.sqrt((x**2).mean())),
).round(2)
print(bucket_stats.to_string())
```

else:
print(‘No ISIN-level matches found.’)

print(’\n’ + ‘=’*80)
print(‘SVENSSON BACKTEST COMPLETE’)
print(’=’*80)