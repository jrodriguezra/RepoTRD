# %% Cell 1 - Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import warnings
warnings.filterwarnings(‘ignore’)

FILE_PATH = ‘Data_historica.xlsx’

# — Config —

LOOKBACK = 252
CURVE_STEP = 10
CURVE_MAX = 600
N_PCS = 2

# %% Cell 2 - Functions

def monotone_convex_interpolate(tenors, yields, target_tenors):
t = np.array(tenors, dtype=float)
y = np.array(yields, dtype=float)
n = len(t)
if n < 2:
return np.full(len(target_tenors), np.nan)
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
result = np.zeros(len(target_tenors))
for k, tt in enumerate(target_tenors):
if tt <= t[0]:
result[k] = y[0]
elif tt >= t[-1]:
result[k] = y[-1]
else:
idx = np.searchsorted(t, tt, side=‘right’) - 1
idx = min(idx, n - 2)
t0, t1 = t[idx], t[idx + 1]
y0, y1 = y[idx], y[idx + 1]
dt = t1 - t0
x = (tt - t0) / dt
g = (t1 * y1 - t0 * y0) / dt
a = f_inst[idx] - g
b = f_inst[idx + 1] - g
integral_norm = g * x + a * (x - 2*x**2 + x**3) + b * (-x**2 + x**3)
tty = t0 * y0 + dt * integral_norm
result[k] = tty / tt
return result

def build_daily_instruments(df):
# returns dict: {date: dataframe with ISIN, Yield, Dias}
instruments = {}
for fecha, grp in df.groupby(‘Fecha’):
g = grp[grp[‘ISIN’] != ‘TIBO’].copy()
instruments[fecha] = g[[‘ISIN’, ‘Yield’, ‘Dias’]].reset_index(drop=True)
return instruments

def get_tibo(df, date):
tibo_rows = df[(df[‘Fecha’] == date) & (df[‘ISIN’] == ‘TIBO’)]
if len(tibo_rows) > 0:
return tibo_rows[‘Yield’].values[0]
return np.nan

def forward_roll(tenors, yields, tibo, n_days):
t = np.array(tenors, dtype=float)
y = np.array(yields, dtype=float)
new_t = t - n_days
mask = new_t > 0
y_fwd = np.full_like(y, np.nan)
if mask.sum() > 0:
df_old = (1 + y[mask]) ** (t[mask] / 360)
df_tibo = (1 + tibo) ** (n_days / 360)
y_fwd[mask] = (df_old / df_tibo) ** (360 / new_t[mask]) - 1
return new_t, y_fwd, mask

def build_daily_curves_for_pca(df, grid):
# for PCA: interpolate each day’s raw instruments onto fixed grid
curves = {}
for fecha, grp in df.groupby(‘Fecha’):
g = grp[grp[‘ISIN’] != ‘TIBO’].sort_values(‘Dias’)
tenors = g[‘Dias’].values.astype(float)
yields = g[‘Yield’].values.astype(float)
unique_tenors = np.unique(tenors)
if len(unique_tenors) < len(tenors):
avg_yields = np.array([yields[tenors == t].mean() for t in unique_tenors])
tenors, yields = unique_tenors, avg_yields
if len(tenors) >= 2:
min_t, max_t = tenors.min(), tenors.max()
interp = monotone_convex_interpolate(tenors, yields, grid)
row = np.copy(interp)
row[grid < min_t] = np.nan
curves[fecha] = row
return curves

def run_pca(curve_matrix, n_pcs=2):
mu = curve_matrix.mean(axis=0)
X = curve_matrix - mu
cov = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
return eigenvalues, eigenvectors[:, :n_pcs], mu

def interpolate_loadings_linear(grid, loadings, target_tenors):
n_pcs = loadings.shape[1]
result = np.zeros((len(target_tenors), n_pcs))
for pc in range(n_pcs):
result[:, pc] = np.interp(target_tenors, grid, loadings[:, pc])
return result

def min_norm_dz(loadings_at_shock, surprises):
L_tau = loadings_at_shock
s = np.array(surprises).reshape(-1, 1)
LLt = L_tau @ L_tau.T
try:
LLt_inv = np.linalg.inv(LLt)
except np.linalg.LinAlgError:
return np.zeros(L_tau.shape[1])
dz = L_tau.T @ LLt_inv @ s
return dz.flatten()

# %% Cell 3 - Load Data & Prepare

df = pd.read_excel(FILE_PATH)
df.columns = [‘Fecha’, ‘ISIN’, ‘Yield’, ‘Dias’]
df[‘Fecha’] = pd.to_datetime(df[‘Fecha’], dayfirst=True)
df = df.dropna(subset=[‘Yield’, ‘Dias’])
df = df[df[‘Dias’] > 0]
df[‘Yield’] = df[‘Yield’] / 100

all_dates = sorted(df[‘Fecha’].unique())
print(f’Total dates in file: {len(all_dates)}’)
print(f’Date range: {all_dates[0].date()} to {all_dates[-1].date()}’)

# build instrument dict

daily_instruments = build_daily_instruments(df)

# fixed grid for PCA

full_grid = np.arange(CURVE_STEP, CURVE_MAX + 1, CURVE_STEP)
pca_curves = build_daily_curves_for_pca(df, full_grid)

# %% Cell 4 - Backtest Loop

# backtest starts after LOOKBACK days

backtest_start_idx = LOOKBACK
backtest_dates = all_dates[backtest_start_idx:]
print(f’Backtest dates: {len(backtest_dates)} ({backtest_dates[0].date()} to {backtest_dates[-1].date()})’)

# storage

isin_results = []  # per-ISIN per-date results
grid_results = []  # per-grid-node per-date results
daily_summaries = []  # per-date summary
day_by_day_curves = {}  # {date: {tenors, fwd_roll, pca_adj, actual}}

for d_idx, current_date in enumerate(backtest_dates):
# find previous available date
current_pos = all_dates.index(current_date)
prev_date = all_dates[current_pos - 1]
fwd_roll_days = (current_date - prev_date).days

```
# get TIBO for current date
tibo = get_tibo(df, current_date)
if np.isnan(tibo):
    continue

# get previous date instruments
if prev_date not in daily_instruments:
    continue
prev_inst = daily_instruments[prev_date]
prev_tenors = prev_inst['Dias'].values.astype(float)
prev_yields = prev_inst['Yield'].values.astype(float)
prev_isins = prev_inst['ISIN'].values

# forward roll
fwd_tenors, fwd_yields, fwd_mask = forward_roll(prev_tenors, prev_yields, tibo, fwd_roll_days)
if fwd_mask.sum() < 2:
    continue

fwd_tenors_valid = fwd_tenors[fwd_mask]
fwd_yields_valid = fwd_yields[fwd_mask]
fwd_isins_valid = prev_isins[fwd_mask]

# get current date instruments (actuals)
if current_date not in daily_instruments:
    continue
curr_inst = daily_instruments[current_date]
curr_isins = set(curr_inst['ISIN'].values)
prev_isins_set = set(fwd_isins_valid)

# identify new issues: ISINs in current date but not in previous date
new_isins = curr_isins - prev_isins_set
has_new_issues = len(new_isins) > 0

if has_new_issues:
    new_inst = curr_inst[curr_inst['ISIN'].isin(new_isins)]
    new_tenors = new_inst['Dias'].values.astype(float)
    new_yields = new_inst['Yield'].values.astype(float)

    # build PCA on trailing LOOKBACK window
    pca_dates = [d for d in all_dates[:current_pos] if d in pca_curves]
    pca_dates = pca_dates[-LOOKBACK:]
    if len(pca_dates) < 50:
        # not enough data for PCA, fall back to fwd roll
        has_new_issues = False

if has_new_issues:
    # stack PCA curves
    pca_matrix_raw = np.array([pca_curves[d] for d in pca_dates])
    pca_df = pd.DataFrame(pca_matrix_raw, columns=full_grid)

    # drop columns with >5% nans
    valid_cols = pca_df.columns[pca_df.isna().mean() < 0.05]
    pca_clean = pca_df[valid_cols].dropna()
    grid_valid = valid_cols.values.astype(float)

    if len(pca_clean) < 50 or len(grid_valid) < 5:
        has_new_issues = False

if has_new_issues:
    # run PCA
    eigenvalues, loadings, mu = run_pca(pca_clean.values, n_pcs=N_PCS)

    # surprise: monotone convex interpolate fwd roll curve at new issue tenors
    fwd_at_new = monotone_convex_interpolate(fwd_tenors_valid, fwd_yields_valid, new_tenors)
    surprises = new_yields - fwd_at_new

    # filter out new issues where tenor is outside grid range
    in_range = (new_tenors >= grid_valid.min()) & (new_tenors <= grid_valid.max())
    if in_range.sum() == 0:
        has_new_issues = False

if has_new_issues:
    new_tenors_valid = new_tenors[in_range]
    surprises_valid = surprises[in_range]
    new_yields_valid = new_yields[in_range]

    # min-norm PCA adjustment
    L_new = interpolate_loadings_linear(grid_valid, loadings, new_tenors_valid)
    dz = min_norm_dz(L_new, surprises_valid)

    # adjust instrument yields via linearly interpolated loadings
    L_instruments = interpolate_loadings_linear(grid_valid, loadings, fwd_tenors_valid)
    dy_instruments = L_instruments @ dz
    adj_yields_instruments = fwd_yields_valid + dy_instruments

    # merge new issues into adjusted set for curve building
    all_adj_tenors = np.concatenate([fwd_tenors_valid, new_tenors])
    all_adj_yields_raw = np.concatenate([adj_yields_instruments, new_yields])
    sort_idx = np.argsort(all_adj_tenors)
    all_adj_tenors = all_adj_tenors[sort_idx]
    all_adj_yields_raw = all_adj_yields_raw[sort_idx]
else:
    adj_yields_instruments = fwd_yields_valid.copy()
    dy_instruments = np.zeros_like(fwd_yields_valid)
    all_adj_tenors = fwd_tenors_valid.copy()
    all_adj_yields_raw = adj_yields_instruments.copy()

# build display grid curves
max_tenor = all_adj_tenors.max()
display_max = min(CURVE_MAX, int(np.ceil(max_tenor / CURVE_STEP) * CURVE_STEP))
display_grid = np.arange(CURVE_STEP, display_max + 1, CURVE_STEP)
display_grid = display_grid[display_grid >= all_adj_tenors.min()]

adj_curve = monotone_convex_interpolate(all_adj_tenors, all_adj_yields_raw, display_grid)
fwd_curve = monotone_convex_interpolate(fwd_tenors_valid, fwd_yields_valid, display_grid)

# actual curve on same grid
curr_all = curr_inst.sort_values('Dias')
curr_tenors_all = curr_all['Dias'].values.astype(float)
curr_yields_all = curr_all['Yield'].values.astype(float)
if len(curr_tenors_all) >= 2:
    actual_curve = monotone_convex_interpolate(curr_tenors_all, curr_yields_all, display_grid)
else:
    continue

# store grid-level results
for i, tenor in enumerate(display_grid):
    grid_results.append({
        'Date': current_date,
        'Tenor': int(tenor),
        'FwdRoll': fwd_curve[i],
        'PCA_Adj': adj_curve[i],
        'Actual': actual_curve[i],
        'Err_FwdRoll_bps': (fwd_curve[i] - actual_curve[i]) * 10000,
        'Err_PCA_bps': (adj_curve[i] - actual_curve[i]) * 10000,
        'Has_New_Issue': has_new_issues,
    })

# store ISIN-level results (match by ISIN)
curr_isin_map = dict(zip(curr_inst['ISIN'].values, curr_inst['Yield'].values))
for i, isin in enumerate(fwd_isins_valid):
    if isin in curr_isin_map:
        actual_yield = curr_isin_map[isin]
        isin_results.append({
            'Date': current_date,
            'ISIN': isin,
            'Tenor': int(fwd_tenors_valid[i]),
            'FwdRoll': fwd_yields_valid[i],
            'PCA_Adj': adj_yields_instruments[i],
            'Actual': actual_yield,
            'Err_FwdRoll_bps': (fwd_yields_valid[i] - actual_yield) * 10000,
            'Err_PCA_bps': (adj_yields_instruments[i] - actual_yield) * 10000,
            'Has_New_Issue': has_new_issues,
        })

# store day-by-day curves
day_by_day_curves[current_date] = {
    'grid': display_grid,
    'fwd_roll': fwd_curve,
    'pca_adj': adj_curve,
    'actual': actual_curve,
}

# daily summary
grid_errs_fwd = (fwd_curve - actual_curve) * 10000
grid_errs_pca = (adj_curve - actual_curve) * 10000
daily_summaries.append({
    'Date': current_date,
    'Has_New_Issue': has_new_issues,
    'N_New_Issues': len(new_isins) if has_new_issues else 0,
    'FwdRoll_MAE_bps': np.abs(grid_errs_fwd).mean(),
    'PCA_MAE_bps': np.abs(grid_errs_pca).mean(),
    'FwdRoll_Max_bps': np.abs(grid_errs_fwd).max(),
    'PCA_Max_bps': np.abs(grid_errs_pca).max(),
    'FwdRoll_RMSE_bps': np.sqrt((grid_errs_fwd**2).mean()),
    'PCA_RMSE_bps': np.sqrt((grid_errs_pca**2).mean()),
    'Fwd_Roll_Days': fwd_roll_days,
})

if (d_idx + 1) % 10 == 0:
    print(f'  Processed {d_idx + 1}/{len(backtest_dates)} dates...')
```

print(f’\nBacktest complete: {len(daily_summaries)} dates processed’)

# convert to dataframes

isin_df = pd.DataFrame(isin_results)
grid_df = pd.DataFrame(grid_results)
daily_df = pd.DataFrame(daily_summaries)

# %% Cell 5 - Summary Statistics

print(’\n’ + ‘=’*80)
print(‘AGGREGATE BACKTEST RESULTS’)
print(’=’*80)

print(f’\nDates processed: {len(daily_df)}’)
print(f’  New issue days: {daily_df[“Has_New_Issue”].sum()}’)
print(f’  No new issue days: {(~daily_df[“Has_New_Issue”]).sum()}’)

print(f’\n— Daily MAE (bps) —’)
print(f’  Fwd Roll:    median={daily_df[“FwdRoll_MAE_bps”].median():.2f}  mean={daily_df[“FwdRoll_MAE_bps”].mean():.2f}  p75={daily_df[“FwdRoll_MAE_bps”].quantile(0.75):.2f}  p95={daily_df[“FwdRoll_MAE_bps”].quantile(0.95):.2f}  max={daily_df[“FwdRoll_MAE_bps”].max():.2f}’)
print(f’  PCA Adj:     median={daily_df[“PCA_MAE_bps”].median():.2f}  mean={daily_df[“PCA_MAE_bps”].mean():.2f}  p75={daily_df[“PCA_MAE_bps”].quantile(0.75):.2f}  p95={daily_df[“PCA_MAE_bps”].quantile(0.95):.2f}  max={daily_df[“PCA_MAE_bps”].max():.2f}’)

print(f’\n— Daily Max Abs Error (bps) —’)
print(f’  Fwd Roll:    median={daily_df[“FwdRoll_Max_bps”].median():.2f}  mean={daily_df[“FwdRoll_Max_bps”].mean():.2f}  p95={daily_df[“FwdRoll_Max_bps”].quantile(0.95):.2f}  max={daily_df[“FwdRoll_Max_bps”].max():.2f}’)
print(f’  PCA Adj:     median={daily_df[“PCA_Max_bps”].median():.2f}  mean={daily_df[“PCA_Max_bps”].mean():.2f}  p95={daily_df[“PCA_Max_bps”].quantile(0.95):.2f}  max={daily_df[“PCA_Max_bps”].max():.2f}’)

# split by new issue vs no new issue

ni_days = daily_df[daily_df[‘Has_New_Issue’]]
no_ni_days = daily_df[~daily_df[‘Has_New_Issue’]]
print(f’\n— New Issue Days Only ({len(ni_days)}) —’)
if len(ni_days) > 0:
print(f’  Fwd Roll MAE: median={ni_days[“FwdRoll_MAE_bps”].median():.2f}  mean={ni_days[“FwdRoll_MAE_bps”].mean():.2f}’)
print(f’  PCA Adj MAE:  median={ni_days[“PCA_MAE_bps”].median():.2f}  mean={ni_days[“PCA_MAE_bps”].mean():.2f}’)
print(f’  Improvement:  {(1 - ni_days[“PCA_MAE_bps”].mean() / ni_days[“FwdRoll_MAE_bps”].mean()) * 100:.1f}% mean MAE reduction’)

print(f’\n— No New Issue Days ({len(no_ni_days)}) —’)
if len(no_ni_days) > 0:
print(f’  Fwd Roll MAE: median={no_ni_days[“FwdRoll_MAE_bps”].median():.2f}  mean={no_ni_days[“FwdRoll_MAE_bps”].mean():.2f}’)

# %% Cell 6 - MAE and RMSE per Grid Node

print(’\n’ + ‘=’*80)
print(‘MAE AND RMSE PER GRID NODE (bps)’)
print(’=’*80)

tenor_stats = grid_df.groupby(‘Tenor’).agg(
FwdRoll_MAE=(‘Err_FwdRoll_bps’, lambda x: np.abs(x).mean()),
PCA_MAE=(‘Err_PCA_bps’, lambda x: np.abs(x).mean()),
FwdRoll_RMSE=(‘Err_FwdRoll_bps’, lambda x: np.sqrt((x**2).mean())),
PCA_RMSE=(‘Err_PCA_bps’, lambda x: np.sqrt((x**2).mean())),
FwdRoll_Median=(‘Err_FwdRoll_bps’, lambda x: np.abs(x).median()),
PCA_Median=(‘Err_PCA_bps’, lambda x: np.abs(x).median()),
N_Obs=(‘Err_PCA_bps’, ‘count’),
).round(2)

print(tenor_stats.to_string())

# %% Cell 7 - Plot 1: Max Deviation per Tenor from Grid

fig, ax = plt.subplots(figsize=(14, 6))
max_dev_fwd = grid_df.groupby(‘Tenor’)[‘Err_FwdRoll_bps’].apply(lambda x: np.abs(x).max())
max_dev_pca = grid_df.groupby(‘Tenor’)[‘Err_PCA_bps’].apply(lambda x: np.abs(x).max())
tenors_plot = max_dev_fwd.index.values
width = CURVE_STEP * 0.35
ax.bar(tenors_plot - width/2, max_dev_fwd.values, width=width, color=’#2C5F8A’, alpha=0.8, label=‘Fwd Roll’)
ax.bar(tenors_plot + width/2, max_dev_pca.values, width=width, color=’#D4553A’, alpha=0.8, label=‘PCA Adjusted’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Max Absolute Error (bps)’)
ax.set_title(‘Max Deviation per Grid Tenor’)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(‘backtest_max_deviation.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 8 - Plot 2: Boxplot of PCA Adj Error vs Actual per Node

# select a subset of tenors for readability

all_tenors_in_data = sorted(grid_df[‘Tenor’].unique())

# pick every other or every 3rd if too many

if len(all_tenors_in_data) > 20:
plot_tenors = all_tenors_in_data[::3]
else:
plot_tenors = all_tenors_in_data

fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# fwd roll boxplot

ax = axes[0]
box_data_fwd = [grid_df[grid_df[‘Tenor’] == t][‘Err_FwdRoll_bps’].values for t in plot_tenors]
bp1 = ax.boxplot(box_data_fwd, positions=range(len(plot_tenors)), widths=0.6,
patch_artist=True, showfliers=True, flierprops={‘markersize’: 2})
for patch in bp1[‘boxes’]:
patch.set_facecolor(’#2C5F8A’)
patch.set_alpha(0.6)
ax.set_xticks(range(len(plot_tenors)))
ax.set_xticklabels([str(t) for t in plot_tenors], rotation=45, fontsize=8)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Error (bps)’)
ax.set_title(‘Fwd Roll Error Distribution per Node’)
ax.grid(True, alpha=0.3)

# pca adj boxplot

ax = axes[1]
box_data_pca = [grid_df[grid_df[‘Tenor’] == t][‘Err_PCA_bps’].values for t in plot_tenors]
bp2 = ax.boxplot(box_data_pca, positions=range(len(plot_tenors)), widths=0.6,
patch_artist=True, showfliers=True, flierprops={‘markersize’: 2})
for patch in bp2[‘boxes’]:
patch.set_facecolor(’#D4553A’)
patch.set_alpha(0.6)
ax.set_xticks(range(len(plot_tenors)))
ax.set_xticklabels([str(t) for t in plot_tenors], rotation=45, fontsize=8)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Error (bps)’)
ax.set_title(‘PCA Adjusted Error Distribution per Node’)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(‘backtest_boxplots.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 9 - Plot 3: Line Plot of Each Node Error Across Dates

# pick subset of representative tenors

repr_tenors = [30, 60, 90, 120, 180, 270, 360]
repr_tenors = [t for t in repr_tenors if t in grid_df[‘Tenor’].unique()]

fig, axes = plt.subplots(len(repr_tenors), 1, figsize=(16, 3 * len(repr_tenors)), sharex=True)
if len(repr_tenors) == 1:
axes = [axes]

for i, tenor in enumerate(repr_tenors):
ax = axes[i]
subset = grid_df[grid_df[‘Tenor’] == tenor].sort_values(‘Date’)
ax.plot(subset[‘Date’], subset[‘Err_FwdRoll_bps’], color=’#2C5F8A’, alpha=0.6, linewidth=0.8, label=‘Fwd Roll’)
ax.plot(subset[‘Date’], subset[‘Err_PCA_bps’], color=’#D4553A’, alpha=0.8, linewidth=0.8, label=‘PCA Adj’)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_ylabel(‘Error (bps)’)
ax.set_title(f’{tenor}d’, fontsize=10)
ax.legend(fontsize=7, loc=‘upper right’)
ax.grid(True, alpha=0.3)

plt.xlabel(‘Date’)
plt.suptitle(‘Error Time Series by Tenor (PCA Adj vs Fwd Roll)’, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘backtest_error_timeseries.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 10 - Plot 4: Bell Curve Grid of Error Distributions

n_cols = 4
n_rows = int(np.ceil(len(plot_tenors) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
axes_flat = axes.flatten()

for i, tenor in enumerate(plot_tenors):
ax = axes_flat[i]
errs = grid_df[grid_df[‘Tenor’] == tenor][‘Err_PCA_bps’].dropna().values
if len(errs) < 5:
ax.set_visible(False)
continue

```
# histogram
ax.hist(errs, bins=25, density=True, color='#D4553A', alpha=0.5, edgecolor='white', linewidth=0.5)

# fitted normal
mu_fit, std_fit = errs.mean(), errs.std()
x_range = np.linspace(errs.min() - 1, errs.max() + 1, 200)
ax.plot(x_range, stats.norm.pdf(x_range, mu_fit, std_fit), color='black', linewidth=1.2)

# stats annotation
skew = stats.skew(errs)
kurt = stats.kurtosis(errs)
ax.set_title(f'{tenor}d', fontsize=9)
ax.text(0.95, 0.95, f'μ={mu_fit:.1f}\nσ={std_fit:.1f}\nskew={skew:.2f}\nkurt={kurt:.2f}',
        transform=ax.transAxes, fontsize=6, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax.set_xlabel('bps', fontsize=7)
ax.tick_params(labelsize=6)
```

# hide unused subplots

for j in range(len(plot_tenors), len(axes_flat)):
axes_flat[j].set_visible(False)

plt.suptitle(‘PCA Adjusted Error Distributions by Tenor’, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘backtest_error_distributions.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# %% Cell 11 - Z-Score Test for Directional Predictive Power

print(’\n’ + ‘=’*80)
print(‘DIRECTIONAL PREDICTIVE POWER - Z-SCORE TESTS’)
print(’=’*80)
print(’\nTest: does the PCA adjustment move in the correct direction?’)
print(‘H0: adjustment direction is random (50/50)’)
print(‘Metric: sign(PCA_adj - FwdRoll) == sign(Actual - FwdRoll)\n’)

# only on new issue days where adjustment is nonzero

ni_grid = grid_df[grid_df[‘Has_New_Issue’]].copy()
ni_grid[‘Adj_Direction’] = np.sign(ni_grid[‘PCA_Adj’] - ni_grid[‘FwdRoll’])
ni_grid[‘Actual_Direction’] = np.sign(ni_grid[‘Actual’] - ni_grid[‘FwdRoll’])
ni_grid[‘Correct’] = (ni_grid[‘Adj_Direction’] == ni_grid[‘Actual_Direction’]) & (ni_grid[‘Adj_Direction’] != 0)
ni_grid[‘Nonzero’] = (ni_grid[‘Adj_Direction’] != 0) & (ni_grid[‘Actual_Direction’] != 0)

print(f’{“Tenor”:>8s}  {“N”:>5s}  {“Correct”:>8s}  {“Hit%”:>6s}  {“Z-stat”:>7s}  {“p-value”:>8s}  {“Sig”:>4s}’)
print(’-’ * 55)

ztest_results = []
for tenor in sorted(ni_grid[‘Tenor’].unique()):
subset = ni_grid[ni_grid[‘Tenor’] == tenor]
nonzero = subset[subset[‘Nonzero’]]
n = len(nonzero)
if n < 10:
continue
correct = nonzero[‘Correct’].sum()
hit_rate = correct / n
# z-test vs 50%
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
print(f’  Fwd Roll MAE:  {np.abs(isin_df[“Err_FwdRoll_bps”]).mean():.2f}’)
print(f’  PCA Adj MAE:   {np.abs(isin_df[“Err_PCA_bps”]).mean():.2f}’)
print(f’  Fwd Roll RMSE: {np.sqrt((isin_df[“Err_FwdRoll_bps”]**2).mean()):.2f}’)
print(f’  PCA Adj RMSE:  {np.sqrt((isin_df[“Err_PCA_bps”]**2).mean()):.2f}’)

```
# bucket by tenor
bins = [0, 90, 180, 360, 9999]
labels = ['<90d', '90-180d', '180-360d', '>360d']
isin_df['Bucket'] = pd.cut(isin_df['Tenor'], bins=bins, labels=labels)

print(f'\n--- By Tenor Bucket ---')
bucket_stats = isin_df.groupby('Bucket', observed=True).agg(
    N=('Err_PCA_bps', 'count'),
    FwdRoll_MAE=('Err_FwdRoll_bps', lambda x: np.abs(x).mean()),
    PCA_MAE=('Err_PCA_bps', lambda x: np.abs(x).mean()),
    FwdRoll_RMSE=('Err_FwdRoll_bps', lambda x: np.sqrt((x**2).mean())),
    PCA_RMSE=('Err_PCA_bps', lambda x: np.sqrt((x**2).mean())),
).round(2)
print(bucket_stats.to_string())
```

else:
print(‘No ISIN-level matches found.’)

print(’\n’ + ‘=’*80)
print(‘BACKTEST COMPLETE’)
print(’=’*80)