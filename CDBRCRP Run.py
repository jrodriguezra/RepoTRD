# %% Cell 1 - Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FILE_PATH = ‘Data_historica.xlsx’

# — Manual Inputs —

LOOKBACK = 252
INSTRUMENT_DATE = ‘2025-04-25’  # date to pull yesterday’s outstanding CDs
CURVE_STEP = 10
CURVE_MAX = 600
FWD_ROLL_DAYS = 1
TIBO = 0.0475  # today’s ON rate as decimal (e.g., 4.75% -> 0.0475)

# new issues: list of (tenor_days, yield_decimal)

# e.g., [(90, 0.0480), (180, 0.0510)]

NEW_ISSUES = [(90, 0.0480)]

# %% Cell 2 - Functions

def monotone_convex_interpolate(tenors, yields, target_tenors):
t = np.array(tenors, dtype=float)
y = np.array(yields, dtype=float)
n = len(t)
if n < 2:
return np.full(len(target_tenors), np.nan)
ty = t * y
# sector discrete forwards
f = np.zeros(n)
f[0] = y[0]
for i in range(1, n):
f[i] = (ty[i] - ty[i - 1]) / (t[i] - t[i - 1])
# instantaneous forwards at knots (hagan-west)
f_inst = np.zeros(n)
f_inst[0] = f[0]
f_inst[-1] = f[-1]
for i in range(1, n - 1):
dt_l = t[i] - t[i - 1]
dt_r = t[i + 1] - t[i]
f_inst[i] = (dt_r * f[i] + dt_l * f[i + 1]) / (dt_l + dt_r)
# monotonicity correction
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
# interpolate
result = np.zeros(len(target_tenors))
for k, tt in enumerate(target_tenors):
if tt <= t[0]:
result[k] = y[0]
elif tt >= t[-1]:
result[k] = y[-1]  # freeze at last yield
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

def build_daily_curves(df):
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
curves[fecha] = (tenors, yields)
return curves

def interpolate_to_grid(curves, grid):
dates = sorted(curves.keys())
data = {}
for d in dates:
tenors, yields = curves[d]
min_t, max_t = tenors.min(), tenors.max()
interp = monotone_convex_interpolate(tenors, yields, grid)
# freeze beyond max tenor (no extrapolation), mask below min
row = np.copy(interp)
row[grid < min_t] = np.nan
# beyond max_t: already handled by monotone_convex (returns y[-1])
data[d] = row
return pd.DataFrame(data, index=grid).T.sort_index()

def forward_roll(tenors, yields, tibo, n_days):
# y_fwd = ((1+y)^(t/360) / (1+tibo)^(n/360))^(360/(t-n)) - 1
t = np.array(tenors, dtype=float)
y = np.array(yields, dtype=float)
new_t = t - n_days
mask = new_t > 0
y_fwd = np.full_like(y, np.nan)
df_old = (1 + y[mask]) ** (t[mask] / 360)
df_tibo = (1 + tibo) ** (n_days / 360)
y_fwd[mask] = (df_old / df_tibo) ** (360 / new_t[mask]) - 1
return new_t, y_fwd, mask

def run_pca(curve_matrix, n_pcs=2):
mu = curve_matrix.mean(axis=0)
X = curve_matrix - mu
cov = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
return eigenvalues, eigenvectors[:, :n_pcs], mu

def interpolate_loadings(grid, loadings, target_tenors):
# linearly interpolate each PC’s loading at target tenors
n_pcs = loadings.shape[1]
result = np.zeros((len(target_tenors), n_pcs))
for pc in range(n_pcs):
result[:, pc] = np.interp(target_tenors, grid, loadings[:, pc])
return result

def min_norm_adjustment(loadings_grid, loadings_new_issue, surprises):
# loadings_grid: (n_grid x k)
# loadings_new_issue: (m x k) where m = number of new issues
# surprises: (m,) vector in yield units
L_tau = loadings_new_issue  # (m x k)
s = np.array(surprises).reshape(-1, 1)  # (m x 1)
LLt = L_tau @ L_tau.T  # (m x m)
LLt_inv = np.linalg.inv(LLt)
dz = L_tau.T @ LLt_inv @ s  # (k x 1)
dy_grid = loadings_grid @ dz  # (n_grid x 1)
return dz.flatten(), dy_grid.flatten()

# %% Cell 3 - Pipeline

# load data

df = pd.read_excel(FILE_PATH)
df.columns = [‘Fecha’, ‘ISIN’, ‘Yield’, ‘Dias’]
df[‘Fecha’] = pd.to_datetime(df[‘Fecha’], dayfirst=True)
df = df.dropna(subset=[‘Yield’, ‘Dias’])
df = df[df[‘Dias’] > 0]

# parse instrument date

inst_date = pd.to_datetime(INSTRUMENT_DATE, dayfirst=False)

# — Step 1: Pull yesterday’s instruments —

inst_df = df[(df[‘Fecha’] == inst_date) & (df[‘ISIN’] != ‘TIBO’)].copy()
inst_df = inst_df.sort_values(‘Dias’)
print(f’Instruments on {inst_date.date()}: {len(inst_df)}’)
print(inst_df[[‘ISIN’, ‘Yield’, ‘Dias’]].to_string(index=False))

yest_tenors = inst_df[‘Dias’].values.astype(float)
yest_yields = inst_df[‘Yield’].values.astype(float)

# — Step 2: Forward roll —

fwd_tenors, fwd_yields, fwd_mask = forward_roll(yest_tenors, yest_yields, TIBO, FWD_ROLL_DAYS)
print(f’\nForward-rolled instruments ({FWD_ROLL_DAYS}d roll, TIBO={TIBO*100:.2f}%):’)
for i in range(len(fwd_tenors)):
if fwd_mask[i]:
print(f’  {yest_tenors[i]:.0f}d -> {fwd_tenors[i]:.0f}d: {fwd_yields[i]*100:.4f}%’)

# filter valid (positive tenor after roll)

fwd_tenors_valid = fwd_tenors[fwd_mask]
fwd_yields_valid = fwd_yields[fwd_mask]

# — Step 3: Build grid and historical PCA —

max_tenor_today = max(fwd_tenors_valid.max(), max(t for t, _ in NEW_ISSUES))
grid_max = min(CURVE_MAX, int(np.ceil(max_tenor_today / CURVE_STEP) * CURVE_STEP))
grid = np.arange(CURVE_STEP, grid_max + 1, CURVE_STEP)
print(f’\nGrid: {grid[0]}d to {grid[-1]}d, step {CURVE_STEP}d, {len(grid)} nodes’)

# build all daily curves and interpolate to grid

raw_curves = build_daily_curves(df)
grid_df = interpolate_to_grid(raw_curves, grid)

# trailing window ending at instrument date

all_dates = grid_df.index
dates_up_to = all_dates[all_dates <= inst_date]
if len(dates_up_to) < LOOKBACK:
print(f’WARNING: only {len(dates_up_to)} dates available, need {LOOKBACK}’)
window = grid_df.loc[dates_up_to[-LOOKBACK:]]

# drop grid tenors with nans in window

valid_cols = window.columns[window.isna().mean() < 0.05]
window_clean = window[valid_cols].dropna()
grid_valid = valid_cols.values.astype(float)
print(f’PCA window: {len(window_clean)} obs, {len(grid_valid)} valid tenors ({grid_valid[0]:.0f}d-{grid_valid[-1]:.0f}d)’)

# run PCA

eigenvalues, loadings, mu = run_pca(window_clean.values, n_pcs=2)
var_exp = eigenvalues[:2] / eigenvalues.sum() * 100
print(f’PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}% (total: {var_exp.sum():.1f}%)’)

# — Step 4: Forward-rolled curve on grid —

fwd_curve_grid = monotone_convex_interpolate(fwd_tenors_valid, fwd_yields_valid, grid_valid)

# — Step 5: Compute surprises —

new_issue_tenors = np.array([t for t, _ in NEW_ISSUES])
new_issue_yields = np.array([y for _, y in NEW_ISSUES])

# interpolate fwd-rolled curve at new issue tenors

fwd_at_new = np.interp(new_issue_tenors, fwd_tenors_valid, fwd_yields_valid)

# also try monotone convex for consistency

fwd_at_new_mc = monotone_convex_interpolate(fwd_tenors_valid, fwd_yields_valid, new_issue_tenors)
surprises = new_issue_yields - fwd_at_new_mc

print(f’\nNew issue surprises:’)
for i, (t, y) in enumerate(NEW_ISSUES):
print(f’  {t}d: auction={y*100:.4f}%, fwd_implied={fwd_at_new_mc[i]*100:.4f}%, surprise={surprises[i]*10000:.2f}bps’)

# — Step 6: Min-norm PCA adjustment —

# interpolate loadings at new issue tenors

L_new = interpolate_loadings(grid_valid, loadings, new_issue_tenors)
dz, dy_grid = min_norm_adjustment(loadings, L_new, surprises)
print(f’\nPC score shocks: PC1={dz[0]*10000:.2f}bps, PC2={dz[1]*10000:.2f}bps’)

# adjusted grid curve

adj_curve_grid = fwd_curve_grid + dy_grid

# — Step 7: Interpolate adjustment back to instrument tenors —

dy_instruments = np.interp(fwd_tenors_valid, grid_valid, dy_grid)
adj_yields_instruments = fwd_yields_valid + dy_instruments

# — Step 8: Yesterday’s actual curve on grid (for DoD delta) —

yest_curve_grid = monotone_convex_interpolate(yest_tenors, yest_yields, grid_valid)
dod_grid = adj_curve_grid - yest_curve_grid

# yesterday’s actual at instrument tenors (for DoD at instruments)

dod_instruments = adj_yields_instruments - yest_yields[fwd_mask]

# delta vs fwd roll

delta_vs_fwd_grid = dy_grid
delta_vs_fwd_instruments = dy_instruments

# %% Cell 4 - Output Tables

print(’\n’ + ‘=’*80)
print(‘10d GRID CURVE OUTPUT’)
print(’=’*80)
grid_output = pd.DataFrame({
‘Tenor’: grid_valid.astype(int),
‘Yest_Actual’: yest_curve_grid * 100,
‘Fwd_Roll’: fwd_curve_grid * 100,
‘PCA_Adjusted’: adj_curve_grid * 100,
‘Delta_vs_FwdRoll_bps’: delta_vs_fwd_grid * 10000,
‘DoD_Delta_bps’: dod_grid * 10000,
})
grid_output = grid_output.round(4)
print(grid_output.to_string(index=False))

print(’\n’ + ‘=’*80)
print(‘INSTRUMENT-LEVEL OUTPUT’)
print(’=’*80)
inst_output = pd.DataFrame({
‘ISIN’: inst_df[‘ISIN’].values[fwd_mask],
‘Yest_Tenor’: yest_tenors[fwd_mask].astype(int),
‘Today_Tenor’: fwd_tenors_valid.astype(int),
‘Yest_Yield’: yest_yields[fwd_mask] * 100,
‘Fwd_Roll’: fwd_yields_valid * 100,
‘PCA_Adjusted’: adj_yields_instruments * 100,
‘Delta_vs_FwdRoll_bps’: delta_vs_fwd_instruments * 10000,
‘DoD_Delta_bps’: dod_instruments * 10000,
})
inst_output = inst_output.round(4)
print(inst_output.to_string(index=False))

# %% Cell 5 - Visualization

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# plot 1: curves comparison on grid

ax = axes[0, 0]
ax.plot(grid_valid, yest_curve_grid * 100, ‘s-’, color=’#888888’, markersize=3, linewidth=1, label=‘Yesterday Actual’)
ax.plot(grid_valid, fwd_curve_grid * 100, ‘o-’, color=’#2C5F8A’, markersize=3, linewidth=1.5, label=f’Fwd Roll ({FWD_ROLL_DAYS}d)’)
ax.plot(grid_valid, adj_curve_grid * 100, ‘D-’, color=’#D4553A’, markersize=3, linewidth=1.5, label=‘PCA Adjusted’)
for t, y in NEW_ISSUES:
ax.plot(t, y * 100, ‘*’, color=’#4A9B6E’, markersize=15, zorder=5, label=f’New Issue ({t}d)’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘CDBCRP Curve: Fwd Roll vs PCA Adjusted’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 2: delta vs fwd roll on grid

ax = axes[0, 1]
ax.bar(grid_valid, delta_vs_fwd_grid * 10000, width=CURVE_STEP * 0.7, color=’#D4553A’, alpha=0.7)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘bps’)
ax.set_title(‘PCA Adjustment (vs Fwd Roll)’)
ax.grid(True, alpha=0.3)

# plot 3: DoD delta on grid

ax = axes[1, 0]
ax.bar(grid_valid, dod_grid * 10000, width=CURVE_STEP * 0.7, color=’#2C5F8A’, alpha=0.7)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘bps’)
ax.set_title(‘Day-over-Day Change (Adjusted vs Yesterday Actual)’)
ax.grid(True, alpha=0.3)

# plot 4: instrument-level comparison

ax = axes[1, 1]
ax.plot(fwd_tenors_valid, fwd_yields_valid * 100, ‘o-’, color=’#2C5F8A’, markersize=5, linewidth=1, label=‘Fwd Roll’)
ax.plot(fwd_tenors_valid, adj_yields_instruments * 100, ‘D-’, color=’#D4553A’, markersize=5, linewidth=1, label=‘PCA Adjusted’)
for t, y in NEW_ISSUES:
ax.plot(t, y * 100, ‘*’, color=’#4A9B6E’, markersize=15, zorder=5, label=f’New Issue ({t}d)’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘Instrument-Level: Fwd Roll vs PCA Adjusted’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f’CDBCRP Forecast — {inst_date.date()} + {FWD_ROLL_DAYS}d Roll | TIBO={TIBO*100:.2f}%’, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘cdbcrp_forecast_output.png’, dpi=150, bbox_inches=‘tight’)
plt.show()