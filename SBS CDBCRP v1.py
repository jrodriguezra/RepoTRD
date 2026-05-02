# %% Cell 1 - Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings(‘ignore’)

FILE_PATH = ‘Data_historica.xlsx’

# — Manual Inputs —

TODAY_DATE = ‘2025-04-28’  # date t
TIBO = 0.0425  # today’s TIBO as decimal

# new issues: list of (tenor_days, yield_decimal)

# set to [] for no new issues

NEW_ISSUES = [(93, 0.0480)]

# curve grid

CURVE_STEP = 10
CURVE_MAX = 600

# %% Cell 2 - Functions

def svensson(m, params):
# m: tenor in days
# params: [b0, b1, b2, b3, t1, t2]
# returns: yield as decimal
b0, b1, b2, b3, t1, t2 = params
m = np.asarray(m, dtype=float)
m = np.maximum(m, 0.01)

```
mt1 = m / t1
mt2 = m / t2

term1 = np.where(mt1 > 1e-8, (1 - np.exp(-mt1)) / mt1, 1.0)
term2 = np.where(mt1 > 1e-8, term1 - np.exp(-mt1), 0.0)
term3 = np.where(mt2 > 1e-8, (1 - np.exp(-mt2)) / mt2 - np.exp(-mt2), 0.0)

return b0 + b1 * term1 + b2 * term2 + b3 * term3
```

def svensson_objective(params, tenors, yields, durations):
# duration-weighted squared yield errors
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

# tau bounded: strictly positive, capped at max tenor
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

# method 1: L-BFGS-B (supports bounds natively)
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

# method 2: Nelder-Mead (no bounds, use penalty in objective)
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

# method 3: Powell (no bounds, use penalty in objective)
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
    print('WARNING: Svensson fit failed, returning initial params')
    return np.array(initial_params), np.inf

return best_result.x, best_result.fun
```

def build_fitting_set(prev_instruments, tibo, new_issues, day_gap):
tenors = []
yields = []
labels = []

```
# TIBO at 1d
tenors.append(1.0)
yields.append(tibo)
labels.append('TIBO')

# shifted t-1 instruments
for _, row in prev_instruments.iterrows():
    new_tenor = row['Dias'] - day_gap
    if new_tenor > 0:
        tenors.append(float(new_tenor))
        yields.append(float(row['Yield']))
        labels.append(row['ISIN'])

# new issues
for tenor, yld in new_issues:
    tenors.append(float(tenor))
    yields.append(float(yld))
    labels.append(f'NEW_{tenor}d')

tenors = np.array(tenors)
yields = np.array(yields)

sort_idx = np.argsort(tenors)
tenors = tenors[sort_idx]
yields = yields[sort_idx]
labels = [labels[i] for i in sort_idx]

return tenors, yields, labels
```

# %% Cell 3 - Pipeline

# load data

df = pd.read_excel(FILE_PATH)
df.columns = [‘Fecha’, ‘ISIN’, ‘Yield’, ‘Dias’]
df[‘Fecha’] = pd.to_datetime(df[‘Fecha’], dayfirst=True)
df = df.dropna(subset=[‘Yield’, ‘Dias’])
df = df[df[‘Dias’] > 0]
df[‘Yield’] = df[‘Yield’] / 100

today = pd.to_datetime(TODAY_DATE, dayfirst=False)
all_dates = sorted(df[‘Fecha’].unique())

# find previous available date

prev_dates = [d for d in all_dates if d < today]
if len(prev_dates) == 0:
raise ValueError(f’No dates before {today.date()} in database’)
prev_date = prev_dates[-1]
day_gap = (today - prev_date).days

print(f’Today: {today.date()}’)
print(f’Previous date: {prev_date.date()} (gap: {day_gap} calendar days)’)
print(f’TIBO: {TIBO*100:.2f}%’)

# pull t-1 instruments

prev_inst = df[(df[‘Fecha’] == prev_date) & (df[‘ISIN’] != ‘TIBO’)].copy()
prev_inst = prev_inst.sort_values(‘Dias’)
print(f’\nT-1 instruments: {len(prev_inst)}’)

# build fitting set

fit_tenors, fit_yields, fit_labels = build_fitting_set(prev_inst, TIBO, NEW_ISSUES, day_gap)
print(f’Fitting set: {len(fit_tenors)} points (1 TIBO + {len(prev_inst)} shifted + {len(NEW_ISSUES)} new)’)

print(f’\nFitting set:’)
for i in range(len(fit_tenors)):
print(f’  {fit_labels[i]:>15s}  {fit_tenors[i]:>6.0f}d  {fit_yields[i]*100:.4f}%’)

# determine max tenor for tau bounds

max_tenor_fit = fit_tenors.max()

# fit Svensson

print(f’\nFitting Svensson (tau bounded to [{1.0:.0f}, {max_tenor_fit:.0f}]d)…’)
params, cost = fit_svensson(fit_tenors, fit_yields, max_tenor=max_tenor_fit)
b0, b1, b2, b3, t1, t2 = params
print(f’  b0={b0*100:.4f}%  b1={b1*100:.4f}%  b2={b2*100:.4f}%  b3={b3*100:.4f}%’)
print(f’  t1={t1:.1f}d  t2={t2:.1f}d’)
print(f’  Objective cost: {cost:.2e}’)

# fitting errors

fitted_yields = svensson(fit_tenors, params)
fit_errors = (fit_yields - fitted_yields) * 10000
print(f’\nFitting errors (bps):’)
print(f’  MAE:  {np.abs(fit_errors).mean():.2f}’)
print(f’  RMSE: {np.sqrt((fit_errors**2).mean()):.2f}’)
print(f’  Max:  {np.abs(fit_errors).max():.2f}’)

# — Instrument-level yields —

inst_tenors = []
inst_isins = []

for _, row in prev_inst.iterrows():
new_tenor = row[‘Dias’] - day_gap
if new_tenor > 0:
inst_tenors.append(float(new_tenor))
inst_isins.append(row[‘ISIN’])

for tenor, yld in NEW_ISSUES:
inst_tenors.append(float(tenor))
inst_isins.append(f’NEW_{tenor}d’)

inst_tenors = np.array(inst_tenors)
inst_yields_svensson = svensson(inst_tenors, params)

# get actual yields for today if available

actual_map = {}
if today in all_dates:
today_inst = df[(df[‘Fecha’] == today) & (df[‘ISIN’] != ‘TIBO’)]
actual_map = dict(zip(today_inst[‘ISIN’].values, today_inst[‘Yield’].values))

# — 10d grid curve —

max_tenor = inst_tenors.max()
grid_max = min(CURVE_MAX, int(np.ceil(max_tenor / CURVE_STEP) * CURVE_STEP))
display_grid = np.arange(CURVE_STEP, grid_max + 1, CURVE_STEP)
grid_yields = svensson(display_grid, params)

# %% Cell 4 - Output Tables

print(’\n’ + ‘=’*80)
print(‘INSTRUMENT-LEVEL OUTPUT’)
print(’=’*80)

inst_output_data = []
for i in range(len(inst_tenors)):
isin = inst_isins[i]
tenor = inst_tenors[i]
sv_yield = inst_yields_svensson[i]
actual = actual_map.get(isin, np.nan)
err = (sv_yield - actual) * 10000 if not np.isnan(actual) else np.nan
inst_output_data.append({
‘ISIN’: isin,
‘Tenor’: int(tenor),
‘Svensson_%’: sv_yield * 100,
‘Actual_%’: actual * 100 if not np.isnan(actual) else np.nan,
‘Error_bps’: err,
})

inst_output = pd.DataFrame(inst_output_data)
print(inst_output.to_string(index=False))

print(’\n’ + ‘=’*80)
print(‘10d GRID CURVE’)
print(’=’*80)

grid_output = pd.DataFrame({
‘Tenor’: display_grid.astype(int),
‘Svensson_%’: grid_yields * 100,
})
grid_output = grid_output.round(4)
print(grid_output.to_string(index=False))

# %% Cell 5 - Visualization

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# plot 1: fitted curve vs fitting points

ax = axes[0, 0]
smooth_tenors = np.linspace(1, display_grid[-1], 500)
smooth_yields = svensson(smooth_tenors, params) * 100
ax.plot(smooth_tenors, smooth_yields, ‘-’, color=’#D4553A’, linewidth=2, label=‘Svensson fit’)
ax.plot(fit_tenors, fit_yields * 100, ‘o’, color=’#2C5F8A’, markersize=5, alpha=0.7, label=‘Fitting set’)
if len(NEW_ISSUES) > 0:
for t, y in NEW_ISSUES:
ax.plot(t, y * 100, ‘*’, color=’#4A9B6E’, markersize=15, zorder=5, label=f’New Issue ({t}d)’)
ax.plot(1, TIBO * 100, ‘D’, color=’#E8A838’, markersize=10, zorder=5, label=‘TIBO’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘Svensson Curve Fit’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 2: fitting errors

ax = axes[0, 1]
bar_widths = np.maximum(np.diff(np.concatenate([[0], fit_tenors])), 2)
ax.bar(fit_tenors, fit_errors, width=bar_widths, color=’#D4553A’, alpha=0.7)
ax.axhline(y=0, color=‘black’, linewidth=0.5)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Error (bps)’)
ax.set_title(f’Fitting Errors (MAE={np.abs(fit_errors).mean():.2f} bps)’)
ax.grid(True, alpha=0.3)

# plot 3: 10d grid curve

ax = axes[1, 0]
ax.plot(display_grid, grid_yields * 100, ‘D-’, color=’#D4553A’, markersize=4, linewidth=1.5, label=‘Svensson’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘10d Grid Curve’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 4: instrument-level svensson vs actual

ax = axes[1, 1]
sort_idx = np.argsort(inst_tenors)
ax.plot(inst_tenors[sort_idx], inst_yields_svensson[sort_idx] * 100, ‘D-’, color=’#D4553A’, markersize=5, linewidth=1, label=‘Svensson’)
actuals_available = [(row[‘Tenor’], row[‘Actual_%’]) for *, row in inst_output.iterrows() if not np.isnan(row[’Actual*%’])]
if len(actuals_available) > 0:
act_t, act_y = zip(*actuals_available)
ax.plot(act_t, act_y, ‘o’, color=’#2C5F8A’, markersize=5, alpha=0.7, label=‘Actual’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘Instrument-Level: Svensson vs Actual’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

title_str = f’CDBCRP Svensson — {today.date()} | TIBO={TIBO*100:.2f}%’
if len(NEW_ISSUES) > 0:
title_str += f’ | {len(NEW_ISSUES)} New Issue(s)’
else:
title_str += ’ | No New Issues’
plt.suptitle(title_str, fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(‘cdbcrp_svensson_output.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# print parameters for warm start next day

print(’\n— Parameters for warm start —’)
print(f’initial_params = [{b0:.8f}, {b1:.8f}, {b2:.8f}, {b3:.8f}, {t1:.4f}, {t2:.4f}]’)