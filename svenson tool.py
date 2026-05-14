# %% Cell 1 - Imports & Config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import timedelta
import warnings
warnings.filterwarnings(‘ignore’)

FILE_PATH = ‘Data_historica.xlsx’

# — Manual Inputs —

TODAY_DATE = ‘2025-04-28’  # date t
TIBO = 0.0425  # today’s TIBO as decimal

# new issues: list of (tenor_days, yield_decimal)

# set to [] for no new issues

NEW_ISSUES = [(93, 0.0480)]

# curve display tenors: 1d (TIBO) then 30d steps to 720d

CURVE_TENORS = [1] + list(range(30, 721, 30))

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

def yield_to_price(y, tenor_days):
y = np.asarray(y, dtype=float)
tenor_days = np.asarray(tenor_days, dtype=float)
return 100.0 / (1 + y) ** (tenor_days / 360.0)

def zc_duration(tenor_days):
return np.asarray(tenor_days, dtype=float) / 360.0

def sbs_objective(params, tenors, valid_prices, durations):
b0, b1, b2, b3, t1, t2 = params
if t1 <= 0 or t2 <= 0:
return 1e10
if b0 + b1 < 0:
return 1e10
sv_yields = svensson(tenors, params)
if np.any(sv_yields <= -1):
return 1e10
estimated_prices = yield_to_price(sv_yields, tenors)
errors = (valid_prices - estimated_prices) / (valid_prices * durations)
return np.sum(errors ** 2)

def fit_svensson_sbs(tenors, yields, initial_params=None, max_tenor=600.0):
tenors = np.asarray(tenors, dtype=float)
yields = np.asarray(yields, dtype=float)
valid_prices = yield_to_price(yields, tenors)
durations = zc_duration(tenors)

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

def make_new_issue_isin(today, tenor_days):
# format: CD + ddMMMYY where date is maturity date
maturity = today + timedelta(days=int(tenor_days))
months_es = {
1: ‘ENE’, 2: ‘FEB’, 3: ‘MAR’, 4: ‘ABR’, 5: ‘MAY’, 6: ‘JUN’,
7: ‘JUL’, 8: ‘AGO’, 9: ‘SET’, 10: ‘OCT’, 11: ‘NOV’, 12: ‘DIC’
}
dd = f’{maturity.day:02d}’
mmm = months_es[maturity.month]
yy = f’{maturity.year % 100:02d}’
return f’CD{dd}{mmm}{yy}’

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

# — Build fitting set —

fit_tenors = [1.0]
fit_yields = [TIBO]
fit_labels = [‘TIBO’]

# shifted t-1 instruments

for _, row in prev_inst.iterrows():
new_tenor = row[‘Dias’] - day_gap
if new_tenor > 1:
fit_tenors.append(float(new_tenor))
fit_yields.append(float(row[‘Yield’]))
fit_labels.append(row[‘ISIN’])

# new issues

has_new_issues = len(NEW_ISSUES) > 0
new_issue_isins = []
if has_new_issues:
for tenor, yld in NEW_ISSUES:
isin = make_new_issue_isin(today, tenor)
fit_tenors.append(float(tenor))
fit_yields.append(float(yld))
fit_labels.append(isin)
new_issue_isins.append(isin)

fit_tenors = np.array(fit_tenors)
fit_yields = np.array(fit_yields)

sort_idx = np.argsort(fit_tenors)
fit_tenors = fit_tenors[sort_idx]
fit_yields = fit_yields[sort_idx]
fit_labels = [fit_labels[i] for i in sort_idx]

print(f’Fitting set: {len(fit_tenors)} points’)
print(f’\nFitting set:’)
for i in range(len(fit_tenors)):
print(f’  {fit_labels[i]:>15s}  {fit_tenors[i]:>6.0f}d  {fit_yields[i]*100:.4f}%’)

# — Fit Svensson (SBS price-space) —

max_tenor_fit = fit_tenors.max()
print(f’\nFitting Svensson (SBS price-space, tau bounded to [{1.0:.0f}, {max_tenor_fit:.0f}]d)…’)
params, cost = fit_svensson_sbs(fit_tenors, fit_yields, max_tenor=max_tenor_fit)
b0, b1, b2, b3, t1, t2 = params
print(f’  b0={b0*100:.4f}%  b1={b1*100:.4f}%  b2={b2*100:.4f}%  b3={b3*100:.4f}%’)
print(f’  t1={t1:.1f}d  t2={t2:.1f}d’)
print(f’  Objective cost: {cost:.2e}’)
print(f’  r(0) = b0+b1 = {(b0+b1)*100:.4f}%’)

# fitting errors

fitted_yields = svensson(fit_tenors, params)
fit_errors = (fit_yields - fitted_yields) * 10000
print(f’\nFitting errors (bps):’)
print(f’  MAE:  {np.abs(fit_errors).mean():.2f}’)
print(f’  RMSE: {np.sqrt((fit_errors**2).mean()):.2f}’)
print(f’  Max:  {np.abs(fit_errors).max():.2f}’)

# — Instrument-level output —

# all shifted t-1 instruments + new issues

inst_data = []

# shifted t-1 instruments

for *, row in prev_inst.iterrows():
new_tenor = row[‘Dias’] - day_gap
if new_tenor > 1:
sv_yield = float(svensson(np.array([new_tenor]), params)[0])
inst_data.append({
‘ISIN’: row[‘ISIN’],
‘Tenor’: int(new_tenor),
’Svensson*%’: sv_yield * 100,
‘Type’: ‘Existing’,
})

# new issues with generated ISINs

if has_new_issues:
for i, (tenor, yld) in enumerate(NEW_ISSUES):
isin = new_issue_isins[i]
sv_yield = float(svensson(np.array([tenor]), params)[0])
inst_data.append({
‘ISIN’: isin,
‘Tenor’: int(tenor),
‘Svensson_%’: sv_yield * 100,
‘Type’: ‘New Issue’,
})

inst_output = pd.DataFrame(inst_data).sort_values(‘Tenor’)

# get actual yields for today if available

actual_map = {}
if today in all_dates:
today_inst = df[(df[‘Fecha’] == today) & (df[‘ISIN’] != ‘TIBO’)]
actual_map = dict(zip(today_inst[‘ISIN’].values, today_inst[‘Yield’].values))

if len(actual_map) > 0:
inst_output[‘Actual_%’] = inst_output[‘ISIN’].map(lambda x: actual_map.get(x, np.nan) * 100 if x in actual_map else np.nan)
inst_output[‘Error_bps’] = (inst_output[‘Svensson_%’] - inst_output[‘Actual_%’]) * 100
else:
inst_output[‘Actual_%’] = np.nan
inst_output[‘Error_bps’] = np.nan

# — Curve output at custom tenors —

curve_tenors = np.array(CURVE_TENORS, dtype=float)
curve_yields = svensson(curve_tenors, params)

# %% Cell 4 - Output Tables

print(’\n’ + ‘=’*80)
print(‘INSTRUMENT-LEVEL OUTPUT’)
print(’=’*80)
print(inst_output.to_string(index=False))

print(’\n’ + ‘=’*80)
print(‘CURVE OUTPUT’)
print(’=’*80)
curve_output = pd.DataFrame({
‘Tenor’: curve_tenors.astype(int),
‘Svensson_%’: curve_yields * 100,
})
curve_output = curve_output.round(4)
print(curve_output.to_string(index=False))

# %% Cell 5 - Visualization

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# plot 1: fitted curve vs fitting points

ax = axes[0, 0]
smooth_tenors = np.linspace(1, max(curve_tenors[-1], max_tenor_fit), 500)
smooth_yields = svensson(smooth_tenors, params) * 100
ax.plot(smooth_tenors, smooth_yields, ‘-’, color=’#D4553A’, linewidth=2, label=‘Svensson fit’)
ax.plot(fit_tenors, fit_yields * 100, ‘o’, color=’#2C5F8A’, markersize=5, alpha=0.7, label=‘Fitting set’)
if has_new_issues:
for i, (t, y) in enumerate(NEW_ISSUES):
ax.plot(t, y * 100, ‘*’, color=’#4A9B6E’, markersize=15, zorder=5, label=f’New: {new_issue_isins[i]}’)
ax.plot(1, TIBO * 100, ‘D’, color=’#E8A838’, markersize=10, zorder=5, label=‘TIBO’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘Svensson Curve Fit (SBS Price-Space)’)
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

# plot 3: curve at custom tenors

ax = axes[1, 0]
ax.plot(curve_tenors, curve_yields * 100, ‘D-’, color=’#D4553A’, markersize=4, linewidth=1.5, label=‘Svensson’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘Curve at Display Tenors’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# plot 4: instrument-level

ax = axes[1, 1]
existing = inst_output[inst_output[‘Type’] == ‘Existing’]
new = inst_output[inst_output[‘Type’] == ‘New Issue’]
ax.plot(existing[‘Tenor’], existing[‘Svensson_%’], ‘D’, color=’#D4553A’, markersize=4, label=‘Existing (Svensson)’)
if len(new) > 0:
ax.plot(new[‘Tenor’], new[‘Svensson_%’], ‘*’, color=’#4A9B6E’, markersize=12, label=‘New Issue (Svensson)’)
if len(actual_map) > 0:
act_rows = inst_output.dropna(subset=[‘Actual_%’])
if len(act_rows) > 0:
ax.plot(act_rows[‘Tenor’], act_rows[‘Actual_%’], ‘o’, color=’#2C5F8A’, markersize=4, alpha=0.7, label=‘Actual’)
ax.set_xlabel(‘Tenor (days)’)
ax.set_ylabel(‘Yield (%)’)
ax.set_title(‘Instrument-Level Output’)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

title_str = f’CDBCRP Svensson (SBS) — {today.date()} | TIBO={TIBO*100:.2f}%’
if has_new_issues:
ni_str = ‘, ‘.join([f’{new_issue_isins[i]} ({NEW_ISSUES[i][0]}d)’ for i in range(len(NEW_ISSUES))])
title_str += f’ | New: {ni_str}’
else:
title_str += ’ | No New Issues’
plt.suptitle(title_str, fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(‘cdbcrp_svensson_sbs_output.png’, dpi=150, bbox_inches=‘tight’)
plt.show()

# warm start for next day

print(’\n— Parameters for warm start —’)
print(f’initial_params = [{b0:.8f}, {b1:.8f}, {b2:.8f}, {b3:.8f}, {t1:.4f}, {t2:.4f}]’)