import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Monotone Convex (Hagan-West 2006) core functions
# ---------------------------------------------------------------------------

def mc_node_forwards(t, f):
    # t: array of knot times (years), f: array of discrete forwards at knots
    # Returns continuous forward estimates at each knot using the H-W scheme
    n = len(t)
    fd = np.empty(n)
    # interior nodes
    for i in range(1, n - 1):
        w1 = (t[i] - t[i-1]) / (t[i+1] - t[i-1])
        w2 = (t[i+1] - t[i]) / (t[i+1] - t[i-1])
        fd[i] = f[i-1] * w2 + f[i] * w1
    # boundary nodes (extrapolate to preserve monotonicity)
    fd[0] = f[0] - 0.5 * (fd[1] - f[0])
    fd[-1] = f[-1] - 0.5 * (fd[-2] - f[-1])
    return fd


def mc_correct_fwd(f0, f1, fd0, fd1, t0, t1):
    # Ensure the forward is monotone on segment [t0, t1].
    # Returns corrected (fd0, fd1) if needed.
    g0, g1 = fd0 - f0, fd1 - f0
    # check if corrections needed
    if g0 * g1 >= 0 and abs(g0) <= 3 * abs(f1 - f0) and abs(g1) <= 3 * abs(f1 - f0):
        return fd0, fd1
    # apply corrections per H-W section 3
    if g0 * (g0 + g1) < 0:
        fd0 = f0 - g1
        g0 = -g1
    if g1 * (g0 + g1) < 0:
        fd1 = f0 + 2 * (f1 - f0) - g0 * (t1 - t0)
    return fd0, fd1


def mc_yield_at(x, t, y, fd):
    # x: scalar query time in years
    # t: knot times, y: zero/par yields at knots, fd: node forwards
    # Returns interpolated yield at x using monotone convex scheme
    n = len(t)
    if x <= t[0]:
        return y[0]
    if x >= t[-1]:
        return y[-1]
    # find segment
    i = np.searchsorted(t, x, side='right') - 1
    i = min(i, n - 2)
    t0, t1 = t[i], t[i + 1]
    y0, y1 = y[i], y[i + 1]
    # discrete forward over segment
    f_seg = (y1 * t1 - y0 * t0) / (t1 - t0)
    fd0 = fd[i]
    fd1 = fd[i + 1]
    fd0, fd1 = mc_correct_fwd(f_seg, f_seg, fd0, fd1, t0, t1)
    # local coordinate in [0,1]
    s = (x - t0) / (t1 - t0)
    g0, g1 = fd0 - f_seg, fd1 - f_seg
    # H-W forward within segment
    if abs(g0) < 1e-12 and abs(g1) < 1e-12:
        fwd_x = f_seg
    elif abs(2 * g0 + g1) < 1e-12 or abs(g0 + 2 * g1) < 1e-12:
        fwd_x = f_seg + g0 * (1 - 4 * s + 3 * s**2) + g1 * (-2 * s + 3 * s**2)
    else:
        eta = (g0 + g1) / (g0 + g1 - (f_seg - y0) * 0)  # fallback
        fwd_x = f_seg + g0 * (1 - 4 * s + 3 * s**2) + g1 * (-2 * s + 3 * s**2)
    # area under forward = x * yield(x) - t0 * y0
    area_0_to_t0 = y0 * t0
    h = t1 - t0
    area_seg = f_seg * h * s + g0 * h * s**2 * (1 - (4/3) * s + s**2) + g1 * h * s**2 * (-1 + s)
    total_area = area_0_to_t0 + area_seg
    return total_area / x


def mc_interpolate(query_times, knot_times, knot_yields):
    # query_times: array of times to interpolate (years)
    # knot_times, knot_yields: sorted arrays of node times and yields
    t = np.array(knot_times, dtype=float)
    y = np.array(knot_yields, dtype=float)
    n = len(t)
    if n == 1:
        return np.full(len(query_times), y[0])
    # compute discrete forwards
    f = np.empty(n - 1)
    for i in range(n - 1):
        f[i] = (y[i + 1] * t[i + 1] - y[i] * t[i]) / (t[i + 1] - t[i])
    fd = mc_node_forwards(t, np.concatenate([[f[0]], f, [f[-1]]]))[:n]
    # fd has n elements aligned to knot nodes
    # recompute properly: fd needs to be length n
    # node forwards per H-W eq 4.1
    fd2 = np.empty(n)
    # interior
    for i in range(1, n - 1):
        dt_prev = t[i] - t[i - 1]
        dt_next = t[i + 1] - t[i]
        fd2[i] = (f[i - 1] * dt_next + f[i] * dt_prev) / (dt_prev + dt_next)
    fd2[0] = f[0] - 0.5 * (fd2[1] - f[0]) if n > 2 else f[0]
    fd2[-1] = f[-1] - 0.5 * (fd2[-2] - f[-1]) if n > 2 else f[-1]
    results = np.array([mc_yield_at(x, t, y, fd2) for x in query_times])
    return results


# ---------------------------------------------------------------------------
# OTR flagging
# ---------------------------------------------------------------------------

def flag_otr(bonds_df):
    # bonds_df: DataFrame with columns years_to_maturity, issue_date, ticker
    # Returns subset of bonds that are OTR per integer-year bucket
    bonds_df = bonds_df.copy()
    bonds_df['bucket'] = bonds_df['years_to_maturity'].round().astype(int)
    # per bucket keep most recently issued
    idx = bonds_df.groupby('bucket')['issue_date'].idxmax()
    return bonds_df.loc[idx].sort_values('years_to_maturity')


# ---------------------------------------------------------------------------
# Per-country CMT builder
# ---------------------------------------------------------------------------

CMT_TENORS = list(range(5, 16))  # 5..15


def build_country_cmt(dates, yield_panel, meta_bonds):
    # dates: list of obs dates
    # yield_panel: DataFrame indexed by date, columns = bond tickers matching meta_bonds index
    # meta_bonds: DataFrame with index=ticker, columns: issue_date, maturity_date, daycount
    # Returns DataFrame with Fecha + tenor columns in percent
    records = []
    tenor_cols = [f'{t}Y' for t in CMT_TENORS]
    for obs in dates:
        active = meta_bonds[
            (meta_bonds['issue_date'] <= obs) & (meta_bonds['maturity_date'] > obs)
        ].copy()
        if active.empty:
            continue
        row = yield_panel.loc[obs, active.index]
        row = row.dropna()
        if row.empty:
            continue
        active = active.loc[row.index].copy()
        active['yield'] = row.values
        active['years_to_maturity'] = (active['maturity_date'] - obs).dt.days / 365.0
        otr = flag_otr(active[['years_to_maturity', 'issue_date', 'yield']].copy())
        if otr.empty or len(otr) < 2:
            continue
        kt = otr['years_to_maturity'].values
        ky = otr['yield'].values
        min_t, max_t = kt[0], kt[-1]
        # interpolate only within range
        result = {}
        for tenor in CMT_TENORS:
            if tenor < min_t or tenor > max_t:
                result[f'{tenor}Y'] = np.nan
            else:
                vals = mc_interpolate([float(tenor)], kt, ky)
                result[f'{tenor}Y'] = vals[0]
        # short-end edge case: fill leading NaNs with nearest interpolated value
        first_valid = None
        for tc in tenor_cols:
            if not np.isnan(result[tc]):
                first_valid = result[tc]
                break
        if first_valid is not None:
            for tc in tenor_cols:
                if np.isnan(result[tc]):
                    result[tc] = first_valid
                else:
                    break
        result['Fecha'] = obs
        records.append(result)
    if not records:
        return pd.DataFrame(columns=['Fecha'] + tenor_cols)
    df = pd.DataFrame(records)[['Fecha'] + tenor_cols]
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(filepath='df_rv.xlsx'):
    xl = pd.ExcelFile(filepath)
    raw = xl.parse('LC', header=None)

    # metadata rows 0-4 (0-indexed), bonds in columns 1+
    bond_cols = raw.columns[1:]
    meta = pd.DataFrame({
        'type':          raw.loc[0, bond_cols].values,
        'issue_date':    pd.to_datetime(raw.loc[1, bond_cols].values, dayfirst=True, errors='coerce'),
        'maturity_date': pd.to_datetime(raw.loc[2, bond_cols].values, dayfirst=True, errors='coerce'),
        'daycount':      raw.loc[3, bond_cols].values.astype(str).str.strip(),
        'ticker':        raw.loc[4, bond_cols].values,
    })
    meta.index = meta['ticker']

    # yield panel: rows 5+
    data_raw = raw.iloc[5:].copy()
    data_raw.columns = ['date'] + list(meta['ticker'])
    data_raw['date'] = pd.to_datetime(data_raw['date'], dayfirst=True, errors='coerce')
    data_raw = data_raw.dropna(subset=['date'])
    data_raw = data_raw.set_index('date')
    data_raw = data_raw.apply(pd.to_numeric, errors='coerce')

    # ffill with limit=5 per bond
    data_raw = data_raw.ffill(limit=5)

    country_types = meta['type'].unique()
    cmt_dfs = {}

    for ctype in country_types:
        bonds = meta[meta['type'] == ctype].copy()
        panel = data_raw[bonds['ticker']].copy()
        # drop dates where all bonds are NaN
        panel = panel.dropna(how='all')
        dates = panel.index.tolist()
        df_cmt = build_country_cmt(dates, panel, bonds)
        key = f'df_{ctype.lower()}_cmt'
        cmt_dfs[key] = df_cmt

    # conversion to 360d annual effective
    dc_map = {r['type']: r['daycount'] for _, r in meta.drop_duplicates('type').iterrows()}
    country_code_map = {'PERUGB': 'PE', 'COLTES': 'CO', 'MBONO': 'MX', 'BNTNF': 'BR', 'BTPCL': 'CL'}
    tenor_cols = [f'{t}Y' for t in CMT_TENORS]

    def convert_360(df, daycount):
        df = df.copy()
        dc = str(daycount).strip()
        if dc == '365':
            for c in tenor_cols:
                df[c] = ((1 + df[c] / 100) ** (365 / 360) - 1) * 100
        elif dc == '252':
            for c in tenor_cols:
                df[c] = ((1 + df[c] / 100) ** (252 / 360) - 1) * 100
        # dc == '360': no conversion
        return df

    harmonized_parts = []
    for ctype in country_types:
        key = f'df_{ctype.lower()}_cmt'
        df_native = cmt_dfs[key]
        dc = dc_map.get(ctype, '360')
        df_360 = convert_360(df_native, dc)
        df_360['country'] = country_code_map.get(ctype, ctype)
        harmonized_parts.append(df_360)

    df_cmt_harmonized = pd.concat(harmonized_parts, ignore_index=True)
    df_cmt_harmonized = df_cmt_harmonized[['Fecha', 'country'] + tenor_cols]

    return cmt_dfs, df_cmt_harmonized


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cmt_dfs, df_cmt_harmonized = run_pipeline('df_rv.xlsx')
    df_perugb_cmt = cmt_dfs['df_perugb_cmt']
    df_coltes_cmt = cmt_dfs['df_coltes_cmt']
    df_mbono_cmt  = cmt_dfs['df_mbono_cmt']
    df_bntnf_cmt  = cmt_dfs['df_bntnf_cmt']
    df_btpcl_cmt  = cmt_dfs['df_btpcl_cmt']
    print('Done. Shapes:')
    for k, v in cmt_dfs.items():
        print(f'  {k}: {v.shape}')
    print(f'  df_cmt_harmonized: {df_cmt_harmonized.shape}')
