import numpy as np
import pandas as pd
import requests
import time
import os
import pickle
from datetime import datetime, date


# ── Retry helper ─────────────────────────────────────────────────────────────

def _fetch_with_retry(url, headers=None, max_retries=3, timeout=30):
    # Fetch URL with exponential backoff
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as e:
            if attempt == max_retries - 1:
                print(f"ERROR: Failed after {max_retries} attempts: {url}")
                print(f"  Last error: {e}")
                raise
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt + 1}/{max_retries} in {wait}s: {e}")
            time.sleep(wait)


# ── A1: USDPEN from BCRP API ────────────────────────────────────────────────

SPANISH_MONTHS = {
    "Ene": "Jan", "Feb": "Feb", "Mar": "Mar", "Abr": "Apr",
    "May": "May", "Jun": "Jun", "Jul": "Jul", "Ago": "Aug",
    "Sep": "Sep", "Oct": "Oct", "Nov": "Nov", "Dic": "Dec"
}


def _parse_bcrp_date(date_str):
    # Parse BCRP date with Spanish month abbreviations
    for es, en in SPANISH_MONTHS.items():
        date_str = date_str.replace(es, en)
    try:
        return pd.to_datetime(date_str, format="%d.%b.%y")
    except ValueError:
        return pd.to_datetime(date_str, dayfirst=True)


def fetch_usdpen(start_date="2018-1-1", end_date=None):
    # Download USDPEN bid/ask/mid from BCRP API
    if end_date is None:
        today = date.today()
        end_date = f"{today.year}-{today.month}-{today.day}"

    url = (
        f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
        f"PD04645PD-PD04646PD/json/{start_date}/{end_date}/ing"
    )
    print(f"Fetching USDPEN from BCRP API...")
    resp = _fetch_with_retry(url)
    data = resp.json()

    rows = []
    for period in data.get("periods", []):
        dt = _parse_bcrp_date(period["name"])
        values = period["values"]
        bid_str = values[0] if len(values) > 0 else "n.d."
        ask_str = values[1] if len(values) > 1 else "n.d."

        bid = np.nan if bid_str == "n.d." else float(bid_str)
        ask = np.nan if ask_str == "n.d." else float(ask_str)
        mid = (bid + ask) / 2.0 if not (np.isnan(bid) or np.isnan(ask)) else np.nan

        rows.append({"date": dt.date(), "usdpen_bid": bid, "usdpen_ask": ask, "usdpen_mid": mid})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  USDPEN: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ── A2: USDCOP from Colombia Datos Abiertos ─────────────────────────────────

def fetch_usdcop(start_date="2018-01-01"):
    # Download USDCOP TRM from Datos Abiertos Colombia
    url = (
        f"https://www.datos.gov.co/resource/32sa-8pi3.json"
        f"?$order=vigenciadesde DESC"
        f"&$limit=50000"
        f"&$where=vigenciadesde>='{start_date}T00:00:00.000'"
    )
    print(f"Fetching USDCOP from datos.gov.co...")
    resp = _fetch_with_retry(url)
    data = resp.json()

    rows = []
    for rec in data:
        val = float(rec["valor"])
        dt = pd.to_datetime(rec["vigenciadesde"]).date()
        rows.append({"date": dt, "usdcop": val})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date", keep="last")
    print(f"  USDCOP: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ── A3: USDCLP from mindicador.cl ───────────────────────────────────────────

def fetch_usdclp(start_year=2018, end_year=None):
    # Download USDCLP from mindicador.cl year by year
    if end_year is None:
        end_year = date.today().year

    print(f"Fetching USDCLP from mindicador.cl...")
    all_rows = []
    for year in range(start_year, end_year + 1):
        url = f"https://mindicador.cl/api/dolar/{year}"
        try:
            resp = _fetch_with_retry(url)
            data = resp.json()
            for item in data.get("serie", []):
                dt = pd.to_datetime(item["fecha"]).date()
                val = float(item["valor"])
                all_rows.append({"date": dt, "usdclp": val})
            print(f"    {year}: {len(data.get('serie', []))} rows")
        except Exception as e:
            print(f"    {year}: FAILED - {e}")

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date", keep="last")
    print(f"  USDCLP: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ── A4: USDBRL PTAX from BCB ────────────────────────────────────────────────

def fetch_usdbrl(start_date="01-01-2018", end_date=None):
    # Download USDBRL PTAX from BCB API, paginating by year if needed
    if end_date is None:
        today = date.today()
        end_date = f"{today.month:02d}-{today.day:02d}-{today.year}"

    print(f"Fetching USDBRL PTAX from BCB...")

    # Split into yearly chunks to avoid API limits
    start_dt = datetime.strptime(start_date, "%m-%d-%Y")
    end_dt = datetime.strptime(end_date, "%m-%d-%Y")

    all_rows = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(
            datetime(current_start.year, 12, 31),
            end_dt
        )
        s = f"{current_start.month:02d}-{current_start.day:02d}-{current_start.year}"
        e = f"{current_end.month:02d}-{current_end.day:02d}-{current_end.year}"

        url = (
            f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            f"CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinal=@dataFinal)"
            f"?@dataInicial='{s}'&@dataFinal='{e}'"
            f"&$top=10000&$format=json"
            f"&$select=cotacaoCompra,cotacaoVenda,dataHoraCotacao"
        )
        try:
            resp = _fetch_with_retry(url)
            data = resp.json()
            for item in data.get("value", []):
                dt_str = item["dataHoraCotacao"]
                dt = pd.to_datetime(dt_str)
                bid = float(item["cotacaoCompra"])
                ask = float(item["cotacaoVenda"])
                mid = (bid + ask) / 2.0
                all_rows.append({"datetime": dt, "date": dt.date(), "usdbrl": mid})
            print(f"    {current_start.year}: {len(data.get('value', []))} quotes")
        except Exception as e:
            print(f"    {current_start.year}: FAILED - {e}")

        current_start = datetime(current_start.year + 1, 1, 1)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("  WARNING: No USDBRL data retrieved")
        return pd.DataFrame(columns=["date", "usdbrl"])

    # Keep last quote per day (closing PTAX)
    df = df.sort_values("datetime")
    df = df.groupby("date").last().reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "usdbrl"]].sort_values("date").reset_index(drop=True)
    print(f"  USDBRL: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ── A5: USDMXN from Banxico or yfinance fallback ────────────────────────────

def fetch_usdmxn(banxico_token=None, start_date="2018-01-01", end_date=None):
    # Download USDMXN FIX from Banxico API or yfinance fallback
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    if banxico_token:
        return _fetch_usdmxn_banxico(banxico_token, start_date, end_date)
    else:
        print("  WARNING: No Banxico token provided. Using yfinance fallback.")
        return _fetch_usdmxn_yfinance(start_date, end_date)


def _fetch_usdmxn_banxico(token, start_date, end_date):
    # Fetch USDMXN FIX from Banxico SIE API
    print(f"Fetching USDMXN from Banxico API...")
    url = (
        f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
        f"SF43718/datos/{start_date}/{end_date}?mediaType=json"
    )
    headers = {"Bmx-Token": token}
    resp = _fetch_with_retry(url, headers=headers)
    data = resp.json()

    rows = []
    series_data = data.get("bmx", {}).get("series", [{}])[0].get("datos", [])
    for item in series_data:
        fecha = item["fecha"]  # dd/mm/yyyy
        dato = item["dato"]
        dt = pd.to_datetime(fecha, format="%d/%m/%Y").date()
        val = np.nan if dato == "N/E" else float(dato.replace(",", ""))
        rows.append({"date": dt, "usdmxn": val})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  USDMXN: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def _fetch_usdmxn_yfinance(start_date, end_date):
    # Fallback: fetch USDMXN from yfinance
    try:
        import yfinance as yf
        print(f"Fetching USDMXN from yfinance...")
        mxn = yf.download("MXN=X", start=start_date, end=end_date, progress=False)
        df = pd.DataFrame({
            "date": mxn.index,
            "usdmxn": mxn["Close"].values.flatten()
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        print(f"  USDMXN (yfinance): {len(df)} rows")
        return df
    except ImportError:
        print("  ERROR: yfinance not installed. Cannot fetch USDMXN.")
        return pd.DataFrame(columns=["date", "usdmxn"])


# ── A6: Merge and align ─────────────────────────────────────────────────────

def merge_fx_data(df_pen, df_cop, df_clp, df_brl, df_mxn, output_path="output"):
    # Merge all FX DataFrames, compute log returns, save outputs
    print("\nMerging FX data...")

    # Normalize date columns
    for df in [df_pen, df_cop, df_clp, df_brl, df_mxn]:
        df["date"] = pd.to_datetime(df["date"])

    # Inner join on date
    merged = df_pen[["date", "usdpen_mid"]].merge(
        df_cop[["date", "usdcop"]], on="date", how="inner"
    ).merge(
        df_clp[["date", "usdclp"]], on="date", how="inner"
    ).merge(
        df_brl[["date", "usdbrl"]], on="date", how="inner"
    ).merge(
        df_mxn[["date", "usdmxn"]], on="date", how="inner"
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward-fill up to 2 days for minor holiday mismatches
    merged = merged.set_index("date").asfreq("B")
    merged = merged.ffill(limit=2)
    merged = merged.dropna().reset_index()

    print(f"  Merged: {len(merged)} rows, {merged['date'].min().date()} to {merged['date'].max().date()}")

    # Compute log returns
    level_cols = ["usdpen_mid", "usdcop", "usdclp", "usdbrl", "usdmxn"]
    return_names = ["r_pen", "r_cop", "r_clp", "r_brl", "r_mxn"]

    returns_df = merged[["date"]].copy()
    for col, rname in zip(level_cols, return_names):
        returns_df[rname] = np.log(merged[col] / merged[col].shift(1))

    returns_df = returns_df.dropna().reset_index(drop=True)

    # Save outputs
    os.makedirs(output_path, exist_ok=True)
    merged.to_csv(os.path.join(output_path, "fx_latam_levels.csv"), index=False)
    returns_df.to_csv(os.path.join(output_path, "fx_latam_logreturns.csv"), index=False)

    with open(os.path.join(output_path, "fx_latam_merged.pkl"), "wb") as f:
        pickle.dump({"levels": merged, "returns": returns_df}, f)

    # Summary
    print(f"\n  Summary:")
    print(f"    Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"    Observations: {len(returns_df)}")

    # Check for gaps > 3 business days
    date_diff = merged["date"].diff().dt.days
    gaps = date_diff[date_diff > 5]  # > 5 calendar days ~ > 3 business days
    if len(gaps) > 0:
        print(f"    Gaps > 3 business days:")
        for idx in gaps.index:
            print(f"      {merged['date'].iloc[idx-1].date()} -> {merged['date'].iloc[idx].date()} ({int(date_diff.iloc[idx])} cal days)")
    else:
        print(f"    No gaps > 3 business days")

    return merged, returns_df
