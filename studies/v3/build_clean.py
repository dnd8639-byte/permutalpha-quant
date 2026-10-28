"""Turn per-contract daily prices into ROLL-GAP-FREE daily and monthly futures returns.

Rule (no look-ahead):
  * On day t we hold the contract that had the most volume on day t-1,
    excluding any contract within 5 trading days of its final bar (funds roll before expiry;
    expiry dates are known in advance from contract specs, so this is not look-ahead).
  * Return on day t = close_t / close_{t-1} of THAT SAME contract - 1.
    A roll never compares two different contracts, so the splice jump never enters a return.
Usage: python build_clean.py   (reads ../state/v3/contracts/*.csv, writes ../state/v3/clean_*.csv)
"""
import glob, os
import numpy as np
import pandas as pd

SRC = "../state/v3/contracts"
DST = "../state/v3"
EXPIRY_BUFFER = 5


def load_contracts(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["ts_event"].astype(str).str[:10])
    # CME symbols reuse a one-digit year (ESZ0 = Dec 2010 AND Dec 2020). instrument_id is unique
    # per contract, so the contract key is symbol + instrument_id.
    if "instrument_id" in df.columns:
        df["symbol"] = df["symbol"].astype(str) + "_" + df["instrument_id"].astype(str)
    df = df[["date", "symbol", "close", "volume"]].dropna(subset=["close"])
    return df.groupby(["date", "symbol"], as_index=False).last()


def clean_returns(df):
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    vol = df.pivot(index="date", columns="symbol", values="volume").reindex(close.index).fillna(0.0)
    # eligibility: not in the last EXPIRY_BUFFER trading days of the contract's life
    pos = pd.Series(np.arange(len(close)), index=close.index)
    last_pos = close.apply(lambda c: pos[c.last_valid_index()])
    elig = pd.DataFrame(pos.values[:, None] <= (last_pos.values[None, :] - EXPIRY_BUFFER),
                        index=close.index, columns=close.columns)
    v = vol.where(elig & close.notna(), -1.0)
    held = v.idxmax(axis=1).where(v.max(axis=1) > 0).shift(1)        # chosen with t-1 volume
    c_prev = close.shift(1)
    r = pd.Series(np.nan, index=close.index)
    rolls = 0
    for i in range(1, len(close)):
        h = held.iloc[i]
        if isinstance(h, str):
            a, b = c_prev.iloc[i][h], close.iloc[i][h]
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                r.iloc[i] = b / a - 1
        if i > 1 and held.iloc[i] != held.iloc[i - 1]:
            rolls += 1
    return r, held, rolls


MONTHS = {c: i + 1 for i, c in enumerate("FGHJKMNQUVXZ")}


def nominal_expiry(key, first_date):
    """Delivery month from the contract code (e.g. ESZ6 -> Dec 2026), known in advance.
    One-digit years are resolved to the first matching year on/after the contract's first bar."""
    import re
    sym = key.split("_")[0]
    m = re.match(r"^[A-Z0-9]+?([FGHJKMNQUVXZ])(\d{1,2})$", sym)
    if not m:
        return pd.NaT
    mon, yy = MONTHS[m.group(1)], m.group(2)
    if len(yy) == 2:
        year = 2000 + int(yy)
    else:
        y0 = first_date.year
        year = next(y for y in range(y0, y0 + 12) if y % 10 == int(yy))
    return pd.Timestamp(year=year, month=mon, day=15)


def carry_and_rolls(df, held):
    """Carry on the last trading day of each month from the two nearest eligible contracts
    (ordered by delivery month from the contract code), annualized by the gap between
    their delivery months; and the number of rolls in each month."""
    close = df.pivot(index="date", columns="symbol", values="close").sort_index()
    pos = pd.Series(np.arange(len(close)), index=close.index)
    last_pos = close.apply(lambda c: pos[c.last_valid_index()])
    first = close.apply(lambda c: c.first_valid_index())
    expiry = pd.Series({k: nominal_expiry(k, first[k]) for k in close.columns})
    month_end_days = close.index.to_series().groupby(close.index.to_period("M")).max()
    carry = {}
    for per, d in month_end_days.items():
        i = pos[d]
        row = close.loc[d]
        ok = row.notna() & (last_pos - i >= EXPIRY_BUFFER) & expiry.notna() & (expiry > d)
        c = expiry[ok].sort_values()
        if len(c) >= 2:
            s1, s2 = c.index[0], c.index[1]
            days = (c.iloc[1] - c.iloc[0]).days
            if days > 0 and row[s2] > 0:
                carry[per.to_timestamp("M")] = (row[s1] - row[s2]) / row[s2] * 365 / days
    changed = (held != held.shift(1)) & held.notna() & held.shift(1).notna()
    rolls = changed.astype(int).resample("ME").sum()
    return pd.Series(carry, dtype=float), rolls


def to_monthly(r):
    return (1 + r.fillna(0)).resample("ME").prod() - 1


if __name__ == "__main__":
    daily, monthly, report, carry, rolls_m = {}, {}, [], {}, {}
    for path in sorted(glob.glob(f"{SRC}/*.csv")):
        root = os.path.basename(path)[:-4]
        df = load_contracts(path)
        r, held, rolls = clean_returns(df)
        first = r.first_valid_index()
        r = r.loc[first:]
        daily[root] = r
        cy, rl = carry_and_rolls(df, held)
        carry[root], rolls_m[root] = cy, rl
        m = to_monthly(r)
        monthly[root] = m.loc[m.index >= (first + pd.offsets.MonthEnd(0))]
        miss = r.isna().mean()
        report.append(f"{root:4s} {first.date()} -> {r.index[-1].date()}  contracts {df['symbol'].nunique():3d}  "
                      f"rolls {rolls:3d}  missing-day share {miss:.2%}  worst day {r.min():+.2%}  best {r.max():+.2%}")
    pd.DataFrame(daily).to_csv(f"{DST}/clean_daily.csv")
    pd.DataFrame(monthly).to_csv(f"{DST}/clean_monthly.csv")
    pd.DataFrame(carry).to_csv(f"{DST}/clean_carry.csv")
    pd.DataFrame(rolls_m).to_csv(f"{DST}/clean_rolls.csv")
    open(f"{DST}/build_report.txt", "w").write("\n".join(report) + "\n")
    print("\n".join(report))
