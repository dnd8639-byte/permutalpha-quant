# RESULTS v3: clean futures data + new strategies

Run 2026-09-23 against `PREREGISTRATION_v3.md` (including Amendments v3-A1 and v3-A2, both recorded **before** any strategy ran on real data).
Full output: `v3/results_v3.txt`. Monthly returns: `v3/returns_v3.csv`. Clean data: `state/v3/clean_*.csv`.

## Verdict (preregistered rule)

**NO EDGE FOUND.** None of T4–T7 passes G1 + G2 + G3.

| strategy | Sharpe (net) | 95% CI | NW t | DSR | 2018+ | gross (no costs) | G1 | G2 | G3 |
|---|---|---|---|---|---|---|---|---|---|
| T0 mom(1,4), clean data | +0.01 | [−0.60, +0.57] | 0.05 | 0.20 | +0.01 | — | F | F | P |
| **T4** 12-month trend | +0.22 | [−0.28, +0.67] | 0.78 | 0.47 | +0.23 | +0.42 | F | F | P |
| **T5** cross-sectional momentum | +0.31 | [−0.19, +0.80] | 1.06 | 0.60 | +0.26 | +0.50 | F | F | P |
| **T6** carry | +0.10 | [−0.28, +0.47] | 0.51 | 0.30 | +0.11 | +0.40 | F | F | P |
| **T7** blend | +0.34 | [−0.28, +0.90] | 1.01 | 0.64 | +0.28 | — | F | F | P |

With N = 11 trials and 172 months, a strategy needed a net Sharpe of about **0.68** to pass G2. **Even before costs, none gets there.** The verdict does not depend on the cost model.

## What the evidence says

- **The strategies are implemented correctly.** T4 correlates **+0.82** with AQR's published TSMOM factor, and T5 +0.71. Carry has a positive cross-sectional information coefficient (IC +0.034, t 1.85). The signals point the right way; they are just weak.
- **All effects are positive but small.** Every strategy is positive in both 2010–2017 and 2018+, and all pass G3. The problem is size, not sign: 16 years is too short to separate a Sharpe of 0.2–0.35 from luck after 11 trials.
- **Costs matter.** Trading and roll costs take 0.2–0.3 off the Sharpe. The preregistered roll cost (two outright legs per roll) is conservative, since real rolls usually go through cheaper calendar spreads. Even at zero cost the verdict stands.
- **The long-history cross-check agrees with v2.** 12-month trend on the 1973+ macro data: Sharpe +0.57 overall, **+0.71 before 2016 and −0.02 after**.

## The data fix (what you paid for)

- The median contamination per market in Yahoo `=F` returns is **1.6%/yr**. The worst are **NG +23.9%/yr**, ZW +10.0%/yr and HE +9.5%/yr, where Yahoo overstates returns because contango roll-downs are hidden in the splice.
- Financial futures were barely affected (correlation > 0.95, contamination < 2%/yr). Commodities and livestock were badly affected (correlation as low as 0.74).
- mom(1,4) on the same markets and months: Yahoo Sharpe +0.08, clean **+0.01**. **The data flaw did not hide an edge.**

## Bugs found and fixed before any result (continuing the v1/v2 record)

4. **HTTP 206 treated as failure.** Databento's "success, partially resolved symbols" code made the downloader discard paid data. This cost one extra download.
5. **CME symbol reuse.** `ESZ0` means both Dec 2010 and Dec 2020. Contracts are now keyed by `instrument_id` (v3-A1).
6. **End-of-sample expiry proxy.** Live contracts' "last bar" is the download date, which produced impossible carry values (ES −149%/yr). Contracts are now spaced by the delivery month in the contract code (v3-A2).

## Limitations

- 16 years of clean data (CME from mid-2010). The ICE softs are excluded.
- BZ (CME Brent) is thin, and its volume-based rolls flip often (759 rolls). This adds cost to every strategy.
- Commodity carry includes seasonal spreads (hogs, natural gas, gasoline), a known weakness of nearest-vs-next carry.
