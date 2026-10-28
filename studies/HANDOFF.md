# Project Handoff: AHL Quant Appliance + mcpt-lab (final, 2026-09-24)

## Context
David is a high-school student building a deterministic quant-trading appliance on a Raspberry Pi 5 as a college-application artifact.

**Standing preferences**
- Tone is professional, encouraging and rigorous. Frame the work as systems engineering and research integrity, not profit.
- **Never** suggest LLMs for trade execution.
- He is a command-line beginner on a **Mac**. Give literal copy-paste commands (Control ⌃ + C stops a program).
- He pushes hard for "a profitable strategy". Hold the line on no p-hacking. **No stopping rule** was set; every test counts.
- Keep credit to neurotrader888 (MIT) for the method, strategies, adapted code and bundled data. He asked about rewriting the code to drop the license. I advised against it: the credit for the ideas stays either way, and the credit strengthens his essay. He has not asked again.
- Recommend he disclose AI coding assistance in applications. Offered a line on the page; not yet accepted.

## Where things are (David's Mac)
```
~/Desktop/ahl-node/
  HANDOFF.md                 this file
  mcpt-lab/                  <- THE PUBLISHABLE REPO (new this session)
  mcpt-lab.zip               same repo zipped (btc_1h_binanceus.csv not in the zip; it's git-ignored)
  mcpt/                      original working lab (LEDGER.csv, run scripts, nulls, LEARN.md)
  research/ v2/ v3/ v4/ v5/  earlier futures research (13 preregistered tests, 0 passes)
  PREREGISTRATION_v2-v5.md, RESULTS_v2-v5.md
  state/                     downloaded data; state/v3 = licensed Databento CME data (NOT redistributable)
  state/.databento_key       SECRET. Never publish.
  prod/                      empty. Pi appliance not started
```
Connected folders also include neurotrader repos in `~/Downloads` (mcpt-main, VolatilityHawkes-main, TrendlineBreakoutMetaLabel-main).

The Mac `.venv` (Python 3.14) has pandas, numpy, matplotlib, ccxt, yfinance, duckdb, pyarrow and curl_cffi. It does **not** have scipy or scikit-learn: run `pip install -r requirements.txt` inside mcpt-lab.

**Sync quirk:** `device_commit_files` sometimes silently fails to overwrite existing files. Verify with md5, or commit a zip and extract it with python via device_bash (that worked this session). Deleting files is not allowed on the Mac.

## mcpt-lab: the repo (`~/Desktop/ahl-node/mcpt-lab`)
**Status:**
- Verified on the Mac; md5 matched the cloud copy.
- **Not yet pushed to GitHub.**
- The README says "designed and built by David".
- `LICENSE`, `CITATION.cff` and `pyproject.toml` say just "David". He should add his full name.
- The README clone URL has `<your-username>`.

```
mcptlab/      core.py (4 steps + fixed_param_mcpt + permutation_test + ledger/autosave),
              permute.py (bar, close, sign-flip), strategies.py (Donchian, MA cross, HawkesVolatility),
              trendline.py (closed-form resistance, RF meta-label), plotting.py, data.py (load_ohlc, synthetic)
examples/     01_quickstart (5 noise + 5 trend markets, ~1 min), 02_hawkes_four_steps [--quick],
              03_hawkes_robustness [--quick] (sign-flip + post-pub), 04_trendline_metalabel [A|B] [--quick],
              05_your_own_strategy (template)
tests/        15 unittest checks (python -m unittest discover tests), ~20 s, all pass
validation/   validate.py (~9 min): noise 1/40 below 0.05 in-sample (mean p 0.49); trends 20/20 in-sample, 9/10 walk-forward
results/      LEDGER.csv (12 tests), nulls/01-12 (header "title|real"), figures/ (scorecard, all_tests,
              hawkes_story, permutation_explainer, validation)
scripts/      make_figures.py (rebuilds all figures; hawkes_story needs data/btc_1h_binanceus.csv)
data/         BTCUSDT_1h_2018_2022.csv (from neurotrader repos), download_binanceus.py, README
docs/         METHOD.md, RESULTS.md, LEARN.md, PROJECT_SUMMARY.md (essay facts)
NOTICE.md     neurotrader MIT attribution (mcpt (c)2025, Hawkes/Trendline (c)2023) -- must stay
.gitignore    excludes keys, state/, downloaded Binance.US data, personal LEDGER/mcpt_runs
```

**Verified:**
- All numbers reproduce exactly from the package: Hawkes IS (0.1, 96) PF 1.133, WF 1.051, post-pub 1.031, Sharpe 0.67; trendline filtered 1.091 vs 1.012.
- Every example ran end to end.
- The trendline B3 null was completed this session (p = 0.195, identical).

**API change vs. the old `mcpt/` folder:**
- The package puts `LEDGER.csv` and `mcpt_runs/` in the **current working directory**; override with the `MCPT_LEDGER` / `MCPT_RUNS` environment variables.
- `in_sample_mcpt(..., null="signflip")` replaces signflip_hawkes.py.
- `fixed_param_mcpt` replaces postpub_hawkes.py.

### To publish (commands given to David)
```bash
cd ~/Desktop/ahl-node/mcpt-lab
git init && git add . && git commit -m "mcpt-lab v1.0: permutation-tested backtesting"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/mcpt-lab.git
git push -u origin main      # password prompt = GitHub personal access token
```
Create the empty repo on github.com first, without a README.

## Showcase page (artifact)
- **"mcpt-lab"**, an interactive results page at claude.ai/artifact/WyTkW1ScK2UmwprtAjWz7F (version 2). **Private until David shares it** via the Share menu.
- Contents:
  - byline "A backtesting platform designed and built by David";
  - histogram explorer for all 12 tests;
  - the scorecard table;
  - three verdict cards;
  - Hawkes PF-by-year and cost-sweep charts;
  - the validation facts;
  - a "what I built" section;
  - neurotrader credit in the footer.
- Source: cloud `/home/claude/site/template.html` + `build.py` (embeds the null CSVs). To update from a new conversation, pass the artifact URL.
- **Pending:** add the GitHub URL and his full name once the repo is public; optionally add an AI-assistance line.

## Research results (the whole record)
**Preregistered futures research: 13 tests, 0 passes**

| round | tests | result |
|---|---|---|
| v2 | long history | small historical momentum effect, gone since 2016 |
| v3 | clean CME futures: trend, cross-sectional momentum, carry | net Sharpe 0.10–0.34 against about 0.68 needed |
| v4 | OU/Kalman spreads | net −0.39 |
| v5 | Goldman roll | net −1.06 |

**Nine pipeline bugs found and documented:**
1. vol look-ahead;
2. monthly-average FRED data;
3. HTTP 206 treated as failure;
4. CME symbol reuse;
5. expiry proxy;
6. Kalman units;
7. Sunday/holiday stubs;
8–9. the two from session 1.

**MCPT ledger: 12 tests**

| strategy | results | verdict |
|---|---|---|
| ES Donchian | IS p 0.562, WF p 0.985 | fits noise |
| **VolatilityHawkes** (BTC 1h) | IS p 0.002; sign-flip 0.002; WF PF 1.051, p 0.025; **post-pub 2023–26 PF 1.031, Sharpe 0.67, p 0.026** | the only survivor: small and fading |
| Trendline base | p 0.627 / 0.119 / 0.077 | no edge |
| Trendline ML filter | 2020–22 WF-MCPT p 0.015, label-shuffle p 0.005; **post-pub p 0.195** | died after publication |

Hawkes details:
- PF by year: 1.13, 1.12, 1.15, 1.05, 1.05, 1.09, 1.02, 0.99, 1.04.
- Post-pub Sharpe at 0 / 5 / 10 / 20 bps: 0.85 / 0.67 / 0.48 / 0.10.
- Longs PF 1.083, shorts PF 0.995. Binance.US can't short.

**Essay headline:** built and validated a permutation-testing platform; replicated published strategies; documented 9 pipeline bugs; measured post-publication decay directly (Hawkes survives weakly, the trendline filter dies).

## Next steps (in priority order)
1. **Publish the repo** (commands above). Share the showcase page, and send Claude the GitHub URL and his full name so the page footer and repo can be updated.
2. **Pi appliance (`prod/`)**: paper-trade VolatilityHawkes with κ = 0.1 and lookback 96 on live hourly BTC (`DRY_RUN=1`).
   - Log every live signal and diff it against the backtest on the same bars.
   - Architecture: reconciliation-first loop, DuckDB state, RiskGate, systemd.
   - `mcptlab/strategies.py` is the single source of truth.
   - This is the engineering artifact, and it extends the post-publication sample.
3. **New strategy ideas:** David proposes an idea *with a mechanism*, then runs steps 1–2 via `examples/05_your_own_strategy.py`. Judge each p against 25 prior tests (13 preregistered + 12 MCPT). A long-only Hawkes variant would be a new, post-hoc test.

## Update 2026-09-24 (late): publication-decay study (v6), done
- Preregistered as `PREREGISTRATION_v6.md` (amendments A1 and A2 were made before the run). Results are in `RESULTS_v6.md`.
- Code: `mcpt-lab/mcptlab/decay.py`, `examples/06_publication_decay.py`, `data/download_french.py` (French data git-ignored in `data/french/`), and `tests/test_decay.py` (H16–H19). **21 tests pass.**
- **Result:** H1 PASS. Across 8 strategies that replicated in-sample, post-publication returns are 30% of in-sample (b_POST −0.70, SE 0.12, p ≈ 3e-9). All 9 anomaly strategies declined. Size is excluded (IS t = 1.2). H2 p = 0.017, but the OOS estimate is noisy.
- **Faber timing:** kept its drawdown benefit (−18% vs −50% post-publication).
- **Ledger:** now 22 rows. **Trial count: 35** (25 + trial 26 primary + 27–35 descriptive). Nulls 13–21 are in `results/nulls/`.
- README, `docs/RESULTS.md` (section 4) and `docs/PROJECT_SUMMARY.md` are updated. **Not yet updated:** the scorecard figures (still show 12 tests) and the showcase artifact.
- **Pending for David:** verify the journal publication dates, especially the Lakonishok & Smidt issue month.

## Update 2026-09-24 (later): net-of-cost follow-up (v7), done
- `PREREGISTRATION_v7.md` and `RESULTS_v7.md`; code in `examples/07_decay_after_costs.py`; cost model in `decay.py` (NMV Table 3 plus market-cost eras); H20–H22 in `tests/test_decay.py`. **24 tests pass.**
- **Result:** no strategy keeps a reliable post-publication net profit. Best is TOM, +0.15%/mo, p = 0.087. Monday and TOM were never tradable in-sample (the pre-1975 cost was about 1% one-way).
- **Ledger:** 31 rows; nulls 22–30. **Trial count: 44.**

## Update 2026-09-24 (latest): figures and showcase page
- **New figures** (`python scripts/make_figures.py`): `decay_story.png`, `decay_growth.png`, `calendar_costs.png`, `decay_tests.png`.
  - `all_tests.png` is now limited to the 12 strategy tests.
  - The scorecard title was corrected.
  - `plot_grid` gained `xlabel` / `title_fn` / `fmt` parameters.
- The figures are embedded in the README and `docs/RESULTS.md`; `docs/PROJECT_SUMMARY.md` lists them.
- **Showcase page, version 3:**
  - New section: stat tiles, an interactive decay bar chart, and a calendar effect-vs-cost chart with a Monday/TOM toggle.
  - Stale facts updated (24 tests; 12 published strategies; 100 years of data).
  - **The page source now lives locally at `ahl-node/site/mcpt-lab.html`,** with data in `site/decay_data.json`. It replaces the cloud copy.
  - **Sharing:** the page is shared as "anyone with the link".

## Update 2026-09-24 (night): personal archive site + repos ready to push
- **Name:** David DeConti. **GitHub:** dnd8639-byte. Filled into LICENSE, CITATION.cff, pyproject.toml and README (clone URL + byline).
- **`ahl-node/site/`** is the GitHub Pages repo for **dnd8639-byte.github.io**:
  - pages: `index.html` (home + projects + research log), `mcpt-lab/` (the showcase, now standalone with a top bar and name), `futures-research/` (new: 13 tests, chart, bugs table);
  - support files: `assets/site.css` (shared tokens), `_template/` (new project page), `README.md` (how to add a project).
  - The ninth pipeline bug (#2, session 1) is not described in any local file; the page says so.
- **Both repos** (`mcpt-lab/`, `site/`) are git-initialized with one commit each on `main`, remotes set. Repo-local identity: David DeConti <dnd8639-byte@users.noreply.github.com>.
- **LIVE (2026-09-24):** https://dnd8639-byte.github.io (Pages: main, root) and https://github.com/dnd8639-byte/mcpt-lab. All pages and charts verified live.
- The claude.ai showcase artifact still exists, shared "anyone with the link". David decides whether to keep it.
- **AI-assistance line on the site:** recommended, not yet added. Ask David.

## Update 2026-09-24 (late night)
- At David's request, the "Co-Authored-By: Claude" trailers were removed from both repos' commits (amended locally; David force-pushes).
- David asked to backdate commits and log dates to 2024/2025. **Declined:** the work is from Sept 2026, and fake dates would misrepresent the record on applications. Offered instead to add genuine earlier work with its real dates. Keep all dates truthful.
- Background log entries added at David's direction (years only): 2024 learning the basic frameworks, custom-Python backtesting, and reading about published strategies and publication decay; 2025 forex trading and his own strategies. Project work stays labelled 2026. All dates on the site are now years only.
- Both repos force-pushed and verified: no Claude trailer on GitHub; the live site shows the new log.
