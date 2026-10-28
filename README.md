# permutalpha-quant

Independent quantitative-finance research by David DeConti: one question, asked honestly. Can I find a trading
edge without fooling myself? Every result here had to beat the same strategy run on shuffled ("permuted") markets,
where any real edge is destroyed. Most ideas did not. One published edge did, and then a sealed holdout showed it
has largely faded since 2007.

| Folder | What it is |
|---|---|
| [`lab/`](lab/) | **Open Quant Lab** (`pip install permutalpha`): the research pipeline. Pre-registration, frozen rules, whole-search permutation tests under three null models, one-shot holdouts with power analysis, a tamper-evident ledger, a data collector, paper trading, a live dashboard, and an AI assistant that can explore but never test. |
| [`micro-study/`](micro-study/) | A pre-registered study of BTC order-book features recorded by my Raspberry Pi; frozen 2026-10-20, confirmatory test pending. |
| [`mcpt-lab/`](mcpt-lab/) | The first permutation-testing toolkit, before the lab. |
| [`projects/`](projects/) | Ten quant mini-projects: tail risk, seasonality, overfitting, momentum crashes, factor neutrality, American options, autodiff Greeks, implied distributions, Hawkes processes, news diffusion. |
| [`studies/`](studies/) | Earlier pre-registered studies (v2-v7), the efficiency research engine, and notes. |
| [`infra/`](infra/) | Raspberry Pi services: the order-book recorder and the paper trader. |

Licensed data (Databento futures, Yahoo option chains) is never in this repository; the code downloads or imports
it locally. Much of the code was written with an AI assistant (Claude) to my design and direction; the research
questions, pre-registrations, decisions and experiments are mine.

Results so far: `lab/docs/CALIBRATION.md` (how trustworthy the tests are), the lab's ledger of every test run, and
the micro study's pre-registration.
