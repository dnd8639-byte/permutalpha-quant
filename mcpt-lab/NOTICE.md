# Third-party notices

Parts of this project are adapted from open-source work by **neurotrader888**, used under the MIT License.

| this repo | adapted from | what was changed |
|---|---|---|
| `mcptlab/permute.py` (`permute_ohlc`) | [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) `bar_permute.py` | vectorized (identical output); added `permute_close`, `signflip_ohlc` |
| `mcptlab/core.py` (four-step process) | [neurotrader888/mcpt](https://github.com/neurotrader888/mcpt) | trading costs, walk-forward MCPT with any strategy, fixed-parameter and custom nulls, ledger |
| `mcptlab/strategies.py` (`HawkesVolatility`) | [neurotrader888/VolatilityHawkes](https://github.com/neurotrader888/VolatilityHawkes) | ATR reimplemented without pandas_ta; reproduces the original exactly |
| `mcptlab/trendline.py` | [neurotrader888/TrendlineBreakoutMetaLabel](https://github.com/neurotrader888/TrendlineBreakoutMetaLabel) | exact closed-form resistance line (about 150× faster; 1,172 of 1,175 trades identical) |
| `data/BTCUSDT_1h_2018_2022.csv` | `BTCUSDT3600.csv` in the two repositories above | renamed only (original market data from Binance) |

The original license text:

```
MIT License

Copyright (c) 2025 neurotrader888   (mcpt)
Copyright (c) 2023 neurotrader888   (VolatilityHawkes, TrendlineBreakoutMetaLabel)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
