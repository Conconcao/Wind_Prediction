# Implementation Sources

This experiment code prefers mature upstream libraries and official
documentation instead of ad-hoc reimplementation where possible.

## Current baseline implementation sources

1. LightGBM
   - GitHub: <https://github.com/microsoft/LightGBM>
   - Python API: <https://lightgbm.readthedocs.io/en/latest/Python-API.html>
   - Early stopping callback:
     <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.html>

2. statsmodels ARIMA / SARIMA
   - GitHub: <https://github.com/statsmodels/statsmodels>
   - ARIMA docs:
     <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>
   - Rolling append update docs:
     <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.append.html>

3. pandas time-series feature handling
   - GroupBy forward fill docs:
     <https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.ffill.html>
   - Rolling std docs:
     <https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html>

## Planned deep-learning implementation sources

4. PyTorch GRU
   - Docs:
     <https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html>

5. PyTorch Dataset / DataLoader
   - Dataset docs:
     <https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>
   - DataLoader docs:
     <https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>

6. NumPy disk-backed arrays
   - `open_memmap` docs:
     <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.open_memmap.html>

7. Graph WaveNet reference
   - Paper: <https://arxiv.org/abs/1906.00121>
   - GitHub: <https://github.com/nnzhan/Graph-WaveNet>
   - Reference model implementation:
     <https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py>

8. Temporal Convolutional Network reference
   - GitHub: <https://github.com/locuslab/TCN>
   - PyTorch Conv1d docs:
     <https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html>
