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

9. AGCRN reference
   - GitHub: <https://github.com/LeiBAI/AGCRN>
   - Reference model files:
     <https://github.com/LeiBAI/AGCRN/blob/master/model/AGCRN.py>
     <https://github.com/LeiBAI/AGCRN/blob/master/model/AGCRNCell.py>
     <https://github.com/LeiBAI/AGCRN/blob/master/model/AGCN.py>

10. MTGNN reference
   - GitHub: <https://github.com/nnzhan/MTGNN>
   - Reference model files:
     <https://github.com/nnzhan/MTGNN/blob/master/net.py>
     <https://github.com/nnzhan/MTGNN/blob/master/layer.py>

11. ModernTCN reference
   - GitHub: <https://github.com/luodhhh/ModernTCN>
   - Reference model files:
     <https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-short-term/models/ModernTCN.py>
     <https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-short-term/models/ModernTCN_Layer.py>

12. Closed-form Continuous-time Neural Network reference
   - Paper: <https://www.nature.com/articles/s42256-022-00556-7>
   - GitHub / official package: <https://github.com/mlech26l/ncps>
   - PyTorch CfC implementation:
     <https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc.py>
