# Xinyang 15-Min Wind Speed Forecasting: Literature Review and Experiment Plan

## 1. Goal

This experiment focuses on `15-minute-ahead` wind speed forecasting for the
`xinyang` wind farm. The main objective is to build a deep-learning-first
benchmark suite that is realistic under strict causal deployment constraints,
while retaining lightweight local baselines such as persistence, SARIMA, and a
simple machine learning regressor.

## 2. Local Data Conditions

### 2.1 Available xinyang data

Main files already available locally:

- `ALL_TURBINES_15min_202501-202512_QC2.parquet`
- `ALL_TURBINES_1min_202501-202512.parquet`
- `pre_QC_气象观测数据.xlsx`
- `风机基本信息.csv`
- optional later-stage weather forecast:
  `data/open_nwp/gdex_gfs_xinyang_2025-07-01_2025-12-31_hourly.parquet`

### 2.2 What the data supports

From local inspection:

- `46` turbines
- turbine IDs like `S01` to `S46`
- one homogeneous manufacturer group: `远景能源`
- rated capacity: `2.2 MW`
- hub height: `125 m`

The 15-minute SCADA table contains:

- `1,423,698` rows
- `34,208` unique timestamps
- time span:
  `2025-01-01 00:15:00` to `2026-01-01 00:00:00`
- target-ready field:
  `平均风速`
- no missing values in `平均风速`
- `cnt_raw` range: `9` to `15`

The 1-minute SCADA table can support higher-resolution lag and variability
features. Example fields include:

- `风机`
- `时间`
- `风速`
- `有功功率`
- `机舱方位角`
- `风向`

The tower meteorology workbook contains a long-format `data_preQC` sheet with:

- timestamps at `15-minute` resolution
- heights: `10, 30, 50, 70, 125 m`
- variables:
  `温度, 湿度, 风速, 风向, 气压`

Coverage versus the SCADA 15-minute timestamps is about `97.39%`, so tower
features are usable as auxiliary inputs after left-join plus missing masks.

### 2.3 Important implication

The local data strongly favors a `multi-turbine short-horizon forecasting`
setup:

- many turbines in one farm
- same turbine platform
- explicit spatial layout via latitude/longitude
- rich recent-history signals from 15-minute and 1-minute SCADA
- tower observations at multiple heights

This is a good fit for graph-based spatio-temporal deep models, with simpler
recurrent and convolutional models as strong lower-complexity baselines.

## 3. Literature Findings

### 3.1 Review papers

Recent review papers consistently show that deep learning dominates modern
short-term wind forecasting, especially when high-frequency observations are
available and the forecasting horizon is short.

Key literature:

1. Alves et al., 2023, systematic review
   - link: <https://www.mdpi.com/2073-431X/12/10/206>
   - takeaway:
     deep learning outperformed traditional methods in the reviewed short-term
     wind nowcasting literature, and high-resolution data was repeatedly
     emphasized as beneficial

2. Wang et al., 2021, deep neural network review
   - link: <https://doi.org/10.1016/j.apenergy.2021.117766>
   - takeaway:
     the most useful distinction is not "RNN vs Transformer" in isolation, but
     rather how feature extraction, temporal relationship learning, and hybrid
     design are combined

3. Abdoos et al., 2021, deep-learning taxonomy review
   - link: <https://doi.org/10.1016/j.egyai.2021.100060>
   - takeaway:
     recurrent models remain dominant, hybrid methods are common, and
     multi-step and probabilistic setups are rising, but dataset structure still
     governs which architecture works best

### 3.2 Recurrent models remain strong for short horizons

Two papers are especially relevant for a local baseline stack:

4. Neshat et al., 2020, fine-tuned LSTM
   - link: <https://www.sciencedirect.com/science/article/pii/S0196890420303629>
   - takeaway:
     careful input design and hyperparameter tuning materially affect LSTM
     performance in short-term wind speed forecasting

5. Wang et al., 2022, optimized GRU input selection
   - link: <https://doi.org/10.1016/j.energy.2021.122960>
   - takeaway:
     GRU remains a strong short-term baseline, and explicit input screening can
     improve robustness instead of blindly feeding every available variable

### 3.3 Spatial modeling matters when multiple turbines are observed together

This matters directly for xinyang because there are `46` turbines in one farm.

6. Sun et al., 2022, spatiotemporal joint learning
   - link: <https://ideas.repec.org/a/eee/renene/v183y2022icp148-159.html>
   - takeaway:
     multi-location wind speed prediction benefits from jointly modeling spatial
     and temporal dependencies rather than treating each location independently

7. Wang et al., 2024, AG-MGAT
   - link: <https://doi.org/10.1016/j.apenergy.2024.123477>
   - takeaway:
     multi-graph attention is useful when latent turbine-to-turbine
     relationships are not fully captured by geography alone

8. Cai and Li, 2024, dynamic spatio-temporal directed graph attention
   - link: <https://doi.org/10.1016/j.apenergy.2024.124124>
   - takeaway:
     dynamic directed graphs are attractive when directional flow relationships
     change over time

### 3.4 Transformer models are promising, but not the first thing to trust

9. Qiu et al., 2024, WindFormer
   - link: <https://www.mdpi.com/2076-3417/14/15/6741>
   - takeaway:
     transformer-style multivariate models can capture richer feature
     interactions using wind speed together with auxiliary meteorological
     variables such as humidity, temperature, and power

My inference from the literature and your data:

- transformers are worth keeping as an ablation
- but for the first causal 15-minute experiment on one farm, a graph-temporal
  model and a GRU/TCN stack are lower-risk primary choices
- the reason is that xinyang has many spatially related turbines but only one
  year of data, so strong spatial inductive bias is likely more valuable than a
  very flexible attention-only model

### 3.5 Why keep classical baselines

10. Liu et al., 2021, SARIMA vs GRU/LSTM
    - link: <https://doi.org/10.1016/j.energy.2021.120492>
    - takeaway:
      SARIMA is still a valid classical benchmark for short-term wind speed
      forecasting, even when deep models ultimately perform better

This justifies preserving:

- persistence
- seasonal persistence
- SARIMA
- one simple tabular ML model

## 4. Recommended Forecasting Task

### 4.1 Primary task

Use a `causal deterministic one-step-ahead` setup:

- forecast origin: time `t`
- target: turbine-level `平均风速` at `t + 15 min`
- output form: all `46` turbines predicted together for the main deep model

### 4.2 Why not start with site-average only

Site-average wind speed is useful as a diagnostic series, but it should not be
the main target because it removes:

- wake-related turbine differences
- layout effects
- directional asymmetry across the farm
- the main advantage of graph-based spatial models

So the recommendation is:

- primary benchmark: `46-turbine multi-output forecast`
- secondary diagnostic: `farm-mean wind speed forecast`

## 5. Data Split

Use a strict chronological `7:2:1` split based on unique timestamps, not random
rows.

### 5.1 Fixed split boundaries

- train:
  `2025-01-01 00:15:00` to `2025-09-15 16:00:00`
- validation:
  `2025-09-15 16:15:00` to `2025-11-26 08:45:00`
- test:
  `2025-11-26 09:00:00` to `2026-01-01 00:00:00`

### 5.2 Leakage rule

All scaling, feature normalization, imputation statistics, and any learned
encoders must be fit on the training split only.

For validation and test windows, it is acceptable to use a short lookback that
extends into the immediately previous split, because those are past
observations available at forecast time. The rule is:

- targets must stay inside the split
- history may reach backward
- no future information may be used

## 6. Feature Plan

Use a phased feature strategy instead of throwing everything into the first
model.

### 6.1 Feature set F0: 15-minute SCADA only

Per turbine at each 15-minute timestamp:

- `平均风速`
- `最大风速`
- `最小风速`
- `风速标准差`
- `平均有功功率`
- `有功功率标准差`
- `平均风向`
- `风向标准差`
- `平均机舱方位角`
- `机舱方位角标准差`
- `cnt_raw`

This should be the first feature set for all baselines and the first deep
models.

### 6.2 Feature set F1: calendar encoding

Add:

- sin/cos of hour-of-day
- sin/cos of day-of-year
- optional month index

### 6.3 Feature set F2: tower met features

Pivot `pre_QC_气象观测数据.xlsx` from long format to wide format by height:

- `ws_10m, ws_30m, ws_50m, ws_70m, ws_125m`
- `wd_10m, ...`
- `temp_10m, ...`
- `rh_10m, ...`
- `pressure_10m, ...`

Also add missing flags per block because tower coverage is not perfectly
complete.

### 6.4 Feature set F3: 1-minute derived recent dynamics

Derive from the 1-minute SCADA table over rolling windows such as:

- past `15 min`
- past `30 min`
- past `60 min`

Suggested aggregations:

- mean
- std
- min
- max
- ramp
- last-minus-mean

for:

- wind speed
- active power
- nacelle angle
- wind direction

### 6.5 Feature set F4: optional NWP ablation only

The open NWP data is hourly and only available from `2025-07-01`, so it should
not be part of the primary 15-minute benchmark.

Recommendation:

- keep NWP out of the main benchmark
- only run it later as a `Jul-Dec subset ablation`

Reason:

- temporal resolution mismatch
- partial-year coverage
- likely limited benefit for a 15-minute one-step horizon

## 7. Model Stack

## 7.1 Local baselines

These should run locally on CPU first.

1. Persistence
   - predict next 15-minute wind speed using current 15-minute wind speed

2. Seasonal persistence
   - use same slot from previous day
   - season length = `96` steps per day

3. SARIMA
   - season length = `96`
   - start with farm-mean wind speed and then extend to per-turbine models
   - to keep local runtime reasonable, use capped order search

4. LightGBM
   - tabular lag-feature baseline
   - train pooled across turbines
   - include turbine ID as categorical or embedding-like integer feature

## 7.2 Deep-learning-first shortlist

### Model A: GRU seq2one baseline

Why keep it:

- supported by the literature as a strong short-term baseline
- cheap to train
- easy to debug locally

Recommendation:

- causal multivariate GRU
- 2 layers
- hidden size around `128`
- dropout around `0.1`

### Model B: TCN or CNN-GRU

Why keep it:

- efficient at capturing short-range temporal structure
- often more stable than RNN-only models on short horizons
- good second deep baseline before graph models

Recommendation:

- causal dilated TCN
- kernel size `3`
- dilation levels like `1, 2, 4, 8, 16`

### Model C: Spatio-temporal graph network

This is the recommended primary model.

Why it best matches xinyang:

- `46` turbines in one farm
- explicit spatial layout
- likely directional/wake coupling
- one-step ultra-short-term task where neighborhood structure matters

Recommended implementation direction:

- GraphWaveNet-style or AG-MGAT-inspired model
- node = turbine
- node features = recent per-turbine SCADA plus optional tower context
- adjacency sources:
  - geographic distance graph
  - training-set correlation graph
  - learned adaptive graph

If model budget must stay narrow, this should be the main advanced model.

### Model D: Transformer ablation

Use later as an ablation, not as the first target.

Two reasonable directions:

- WindFormer-style multivariate transformer
- PatchTST-style time-series transformer

Rationale:

- useful comparison against graph models
- strong multivariate feature fusion
- but less directly aligned with turbine-to-turbine inductive bias

## 8. Training Setup

### 8.1 Default lookback search

Test:

- `16` steps = `4 h`
- `32` steps = `8 h`
- `96` steps = `24 h`

Recommended default starting point:

- `32` steps

### 8.2 Loss and optimization

- loss:
  `Huber` first, `MSE` as ablation
- optimizer:
  `AdamW`
- initial learning rate:
  `1e-3`
- early stopping:
  patience `10`
- max epochs:
  `50` for local debugging, more later on HPC

### 8.3 Standardization

- z-score continuous inputs using train split only
- standardize target for neural training if needed, but invert before reporting

## 9. Evaluation

Use metrics that remain stable near zero wind speed.

### 9.1 Main metrics

- MAE
- RMSE
- R2
- skill score against persistence:
  `1 - RMSE_model / RMSE_persistence`

Avoid MAPE as the main metric because wind speed can get close to zero.

### 9.2 Reporting levels

Report all metrics at two levels:

1. macro average across turbines
2. farm-mean series

### 9.3 Stratified diagnostics

Also report:

- low-wind regime
- medium-wind regime
- high-wind regime
- daytime vs nighttime

## 10. Recommended Execution Order

### Phase 1

- build F0 features
- run persistence, seasonal persistence, SARIMA, LightGBM

### Phase 2

- run GRU and TCN on F0
- choose lookback window from validation

### Phase 3

- add F2 tower features
- train spatio-temporal graph model

### Phase 4

- add F3 one-minute derived features
- compare graph model gain against simpler deep models

### Phase 5

- optional transformer ablation
- optional Jul-Dec NWP ablation

## 11. Final Recommendation

Given the literature and the actual xinyang data, the most sensible first
experiment is:

- target:
  `46-turbine 15-minute-ahead mean wind speed`
- main features:
  `F0 + F1`, then `F0 + F1 + F2`
- local baselines:
  `persistence + seasonal persistence + SARIMA + LightGBM`
- deep baselines:
  `GRU + TCN`
- primary advanced model:
  `spatio-temporal graph network`
- transformer:
  `ablation only after graph model is running`

This gives a practical path that is literature-backed, compatible with your
data, strict about leakage, and efficient enough to debug locally before moving
to server-side training jobs.
