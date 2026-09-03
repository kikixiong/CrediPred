# WWW RQ1 Fixed-Split Strong Baselines

## Frozen protocol

- Use the existing group-disjoint split with partition SHA-256
  `b896a886a4b8981043820e2326492f5ffd8c20c63006380f9df8f2544231b5d7`.
- The 9,199 mapped labelled domains are fixed as
  `fit/select/calibrate/test = 5519/920/1380/1380`.
- The split seed is `2027`; neural model seeds are exactly `42` through `46`.
- Optimization jobs receive a role-masked `fit/select` target artifact. Calibration
  jobs receive a separate `calibrate` artifact. The `test` artifact is stored under
  `sealed/` and is admitted only after the pretest exposure ledger passes.

## Registered matrix

- Train `FF`, `GCN`, `SAGE`, and `GAT` for all five neural seeds.
- Derive `GAT-mlp` and `GAT-topology` from each seed's frozen GAT parent cache.
- Average the five aligned GAT prediction artifacts for `GAT-ensemble`.
- Fit `GlobalMedian` once from fit targets only. It supplies a point prediction and
  symmetric split conformal interval calibrated at alpha `0.10`.
- Fit deterministic `xgboost-cpu==3.2.0` once. The fixed two-entry depth grid is
  fit on `fit` and selected on `select`, independently for the squared-error
  midpoint and native `.05/.95` quantile objectives. It supplies simple conformal
  and CQR; there is no alternate boosting implementation or runtime fallback.
- Fit deterministic regression label propagation once on the complete archived
  45,030,252-node, 1,014,523,551-edge directed graph. Only fit targets are seeds.
  The fixed damping/iteration grid is selected by select MAE. Unreached nodes use
  the fit median. A CPU full-graph throughput/memory probe must pass before the
  baseline runs; failure stops the baseline rather than changing its graph.

Every arm uses the same 1,380-domain sealed test population and reports MAE, RMSE,
Spearman, and, for each legally available UQ method, coverage, mean width, median
width, and interval score. The complete sealed inventory is 34 prediction/audit
artifacts and 66 unaggregated metric rows.

## Execution gates

1. Build one locked, isolated Python environment including the `rq1` dependency
   group and verify XGBoost reports version `3.2.0`.
2. Execute the synthetic strong-baseline smoke from an immutable source snapshot.
3. Submit seed-42 GAT alone and observe real startup without OOM before releasing
   the remaining pretest DAG.
4. Gate every full GAT parent cache on the representative cache throughput probe,
   and gate RegressionLP on its independent full-graph CPU probe.
5. Collect pretest only after all fit/select/calibrate artifacts complete. Reject
   any sealed-test prediction, audit, or result artifact in this phase.
6. After checking the pretest exposure ledger, submit one seed-42 GAT sealed-test
   prediction and observe startup before releasing the remaining sealed DAG.
7. Collect the 66 sealed rows without tuning, aggregation, seed changes, or retries
   that alter the scientific configuration.
