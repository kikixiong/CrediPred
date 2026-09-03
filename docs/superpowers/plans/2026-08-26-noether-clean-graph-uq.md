# Noether Clean Graph/UQ Execution Plan

**Goal:** Produce a leakage-free CrediBench Dec-2024 graph/UQ rerun for WWW RQ1, then launch and monitor it on Noether.

**Scope:** Reuse the existing PyG dataset, `Model`, quantile loss, CQR utilities, correction GNN, and Noether H100 environment. Add only the experiment-role separation and reporting that are missing. Do not modify the historical experiment outputs or the dirty `mu_star` worktree.

## Task 1 — Protocol, split, and population ledger (medium risk)

Files:

- Add `credipred/experiments/gnn_experiments/clean_uq_protocol.py`.
- Add `test/test_clean_uq_protocol.py`.

Behavior:

- Build one deterministic group-preserving `fit/select/calibrate/test` split with target ratios `60/10/15/15`.
- Keep the split seed separate from model seeds.
- Emit one compact protocol JSON and population ledger; report exclusions rather than hiding the 11,520-to-mapped-node funnel.
- Prefer a genuinely unused mapped/labelled cohort as sealed test when the ledger proves one exists; otherwise use the frozen four-way split.

Verification:

- RED/GREEN tests for deterministic assignment, group integrity, mutual disjointness, and complete once-only assignment.
- No full repository test run at this stage.

## Task 2 — Clean trainer and audit (medium risk)

Files:

- Add `credipred/experiments/gnn_experiments/clean_uq_train.py`.
- Add `credipred/experiments/gnn_experiments/clean_uq_audit.py`.
- Add a small no-edge correction model beside the existing correction GNN.
- Add focused tests under `test/`.

Behavior:

- Generic three-head quantile trainer for `FF`, `GCN`, `SAGE`, and `GAT`.
- Training receives only fit and select roles; no calibrate/test loader or metric exists in the epoch loop. Checkpoint selection uses select pinball loss and an immutable state snapshot.
- Quantile heads are ordered `[midpoint, lower, upper]` and target `.50/.05/.95`; primary conformal miscoverage is `.10`.
- The training CLI has small explicit stages for base fit, frozen-parent prediction cache, correction fit, and role-specific prediction. Every stage accepts `--run-dir` and one `--seed`; historical fixed weight/cache paths are not reused.
- Independent audit first writes calibration state (`qhat` values) from the calibration role, then a separate invocation reads that state and the sealed-test predictions. It computes simple absolute-residual conformal and CQR, raw crossing, coverage, mean/median width, interval score, midpoint MAE/RMSE/Spearman, and per-seed JSON/CSV rows.
- For evaluation, report raw lower/upper crossing; then sort raw endpoints, conformalize, and clip final intervals to `[0,1]` for both simple conformal and CQR.
- Correction arms share a frozen GAT parent: one no-edge MLP and one topology-aware GNN. Both use the same fit/select roles, optimizer, epoch budget, seed, three-input additive-delta contract, and pinball objective. Disable batch-internal conformal size loss; perform conformal calibration only in the audit stage.
- Derive the GAT ensemble from seed predictions without another training job.

Lean implementation boundary:

- Prefer one generic trainer module and one audit module; a tiny correction-MLP module is allowed.
- Reuse `Model`, `NeighborLoader`, `_quantile_loss`/equivalent corrected pinball logic, `snapshot_state_dict`, `CorrectionGNN`, and the functions in `cqr.py`.
- Make the existing `FF` path callable with no `edge_index` using the smallest signature fix; do not refactor the historical experiment runners.
- Core artifacts may be Torch tensors plus compact JSON/CSV. Do not add W&B, a registry framework, per-epoch test logs, or per-artifact receipts.

Verification:

- RED/GREEN tests for the FF forward path, correction output contract, finite-sample qhat, and the trainer role contract.
- Run only the affected tests locally; run torch/PyG integration smoke in the configured Noether environment.

## Task 3 — Noether launch path (medium risk)

Files:

- Add one reusable H100 array job under `scripts/noether/`.
- Add one small submit script that wires `afterok` dependencies.
- Add one resolved Dec-2024 experiment configuration.

Formal matrix:

- `FF/GCN/SAGE/GAT × seeds 42/43/44/45/46`: 20 base jobs.
- `GAT parent cache × 5`: 5 inference jobs. GAT role predictions are exact
  slices of this cache so the parent and both correction arms remain paired.
- `no-edge/topology correction × 5`: 10 correction jobs.
- Serialized calibration/test audit and aggregation stages; they perform no
  optimization and never feed metrics back into training.
- One final full-graph export only for the selected producer.

Sealed-test boundary:

- All five model seeds are frozen in the protocol before any training.
- `pretest` may fit/select and create calibration predictions/state, but cannot
  create test artifacts. `sealed-test` is a separate submission that first
  verifies all five seeds, six arms, caches, calibration states, and ensemble
  membership.
- No seed is added or removed after the sealed-test phase begins.

Verification and launch:

- Read-only Noether preflight: repository, Python environment, CUDA/PyG, graph/mapping/labels, partition, and writable fresh output root.
- Seed-42 short smoke over the complete GAT → cache → both corrections → audit chain.
- After GAT fit, measure real-graph parent-cache throughput on 32,768 seed
  nodes. If projected runtime exceeds the 20-hour safety budget, exit nonzero
  so the full-cache `afterok` dependency does not start under a 24-hour job.
- If smoke succeeds, submit only the serialized `pretest` chain with bounded
  concurrency. Submit `sealed-test` only after the full pretest ledger is
  complete and reviewed.

## Task 4 — Results and exposure ledger (low risk)

- Pull back compact CSV/JSON/log/config artifacts, not all checkpoints or full tensors.
- Produce the domain ledger and benchmark exposure ledger.
- Run one final completeness check over expected model/seed/arm rows, missing predictions, and sealed-test denominators.
- Temporal or natural-component stress is a separate follow-up only if the Noether inventory proves the required snapshots/components exist.

## Lightweight constraints

- One source/input manifest and one final results manifest; no per-stage receipt hierarchy.
- Hash only the frozen graph/mapping/label inputs, the split, and the final producer artifact.
- No repeated environment rebuild, full-graph export per seed, or alpha-by-model Cartesian expansion.
- Preserve null, harmful, or unstable results without changing the matrix after test opening.
