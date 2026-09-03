# AutoDL P0 Corrected-Loss to AVeriTeC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to execute this plan task-by-task. Use the
> checkbox items as the execution ledger. Do not convert a preflight artifact
> into a formal result.

**Goal:** Complete the shortest auditable thesis evidence chain:
corrected-loss CrediPred at alpha `0.10`, Standard CQR, a budget-matched
no-edge/topology validation, one predeclared 45,030,252-node export, and the
full-cohort AVeriTeC Base/Midpoint/Conservative comparison.

**Architecture:** The frozen CrediGraph/CrediBench tensors are the graph-scale
source layer. CrediPred produces and validates domain-level midpoint/interval
artifacts. CrediSearch consumes only a frozen, manifest-verified export and
maps it to a frozen AVeriTeC candidate/content snapshot. The three AVeriTeC
arms change candidate order only; each arm independently serialises context,
generates output, parses it, and runs the bound evaluator. Thesis files consume
only validated manifests and compact result artifacts.

**Tech Stack:** Ubuntu 22.04, Python 3.10 for CrediPred, Python 3.11 for
CrediSearch, `uv==0.11.29`, PyTorch `2.7.1+cu128`, PyG `2.6.1` with matching
CUDA 12.8 extension wheels, one RTX PRO 6000 Blackwell GPU, `tmux`, pytest,
Ruff, mypy, JSON/JSONL/Parquet manifests, LaTeX/Overleaf.

## Controlling Scientific Mainline

The thesis asks whether a domain-level institutional quality prior, extended
to more scoreable domains through CrediGraph, changes evidence exposure and
benchmark output when it is used only to reorder a fixed candidate/content
pool. It does **not** treat a domain score as page truth, evidence correctness,
claim truth, political neutrality, or a graph-validity guarantee.

The implementation has two different three-way concepts:

| Layer | Variants | Role |
| --- | --- | --- |
| CrediPred producer validation | corrected-loss base, matched no-edge MLP correction, topology GAT correction | Validate the interval producer and identify only the incremental value of topology in the correction stage |
| AVeriTeC end-task arms | Base original order, Midpoint reranking, Conservative reranking | Estimate paired benchmark-local effects of changing evidence order |

The corrected-loss base is itself a graph GAT. Therefore no-edge versus
topology identifies incremental neighbour propagation in the correction
stage; it is not a no-graph versus graph comparison.

The P0 dependency chain is:

```text
frozen CrediGraph inputs
  -> alpha=.10, seed=42 corrected-loss base
  -> deterministic two-fold Standard CQR
  -> matched no-edge and topology corrections from the same parent
  -> predeclared producer/export rule
  -> one complete 45M-node manifested export
  -> sparse AVeriTeC domain overlay
  -> fixed-candidate Base / Midpoint / Conservative full-cohort execution
  -> official AVeriTeC metrics, paired contrasts, and source-exposure audit
```

Alpha `.05/.20`, seeds `43/44`, the full 27-training-unit matrix, CRAG, and any
second model family are P1. They may begin only after P0 artifacts, schemas,
protocol receipts, and reranking execution are stable.

Job 250229 is reused as a separate historical negative boundary. It is not
rerun, it is not the new Standard CQR result, it does not define Conservative,
and no later AVeriTeC result may repair its failed shift-UQ gate.

## Global Execution Constraints

- Do not run a generic smoke, pilot, infrastructure benchmark, 10-claim replay,
  or 20-query replay.
- Do run focused unit/contract tests and one minimum real-data PyG batch that
  exercises the actual CUDA/PyG forward and backward kernels. Mark its receipt
  `formal_result: false`; never promote its output or metrics.
- Reuse the current upper-pinball fix and immutable checkpoint snapshot at
  CrediPred commit `9c21d739852d315d41bc4f3cc6595737d9a58032`.
- Preserve all existing user work. CrediSearch currently has staged changes;
  do not reset, overwrite, or cite that dirty tree as a formal source identity.
- Every formal source tree is clean and bound to a full commit plus bundle or
  archive hash. Every input, config, seed, parent checkpoint, output, and
  manifest has a SHA-256 digest.
- Every run uses a new immutable run root and a new `tmux` session. A failed
  attempt remains on disk with its exit receipt; a fix uses a new run ID.
- Before a formal GPU launch, verify that no equivalent Noether or AutoDL job
  is writing the same canonical result. If a duplicate writer or ambiguous
  ownership exists, stop and report it.
- Do not treat host-visible 208 logical CPUs or about 1 TiB RAM as available.
  Resource decisions and receipts use cgroup limits, `/dev/shm`, and actual
  CUDA visibility.
- Null, harmful, or gate-failing scientific outcomes remain valid completed
  results when the artifact contract passes. Do not change the protocol in
  response to the outcome.

## Authoritative AutoDL Baseline

Use the supplied read-only preflight sampled at
`2026-07-28T20:23:57Z` as the sole setup baseline. Do not repeat broad
inventory probing.

| Resource | Effective value for planning |
| --- | --- |
| SSH | `root@connect.westd.seetacloud.com:13463`, dedicated key only, `IdentitiesOnly=yes`, `BatchMode=yes` |
| CPU | cgroup quota `2500000/100000` = **25 CPUs** |
| RAM | cgroup `memory.max` = **120 GiB**, no swap |
| GPU | exactly 1 RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, UUID `GPU-95684c29-453a-e4bb-2c8f-1cb1dbb43655` |
| VRAM snapshot | 97,887 MiB total; 21,125 MiB used; 76,127 MiB free; no container-visible process |
| `/dev/shm` | **60 GiB**, empty at snapshot |
| data disk | `/root/autodl-tmp`, **450 GiB**, almost empty |
| system disk | 30 GiB overlay; do not store graph artifacts here |
| Python | `/root/miniconda3/bin/python` 3.10.8; base lacks Torch/PyG and must remain unmodified |
| process manager | `tmux` absent and is a hard setup gate; `screen` is not an accepted substitute |
| network | source `/etc/network_turbo` in setup/run subshells; do not persist proxy variables globally |

The 120-GiB effective RAM passes the historical 55--58-GiB peak envelope. The
environment is nevertheless not formal-launch ready until `tmux`, the isolated
CUDA 12.8 environment, matching PyG extensions, and the unexplained roughly
21-GiB GPU allocation have been resolved.

## Remote Layout

```text
/root/autodl-tmp/credipred/
  bootstrap/
  source/
  env/
    credipred-cu128/
  cache/
    uv/
  wheelhouse/
  inputs/
    dec2024_pc1/
      data.pt
      mapping.pt
      INPUTS.sha256
  protocol/
    preflight/
    decisions/
    receipts/
  runs/
    immutable run roots

/root/autodl-tmp/credisearch/
  source/
  env/
    credisearch-py311/
  cache/
  inputs/
  protocol/
  runs/
```

CrediPred and CrediSearch never import from one another. CrediSearch consumes
only CrediPred's frozen export and sibling manifest.

---

### Task 1: Confirm and Freeze the P0 Interpretation

**Files:**

- Read:
  `thesis/overleaf-rewrite/notes/research_ledger.md`
- Read:
  `thesis/overleaf-rewrite/sections/04_research_design.tex`
- Read:
  `thesis/overleaf-rewrite/sections/05_methods.tex`
- Read:
  `thesis/overleaf-rewrite/sections/06_evaluation.tex`
- Modify after approval:
  `thesis/overleaf-rewrite/data/manifests/protocol_reranking_averitec_crag_v1.json`
- Create:
  `CrediPred/experiments/autodl_formal_credipred/protocol/p0_producer_decisions.json`

**Interfaces:**

- Consumes: the current draft thesis protocol and explicit author approval.
- Produces: one pre-outcome producer-decision receipt.

- [ ] **Step 1: Freeze the three required CrediPred research choices**

Recommended decision set for author confirmation:

1. **Midpoint:** explicit output head 0. The correction stage changes interval
   endpoints only, so Midpoint remains the same parent point estimate across
   no-edge and topology.
2. **Non-crossing:** preserve raw outputs for diagnostics; for CQR,
   evaluation, and export only, set lower to `min(raw_lower, raw_upper)` and
   upper to `max(raw_lower, raw_upper)`. Keep midpoint unchanged and report
   midpoint-outside-interval rates.
3. **Size loss:** use stable node-ID hashing with seed 42 to partition only the
   original training population into 60% correction fit, 20% size calibration,
   and 20% size optimisation. Validation selects the checkpoint; the 1,694-node
   formal population is untouched until post-checkpoint evaluation.

Do not formal-launch if any of these semantics remains ambiguous.

- [ ] **Step 2: Freeze the downstream producer rule before seeing matched results**

Recommended rule: export the artifact-valid topology producer regardless of
whether its matched comparison is positive, null, or harmful. Base and no-edge
remain attribution controls. Do not select the numerically best arm after
outcomes; do not silently substitute another producer if topology has an
artifact failure.

- [ ] **Step 3: Record P0 and P1 in the decision receipt**

The receipt records P0 as alpha `.10`, seed `42`, one base, one no-edge, one
topology, one deployment export, and AVeriTeC only. It records all other
alphas, seeds, producers, exports, and CRAG as deferred.

- [ ] **Step 4: Hash the canonical JSON receipt**

Expected: repeated canonical serialisation of the same decision object is
byte-identical; changing any decision changes the SHA-256.

### Task 2: Materialise the Supplied Preflight Receipt and Resolve Setup Gates

**Files:**

- Create:
  `CrediPred/experiments/autodl_formal_credipred/protocol/autodl_preflight_20260728T202357Z.json`
- Create:
  `CrediPred/experiments/autodl_formal_credipred/setup_autodl.sh`
- Create:
  `CrediPred/experiments/autodl_formal_credipred/capture_environment_receipt.py`

**Interfaces:**

- Consumes: only the supplied preflight facts.
- Produces: a checked-in receipt template and an isolated remote setup.

- [ ] **Step 1: Write the preflight receipt using effective resources**

Record both host-visible and cgroup-visible values, but set
`effective_cpu_count=25`, `effective_memory_bytes=128849018880`, and
`effective_shm_bytes=64424509440`. Record the visible GPU UUID, driver
`595.58.03`, compute capability `12.0`, VRAM snapshot, open-files soft limit
`1024`, and that `nvcc` is absent.

- [ ] **Step 2: Install `tmux` without changing the conda base**

The setup script sources `/etc/network_turbo` in its own shell, installs
`tmux`, records the package/version, and fails closed if `tmux -V` does not
exit zero. Do not report `screen` as passing this gate.

- [ ] **Step 3: Create the isolated bootstrap and project environment**

Run from the new AutoDL project root:

```bash
source /etc/network_turbo
/root/miniconda3/bin/python -m venv /root/autodl-tmp/credipred/bootstrap/uv
/root/autodl-tmp/credipred/bootstrap/uv/bin/python -m pip install uv==0.11.29
export UV_CACHE_DIR=/root/autodl-tmp/credipred/cache/uv
export UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/credipred/env/credipred-cu128
cd /root/autodl-tmp/credipred/source/credipred
/root/autodl-tmp/credipred/bootstrap/uv/bin/uv sync \
  --frozen --group dev --no-install-package torch
```

- [ ] **Step 4: Install the Blackwell-compatible Torch/PyG overlay**

Download and hash the exact Linux CPython 3.10 wheels into `wheelhouse/`, then
install them into `credipred-cu128`:

- PyTorch `2.7.1+cu128` from the official PyTorch CUDA 12.8 index.
- `pyg_lib==0.4.0+pt27cu128`.
- `torch_scatter==2.1.2+pt27cu128`.
- `torch_sparse==0.6.18+pt27cu128`.
- `torch_geometric==2.6.1`.

After this overlay, execute only
`/root/autodl-tmp/credipred/env/credipred-cu128/bin/python`. Do not use
`uv run`, because it may resynchronise the locked CUDA 12.6 Torch build.

- [ ] **Step 5: Capture the environment receipt**

The receipt includes wheel SHA-256 digests, `uv pip freeze`, Python version,
Torch/PyG/extension versions, `torch.version.cuda`, `torch.__config__.show()`,
`torch.cuda.get_arch_list()`, GPU name/UUID, driver, cgroup limits,
`/dev/shm`, disk bytes, `tmux -V`, soft/hard open-file limits, and the exact
setup script hash.

Hard acceptance:

```text
torch.version.cuda == "12.8"
torch.cuda.is_available() is True
"sm_120" is present in torch.cuda.get_arch_list()
visible CUDA device count == 1
visible GPU UUID matches the preflight receipt
```

### Task 3: Freeze a Clean Source Snapshot and Stage the Frozen Inputs

**Files:**

- Consume:
  `thesis/overleaf-rewrite/data/manifests/credigraph_dec2024.json`
- Consume:
  `CrediPred` branch `codex/fix-topology-correction-loss`
- Create:
  `CrediPred/experiments/autodl_formal_credipred/protocol/source_receipt.json`
- Create remotely:
  `/root/autodl-tmp/credipred/inputs/dec2024_pc1/INPUTS.sha256`

**Interfaces:**

- Consumes: a clean implementation commit and two frozen tensors.
- Produces: verified source and input trust roots on AutoDL.

- [ ] **Step 1: Preserve and review the current CrediPred state**

Start from
`9c21d739852d315d41bc4f3cc6595737d9a58032`. Preserve the existing untracked
checklist. Apply only the formal-path changes in Tasks 4--7, then bind the
resulting clean full commit and source bundle hash.

- [ ] **Step 2: Resolve an authenticated source-to-AutoDL data route**

The frozen source URIs are on `/data-gauss`. The local laptop does not have
enough free space to stage both tensors. Use a route that does not copy a
private key onto another host and does not invent a public URL. If no such
route is available, stop and report this as the input-transport blocker.

- [ ] **Step 3: Stage `data.pt` and verify it**

Required acceptance:

```text
size = 28120435333 bytes
sha256 = 0c09d2b5aeefae0c5f5306be19dbf0ee9b1b4df85ea02436f14da2280487aff1
```

Training may begin with `data.pt` alone.

- [ ] **Step 4: Stage `mapping.pt` before full export**

Required acceptance:

```text
size = 1483903149 bytes
sha256 = 321e90b87aa3e1af4cb59e513b483970f9843f68019920b39f609309ea4e3927
```

- [ ] **Step 5: Validate structure from the verified tensor**

Load only the hash-verified `data.pt` and record 45,030,252 nodes,
1,014,523,551 directed edges, feature/output dimensions, mask counts, and split
hashes. Do not use the historical inventory log as the sole runtime binding.

### Task 4: Close the Known CrediPred Correctness Gaps

**Files:**

- Modify:
  `credipred/experiments/gnn_experiments/regression_uq_experiment.py`
- Modify:
  `credipred/experiments/gnn_experiments/topology_correction_experiment.py`
- Modify:
  `credipred/conformal_regression/cqr.py`
- Modify:
  `credipred/conformal_regression/correction_gnn.py`
- Modify:
  `credipred/experiments/gnn_experiments/main.py`
- Modify:
  `credipred/utils/save.py`
- Create:
  `credipred/conformal_regression/correction_mlp.py`
- Create:
  `credipred/experiments/gnn_experiments/formal_cqr.py`
- Create:
  `credipred/experiments/gnn_experiments/formal_artifacts.py`
- Test:
  `test/test_no_test_leakage.py`
- Test:
  `test/test_formal_cqr.py`
- Test:
  `test/test_correction_mlp.py`
- Test:
  `test/test_formal_artifacts.py`

**Interfaces:**

- Consumes: existing corrected-loss base/topology code.
- Produces: leakage-safe base and matched correction training with immutable,
  run-local lineage.

- [ ] **Step 1: Remove epoch-time test access**

Write a focused failing test that spies on the test evaluator. Base, no-edge,
and topology training must use train and validation only. The formal
1,694-node population may be read only after the selected checkpoint is
immutable and hashed.

- [ ] **Step 2: Hard-fail missing checkpoints and remove shared outputs**

Reject `best_state_dict is None`. Route logs, checkpoints, predictions,
metrics, loss traces, and manifests under the current run root. Disable or
replace `save_loss_results()` fixed-path output so no run can overwrite another.

- [ ] **Step 3: Bind every cached parent artifact**

`base_preds.pt` is reusable only when its sibling manifest matches the exact
base checkpoint SHA, source SHA, config SHA, input SHA, alpha, seed, shape,
dtype, and row-order contract. A filename match alone is never sufficient.

- [ ] **Step 4: Add deterministic two-fold Standard CQR**

Create folds by a stable hash of formal node ID, namespace, and seed 42.
Requirements:

- the folds are disjoint;
- every one of the 1,694 nodes is evaluated exactly once;
- each fold calibrates the other;
- fold membership and qhat values are hashed;
- repeated execution is byte-stable;
- a deployment qhat is fitted on all 1,694 nodes only after cross-fitted
  evaluation is complete and is labelled separately from evaluation qhats.

- [ ] **Step 5: Add the budget-matched no-edge correction**

Implement a three-input additive `CorrectionMLP` that never receives
`edge_index`, degree, neighbourhood aggregates, node identity, or any other
topology-derived feature. Mechanically choose its hidden width so its trainable
parameter count is within 1% of the topology correction when possible.

Match topology exactly on:

- parent base checkpoint and prediction hash;
- training/size partitions;
- optimizer, learning rate, weight decay;
- epoch and optimiser-step budget;
- batch order and seed;
- midpoint/non-crossing/size-loss semantics;
- validation checkpoint rule;
- formal CQR folds and metrics.

- [ ] **Step 6: Implement the approved endpoint-only correction contract**

Under the recommended decision, correction models output only lower/upper
deltas and leave midpoint head 0 unchanged. Tests assert midpoint identity to
the frozen parent for no-edge and topology.

- [ ] **Step 7: Add canonical run manifests**

Each manifest binds arm, alpha, seed, baseline and implementation commits,
input/config/split hashes, checkpoint and parent hashes, run root, output
schema, software/CUDA/GPU identity, deterministic settings, command, start/end
UTC, exit code, and every emitted artifact digest.

### Task 5: Implement a Complete, Sharded 45M Export

**Files:**

- Create:
  `credipred/experiments/gnn_experiments/export_full_graph.py`
- Test:
  `test/test_full_graph_export.py`
- Modify:
  `credipred/experiments/gnn_experiments/main.py`

**Interfaces:**

- Consumes: one predeclared producer checkpoint, frozen base predictions,
  deployment qhat, and verified graph.
- Produces: complete raw and policy-ready node-order exports.

- [ ] **Step 1: Make toy-graph completeness fail before implementation**

Tests cover missing IDs, duplicate IDs, out-of-range IDs, interrupted shard
writes, non-finite values, wrong parent hash, and a successful out-of-order
loader whose final output is node-ID ordered.

- [ ] **Step 2: Export every node exactly once**

Do not clone base predictions and update only labelled rows. Stream or shard
all node IDs through the selected producer. Track a completeness bitmap or an
equivalent fail-closed index contract.

- [ ] **Step 3: Preserve raw and derived semantics**

Write separately manifested raw predictions and policy-ready predictions. The
policy-ready tensor uses the frozen non-crossing rule and deployment qhat. Each
tensor has:

```text
shape = [45030252, 3]
column order = [midpoint, lower, upper]
dtype = float32
row order = node ID
```

Record shard ranges, per-shard hashes, aggregate hash, raw crossing diagnostics,
midpoint-outside diagnostics, parent checkpoint, CQR receipt, and mapping hash.

- [ ] **Step 4: Make interrupted writes non-promotable**

Write to attempt-local temporary paths. Emit `COMPLETED.json` only after all
shards, row counts, hashes, shape, dtype, finiteness, and parent lineage pass.

### Task 6: Run Focused Verification and the Minimum Real-Batch Preflight

**Files:**

- Create:
  `CrediPred/experiments/autodl_formal_credipred/run_contract_preflight.sh`
- Produce remotely:
  `/root/autodl-tmp/credipred/protocol/preflight/real_batch_receipt.json`

**Interfaces:**

- Consumes: final source/environment/input snapshots.
- Produces: non-result compatibility evidence.

- [ ] **Step 1: Run focused tests with the direct environment Python**

```bash
/root/autodl-tmp/credipred/env/credipred-cu128/bin/python -m pytest -q \
  test/test_quantile_loss.py \
  test/test_checkpoint.py \
  test/test_no_test_leakage.py \
  test/test_formal_cqr.py \
  test/test_correction_mlp.py \
  test/test_formal_artifacts.py \
  test/test_full_graph_export.py
```

- [ ] **Step 2: Run the full relevant repository checks**

```bash
/root/autodl-tmp/credipred/env/credipred-cu128/bin/python -m pytest -q
/root/autodl-tmp/credipred/env/credipred-cu128/bin/python -m ruff check .
/root/autodl-tmp/credipred/env/credipred-cu128/bin/python -m ruff format --check .
/root/autodl-tmp/credipred/env/credipred-cu128/bin/python -m mypy credipred
git diff --check
```

- [ ] **Step 3: Run one real-data CUDA/PyG batch**

From the hash-verified `data.pt`, create one real `NeighborLoader` batch and run
base forward/backward plus no-edge and topology correction forward/backward.
Assert finite loss/gradients, real CUDA placement, expected output shape, and
successful PyG sampling/extension calls. Do not save a model, benchmark
throughput, report a metric, or reuse the batch output.

Keep `num_workers=4` and persistent workers initially. Reduce to 1 or 0 only if
this real-batch contract fails, record the observed failure, and apply the
smallest configuration change in a new receipt.

### Task 7: Freeze Run-Local P0 Configs and the Pre-Launch Receipt

**Files:**

- Consume templates:
  `configs/gnn/quantile/gat_domainrel_quantile_dec_corrected_loss.yaml`
- Consume templates:
  `configs/gnn/quantile/gat_domainrel_topology_correction_dec_corrected_loss.yaml`
- Create per run:
  `runs/${P0_RUN_ID}/configs/resolved.yaml`
- Create:
  `protocol/receipts/p0_pre_execution_receipt.json`

**Interfaces:**

- Consumes: approved research decisions and green contract checks.
- Produces: immutable formal launch inputs.

- [ ] **Step 1: Resolve configs without editing the reusable templates**

Every P0 run-local config sets:

```yaml
MetaArguments:
  processed_location: "../inputs/dec2024_pc1"
  is_scratch_location: false
  global_seed: 42
ExperimentArguments:
  exp_args:
    GAT:
      model_args:
        quantile_alpha: 0.10
        runs: 1
```

Set `weights_directory` and `log_file_path` to the unique run root. Preserve
the existing base 100-epoch and correction 200-epoch schedules initially.
Set `WANDB_MODE=offline`.

- [ ] **Step 2: Recheck only volatile launch state**

Immediately before launch:

1. inspect Noether and AutoDL for an equivalent canonical job/output writer;
2. recheck actual CUDA visibility and the visible GPU UUID;
3. recheck GPU memory/process state and resolve the unexplained roughly
   21-GiB allocation;
4. verify `tmux`, cgroup CPU/RAM, `/dev/shm`, data-disk free bytes, source
   cleanliness, and input hashes.

If the duplicate-job conflict exists, the GPU allocation remains unexplained,
or any identity differs, stop and report rather than launch.

- [ ] **Step 3: Seal the receipt before opening outcomes**

The receipt includes the three research decisions, producer rule, source/input/
environment/config/seed/output-root hashes, exact commands, effective resource
limits, and explicit `formal_execution_authorized=true`.

### Task 8: Execute the Three P0 Producer Units

**Files:**

- Create:
  `CrediPred/experiments/autodl_formal_credipred/launch_p0.sh`
- Produce:
  `/root/autodl-tmp/credipred/runs/${P0_RUN_ID}/`

**Interfaces:**

- Consumes: sealed receipt and direct CUDA 12.8 environment.
- Produces: base, Standard CQR, matched no-edge, and topology artifacts.

- [ ] **Step 1: Launch corrected-loss base in a new `tmux` session**

The run ID encodes UTC, `p0-base`, alpha `a10`, seed `s42`, and source SHA.
The immutable `run.sh` sources `/etc/network_turbo`, sets CPU thread limits
from the 25-CPU cgroup quota, raises the soft open-file limit only within the
hard limit, records the environment, and invokes the direct environment Python.

- [ ] **Step 2: Freeze and hash the selected base checkpoint**

Only validation quantile loss selects the checkpoint. After it is immutable,
produce all-node base predictions, evaluate the formal population once through
the two-fold Standard CQR path, and fit the separately labelled deployment
qhat.

- [ ] **Step 3: Launch matched no-edge and topology runs from the exact parent**

Use two new run roots and two new `tmux` sessions. Both manifests must reference
the exact same base checkpoint and base-prediction hashes. Do not share mutable
files between correction runs.

- [ ] **Step 4: Validate matched outputs without outcome-dependent repair**

For each producer report raw crossing rates, lower/mid/upper ordering
violations, cross-fitted coverage and exact binomial interval, mean/median/
predeclared width quantiles, qhat per fold, midpoint MAE as secondary, and
paired width/MAE contrasts on identical nodes. Preserve negative outcomes.

- [ ] **Step 5: Apply the predeclared producer rule**

If topology is artifact-valid, proceed with its 45M export regardless of the
direction of the matched result. If it is artifact-invalid, stop; do not choose
the best-looking alternative.

### Task 9: Produce and Validate the Required 45M Export

**Files:**

- Produce under the selected run:
  `predictions/full_graph/`
- Produce:
  `manifests/full_graph_export.json`
- Produce:
  `receipts/full_graph_export_completion.json`

**Interfaces:**

- Consumes: selected producer and deployment CQR artifact.
- Produces: one complete CrediSearch-consumable export.

- [ ] **Step 1: Run the sharded exporter in a new `tmux` session**

Use a new export attempt root even though the parent training run is frozen.
Do not overwrite training outputs.

- [ ] **Step 2: Verify all-node completeness and hashes**

Acceptance requires exactly 45,030,252 unique rows, no missing/duplicate node,
finite float32 values, correct column semantics, valid shard/aggregate hashes,
matching input/mapping/checkpoint/CQR parents, and a terminal completion receipt.

- [ ] **Step 3: Persist compact metadata locally**

Pull back manifests, receipts, configs, metrics, and logs. Keep large tensors on
the data disk but record their exact remote paths, byte sizes, and SHA-256
digests.

### Task 10: Stabilise CrediSearch and Implement the New Three-Arm Path

**Files:**

- Review existing staged CrediSearch work before editing.
- Reuse:
  `src/credisearch/data/averitec.py`
- Reuse:
  `src/credisearch/retrieval.py`
- Reuse:
  `src/credisearch/evaluation.py`
- Reuse:
  `src/credisearch/canonicalize.py`
- Reuse:
  `src/credisearch/domains.py`
- Reuse/extend with a new schema:
  `src/credisearch/graph_lookup.py`
- Create:
  `src/credisearch/reranking.py`
- Create:
  `src/credisearch/reranking_artifacts.py`
- Create:
  `src/credisearch/arm_generation.py`
- Create:
  `tests/test_three_arm_reranking.py`
- Create:
  `tests/test_reranking_artifacts.py`
- Create:
  `tests/test_arm_generation.py`

**Interfaces:**

- Consumes: frozen AVeriTeC snapshot and CrediPred export.
- Produces: independent, manifested three-arm contexts and outputs.

- [ ] **Step 1: Form a clean CrediSearch source trust root**

Review and preserve the current staged WIP on `codex/v2-uq-gate@0a851dd`.
Create a clean commit before adding the three-arm path. Do not rely on stale
`docs/STATUS.md` or `docs/HANDOFF.md` over thesis manifests.

- [ ] **Step 2: Create an independent Python 3.11 environment**

Use `uv==0.11.29` and the frozen CrediSearch lock in
`/root/autodl-tmp/credisearch/env/credisearch-py311`. Do not install or import
CrediPred. Add a GPU generation backend only after the model contract is
frozen.

- [ ] **Step 3: Add a new export schema**

Do not relabel the existing `legacy_lower_proposal`/
`legacy_upper_proposal` contract as formal corrected quantiles. Add a versioned
schema naming the producer, raw/policy-ready semantics, CQR receipt, midpoint,
lower, upper, support, and all parent hashes.

- [ ] **Step 4: Reuse sparse graph lookup**

Verify the full export and mapping hashes, memory-map on CPU, and select only
the AVeriTeC target node rows. Preserve exact-host preference and the frozen
registrable-domain fallback. Emit a sparse overlay with missing, invalid,
ambiguous, available, and supported states separated.

- [ ] **Step 5: Implement deterministic three-arm permutations**

Base preserves original rank. Midpoint and Conservative use separately frozen
scoring maps, with original rank as stable tie-break. Missing, unseen, invalid,
or unsupported signal contributes zero, never deletes a candidate, and
preserves original order among no-contribution candidates.

- [ ] **Step 6: Do not reuse the old replay generation semantics**

Existing `replay.py` may supply retrieval-state helpers, but its reenrichment/
shared-output design cannot generate the new end-task arms. Each arm gets its
own context, request, response, parsed output, evaluator output, failure
record, and manifest.

### Task 11: Freeze the AVeriTeC Protocol Before Full-Cohort Generation

**Files:**

- Consume:
  `thesis/overleaf-rewrite/data/manifests/averitec_source_v1.json`
- Modify:
  `thesis/overleaf-rewrite/data/manifests/protocol_reranking_averitec_crag_v1.json`
- Create in CrediSearch:
  `protocol/averitec_three_arm_v1.json`
- Create:
  `protocol/averitec_pre_execution_receipt.json`

**Interfaces:**

- Consumes: source manifests and training-only/pre-final evidence.
- Produces: a hash-bound executable AVeriTeC contract.

- [ ] **Step 1: Freeze the candidate/content snapshot**

Bind claim membership, raw candidates, original ranks, fetched content,
content hashes, URL/domain canonicalisation, duplicates, short/empty pools,
mapping rules, and candidate/content population hashes. No arm may fetch a
replacement or alter membership.

- [ ] **Step 2: Freeze Midpoint and Conservative scoring**

Bind exact midpoint extraction, interval-derived Conservative formula,
normalisation, relevance-credibility fusion, direction, support use, zero
contribution, and tie handling. Record whether Conservative is strictly nested
within Midpoint and differs only by the interval/support adjustment. If not,
label the secondary contrast a composite conservative-policy contrast.

Do not assume that Conservative means the lower bound merely because a lower
endpoint exists.

- [ ] **Step 3: Freeze context and generation**

Bind top-k, serialisation and truncation, model/revision, prompt, decoding,
seed, retry/timeout, parser, per-instance and aggregate budget, and the common
generation-budget identity shared across arms.

- [ ] **Step 4: Freeze evaluator and inference**

Bind the official AVeriTeC evaluator repository/commit, reference and input
hashes, schemas, native denominator, invalid/missing handling, fail-closed
rules, paired resampling unit/algorithm/count/seed/multiplicity/aggregation,
and source-exposure audit conventions.

- [ ] **Step 5: Complete the prior-access audit**

The pinned 500-record dev cohort is final only if prior gold/evaluator-output
access can be ruled out in a documented audit. Otherwise label the analysis
exploratory or freeze an independent final population. The 2,215-record public
test is excluded because no matching candidate/content archive is bound.

- [ ] **Step 6: Seal the protocol**

Set the protocol hash, `frozen_at`, source/config/input/output-root hashes, and
pre-execution receipt before opening final outcomes. If any unique research
choice remains unresolved, stop and report it.

### Task 12: Execute the Full-Cohort AVeriTeC Three-Arm Study

**Files:**

- Produce one immutable run root per arm.
- Produce shared candidate/content and overlay manifests.
- Produce arm-specific context, request/response, parser, evaluator, and output
  manifests.

**Interfaces:**

- Consumes: the sealed AVeriTeC protocol.
- Produces: full-cohort arm outputs and official metrics.

- [ ] **Step 1: Validate contracts with deterministic fixtures**

Run unit/contract tests only. Do not run a 10-claim or other outcome-bearing
subset as a smoke experiment.

- [ ] **Step 2: Materialise all three permutations and contexts**

Verify identical claim and candidate membership across arms, deterministic
byte-stable reranking, stable ties, no silent deletion, and shared budget
identity. Contexts remain arm-specific.

- [ ] **Step 3: Generate each arm independently**

Run full-cohort Base, Midpoint, and Conservative generation in separate
immutable run roots. Record every model call, token/context accounting, retry,
parser failure, coded failure, and hash. Never copy generated/evaluator output
between arms.

- [ ] **Step 4: Run the bound evaluator independently per arm**

Exhausted evaluator failure invalidates an official run. Do not silently change
the denominator or drop failed instances.

- [ ] **Step 5: Compute paired contrasts and the source-exposure audit**

Primary contrasts are Midpoint minus Base and Conservative minus Base.
Midpoint minus Conservative is secondary. The exposure audit reuses the frozen
permutations and contexts and adds no fourth arm or model calls.

### Task 13: Archive Evidence, Update the Thesis, and Decide P1

**Files:**

- Modify after validation:
  `thesis/overleaf-rewrite/notes/research_ledger.md`
- Modify after validation:
  `thesis/overleaf-rewrite/data/results_used.csv`
- Modify result prose/tables only from verified manifests.
- Update CrediPred and CrediSearch status/handoff documents.

**Interfaces:**

- Consumes: completion receipts and verified compact artifacts.
- Produces: evidence-backed thesis updates and an explicit P1 decision.

- [ ] **Step 1: Validate the complete artifact graph**

Trace every paper number back through evaluator output, arm manifest, protocol,
candidate/content snapshot, sparse overlay, full export, producer checkpoint,
source commit, environment, and frozen inputs.

- [ ] **Step 2: Record scientific state without overclaiming**

Keep Job 250229 as its own negative boundary. Label P0 single-seed topology
evidence as preliminary unless later independent seeds support the claim.
Never interpret scoreability as validity or domain credibility as truth.

- [ ] **Step 3: Update paper tables and prose**

Populate only verified cells. Preserve null/harmful outcomes and failures.
State whether the AVeriTeC cohort is final or exploratory based on the access
audit.

- [ ] **Step 4: Open P1 only after P0 stability**

P1 may add alpha `.05/.20`, seeds `43/44`, the complete 27-unit producer
matrix, additional 45M exports only if preregistered, and CRAG under its own
frozen evaluator/metric contract. P1 does not retroactively alter P0.

## Definition of Done

P0 is complete only when:

1. the author-approved producer decisions and export rule are receipt-bound;
2. the effective-resource receipt reports 25 CPUs, 120 GiB RAM, 60 GiB
   `/dev/shm`, 450 GiB data disk, and the single visible Blackwell GPU;
3. `tmux`, the isolated CUDA 12.8 environment, PyG extensions, and real-batch
   preflight pass;
4. source, inputs, configs, seeds, checkpoints, parents, outputs, and run roots
   have exact hashes and no duplicate writer;
5. base, two-fold Standard CQR, no-edge, and topology units terminate with
   leakage-safe manifests;
6. one predeclared producer has a complete, immutable
   `[45030252, 3]` all-node export;
7. the AVeriTeC candidate/content, reranking, generation, evaluator, inference,
   and access-audit contracts are frozen before outcomes;
8. Base, Midpoint, and Conservative complete the same full cohort with
   independent generated and evaluator outputs;
9. primary paired contrasts and source-exposure artifacts validate;
10. the thesis ledger and result tables cite only manifest-backed evidence;
11. no `.05/.20`, seed `43/44`, 27-unit, CRAG, or second-model work is reported
    as part of P0.
