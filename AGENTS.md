# AGENTS.md

Guidance for AI coding agents working in this repo. Read this once at the start
of a session; the conventions and gotchas here will save you from common
mistakes.

## What this repo is

Image metric-learning training and evaluation framework. Hydra for config
composition, Accelerate for distributed training. The framework targets two
loosely related domains:

- **Product / retrieval** (Cars196, SOP, InShop, Products10k, RP2K, …) — recall@k
  is the success metric.
- **Face recognition** (IResNet + AdaFace/MagFace/PartialFC/KP-RPE) — evaluated
  with TAR@FAR, LFW/AgeDB/CFP-style pair protocols, IJB templates, and
  MegaFace-style identification metrics.

Everything is configurable via YAML and a single classification-style training
loop. Pair and proxy losses are supported as embedding losses; pair/batch losses
need a positive-pair sampler such as `dataloader=pk`.

## Repo layout

```
src/
  data/            DataLoader + Dataset wiring (BaseDataset, MXDataset)
  evaluator/       Embedding eval: base, DDP, KNN (sklearn + FAISS),
                   k-reciprocal re-ranking, face verification, IJB, MegaFace
  experiment_tracker/  W&B, MLflow
  loss/            Classification, pair/proxy losses + paper-specific
                   (UniFace UCE, TopoFR, TransFace EHSM, Circle)
  metric/          recall@k, GAP
  model/
    backbone/      timm dispatcher + custom (IResNet, MobileFaceNet, EdgeFace,
                   KP-RPE, RADIO, UniCom). NOTE: dispatch uses family first,
                   then prefix/type inference — see gotchas.
    head/          Embedding head (Linear + BN)
    margin/        13 angular-margin heads (Arc, Cos, AdaFace, MagFace,
                   PartialFC, …)
    modules/       GeM pooling, KP-RPE bias
    model.py       EmbeddingsNet + MLNet (backbone+head+margin)
  optimizer/       Hydra-dispatched (SGD/Adam/AdamW/Lion/RAdam/Sophia-G)
  sampler/         MPerClassSampler, PK, class-balanced, hard-negative
  scheduler/       Cosine / cyclic / multistep / plateau (+ pytorch_warmup)
  trainer/         base.py, ddp.py, distill.py + helpers
                   (margin_regularization.py, loss_inputs.py, targets.py,
                   dpap.py, hard_negative_cache.py, model_averaging.py, xbm.py)
  transform/       albumentations transforms + mixers (mixup, cutmix, DPAP)
  pca.py           Optional PCA over embeddings (offline)
  infer.py         Single-image inference helper

tools/             Hydra entry points (train.py, test.py) + utilities
data/              Dataset preparation scripts (download, filter, split, merge)
configs/           Hydra configs. Composed from config_train.yaml /
                   config_test.yaml / config_pca.yaml.
tests/             pytest. Most tests `importorskip("torch")` so they're
                   skipped in lightweight envs.
pipelines/         Shell scripts orchestrating prepare → train → eval flows
```

## Environment

Conda env name is **`eml`**:

```bash
conda env create -f environment.yml      # one-time
conda activate eml
```

If torch is missing, tests in `tests/` will be **skipped**, not failed — don't
mistake a green run in a broken env for passing tests. Sanity check:

```bash
python -c "import torch; print(torch.__version__)"
```

## Common commands

```bash
# Train (single GPU, Hydra overrides)
python tools/train.py backbone=openclip_vit_b32 dataset=inshop \
    margin=arcface batch_size=32 epochs=10

# Distributed training (configure once with `accelerate config`)
accelerate launch tools/train.py backbone=iresnet100 margin=partialfc_arcface

# Evaluate a checkpoint
python tools/test.py weights=path/to/ckpt.pth evaluation/data=cars196

# Tests
pytest                              # full suite
pytest tests/test_margin_heads.py   # one file
pytest -k "magface"                 # by keyword

# Formatting (black is the only formatter — line length 88, py310 target)
black src/ tests/ tools/
```

Entry points are Hydra apps with `config_path=../configs/`, so overrides use
dotted keys: `python tools/train.py train.trainer.grad_accum_steps=8`.

## How to add a new component

The framework is plugin-style. To add anything:

1. **Implement** the module in `src/<subsystem>/`.
2. **Register** it in `src/<subsystem>/__init__.py` factory function (e.g.
   `get_margin`, `get_loss`, `get_optimizer`). Most factories use a type-name
   string dispatched via `if/elif`.
3. **Add a config** in `configs/<subsystem>/<name>.yaml` mirroring the
   constructor kwargs.
4. **Add a test** in `tests/test_<subsystem>_*.py`. The existing parametrized
   matrix tests (e.g. `tests/test_margin_heads.py::test_all_registered_margin_heads_run_on_cpu`)
   are the canonical pattern — add your type to the list and provide any new
   kwargs in `margin_config()`.
5. **Document** in README.md margins/losses/backbones table.

For losses specifically, `src/loss/__init__.py::get_loss` instantiates via
`hydra.utils.instantiate` from a `_target_` field, so the config alone wires
the class — no `if/elif` to edit.

## Coding conventions

- **Python 3.10+**.
- **Black, line length 88** (configured in `pyproject.toml`). Run before commit.
- **No emojis** in code, comments, or commit messages unless the user asks.
- **No new comments** unless they explain a non-obvious *why*. Don't restate
  what the code does.
- **No new markdown docs** unless the user explicitly asks (README/CHANGELOG
  are fine to update; don't create planning/analysis files).
- **Prefer editing existing files** over creating new ones.
- **Keep PRs surgical** — don't refactor surrounding code while fixing a bug.
- Internal code is trusted; only validate at boundaries (user input, file I/O,
  external APIs).

## Testing conventions

- `pytest` with `pythonpath = .` (so `from src.foo import bar` works).
- Every torch-dependent test starts with `torch = pytest.importorskip("torch")`
  so the suite degrades gracefully when torch isn't installed.
- Heavy tests (DDP, full training loop) live alongside unit tests; there is no
  separate integration/e2e tier yet.
- Parametrized matrix tests are preferred for plugin families (margins, losses,
  backbones) — append new types rather than writing per-type test files.

## Architecture notes the code won't tell you

- **`MLNet.forward(x, label)` returns `(margin_logits, embeddings)`** — both are
  used downstream. The margin module is `model.margin`; under DDP it's
  `accelerator.unwrap_model(model).margin`.
- **Margin regularizers** (e.g. MagFace) expose `regularization_loss()` and
  `regularization_loss_name`. The trainer auto-discovers and adds these via
  `src/trainer/margin_regularization.py`. To add a new regularizer-bearing
  margin, just implement those two attributes on the head — no trainer
  changes needed.
- **Hydra defaults list** in `configs/config_train.yaml` controls what's
  composed. Adding a new backbone YAML alone doesn't activate it; users pass
  `backbone=<name>` to override.
- **`config.n_classes` is mutated at runtime** by `tools/train.py` after
  reading dataset stats. Don't rely on the value being correct at config-load
  time.
- **Backbone factory creates timm models with `num_classes=0`** (feature
  extractor mode) and reads `model.num_features`. Old checkpoints saved before
  this change had `backbone.head.*` weights that no longer exist — loading
  them with `strict=True` will fail. Use `strict=False` in `load_checkpoint`
  or re-save.

## Known gotchas (the things that will bite you)

1. **Backbone dispatcher prefers explicit `family:`** in backbone configs
   (`src/model/backbone/__init__.py`). If `family` is missing, it falls back to
   prefix/type inference for OpenCLIP, UniCom, IResNet, MobileFaceNet, EdgeFace,
   KP-RPE, RADIO, then timm. New backbone configs should declare `family`.
2. **timm `scriptable` is config-driven.** Several modern models (DINOv2,
   Hiera, SigLIP) can fail when `scriptable=True` under some timm versions, so
   leave it false unless you need TorchScript compatibility.
3. **`MagFace.m` is a tuple `(l_margin, u_margin)`**, not a scalar. It does
   not support `incremental_margin` and the trainer raises a clear error if
   you try. `_get_margin_value()` will log the tuple to W&B oddly — cosmetic.
4. **MagFace short-circuits in `get_margin`** before the `dynamic_margin`
   branch. It intentionally does not expose `dynamic_margin` or
   `incremental_margin` config keys.
5. **`config.head.type == "no_head"`** rewrites `config.embeddings_size` at
   runtime to the backbone's output features. Code reading `config.embeddings_size`
   before model construction sees a stale value.
6. **DDP unwrap is explicit** — `self.accelerator.unwrap_model(self.model)`.
   Forgetting this when reading `model.margin` attributes works in eager but
   fails after `accelerator.prepare()`.
7. **Mixup/CutMix targets are `(label1, label2, lam)` tuples**, not tensors.
   Margin heads must handle this via `src/model/margin/utils.py::build_one_hot`.
   New margins should call that helper rather than rolling their own one-hot.
8. **XBM is local per DDP rank.** `train.trainer.xbm.enabled=True` extends
   embedding losses with a per-process queue. It does not all-gather embeddings
   into a synchronized cross-rank memory bank.
9. **k-reciprocal re-ranking is dense `O(N^2)` memory.** Keep
   `evaluation.knn.rerank.enabled=False` for million-scale galleries unless a
   chunked variant is added.
10. **AM-RADIO uses TorchHub.** The RADIO backbone loads through NVIDIA's
    TorchHub entrypoint and needs network/cache access to the repo and weights.
11. **MobileFaceNet pretrained weights are not bundled.** The local
    implementation supports `checkpoint_path`; set it to external weights if
    you need pretrained MobileFaceNet.
12. **`configs/old/`** is deprecated — don't extend it, and treat anything that
   only references it as dead code.
13. **`pipelines/`** contains shell scripts that hardcode paths to a previous
   workstation. They are reference material, not executable from a fresh
   checkout.

## Known gaps (good places to contribute)

These are documented so agents don't waste time hunting for them or
re-implementing partial versions:

- **No bundled official face benchmark assets**. `data/face_benchmarks.py`
  normalizes LFW/AgeDB, CFP-style, IJB, and MegaFace layouts, but users still
  need to obtain the datasets and aligned images themselves.
- **No distributed-grade hard-negative cache service**. The trainer can refresh
  local hard-negative caches, but there is no asynchronous multi-node cache.
- **No hosted tracker integration tests**. W&B, MLflow, TensorBoard, Aim, and
  Neptune are wired, but tests use local doubles for optional hosted services.
- **No optional GPU CI workflow**. CPU pytest, formatting, and default config
  validation run in GitHub Actions.

If asked to add one of these, follow the "How to add a new component" recipe
above and update this file's gaps list.

## When in doubt

- Read the existing implementation of the closest-analog module before
  designing a new one (e.g. before adding a new margin, read
  `src/model/margin/arcface.py` end-to-end).
- For Hydra wiring questions, `configs/config_train.yaml` is the source of
  truth for what gets composed.
- For trainer plumbing, read `src/trainer/base.py` first; `ddp.py` and
  `distill.py` are variations on its structure.
