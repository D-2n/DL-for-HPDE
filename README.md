# HypNO — Hyperbolic Neural Operators for Traffic-Flow PDEs

Reference implementation and full experimental pipeline for **HypNO**, a space–time
graph neural operator for one-dimensional hyperbolic conservation laws with
discontinuous solutions, together with every baseline, dataset generator and
evaluation script used in the accompanying paper.

The operator maps an initial condition directly to the **entire space–time
solution field** in a single forward pass — no time stepping, no CFL restriction
at inference — while retaining the structural ingredients that make classical
finite-volume schemes work on shocks: characteristic speeds, upwind direction,
Rankine–Hugoniot jump conditions and an Oleinik/Lax entropy gate, all injected as
*edge features and multiplicative gates* rather than as soft penalties.

Two PDEs are covered:

| Branch | Equation | State | Ground truth | Model |
|---|---|---|---|---|
| **LWR** | scalar Lighthill–Whitham–Richards, `f(ρ)=ρ(1−ρ)` | `ρ` | exact **Lax–Hopf** | `HypNO-ST3` (`models/hypno_st3.py`) |
| **ARZ** | Aw–Rascle–Zhang system with relaxation | `(ρ, ω)` | **wave-front tracking** / exact Riemann | `HypNO-ARZ` (`arz/model_arz_orig.py`) |

Both use the same backbone; they differ only in the edge-feature set and the
physics gates. Shared discretisation for every model and every experiment:
`nx = nt = 128`, `x ∈ [−1, 1]`, `t ∈ [0, 1]`, ghost-cell (transmissive)
boundaries, and — for ARZ — pressure form **`p(ρ) = ρ`**.

---

## Method in one paragraph

Each sample is a space–time graph whose nodes are the grid points `(xᵢ, tⁿ)`.
A lifting network embeds the local state plus derived physical quantities
(flux, characteristic speed, velocity, relaxation residual) into a latent vector
per node. Twelve message-passing layers then exchange information over a
space–time stencil of half-widths `k_x`, `k_t`, with **two structurally separate
edge MLPs**: adjacent edges (`|Δi| = 1`, same time level) carry genuine *interface*
quantities — jumps `Δρ, Δv, Δω`, characteristic speeds `λ₁, λ₂`, upwind and
entropy indicators — while non-adjacent edges carry only node-local and geometric
features. A multiplicative physics gate built from the upwind direction, the
Oleinik entropy condition and a numerical-domain-of-dependence (characteristic
cone) test re-weights the messages, and aggregation is gate-normalised. A decoder
reads out the full field. Training minimises state MAE plus a discrete
conservation term and a deep-supervision probe.

---

## Repository layout

```
hyperbolic_pde/
  models/
    hypno_st3.py                  HypNO-ST3 — the LWR model (paper architecture)
    hypno_st3_2d.py               2-D extension (exploratory, out of paper scope)
    competitive_architectures/    FNO and DeepONet operator baselines
    pinn_methods/                 PINN / VPINN + shock-detector baselines
    legacy/                       superseded model versions, kept for provenance
  data/
    fvm.py                        Godunov / MUSCL / WENO5 finite volumes + dataset generation
    lax_hopf.py                   exact Lax–Hopf solver (LWR ground truth)
    fvm_2d.py                     2-D finite volumes
  arz/
    model_arz_orig.py             HypNO-ARZ (paper architecture, frozen)
    model_arz{,_mark1,_mark2}.py  later ARZ variants — NOT in the paper
    physics_arz.py                ARZ flux, eigenstructure, pressure forms
    riemann_arz.py                exact ARZ Riemann solver
    wft_arz.py                    wave-front tracking (machine-precision ARZ ground truth)
    reference_arz.py              Strang-split FV reference (finite relaxation time τ)
    datagen_arz.py                stratified ARZ dataset generation
    precompute_baselines_parallel.py  parallel Godunov/HLL/WENO5 baseline cache
    arz_mark0_paper_eval.py       ARZ paper tables and figures
    eval_wilcoxon_cost_arz.py     paired Wilcoxon significance + cost comparison
    arz_conservation_report.py    conservation-defect report
    arz_lax_admissibility.py      Lax-admissibility report
    tests/                        physics, Riemann, WFT and model unit tests
  scripts/
    generate_data.py              LWR dataset generation
    train_hypno_st3.py            LWR training
    train_hypno_arz.py            ARZ training
    paper_eval.py                 LWR paper tables and figures
    shock_paper_export.py         shock-resolution study
    paper_figures.py              paper figure export
    lwr_conservation_report.py    conservation-defect report
    lwr_lax_admissibility.py      Lax-admissibility report
    eval_*.py                     ablations, OOD, super-resolution, scheme comparisons
  configs/                        YAML configs (base + per-experiment overrides)
  tests/                          solver and model unit tests
models_manifest.yaml              every released checkpoint: path, md5, architecture, recipe
papers/                           compiled manuscript
```

Standalone single-file reference solvers used for cross-checking live at the repo
root: `hll_arz_standalone.py`, `weno5_arz_standalone.py`,
`weno5_clip_arz_standalone.py`.

---

## Installation

```bash
git clone https://github.com/D-2n/DL-for-HPDE
cd DL-for-HPDE
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # torch, numpy, scipy, matplotlib, pyyaml
```

A CUDA GPU is needed for training (each model is only ~6 MB of weights, but the
space–time graph is memory-hungry); evaluation runs on CPU, slowly.

Everything is invoked as a module from the repo root, e.g.
`python -m hyperbolic_pde.scripts.train_hypno_st3`.

---

## Quickstart

### 1. Generate data

```bash
# LWR — exact Lax–Hopf ground truth
python -m hyperbolic_pde.scripts.generate_data \
    --config hyperbolic_pde/configs/hyperbolic_pde.yaml

# ARZ — wave-front-tracking ground truth, p(rho) = rho, paper grid
python -m hyperbolic_pde.arz.datagen_arz \
    --out data/arz_dataset.npz --pressure-form rho --fv-solver wft --tau inf \
    --N 5400 --nx 128 --nt 128 --x-min -1 --x-max 1 --t-max 1.0 \
    --families riemann_stratified,piecewise_constant_stratified,piecewise_sine \
    --segments 2,3,5,7,10 --use-exact-riemann --num-workers 16
```

Initial conditions are sampled from three families — `riemann_stratified`,
`piecewise_constant_stratified` and `piecewise_sine` (a sine *staircase*) —
stratified over the number of discontinuities. Held-out segment counts unseen in
training define the **OOD** split used throughout the evaluation.

### 2. Train

```bash
python -m hyperbolic_pde.scripts.train_hypno_st3 \
    --config hyperbolic_pde/configs/hyperbolic_pde.yaml

python -m hyperbolic_pde.scripts.train_hypno_arz \
    --config hyperbolic_pde/configs/hyperbolic_pde_arz_cleps_prho.yaml
```

Training is fully config-driven. Each run creates a run directory containing
`train.log`, periodic checkpoints, a training-curve plot and — importantly — a
**copy of the exact config used**, which is required to reload the checkpoint
(see the caveat below). `--resume_run <dir>` continues from the latest checkpoint.

### 3. Evaluate / reproduce the paper tables

```bash
# LWR: HypNO vs FNO vs WENO5 vs Godunov, stratified by IC family x #discontinuities
python -m hyperbolic_pde.scripts.paper_eval --run-dir <run_dir> --fno-weights <fno.pt>

# ARZ: same, plus both HypNO-ARZ variants
python -m hyperbolic_pde.arz.arz_mark0_paper_eval \
    --ckpt   hyperbolic_pde/runs/final/arz_wft.pt \
    --config <config paired with that checkpoint> \
    --data   data/arz_evaluation.npz \
    --out_dir results/arz_paper_eval

# paired Wilcoxon significance (Holm-corrected) + wall-clock cost
python -m hyperbolic_pde.arz.eval_wilcoxon_cost_arz --out-dir results/wilcoxon
```

Each evaluation writes `summary.csv` (long format), `summary.tex` (paper-ready
tabulars), `summary.txt` and per-cell qualitative figures — ground truth, every
solver, and `|solver − GT|` error maps side by side.

Numerical baselines are expensive, so precompute them once and reuse:

```bash
python -m hyperbolic_pde.arz.precompute_baselines_parallel \
    --data data/arz_evaluation.npz --baselines godunov,hll,weno5 --workers 20
# -> pass the resulting .npz to any eval via --baselines-npz
```

### 4. Structure-preservation reports

Beyond accuracy, the learned operators are checked for the two properties
numerical schemes get for free:

```bash
python -m hyperbolic_pde.arz.arz_conservation_report --ckpt ... --data ... --out_dir ...
python -m hyperbolic_pde.arz.arz_lax_admissibility   --ckpt ... --data ... --out_dir ...
```

- **Conservation defect** — spurious change in the space integral relative to the
  ground-truth boundary flux, normalised by initial mass (lower is better).
- **Lax admissibility rate** — fraction of true shock interfaces at which the
  prediction's characteristics still converge, i.e. no spurious expansion shocks
  (higher is better).

The informative comparison here is HypNO vs FNO: Godunov and HLL are conservative
and entropy-admissible by construction.

### 5. Tests

```bash
python -m pytest hyperbolic_pde/tests hyperbolic_pde/arz/tests
```

These cover the finite-volume solvers, the ARZ eigenstructure and pressure forms,
the exact Riemann solver, wave-front tracking, and model forward/equivalence
smoke tests.

---

## Checkpoints

`models_manifest.yaml` is the authoritative record for every released model:
file path, md5, the architecture **read directly from the saved tensors**, the
structural config that is *not* stored in the weights, and the full training
recipe.

| Name | Role | File |
|---|---|---|
| `hypno_lwr` | HypNO — LWR | `hyperbolic_pde/runs/final/lwr.pt` |
| `hypno_arz` | HypNO — ARZ, WFT-trained | `hyperbolic_pde/runs/final/arz_wft.pt` |
| `hypno_arz_hll` | HypNO — ARZ, HLL-pretrained then WFT fine-tuned | `hyperbolic_pde/runs/final/arz_hll_wft.pt` |
| `fno_lwr`, `fno_arz` | FNO operator baselines | see manifest |

Weights (`*.pt`) and generated datasets (`*.npz`) are excluded from version
control and are distributed alongside the manuscript rather than in the repo.
Checkpoints are bare `state_dict`s and may carry a `_orig_mod.` prefix from
`torch.compile` — strip it on load.

> ### ⚠️ Load each checkpoint with *its own* config
>
> The stencil half-widths `stencil_k_x` / `stencil_k_t` and the physics-gate
> settings are **structural, not stored in the weights**. `load_state_dict`
> therefore succeeds under the *wrong* stencil and silently produces garbage
> (order-of-magnitude worse error) with no warning. `arz_wft.pt` requires
> `k_t = 4`; `arz_hll_wft.pt` requires `k_t = 5`. Always pass the config paired
> with the checkpoint in `models_manifest.yaml`, and check the `kt=` value the
> loader prints.
>
> Likewise, the ARZ pressure form is **`p(ρ) = ρ`** for every dataset, model and
> figure in this work. `physics_arz.py` carries a different *module* default, so
> standalone code must call `set_pressure_form("rho")` explicitly.

---

## Notes for reviewers

- **Paper scope.** The paper covers exactly two models: `HypNO-ST3` for LWR and
  `HypNO-ARZ` (`arz/model_arz_orig.py`) for ARZ. Other files under `arz/`
  (`model_arz_mark1_router.py`, `model_arz_mark2*.py`), the 2-D modules and
  `models/legacy/` are exploratory or superseded work kept for provenance; they
  are not part of any reported result.
- **Ground truth is exact, not another numerical scheme.** LWR uses Lax–Hopf and
  ARZ uses wave-front tracking / the exact Riemann solver, so the finite-volume
  schemes (Godunov, HLL, WENO5) are evaluated as *competitors* on the same
  footing as the learned operators, not as the reference.
- **Baseline CFL.** Numerical baselines are run at `cfl = 0.4` unless a script
  states otherwise; tightening it changes baseline error by a few percent, in the
  direction that would flatter the learned models.
- **Reproducibility.** Dataset generation and evaluation are seeded, and the
  baseline caches are tied to an initial-condition checksum, so re-running an
  evaluation against a cache reproduces the same numbers.

## License & citation

See the manuscript in `papers/` for the method description and the citation to
use.
