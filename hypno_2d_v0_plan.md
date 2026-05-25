# Plan — HypNO-ST3 → 2D v0 (constant-coefficient linear advection)

## Context

The 1D HypNO-ST3 family solves nonlinear scalar conservation laws (LWR).
We're now extending it to 2D, but **deliberately starting with the linear
constant-coefficient case** `u_t + a·u_x + b·u_y = 0` with `a = 1, b = 0.7`,
so the 2D graph construction can be validated on pure directional transport
before nonlinear shock/rarefaction physics is layered on. Ground-truth
solutions are `u(x,y,t) = u_0(x − a·t, y − b·t)`.

Two instruction documents from the user pin down v0:

- `hypno_sd3_to_2d_v0_instructions.md` — geometry, lifting inputs, gates.
- `hypno_2d_message_passing_update_instructions.md` — message-passing
  structure and node update (this one is the more detailed of the two
  and overrides spec-1 where they conflict).

Decisions taken with the user (some override the spec):

1. **CFL/temporal gate**: spec-2's characteristic-alignment gate
   `g_temp = exp(−κ‖(Δr − (a,b)Δt) / (h_s + ε)‖²)` is used. This avoids
   the pure-axial degeneracy in spec-1's original edge-length CFL.
2. **`α_x`, `α_y`, `η` scaling**: **omitted**, overriding spec-2's
   explicit instruction. Match the existing 1D/2D code: gate-normalise
   messages, feed `concat([h, M^x, M^y, M^non])` into `update_net`.
   Rationale: "preserve the 1D update structure"; let `update_net`
   learn the relative weighting.
3. **Gates as features**: **dropped**, overriding spec-2. Gate values
   act only as multiplicative aggregation weights. Edge feature vectors
   carry only static geometry/physics scalars. Rationale: cleaner
   separation, smaller feature vectors, matches existing 1D/2D code.
4. **Boundary handling**: ghost-cell replicate everywhere — model uses
   replicate padding (as existing 2D model does), data generator runs
   with `boundary="ghost"` (already supported by `fvm_2d.py`).
5. **File layout**: new model file + extend `fvm_2d.py` with linear
   advection routines. The existing nonlinear 2D LWR pipeline is left
   untouched.
6. **`HypNO-SD3`** in the specs is treated as a typo for `HypNO-ST3`.

Intended outcome: a self-contained v0 pipeline that learns 2D directional
transport, separate from the existing nonlinear 2D LWR pipeline.

**Ground-truth strategy: up-sampled WENO5, identical pipeline to v1.**

The model is supposed to *beat* first-order upwind / MUSCL / WENO at
evaluation time, which is only meaningful if the training targets are
not themselves outputs of one of those schemes at the same resolution.

**Decision** (per user): use the same up-sampled-WENO5-and-down-average
pipeline for v0 that v1 will use. Specifically — run WENO5 + LF + SSP-RK3
at **8× the training grid resolution**, then **cell-average down** to the
training grid. The only difference between v0 and v1 in the data
generator is the flux:

- **v0**: `f(u) = a·u, g(u) = b·u` (linear, constant `a = 1, b = 0.7`).
- **v1**: `f(u) = g(u) = u(1−u)` (nonlinear LWR, isotropic).

Rationale (per user): pipeline reuse over per-task optimality. v0 must
produce results that flow directly into v1 without changing the data
pipeline. An analytical Lax–Hopf v0 generator would have been marginally
more accurate, but it would have been throwaway code that doesn't
exercise the v1 pipeline. Up-sampled WENO5 at 8× resolution is far more
accurate than any same-grid scheme the model is benchmarked against
(WENO5 error scales as `O((dx/8)⁵)` in smooth regions, much better than
`O(dx⁵)` at the training grid), so "model beats same-grid WENO5" stays
meaningful even with WENO5-generated targets.

Bonus property for v0: the analytical solution
`u(x, y, t) = u_0(x − a·t, y − b·t)` is available for *evaluation*
(linear advection has no shocks; characteristics never cross; Lax–Hopf
collapses to the method of characteristics). We use this analytical
solution as an *additional* eval metric — the model and the up-sampled
WENO5 reference can both be compared against the truly-exact analytical
solution. This is a sanity check on the WENO5 reference itself (the
8× WENO5 down-average should agree with the analytical to within
machine-noise levels for this linear problem).

## Files to create

| File | Purpose |
|---|---|
| `hyperbolic_pde/data/fvm_2d.py` *(extend / refactor)* | Factor the existing WENO5 + LF + SSP-RK3 solver so the flux `f(u), g(u)` is a parameter (currently hard-coded to `u(1−u)`). Add `simulate_weno5_2d(u0, x, y, t_out, f, g, cfl, boundary)` as the generic entry point. Add `generate_dataset_upsampled_2d(num_samples, nx, ny, nt, upsample_factor=8, flux="linear"/"lwr", ...)` that runs WENO5 on the fine grid and cell-averages to the target. Also add `upwind_flux_x`, `upwind_flux_y` standalone helpers — *not* used by the data generator, only used by the lifting layer for the four numerical-flux node features. Also add `analytical_shift_2d` (one-liner: `u_0(x − a·t, y − b·t)` with ghost clamp) as an evaluation helper only. |
| `hyperbolic_pde/models/hypno_linear_2d_v0.py` *(new)* | The v0 model. Forks the structure of `hypno_st3_2d.py`; replaces feature vectors and gates per the two specs (with the three overrides above). |
| `hyperbolic_pde/scripts/generate_data_linear_2d_v0.py` *(new)* | Driver for the new data generator. |
| `hyperbolic_pde/scripts/train_hypno_linear_2d_v0.py` *(new)* | Training script, mirroring `train_hypno_st3_2d.py`. |
| `hyperbolic_pde/scripts/smoke_test_linear_2d_v0.py` *(new)* | Diagnostics from spec-1 §"Diagnostics" + spec-2 §"Implementation Checklist". |
| `hyperbolic_pde/configs/hyperbolic_pde_linear_2d_v0.yaml` *(new)* | Config: `a, b`, grid sizes, model hyperparams, loss weights. |

SLURM scripts deferred until v0 leaves the dev box.

## Reuse from existing code

- `_enumerate_ball_offsets`, `_adjacency_class`, `_pad_space_time` from
  `hypno_st3_2d.py` — 3D Chebyshev ball + replicate padding. Copy or
  import.
- Gate-normalised aggregation pattern at `hypno_st3_2d.py:293-301`:
  `w_k = gate_k / (sum_j gate_j + 1e-3)`.
- Three-message-class routing `adj_x_msg`, `adj_y_msg`, `nonadj_msg`
  at `hypno_st3_2d.py:312-510`. Keep structure; replace feature vectors.
- Shared-decoder + per-layer state probe for deep supervision
  (`hypno_st3.py:1295-1316`).
- Residual update `act(update_net([h, agg]) + W·h)` at
  `hypno_st3_2d.py:508-510`.
- Training loop, loss decomposition (state + probe + conservation),
  checkpointing — copy `train_hypno_st3_2d.py` and adapt.

## Model design

### Lifting (`_LinearAdvectionLiftingLayer2D`)

Node MLP input (12 dims, per spec-1):
```
[u0_{i,j},  x_i, y_j, t_n,  a, b, λ_x, λ_y,
 F̂^x_{i−1/2,j}(u^0),  F̂^x_{i+1/2,j}(u^0),
 F̂^y_{i,j−1/2}(u^0),  F̂^y_{i,j+1/2}(u^0)]
```
- `λ_x = a, λ_y = b` for v0. Constants kept for forward-compat with
  nonlinear extension; 4 redundant input dims cost is negligible.
- The four numerical fluxes are produced by calling the same
  `upwind_flux_x/y` routine the data generator uses, applied to `u^0`.
- For the edge MLP in the lifting layer, follow the same per-class
  routing as the MP layers below (use the same x/y/non-adj edge feature
  vectors). Lifting and MP share the edge feature layout; only the
  per-layer MLP weights differ.

### Adjacent x-edge features (spec-2, gates dropped from feature vec)
```
[a,  sgn(x_k − x_i),  χ^x_up]
```
- `χ^x_up = 1[a·(x_i − x_k) > 0]` — hard flag, MLP input.
- Aggregation weight `g^x = g^x_up · g^x_ent`, with `g^x_ent ≡ 1` for v0.
  Soft form: `g^x_up = σ(a·(x_i − x_k) / (τ·|x_i − x_k| + ε))`.

### Adjacent y-edge features
```
[b,  sgn(y_p − y_j),  χ^y_up]
```
- `χ^y_up = 1[b·(y_j − y_p) > 0]`.
- Aggregation weight `g^y = g^y_up · g^y_ent`, with `g^y_ent ≡ 1`.

### Non-adjacent edge features (spec-2, gates dropped)
```
[Δx,  Δy,  Δt,  a,  b,  χ^non_up]
```
- `Δx = x_i − x_k`, `Δy = y_j − y_p`, `Δt = t_n − t_m`.
- `χ^non_up = 1[a·Δx + b·Δy > 0]`.
- Aggregation weight `g^non = g^non_up · g^non_ent · g_temp`, with
  `g^non_ent ≡ 1` for v0.
  - `g^non_up = σ((a·Δx + b·Δy) / (τ·√(Δx² + Δy²) + ε))`.
  - `g_temp = exp(−κ·‖(Δr − (a,b)·Δt) / (h_s + ε)‖²)` where
    `Δr = (Δx, Δy)` and `h_s = √(dx · dy)` (grid spacing scale).
    `κ` learnable (softplus-parameterised).

### Stencil / adjacency

3D Chebyshev ball, replicate padding:
- `|di_x| ≤ K_s`, `|di_y| ≤ K_s`, `−K_t ≤ dm ≤ 0` (causal).
- `K_s := K_x = K_y` (per spec-2).
- **adj_x**: `dm == 0, |di_x| == 1, di_y == 0`.
- **adj_y**: `dm == 0, di_x == 0, |di_y| == 1`.
- **non-adj**: everything else (including same-time non-axial,
  diagonals, all past-time edges). Self-loop and the four face-adj
  edges are excluded by construction.

### Physics gates summary

| Edge class | `g_up` | `g_ent` | `g_temp` | Combined |
|---|---|---|---|---|
| adj_x | soft x-upwind sigmoid | 1 | not applied | `g_up` |
| adj_y | soft y-upwind sigmoid | 1 | not applied | `g_up` |
| non-adj | soft 2D upwind sigmoid | 1 | char-alignment | `g_up · g_temp` |

`g_ent` is structurally present (multiplier `= 1`) so the slot exists
for the nonlinear follow-up. Sign convention to preserve when
extending: 1D `_ball_physics_gate` uses `is_shock = (u_L < u_R)` for
concave LWR (post-fix convention), and
`g_entropy = 1 − is_shock · (1 − entropy_ok) · (1 − γ_ent)` —
**suppresses** inadmissible expansion shocks. Carry this convention
forward in a comment at the entropy-gate call-site.

### Message MLPs and aggregation

Three message MLPs `adj_x_msg`, `adj_y_msg`, `nonadj_msg` (per spec-2
§1). Each consumes `concat[h_i, h_j, edge_features]` for its class.

Aggregation per class, gate-normalised with the same `+1e-3` additive
floor used by `hypno_st3_2d.py:293`:
```
gate_sum_x   = Σ_{k ∈ x-adj} g^x_k + 1e-3
M^x          = Σ_{k} (g^x_k / gate_sum_x) · m^x_k
   (same for M^y, M^non)
```

### Node update (omits spec-2's α, η)
```
agg     = concat([M^x, M^y, M^non])               # per-class aggregates side by side
upd_in  = concat([h, agg])
h_new   = act( update_net(upd_in) + W·h )
```
No fixed `α_x = Δt/dx`, no learnable `η` — the MLP learns the relative
weighting. Documented in a comment block at the update site so future
self knows this is a deliberate departure from spec-2 §5.

## Data generator design

**Up-sampled WENO5 + cell-average down, identical to the v1 pipeline.**

For each sample:

1. Pick the training grid `(nx, ny, nt)` and `t_out`. Pick an
   up-sample factor `U = 8` for the reference grid `(U·nx, U·ny, U·nt)`.
2. Rasterise the IC onto the **fine** `(U·nx, U·ny)` grid via the
   existing `piecewise_rectangles_ic_2d`.
3. Run WENO5 + LF + SSP-RK3 (Strang-split) on the fine grid for the
   linear flux `f(u) = a·u, g(u) = b·u` (v0) or `f = g = u(1−u)` (v1),
   producing `u_fine[U·nt, U·nx, U·ny]`.
4. Cell-average down to the training grid:
   ```
   u_train[k, i, j] = mean( u_fine[U·k, U·i : U·(i+1), U·j : U·(j+1)] )
   ```
   This is the FV-correct down-sampling (preserves cell averages).
5. Save `{u0_train, u_train, x, y, t_out}` per sample.

Refactor needed in `fvm_2d.py`: the existing solver hard-codes `f = g =
u(1−u)` inside the WENO5 reconstruction and the LF speed estimate. Lift
both into callable parameters so the same routine handles linear (v0)
and LWR (v1):

```python
def simulate_weno5_2d(u0, x, y, t_out, *, f, g, fp_abs_max, cfl, boundary):
    # f(u), g(u): scalar fluxes
    # fp_abs_max(u_min, u_max): upper bound on |f'(u)|, |g'(u)| over [u_min, u_max]
    #   v0:  f = lambda u: a*u,        fp_abs_max = lambda lo, hi: abs(a)
    #   v1:  f = lambda u: u*(1-u),    fp_abs_max = lambda lo, hi: max(|1-2*lo|, |1-2*hi|)
    ...

def generate_dataset_upsampled_2d(
    num_samples, nx, ny, nt, *,
    flux_kind: str,   # "linear" or "lwr"
    a=None, b=None,   # v0 only
    upsample_factor=8,
    cfl=0.4, boundary="ghost", ...,
):
    for sample in range(num_samples):
        rects = sample_piecewise_rectangles_ic_2d(...)
        u0_fine = rasterise(rects, x_fine, y_fine)
        u_fine  = simulate_weno5_2d(u0_fine, ..., f=f_v0_or_v1, g=..., ...)
        u_train, u0_train = cell_average_to(u_fine, factor=upsample_factor)
        save(...)
```

IC families: piecewise-constant rectangles via the existing
`piecewise_rectangles_ic_2d` at `fvm_2d.py:102-132` (rasterised at the
fine resolution, not the training resolution — so rectangle edges line
up at sub-training-cell precision). Optionally add an isotropic Gaussian
family for smooth-IC diagnostic value.

Boundary: `"ghost"` (replicate). Dataset format: `.npz` with
`{u0, u, x, y, t}` plus metadata `{flux_kind, a, b, upsample_factor}`.

**Cost note**: at `U = 8` the fine grid has `64 · 64 · 8 = 32768` cells
per slice (vs `64²·1 = 4096` at training resolution); CFL forces ~8×
more time steps too. So each fine sample is ~512× the cost of a same-
grid sample. For v0 with `nx = ny = 64`, that's manageable on a single
GPU/CPU. For v1 it'll need a cluster job — the existing
`generate_data_2d.sh` SLURM script can be reused.

**Sanity check on the reference**: for v0 we additionally compute the
analytical solution `u(x, y, t) = u_0(x − a·t, y − b·t)` at the fine
grid (closed-form, free), cell-average down, and verify the WENO5 fine
output agrees to within ~`1e-4` on test samples. If it doesn't, the
WENO5 refactor for linear flux has a bug.

Note on the lifting layer's numerical-flux features: those are computed
from `u^0` by the *model* using `upwind_flux_x/y`, decoupled from the
data generator. The lifting flux features encode upwind direction at
the IC; the data generator produces a much-higher-fidelity reference.
Different roles.

## Diagnostics

`smoke_test_linear_2d_v0.py` prints / asserts:

1. Lifting input shape `[B, nt, nx, ny, 12]`.
2. Counts: 2 x-adj, 2 y-adj, `(2K_s+1)^2·(K_t+1) − 1 − 4` non-adj edges per node.
3. For `a=1, b=0.7`:
   - Left x-source → `χ^x_up == 1`, right x-source → `0`.
   - Bottom y-source → `χ^y_up == 1`, top y-source → `0`.
4. `g^x_ent == g^y_ent == g^non_ent == 1` for every edge in v0.
5. `g_temp` applied only on non-adj edges (verify by constructing one
   edge from each class and inspecting the gate function called).
6. Upwind dot product uses `x_i − x_k` (not `x_k − x_i`): construct a
   single edge with known displacement and compare against expected sign.
7. No accidental wrap-around in padding (ghost replicate, not periodic).

## Verification

1. **Diagnostics** — run `smoke_test_linear_2d_v0.py`; all items pass.
2. **Reference sanity** — for one fixed test IC, compute three signals
   on the training grid: (a) the up-sampled-WENO5 reference,
   (b) the analytical cell-averaged solution
   `cell_avg(u_0(x − a·t, y − b·t))`, (c) same-grid WENO5.
   The first two should agree to `~1e-4` MAE. If not, the WENO5
   refactor for the linear flux is broken.
3. **Data generator visual** — generate 10 samples at `nx=ny=32, nt=8`.
   Plot one trajectory; structure should advect rigidly in `(1, 0.7)`
   with only the small WENO5 shock-smearing at rectangle edges.
4. **One-epoch train** — 100 samples, 1 epoch. Loss drops; final-state
   MAE beats trivial `u_0 → u_0` baseline.
5. **Longer train** — ~100 epochs at small scale. Visualise; predicted
   structure should sit at `(a·t, b·t)` offset.
6. **Vs same-grid baselines** — for held-out test ICs, compute MAE of
   each candidate against the analytical truth at the training grid:
   - HypNO-2D-v0 (model)
   - same-grid first-order upwind FVM
   - same-grid MUSCL (2nd-order TVD)
   - same-grid WENO5
   - the up-sampled-WENO5 reference (sanity floor)
   
   Order of MAE we expect (best to worst):
   `up-sampled WENO5 < model ≤ same-grid WENO5 ≤ MUSCL < upwind`.
   The model must beat same-grid upwind and MUSCL convincingly, and
   ideally tie or beat same-grid WENO5.

## Out of scope (intentional)

- Nonlinear 2D LWR (entropy gate becomes live).
- Variable-coefficient linear advection.
- WENO5 / higher-order fluxes (first-order upwind suffices for v0).
- SLURM and CLEPS configs.
- Revisiting the spec-2 `α_x, α_y, η` decision unless v0 actually fails
  to converge — only with evidence.
