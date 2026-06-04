# HypNO-ST3 → 2D LWR (proof of concept)

**Status:** plan only, not yet implemented.
**Date:** 2026-05-13.
**Goal:** extend HypNO-ST3 to a 2D scalar conservation law for a paper-section
proof of concept, with a new parallel module `hypno_st3_2d.py` (no refactor
of the existing 1D code).

---

## 1. PDE

Isotropic scalar 2D LWR in divergence form (conservation of mass holds by
construction):

```
u_t + f(u)_x + g(u)_y = 0
f(u) = g(u) = u(1 - u)
a(u) = f'(u) = 1 - 2u
b(u) = g'(u) = 1 - 2u
```

Wave-speed vector `λ(u) = (a(u), b(u))`, `|λ|_2 = √2 · |1 - 2u|`,
`c_max = √2` (at `u ∈ {0, 1}`).

Domain `(x, y) ∈ [-1, 1]²`, `t ∈ [0, T_max]`.

The flux symmetry `f = g` means `a = b`. We will still store both
separately in the model so that an anisotropic variant
(e.g. `g = 0.5·u(1-u)`) becomes a config flag rather than a code change.

---

## 2. Starting grid: 32 × 32 × 16

For first-pass debugging:
- `nx = ny = 32` → `dx = dy = 2/32 = 0.0625`
- `nt = 16`, `T_max = 1.0` → `dt = 1/16 = 0.0625`
- `c_net_xy = dx / dt = 1.0` per layer in either spatial axis from a
  diagonal temporal-spatial edge; with `k_x = k_y = 3, k_t = 3` and
  `L = 6` layers, spatial reach is `L·k_x·dx = 1.125 > 1` (full domain),
  temporal reach is `L·k_t·dt = 1.125 > T_max`. Both reach bounds clear
  at this small grid — a clean baseline before scaling up.

Dataset size for PoC: 400 train + 50 val + 50 test samples (matches the
ablation-run pattern from the 2026-05-07 memory).

---

## 3. Data generator (new: `hyperbolic_pde/data/fvm_2d.py`)

### Numerical scheme: WENO5 + Strang splitting + RK3

Used for **both** training data and ground-truth evaluation. Lax-Hopf
doesn't generalize cleanly to 2D, so WENO5 plays the role 1D's Lax-Hopf
plays: high-order reference.

Strang dimensional splitting (2nd-order in time, symmetric):

```
u^{*}      = L_x^{dt/2} ( u^n )       # half x-sweep
u^{**}     = L_y^{dt}   ( u^{*} )     # full y-sweep
u^{n+1}    = L_x^{dt/2} ( u^{**} )    # half x-sweep
```

Each 1D sweep `L_x` / `L_y` is solved with WENO5 reconstruction +
Lax-Friedrichs flux split + 3-stage SSP-RK3 in time:

- WENO5 reconstruction: 5th-order in smooth regions, ENO-stable at shocks.
  Uses Jiang–Shu smoothness indicators on a 5-cell stencil.
- Flux splitting: local Lax–Friedrichs `f±(u) = 0.5·(f(u) ± α·u)` with
  `α = max|f'(u)| = 1` for LWR.
- RK3 (Shu–Osher): `u^{(1)} = u + dt·L`, `u^{(2)} = 0.75·u + 0.25·(u^{(1)} + dt·L^{(1)})`,
  `u^{n+1} = (1/3)·u + (2/3)·(u^{(2)} + dt·L^{(2)})`.

CFL: `dt · (|a|_max / dx + |b|_max / dy) ≤ 0.5` (tighter than first-order
because of RK3 stability region). Internal step adaptive; output times
`t_out` linearly interpolated between bracketing steps (same fix from
the 1D FVM-temporal-interpolation memory).

Implementation note: the 1D `godunov_flux` in `fvm.py` is **not** reused —
WENO5 + LF flux is a separate scheme. The 1D WENO5 helper can live in
`fvm_2d.py` for now (it's a private 1D-sweep utility, not a public 1D
data generator).

### Initial conditions

Piecewise constant on rectangles. Random number of axis-aligned
rectangles `r ∈ {1..8}`, each with a constant `u ∈ [u_min, u_max]`.
`u_min = 0.05, u_max = 0.95` (avoid endpoints, same as 1D).

(Sine and 4-quadrant Riemann are deferred; only added if the
piecewise-constant baseline trains.)

### Output

NPZ with keys:
- `u`: `[N, nt, ny, nx]` float32
- `u0`: `[N, ny, nx]` float32 (redundant convenience: `u[:, 0]`)
- `x`: `[nx]` float32, `y`: `[ny]` float32, `t`: `[nt]` float32
- `ic_kind`: `[N]` int8 (0=pwconst, 1=pwsine, 2=riemann)

---

## 4. Model (new: `hyperbolic_pde/models/hypno_st3_2d.py`)

Fork of `hypno_st3.py`. Strategy: keep names parallel to 1D wherever
possible so a side-by-side diff is short and reviewable.

### 4.1 Tensor layout

All latents are `[B, nt, ny, nx, d]`. Node features `u0` is `[B, ny, nx]`,
coords are `x: [nx]`, `y: [ny]`, `t: [nt]`. Padding helper
`_pad_space_time_2d` replicate-pads in all three axes:

```python
h_pad = F.pad(
    h.permute(0, 4, 1, 2, 3),                              # [B, d, nt, ny, nx]
    (k_x, k_x, k_y, k_y, k_t, k_t),
    mode="replicate",
).permute(0, 2, 3, 4, 1)                                   # [B, nt+2k_t, ny+2k_y, nx+2k_x, d]
```

### 4.2 Stencil

```python
def _enumerate_ball_offsets_2d(k_x, k_y, k_t, causal):
    m_range = range(-k_t, 1) if causal else range(-k_t, k_t + 1)
    out = []
    for dm in m_range:
        for dy in range(-k_y, k_y + 1):
            for dx in range(-k_x, k_x + 1):
                if dx == 0 and dy == 0 and dm == 0:
                    continue
                out.append((dx, dy, dm))
    return out
```

For `k_x = k_y = 3, k_t = 3, causal=True`: `7·7·4 - 1 = 195` edges per
node per layer. Heavy — see §7 on memory.

### 4.3 Adjacency classes (three, not two)

| Class      | Predicate                                  | Gate                                  |
|------------|--------------------------------------------|---------------------------------------|
| `adj_x`    | `dm == 0, abs(dx) == 1, dy == 0`           | x-upwind × x-entropy                  |
| `adj_y`    | `dm == 0, dx == 0, abs(dy) == 1`           | y-upwind × y-entropy                  |
| `nonadj`   | everything else                            | CFL (or `time`); identity by default  |

Three edge MLPs: `adj_x_msg`, `adj_y_msg`, `nonadj_msg`. The two adj
MLPs have symmetric input layouts but separate weights.

### 4.4 Node MLP input

```
[u0, x, y, t, f0, a0, g0, b0]    → 8 dims with include_flux=True
[u0, x, y, t, a0, b0]            → 6 dims without flux
```

Note: under isotropic flux `f = g`, the entries `(f0, a0)` and `(g0, b0)`
are duplicates. We keep them as separate fields anyway so that swapping
in an anisotropic flux later is a one-line FVM change with no node-MLP
input dimensionality shift.

### 4.5 Lifting edge MLP input

Unified (analogous to 1D unified edge MLP). One offset gives:
- `du0 = u0_j - u0_i`
- `sign(rel_x), sign(rel_y), rel_t`
- `t_i, t_j`
- per-axis Rankine–Hugoniot speeds when adj in that axis (otherwise 0):
  `a0_ij` if `adj_x`, `b0_ij` if `adj_y`, both 0 on `nonadj`
- `is_adj_x`, `is_adj_y` flags (one-hot among `{adj_x, adj_y, nonadj}`)
- with flux: `u0_i, u0_j, f0_i, f0_j, g0_i, g0_j, a0_i, a0_j, b0_i, b0_j`

Total dims (with flux): `1 + 3 + 2 + 2 + 2 + 2·5 = 20`. Without flux:
drop `f0_i, f0_j, g0_i, g0_j` → 16. Pure-pairwise variant: drop also
`u0_i, u0_j, a0_i, a0_j, b0_i, b0_j` → 10.

### 4.6 MP layer adj edges (one per axis)

**adj_x** features (2d + 12 with flux):
```
h_i, h_j,                                      # 2d
u_i, u_j, f_i, f_j, a_i, a_j,                  # 6 (with flux)
a_ij_x, sign(a_ij_x), upwind_x,                # 3
sign(rel_x)                                    # 1
```

`adj_y` is the mirror with `b_ij_y, sign(b_ij_y), upwind_y, sign(rel_y)`
and `g_i, g_j` instead of `f_i, f_j` (or just `b_i, b_j` without flux).

**nonadj** features (2d + 14 with flux):
```
h_i, h_j,
u_i, u_j,
f_i, f_j, g_i, g_j,                            # both flux components
a_i, a_j, b_i, b_j,
rel_x, rel_y, rel_t,
cfl,                                           # cfl = (|a_i|/dx + |b_i|/dy) · |rel_t|
sign(rel_x), sign(rel_y),
```

### 4.7 Physics gate (analytical)

Per-axis adj edges fire the corresponding upwind × entropy components:

- `adj_x` (`dm=0, |dx|=1, dy=0`):
  - `g_upwind_x = sigmoid(-a_ij_x · sign(rel_x) / T_x)`
  - `g_entropy_x`: Oleinik condition with `u_L, u_R` taken along x
- `adj_y` (`dm=0, dx=0, |dy|=1`):
  - `g_upwind_y = sigmoid(-b_ij_y · sign(rel_y) / T_y)`
  - `g_entropy_y`: same shape, along y

Separate learnable temperatures `T_x, T_y` and `gamma_entropy_x,
gamma_entropy_y` (could be tied via flag).

`nonadj` (`dm ≠ 0` or pure-spatial non-axial): CFL gate
```
cfl = (|a_i|/dx + |b_i|/dy) · |rel_t|
g_cfl = exp(-cfl_scale · relu(cfl - 1)²)
```
plus optional char-cone factor on the 2D wave-speed magnitude.

Pure-spatial diagonal edges (`dm=0, dx≠0, dy≠0` or non-adj along an axis)
are gated to 0 in the MP layer by default (extension of
`mask_same_t_nonadj`).

### 4.8 Aggregation, update, decoder

Identical to 1D: gate-normalized aggregation with `+1e-3` additive
floor, `update_net([h, agg]) + W·h`, GELU. Decoder is shared (used for
both the gate `u_hat` and per-layer deep supervision readouts). Skip:
`u_pred = u0_expanded + decoder(h)` when `skip=True`.

---

## 5. Configs

New: `hyperbolic_pde/configs/hyperbolic_pde_2d.yaml`. Mirrors the 1D base
plus:

```yaml
model:
  stencil_k_x: 3
  stencil_k_y: 3
  stencil_k_t: 3
  d_latent: 64
  d_hidden: 64
  d_hidden_nonadj: 64
  n_layers: 6
  encoder_scaling: physics
  encoder_type: gnn
  skip: true
  include_flux: true
  pure_pairwise_edges: false
  use_checkpoint: true

data:
  nx: 32
  ny: 32
  nt: 16
  x_min: -1.0
  x_max: 1.0
  y_min: -1.0
  y_max: 1.0
  t_max: 1.0
  num_train: 400
  num_val: 50
  num_test: 50
```

(CLEPS variant later if/when this trains; not part of the PoC plan.)

---

## 6. Scripts

- `hyperbolic_pde/scripts/generate_data_2d.py` — driver around `fvm_2d`.
- `hyperbolic_pde/scripts/train_hypno_st3_2d.py` — fork of
  `train_hypno_st3.py`. Same `--config`, `--resume_run` semantics.
- `hyperbolic_pde/scripts/eval_vs_numerical_2d.py` — per-sample MAE / rL2
  vs Godunov reference; per-t error curves; visualization of 2D fields
  (predicted vs truth heatmaps at selected timesteps, plus error map).

---

## 7. Memory budget — important

The stacked edge tensor `[B, nt, ny, nx, n_offsets, feat_dim]` is the
hot tensor. For PoC numbers (`B=4, nt=16, ny=32, nx=32, d=64, n_offsets=195`,
adj_x_msg feat_dim = 2·64+12 = 140):

```
B·nt·ny·nx·n_offsets·feat_dim = 4·16·32·32·195·140 ≈ 5.6e8 floats ≈ 2.2 GB
```

That's just for one MP layer's adj/nonadj concat. With activations,
update_net, decoder probes per layer, and `n_layers=6`, GPU memory will
be tight on a single 40 GB A100. Mitigations, **in this order**:

1. **`use_checkpoint=True`** already in 1D — keeps activations only at
   layer boundaries.
2. **Loop over offsets instead of stacking** when `n_offsets > threshold`
   — explicit `for di, dy, dm in offsets:` with running accumulation.
   Slower wall-clock, ~10× smaller peak. Implement as a flag
   `loop_over_offsets: bool` (default `True` for 2D).
3. **Smaller `d_latent` / `d_hidden`** — PoC doesn't need 128.
4. **Smaller stencil** — `k_x = k_y = 2` first; only widen if accuracy
   demands it.

**Recommendation for PoC**: start with `loop_over_offsets=True`,
`d_latent=64`, `k=2`. Make the model run end-to-end on CPU first (tiny
batch), then push to GPU.

---

## 8. Receptive cone in 2D — what we expect to study

(For the paper section, not a model design item.)

In 2D, the receptive-field rectangle becomes a 3D rectangular prism
`[-L·k_x·dx, L·k_x·dx] × [-L·k_y·dy, L·k_y·dy] × [0, L·k_t·dt]` in
`(rel_x, rel_y, t)`. The continuous characteristic cone is a true 3D
cone `{(rx, ry, t) : √(rx² + ry²) ≤ c_max · t}` with `c_max = √2`.

Two new effects vs 1D:
- The cone is a *circle* in cross-section, the receptive field is a
  *square*. Mismatch is worse than 1D's interval-in-interval.
- Anisotropic stencil (`k_x ≠ k_y`) creates direction-dependent freeze.

§4 of `receptive_cone_theory.md` will need a 2D supplement; out of
scope for this implementation plan but tag in the writeup.

---

## 9. Implementation order

1. **`fvm_2d.py`** — WENO5 1D sweep + Strang splitting + RK3 +
   piecewise-constant rectangles IC. Sanity: (a) a single rectangle
   of higher density relaxes in the expected direction; (b) on a
   smooth sinusoidal IC, convergence rate vs grid spacing is close
   to 5 on the smooth part (3 around shocks is fine).
2. **`generate_data_2d.py`** + tiny dataset (50 samples, `nx=ny=16,
   nt=8`) — smoke test, no model.
3. **`hypno_st3_2d.py` lifting layer** — get one forward pass through
   the lifting on a dummy batch, verify output shape `[B, nt, ny, nx, d]`.
4. **`hypno_st3_2d.py` MP layer** — same, with `n_layers=1`.
5. **Full model with `n_layers=2`, `loop_over_offsets=True`** —
   end-to-end forward pass with the smoke-test dataset.
6. **Training script** — train 5 epochs, confirm loss goes down.
7. **Generate PoC dataset** (400/50/50 at 32×32×16). Train ~50 epochs.
8. **Eval script** — produce MAE/rL2 vs Godunov, error maps, per-t curves.
9. **Riemann 2D sweep** — only if PoC trains successfully.

Stop and reassess after step 6 — that's the earliest point where it's
clear whether the 2D extension trains at all.

---

## 10. Out of scope (for this plan)

- ARZ second-order model. (See `multi_pde_plan_2026-05-07.md`.)
- Cone-stencil 2D variant (analog of `hypno_st3_charcone.py`).
- Network/junction LWR.
- Boundary condition study (we'll use replicate-padding everywhere,
  same as 1D).
- Rotation-equivariant weight sharing across `adj_x`/`adj_y`.
- 2D Riemann sweep / oblique shock fidelity study.
- Sine and 4-quadrant Riemann ICs.
