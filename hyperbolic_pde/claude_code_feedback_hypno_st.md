# Feedback on current `hypno_st.py` implementation

The current implementation is only a **partial match** to the requested refactor.

## Overall assessment

You correctly implemented most of the **encoder / lifting** changes, but you did **not** implement most of the planned **processor / message-passing** changes.

So the status is:

- **lifting / encoder:** mostly correct
- **processor / MP layers:** still mostly old design
- **WENO branch:** still old design
- **PINN shock handling:** still old design

---

## What is correct

### 1. LWR-aware lifting edge features are implemented
This part is good.

You replaced the old generic lifting edge features with a much more LWR-aware set based on:
- `u0_i`, `u0_k`
- `du`, `abs(du)`, `u_avg`
- `rel_x`, `abs_dx`, `slope`
- `f_i`, `f_k`
- `a_i`, `a_k`, `a_ik`
- `sign_a`, `upwind`
- and then append `t`

This matches the intended encoder refactor.

### 2. Precomputation for encoder edge features is implemented
Good.

`precompute_lwr_edge_features(...)` is a useful addition and matches the intended optimization:
- precompute all static `u0, x` dependent edge features once
- append `t` only during forward
- reuse across all time steps

### 3. Node MLP remains simple
Good.

Keeping node features as `[u0_i, x_i, t_j]` is acceptable for now.

---

## What is still missing / incorrect

# 1. Processor layers were NOT refactored as planned

This is the main issue.

Both `_PINNSpaceTimeMPLayer` and `_WENOSpaceTimeMPLayer` still use the old feature design.

## Current spatial MP still uses old features
It still builds messages from:
- `h_i, h_j`
- `x_i, x_j, dx`
- `du0, |du0|`

This is NOT what was planned.

## What it should do instead
Each processor layer should decode a cheap local scalar state from `h`, e.g.

```python
u_hat = state_probe(h).squeeze(-1)
```

or, if density is known to be in `[0,1]`,

```python
u_hat = torch.sigmoid(state_probe(h)).squeeze(-1)
```

Then compute:
- `f_hat = u_hat * (1 - u_hat)`
- `a_hat = 1 - 2 * u_hat`

And build spatial edge features from the **current decoded state**, not just from `u0`.

The processor spatial edge features should look more like:

- `h_i, h_j`
- `rel_x`, `abs_dx`
- `u_hat_i, u_hat_j`
- `du_hat`, `u_avg_hat`
- `slope_hat = du_hat / dx`
- `f_i, f_j`
- `a_i, a_j`
- `a_ij`
- `sign_a`
- `upwind`

### Why
LWR transport direction is state-dependent. If processor edges are built only from `u0`, the transport bias is frozen by the initial condition instead of following the evolving latent state.

---

# 2. Temporal processor features were NOT updated

Currently temporal MP still uses:
- `h_i, h_j`
- `t_i, t_j, dt`
- `x/t`

This is still the old design.

## What it should do instead
Temporal edges should use a CFL-like transport feature derived from decoded local speed.

For example:
- `u_hat_i, u_hat_j`
- `a_i, a_j`
- `|a_i| * dt / dx_local`
- or `|a_ij| * dt / dx_local`

The main temporal physics cue should be **CFL-like propagation scale**, not `x/t`.

`x/t` can remain as an optional extra feature for self-similar/Riemann-like settings, but it should not be the main temporal physics feature.

---

# 3. PINN processor still uses positive clipping in shock regions

This was supposed to be changed, but it was not.

Currently you still do:
```python
msg_capped = msg.clamp(0.0, self.delta)
contrib = torch.where(is_shock, msg_capped, msg)
```

## Why this is a problem
This destroys the sign of the message near shocks.

For conservation-law dynamics, signed corrections matter. Clamping everything to positive values is not a good inductive bias.

## What it should do instead
Replace hard positive clipping with **signed attenuation**.

For example:
```python
alpha = 1.0 - shock_indicator.unsqueeze(-1)   # or another smooth attenuation
contrib = alpha * msg
```

Or use a thresholded attenuation if needed, but preserve sign.

Goal:
- smooth region: keep message mostly unchanged
- shock region: reduce message magnitude
- do NOT force it positive

---

# 4. WENO branch was NOT updated as planned

Currently `_WENOSpaceTimeMPLayer` still computes smoothness indicators from latent field `h`:
- `_spatial_beta(h)`
- `_temporal_beta(h)`

This is still the old latent-space heuristic.

## What it should do instead
WENO smoothness should ideally be computed from a decoded scalar state:
```python
u_hat = state_probe(h).squeeze(-1)
```

Then define smoothness indicators from `u_hat` or `f(u_hat)`.

For example:
- spatial beta from local differences of `u_hat`
- optionally temporal beta from local differences of `u_hat`
- optionally flux-based beta

This makes WENO weighting physically grounded in the actual evolving traffic state, rather than purely in latent feature differences.

### Target behavior
- smooth regions → strong message passing
- discontinuous / shock regions → reduced message passing
- based on decoded physical state, not only latent-space geometry

---

# 5. Unified MP mode was not updated either

If `unified_mp=True`, it still uses the old low-level feature design.

If unified MP is kept, it should also receive richer LWR-aware features:
- decoded `u_hat`
- `f`
- `a`
- `a_ij`
- `sign_a`
- `upwind`
- CFL-like ratio
- `is_spatial`

Do not keep unified MP with only the old coordinate-based feature tuple.

---

## Minor issues / cleanup

### 1. Comments/docstrings are outdated
Several comments still describe the old feature definitions, e.g. coordinate-heavy edge tuples.

These should be updated so comments match actual implementation.

### 2. `a_ik` computation is okay but should be clearer
Current code:
```python
a_ik_raw = (f_k - f_i) / du.abs().clamp(min=1e-6) * du.sign()
```

This is mathematically fine, but less readable than directly expressing:
```python
a_ik = (f_k - f_i) / du
```
with explicit safe handling for small `du`.

Please rewrite this more transparently.

---

## What should be implemented next

Priority order:

### Priority 1: Refactor processor spatial edges
Add a small per-layer state probe and rebuild processor spatial features around decoded state:
- `u_hat`
- `f(u_hat)`
- `a(u_hat)`
- `a_ij`
- `sign_a`
- `upwind`

### Priority 2: Replace PINN shock clipping with signed attenuation
Do not clamp to `[0, delta]`.
Use multiplicative attenuation that preserves message sign.

### Priority 3: Refactor processor temporal edges
Replace `x/t` as the main physics feature with a CFL-like ratio:
- `|a| dt / dx`

### Priority 4: Refactor WENO smoothness
Compute WENO weights from decoded `u_hat` or `f(u_hat)` rather than from latent `h` directly.

### Priority 5: Update unified MP mode consistently
If unified MP remains, it must use the same richer physics-aware features.

---

## Concrete desired end state

### Encoder / lifting
Keep current encoder changes. They are mostly good.

### Processor
Each MP layer should:
1. decode a provisional scalar state from `h`
2. compute local LWR quantities from that decoded state
3. use those quantities in spatial and temporal message features

### PINN mode
Use shock detector to attenuate messages smoothly, not clip them to positive values.

### WENO mode
Use decoded-state smoothness, not just latent smoothness.

---

## Short summary

The current implementation successfully refactors the **encoder** but leaves the **processor** mostly unchanged.

Please implement the processor refactor as originally planned:
- state-aware processor edges
- CFL-aware temporal features
- signed shock attenuation
- decoded-state WENO smoothness
- consistent unified MP support
