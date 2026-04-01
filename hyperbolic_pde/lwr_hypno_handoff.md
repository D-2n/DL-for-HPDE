# Handoff Note: LWR-Specific Physics-Aware Refactor for `HypNO_ST`

This document summarizes:

1. the current implementation state,
2. the reasoning behind the changes we discussed,
3. the agreed changes for the **lifting / encoder**,
4. the planned follow-up changes for the rest of the network,
5. implementation guidance for a coding agent.

Target PDE: **LWR (Lighthill-Whitham-Richards)** traffic flow model, i.e. a **scalar conservation law**

\[
u_t + f(u)_x = 0
\]

with flux typically

\[
f(u) = u(1-u)
\]

and characteristic speed

\[
a(u) = f'(u) = 1 - 2u.
\]

---

## 1. High-level modeling principle

For LWR, the network should not treat message passing as generic local smoothing.
It should treat each edge as a **local transport / interface interaction**.

The core physical quantities are:

- local state values,
- state jumps,
- local slope,
- flux values,
- characteristic speed,
- interface speed,
- upwind direction,
- CFL-like propagation ratios.

The main idea is:

- **node features** describe the local state at a point,
- **edge features** describe how information propagates between nearby points.

This is especially important for hyperbolic PDEs and even more for nonlinear conservation laws like LWR, where propagation direction depends on the state.

---

## 2. Current implementation status

Current model file: `hypno_st.py`

### Current lifting layer

`_SpaceTimeLiftingLayer` currently:

- builds node features from `[u0_i, x_i, t]`,
- aggregates messages from spatial neighbors only,
- uses current edge features:

```python
[u0_i, u0_k, x_i, x_k, t, rel_x, |du0|]
```

This is already better than a plain MLP lifting because it uses locality and relative position, but it is still too generic for LWR.

### Current MP layers

The message-passing layers (`_PINNSpaceTimeMPLayer`, `_WENOSpaceTimeMPLayer`) currently:

- use causal temporal message passing,
- include some physics-flavored quantities,
- still rely too heavily on coordinates and `u0`-based jumps,
- do **not** yet explicitly encode LWR flux / characteristic information inside the edge features.

### Current shock handling

Two modes exist:

- `pinn`: shock regions identified via a coarse decoder and PDE residual,
- `weno`: shock/smoothness weighting inferred from latent-field smoothness.

This is a reasonable scaffold, but several changes are planned later.

---

## 3. Main conclusion from the discussion

The first refactor should focus on the **lifting / encoder only**.

For now, do **not** change the rest of the model unless needed for compatibility.

Priority order:

1. keep node features simple,
2. make lifting edge features LWR-aware,
3. optionally precompute static edge physics terms in the dataset.

---

## 4. Agreed encoder changes

## 4.1 Node features: keep essentially the same

Current node input:

```python
[u0, x, t]
```

Decision: **keep this as the default**.

Reasoning:

- node features should encode the local state and position,
- the truly transport-specific physics enters through **pairwise / edge** quantities,
- so the biggest gain comes from improving edge features, not node features.

### Optional node feature extension

A small optional upgrade is to extend node features to:

```python
[u0, x, t, f(u0), a(u0)]
```

where:

- `f(u0) = u0 * (1 - u0)`
- `a(u0) = 1 - 2 * u0`

This is optional and not required for the first refactor.

---

## 4.2 Edge features: change substantially

### Current edge features

Current lifting edge input is:

```python
[u0_i, u0_k, x_i, x_k, t, rel_x, |du0|]
```

### Problem with current edge features

This gives the network:

- raw values,
- absolute coordinates,
- relative position,
- jump magnitude,
- time coordinate,

but it does **not** explicitly provide the LWR transport physics:

- flux,
- characteristic speed,
- interface speed,
- upwind direction,
- slope / compression.

### Proposed LWR-aware edge features

For a pair of neighboring points `i` and `k`, define:

- `du = u_k - u_i`
- `u_avg = 0.5 * (u_i + u_k)`
- `rel_x = x_k - x_i`
- `abs_dx = |rel_x|`
- `slope = du / rel_x`

Flux values:

- `f_i = u_i * (1 - u_i)`
- `f_k = u_k * (1 - u_k)`

Characteristic speeds:

- `a_i = 1 - 2 * u_i`
- `a_k = 1 - 2 * u_k`

Interface speed:

- `a_ik = (f_k - f_i) / (u_k - u_i)` if `|du|` is not tiny,
- fallback to `1 - 2 * u_avg` when `|du|` is small.

Additional directional features:

- `sign_a = sign(a_ik)`
- `upwind = 1[a_ik * rel_x < 0]`

### Proposed edge feature vector

Use:

```python
edge_in = torch.cat([
    u0_bc, u_k_bc,
    du, du.abs(), u_avg,
    rel_x, abs_dx, slope,
    f_i, f_k,
    a_i, a_k, a_ik,
    sign_a, upwind,
    t_bc
], dim=-1)
```

This intentionally **does not include raw absolute edge coordinates** `x_i` and `x_k`.

---

## 4.3 Why raw coordinates were removed from edge features

The proposed edge feature vector does **not** include raw absolute spatial coordinates.

This is intentional.

### Reasoning

For homogeneous 1D LWR on a uniform domain, local interface physics depends mainly on:

- relative displacement,
- state jump,
- slope,
- flux,
- characteristic direction,
- interface speed.

So in the edge features, relative geometry is more useful than absolute position.

### Separation of roles

- **node features** carry absolute position information,
- **edge features** carry transport / interaction information.

### When raw coordinates should be reintroduced

Raw edge coordinates may be needed later if the PDE/data are not homogeneous, e.g.:

- position-dependent flux `f(x, u)`,
- bottlenecks,
- ramps,
- varying road conditions,
- special boundary behavior.

For the current LWR setting, keeping them out of `edge_in` is the preferred default.

---

## 4.4 Implementation sketch for encoder edge features

### Current code fragment

Current lifting edge construction is based on:

```python
edge_in = torch.cat([u0_bc, u_k_bc, x_bc, x_k_bc, t_bc, rel_x, abs_du], dim=-1)
```

### Replace with

```python
# pairwise state features

du = u_k_bc - u0_bc
u_avg = 0.5 * (u0_bc + u_k_bc)
rel_x = x_k_bc - x_bc
abs_dx = rel_x.abs()

# stable slope
slope = du / rel_x.clamp(min=1e-6).abs() * rel_x.sign()

# LWR flux
f_i = u0_bc * (1.0 - u0_bc)
f_k = u_k_bc * (1.0 - u_k_bc)

# local characteristic speeds
a_i = 1.0 - 2.0 * u0_bc
a_k = 1.0 - 2.0 * u_k_bc

# interface speed
small_jump = du.abs() < 1e-6
a_ik_raw = (f_k - f_i) / du.clamp(min=1e-6).abs() * du.sign()
a_ik = torch.where(small_jump, 1.0 - 2.0 * u_avg, a_ik_raw)

sign_a = torch.sign(a_ik)
upwind = (a_ik * rel_x < 0).float()

edge_in = torch.cat([
    u0_bc, u_k_bc,
    du, du.abs(), u_avg,
    rel_x, abs_dx, slope,
    f_i, f_k,
    a_i, a_k, a_ik,
    sign_a, upwind,
    t_bc
], dim=-1)
```

### Update MLP input dimensions

Because the edge feature size increases, update:

```python
self.edge_mlp = _make_mlp(16, d_hidden, d_latent, 2, activation)
self.gate_net = _make_mlp(16, d_hidden, 1, 2, activation)
```

If `t_bc` is later removed from edge features, reduce the dimension accordingly.

---

## 4.5 About `t_j` / `t_bc`

There was confusion about `t_j`.

Clarification:

- in the math discussion, `t_j` referred to the broadcast time coordinate at a given time slice,
- in the code, the loop variable `j` inside the neighbor loop is a **spatial offset**, not a time index,
- so `t_bc` is really just the time coordinate `t`, broadcast to `[B, nt, nx, 1]`.

### Recommendation

- keep it for now if preserving the current “joint space-time lifting” design,
- but note that for LWR the most important lifting physics comes from `u0` and spatial interactions, not from time itself.

So `t_bc` is allowed, but it is not the core physics feature.

---

## 5. Precomputation plan for faster training

Many of the encoder edge features are static because they depend only on:

- `u0`,
- `x`,
- fixed stencil offsets.

Therefore they can be precomputed in the dataset.

## 5.1 Good candidates for precomputation

Per sample, precompute:

- `f0 = u0 * (1 - u0)`
- `a0 = 1 - 2 * u0`
- shifted neighbor copies for each stencil offset,
- `du`,
- `abs_du`,
- `u_avg`,
- `rel_x`,
- `abs_dx`,
- `slope`,
- `f_i`, `f_k`,
- `a_i`, `a_k`,
- `a_ik`,
- `sign_a`,
- `upwind`

### If `x` is fixed across the dataset

Then these can be global / shared:

- `rel_x`,
- `abs_dx`,
- stencil masks,
- possibly any constant geometric terms.

### What not to precompute

Do **not** precompute time-broadcasted copies unless necessary.

It is fine to expand `t` at runtime because that is cheap.

## 5.2 Suggested tensor layout for precomputed edge features

Recommended dataset output shape:

- either `[B, nx, 2k+1, d_edge]`,
- or `[B, 1, nx, 2k+1, d_edge]`

Then in `forward`, simply expand along time to `[B, nt, nx, 2k+1, d_edge]` if needed.

## 5.3 Tradeoff

Precomputation reduces compute in forward passes but increases:

- dataset storage,
- dataloader bandwidth.

Best compromise:

- precompute the spatial physics terms,
- do **not** precompute time-expanded copies,
- keep `t_bc` broadcasted at runtime.

---

## 6. Planned changes for the rest of the network (not yet to implement)

These are planned next steps after the encoder refactor.

---

## 6.1 Message-passing layers should stop using only `u0`

Current issue:

The MP layers still build important edge terms from `u0` even deep into the network.

This means transport bias remains too tied to the initial condition.

### Planned change

Inside each MP layer, decode a temporary scalar state from the latent field:

```python
self.state_probe = nn.Linear(d_latent, 1)
```

then:

```python
u_hat = torch.sigmoid(self.state_probe(h)).squeeze(-1)
f_hat = u_hat * (1.0 - u_hat)
a_hat = 1.0 - 2.0 * u_hat
```

and use those to build LWR-aware edge features dynamically.

### Reason

For nonlinear conservation laws, propagation direction is **state-dependent**. The network should use the evolving representation, not just the initial state.

---

## 6.2 Spatial MP layers should use interface features like a learned finite-volume scheme

Planned spatial edge features in MP layers:

- `du`,
- `abs_du`,
- `u_avg`,
- `rel_x`,
- `abs_dx`,
- slope,
- `f_i`, `f_j`,
- `a_i`, `a_j`,
- `a_ij`,
- `sign(a_ij)`,
- `upwind`.

This makes the MP update behave more like a learned Godunov / Rusanov interface interaction.

---

## 6.3 Temporal MP features should use CFL-like quantities

Current temporal features rely heavily on:

- absolute times,
- relative time,
- `x/t`.

### Planned change

Use CFL-like propagation features such as:

\[
|a| \Delta t / \Delta x
\]

computed from the decoded state.

### Position of `x/t`

`x/t` can remain as an optional auxiliary feature, especially for Riemann-type self-similar structure, but it should not be the main physical temporal feature.

---

## 6.4 Shock handling in PINN mode should preserve sign

Current PINN mode clamps messages to `[0, delta]` inside shock regions.

### Problem

This destroys message sign in regions where signed corrections matter.

### Planned change

Replace hard positive clamping with smooth attenuation, e.g.:

```python
contrib = alpha * msg
```

where `alpha in [0,1]` is derived from the shock indicator.

This preserves message direction while reducing unstable magnitude near shocks.

---

## 6.5 WENO-style smoothness should eventually be computed on decoded physical state

Current WENO mode computes smoothness indicators from the latent representation `h`.

### Planned change

Compute smoothness weights from:

- decoded scalar state `u_hat`, or
- decoded flux `f(u_hat)`

instead of relying only on latent differences.

This would make the smoothness weighting more physically interpretable and closer to real numerical smoothness indicators.

---

## 7. Concrete implementation tasks for the coding agent

## Phase 1: encoder-only refactor

### Task 1
Modify `_SpaceTimeLiftingLayer` edge feature construction to use LWR-aware features:

- keep node features as `[u0, x, t]`,
- replace edge features with the 16-dimensional set discussed above.

### Task 2
Update `self.edge_mlp` and `self.gate_net` input dimensions accordingly.

### Task 3
Keep the overall lifting structure unchanged:

- same node embedding,
- same spatial neighbor loop,
- same aggregation,
- same combine step.

### Task 4
Optionally add a configuration flag to include/exclude `t_bc` in edge features.

---

## Phase 2: optional precompute support

### Task 5
Add dataset-side precomputation for encoder spatial edge features.

Suggested approach:

- precompute edge physics features per sample and per stencil offset,
- pass them through the dataloader,
- reduce runtime construction inside the lifting layer.

### Task 6
Keep runtime expansion over time instead of storing time-broadcasted copies.

---

## Phase 3: later refactor of MP layers

### Task 7
Add per-layer `state_probe` heads to decode `u_hat` from latent features.

### Task 8
Refactor spatial MP edge features to use decoded LWR physics instead of only `u0`.

### Task 9
Refactor temporal MP edge features to include CFL-like features.

### Task 10
Replace shock-region hard clamping with signed attenuation.

### Task 11
Refactor WENO smoothness weighting to use decoded physical state / flux.

---

## 8. Minimal accepted first implementation

If only one change is implemented now, it should be this:

> Replace the lifting edge features with LWR-aware interface features.

That is the single highest-value modification from the discussion.

---

## 9. Summary of decisions

### Agreed now

- focus only on the encoder / lifting layer first,
- keep node features essentially unchanged,
- replace edge features with LWR-aware interface features,
- raw absolute coordinates do not need to be in edge features,
- precomputation is possible and likely worthwhile.

### Planned later

- use decoded latent state inside MP layers,
- move MP layers toward learned finite-volume behavior,
- use CFL-like temporal features,
- replace sign-destroying shock clamps,
- make WENO weighting more physically grounded.

---

## 10. Reference formulas for implementation

### LWR flux

```python
f(u) = u * (1 - u)
```

### Characteristic speed

```python
a(u) = 1 - 2 * u
```

### Interface speed

```python
a_ik = (f_k - f_i) / (u_k - u_i)
```

with fallback to

```python
1 - 2 * u_avg
```

when `|u_k - u_i|` is very small.

### Upwind flag

```python
upwind = (a_ik * rel_x < 0).float()
```

### Suggested edge feature order

```python
[
    u_i, u_k,
    du, abs_du, u_avg,
    rel_x, abs_dx, slope,
    f_i, f_k,
    a_i, a_k, a_ik,
    sign_a, upwind,
    t
]
```

---

## 11. Final note

The guiding principle for the coding agent should be:

> treat the lifting GNN as a learned local interface encoder for a conservation law, not just as a replacement for an MLP.

