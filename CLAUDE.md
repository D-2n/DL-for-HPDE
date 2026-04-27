# HypNO-ST3 Project Memory (2026-04-20)

## Project Overview
Space-time GNN operator (HypNO-ST3) for solving hyperbolic PDEs, specifically LWR traffic flow:
`u_t + f(u)_x = 0`, `f(u) = u(1-u)`, `a(u) = f'(u) = 1 - 2u`.

Key files:
- `hyperbolic_pde/models/hypno_st3.py` — main model
- `hyperbolic_pde/data/fvm.py` — MUSCL-Hancock FVM data generator
- `hyperbolic_pde/scripts/train_hypno_st3.py` — training script
- `hyperbolic_pde/scripts/generate_data.py` — data generation script
- `hyperbolic_pde/configs/hyperbolic_pde.yaml` — config
- `hypno_st.tex.txt` — paper writeup

---

## Architecture: HypNO-ST3

### Lifting layer (`_SpaceTimeLiftingLayer`)
- Node MLP: input `[u0_i, x_i, t_i, f0_i, a0_i]` (5 dims) → `d_latent`
  - `f0_i = u0*(1-u0)`, `a0_i = 1-2*u0` included explicitly so both `classic` and `physics` encoder modes have these features
- Edge MLP: unified, 14-dim input:
  `u0_i, u0_j, f0_i, f0_j, a0_i, a0_j, du0, sign(rel_x), rel_t, t_i, t_j, a0_ij, sign(a0_ij), is_adj_sp`
- Space-time Chebyshev ball neighbourhood: `|di| <= k_x`, `-k_t <= dm <= 0` (causal)
- Gate-normalized aggregation: `w_k = gate_k / sum(gate_j)`, weights sum to 1
- `encoder_type="mlp"` returns `h_node` immediately (no edge aggregation)
- `encoder_type="gnn"` runs edge MLP + combine MLP

### Message passing (`_PhysicsSpaceTimeMPLayer`)
- Two separate edge MLPs routed by adjacency:
  - `adj_msg`: adjacent-spatial edges (`dm==0, |di|==1`), input dim `2*d_latent + 12`
    - Features: `h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, a_ij, sign(a_ij), upwind, sign(rel_x)`
  - `nonadj_msg`: all other edges (non-adj spatial, temporal, diagonal), input dim `2*d_latent + 10`
    - Features: `h_i, h_j, u_i, u_j, f_i, f_j, a_i, a_j, rel_x, rel_t, cfl, sign(rel_x)`
- Physics gate (adj edges): `gate = g_upwind * g_entropy`
  - `g_upwind = sigmoid(-a_ij * rel_x / T)`
  - `g_entropy`: Oleinik entropy condition
- CFL gate (non-pure-spatial edges): `g_cfl = exp(-scale * relu(cfl - 1)^2)`
- Gate-normalized aggregation (same as lifting)
- Update: `h' = act(update_net([h, agg]) + W*h)`
- Only physics mode supported (PINN, WENO, Classic MP variants removed)

### Decoder
- `skip=True` (default): `u_pred = u0_expanded + decoder(h)`
- `skip=False`: `u_pred = decoder(h)`
- Each MP layer has a `state_probe` linear to decode intermediate `u_hat`

---

## Key fixes made this session

### 1. FVM temporal interpolation fix (`fvm.py`)
**Problem**: periodic error pattern (period ~3-4 timesteps) in predictions.
**Cause**: FVM takes variable internal timesteps; snapshots were recorded at the overshot state, not interpolated to the exact output time.
**Fix**: linear interpolation between bracketing steps:
```python
u_prev = u.copy()
t_prev = t
u = step_fn(u, dx, dt, boundary)
t += dt
while k < nt_out and t >= t_out[k] - 1e-12:
    alpha = (t_out[k] - t_prev) / (t - t_prev) if t > t_prev else 1.0
    u_hist[k] = (u_prev + alpha * (u - u_prev)).astype(np.float32)
    k += 1
```
**Action needed**: regenerate dataset with this fix.

### 2. Gate-normalized aggregation (lifting + MP layers)
**Problem**: old code used softmax attention before physics gate — softmax re-normalization undid the gate suppression.
**Fix**: removed `attn_score` linear entirely; use `gate_k / sum(gate_j)` directly:
```python
gate_sum = torch.stack(gates, dim=-2).sum(dim=-2).clamp(min=1e-12)
agg = agg + (gates[k] / gate_sum) * msg
```

### 3. Separate adj/non-adj MLPs in physics MP layer
Old: single edge MLP for all edges.
New: `adj_msg` and `nonadj_msg` with different feature vectors and optionally different hidden widths (`d_hidden_nonadj`).

### 4. f0_i, a0_i added to node MLP input
`node_mlp` input changed from 3 → 5 dims. `f0_i` and `a0_i` computed once and reused for both `node_in` and `edge_in` (no redundant recomputation).

---

## Config (intended for hypno_st3)
```yaml
model:
  stencil_k_x: 4
  stencil_k_t: 4
  d_latent: 32
  d_hidden: 64
  d_hidden_nonadj: 32
  n_layers: 6
  encoder_scaling: classic  # or physics
  encoder_type: gnn
  skip: true
```

---

## Removed from model
- `_ShockDetectorPINN`
- `_PINNSpaceTimeMPLayer`
- `_ClassicSpaceTimeMPLayer`
- `_WENOSpaceTimeMPLayer`
- `attn_score` linear layers (both lifting and MP)
- `shock_mode`, `weno_*`, `unified_mp`, `shock_delta`, `shock_threshold` params (kept `**_ignored` for backward compat)

---

## Neighbourhood size
With `k_x=4, k_t=4` (causal): `(2*4+1) * (4+1) - 1 = 44` edges per node per layer.
Each node sees `k_x` spatial cells left/right and `k_t` past timesteps (including diagonals).
Receptive field grows by `k_x` in space and `k_t` in time per layer.

---

## CLEPS Cluster Setup

**Access**: `ssh dzdrale@cleps.paris.inria.fr` (must be on INRIA-interne WiFi or INRIA VPN)

**Paths**:
- Repo: `/home/dzdrale/DL-for-HPDE`
- Venv: `/home/dzdrale/hypno_env` (Python 3.9, PyTorch cu124)
- Dataset: `/home/dzdrale/scratch/lwr_1d/hyperbolic_dataset.npz`
- Run outputs/checkpoints: `/home/dzdrale/scratch/runs/`
- SLURM logs: `/home/dzdrale/scratch/logs/`

**SLURM scripts**: `slurm/generate_data.sh`, `slurm/train_hypno_st3.sh`, `slurm/train_hypno_st3_mlp.sh`

**Hostname detection** (for auto config selection):
- Login node: `cleps` → matches `_HOST_CONFIGS["cleps"]`
- Compute nodes: `node0XX` → matches `_HOST_CONFIGS["node0"]`
- GPU nodes: `gpu0XX` → matches `_HOST_CONFIGS["gpu0"]`
- All map to `hyperbolic_pde_cleps.yaml`

**Config**: `hyperbolic_pde/configs/hyperbolic_pde_cleps.yaml` overrides data paths, num_samples, compile=false. MLP variant: `hyperbolic_pde_cleps_mlp.yaml`.

**GPU nodes** (as of 2026-04-23):
- `parq-gpu001`: A100-PCIE-40GB × 3
- `gpu012-013`: A100-SXM4-80GB × 4 (preferred for large runs)
- `gpu016-017, gpu015`: H100-80GB × 4
- `gpu018`: H200-141GB × 4
- Max walltime: 48 hours. User limits: 75 concurrent jobs, 2000 total submissions.

**Key commands**:
```bash
sbatch slurm/generate_data.sh                  # generate dataset
sbatch slurm/train_hypno_st3.sh                # submit GNN encoder training
sbatch slurm/train_hypno_st3_mlp.sh            # submit MLP encoder training
squeue -u $USER                                # check job status
scancel <JOBID>                                # cancel job
ssh <nodename> nvidia-smi                      # check GPU usage on compute node
tail -f /home/dzdrale/scratch/logs/<job>.log   # live log
sinfo -p gpu -o "%N %G %t"                     # check GPU availability
sbatch --exclude=<node> slurm/train_hypno_st3.sh  # avoid a busy node
```

**Python path**: scripts use `export PYTHONPATH=/home/dzdrale/DL-for-HPDE:$PYTHONPATH` (no setup.py/pyproject.toml).

**Known issues**:
- `torch.compile` fails on Python 3.9 — disabled via `compile: false` in cleps config
- Compute node hostnames are `node0XX` or `gpu0XX`, not `cleps` — all handled by `_HOST_CONFIGS`
- Install torch with `--index-url https://download.pytorch.org/whl/cu124` or you get CPU-only
