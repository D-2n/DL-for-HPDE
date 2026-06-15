"""Equivalence check + memory/time benchmark for the ARZ mark2 / mark2.1
chunked edge-message aggregation mode.

Two model classes (`HypNO_ARZ_Mark2`, `HypNO_ARZ_Mark2_1`) gained an
`aggregation_mode` flag ("materialized" | "chunked") with a configurable
`edge_chunk_size`. The chunked mode processes the (dominant) non-adjacent edges
in dense `[B, nt, nx, C, features]` chunks instead of materializing the entire
`[B, nt, nx, E, d]` edge batch at once, trading a little speed for a large drop
in peak activation memory. The adjacent-spatial physics edges (always 2) are
computed exactly as before and folded into the same num/den accumulators, so the
aggregation `agg = sum_k gate_k * msg_k / (sum_k gate_k + eps)` is preserved.

This script:
  1. EQUIVALENCE: same weights + same input, materialized vs chunked, for every
     requested chunk size; reports max-abs / max-rel error on the (rho,v,w)
     outputs and on the gradients. Runs an fp64 pass (tight assert) and reports
     the working-dtype error too.
  2. BENCHMARK: peak GPU memory, forward time, backward time, and full
     train-step time for materialized and each chunk size.
  3. OOM FRONTIER: largest (batch, nx, nt) each mode handles before OOM.
  4. Prints a summary table and a one-line best-tradeoff recommendation.

Run on a GPU node for the memory/time sweep; the fp64 equivalence gate runs fine
on CPU.

    python -m hyperbolic_pde.arz.bench_chunked_aggregation --device cuda
    python -m hyperbolic_pde.arz.bench_chunked_aggregation --device cpu --equiv-only
"""
from __future__ import annotations

import argparse
import time
from copy import deepcopy

import torch

from hyperbolic_pde.arz import physics_arz as _P_MOD
from hyperbolic_pde.arz.model_arz_mark2 import HypNO_ARZ_Mark2
from hyperbolic_pde.arz.model_arz_mark2_1 import HypNO_ARZ_Mark2_1

_MODEL_CLASSES = {"mark2": HypNO_ARZ_Mark2, "mark2_1": HypNO_ARZ_Mark2_1}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_inputs(B, nx, nt, device, dtype, seed=0):
    """Random but physically valid (rho>0) ARZ inputs on a unit grid."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    rho0 = (0.2 + 0.6 * torch.rand(B, nx, generator=g)).to(device=device, dtype=dtype)
    # w0 = v0 + p(rho0); pick v0 in a sane band so w is well-scaled.
    v0 = (0.1 + 0.8 * torch.rand(B, nx, generator=g)).to(device=device, dtype=dtype)
    p0 = _P_MOD.pressure(rho0)
    w0 = v0 + p0
    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
    return rho0, w0, x, t


def build_pair(model_cls, kwargs, edge_chunk_size, device, dtype, seed=0):
    """Build a materialized and a chunked model sharing identical weights."""
    torch.manual_seed(seed)
    mat = model_cls(aggregation_mode="materialized", **kwargs).to(device=device, dtype=dtype)
    chk = model_cls(
        aggregation_mode="chunked", edge_chunk_size=edge_chunk_size, **kwargs
    ).to(device=device, dtype=dtype)
    chk.load_state_dict(deepcopy(mat.state_dict()))
    mat.eval()
    chk.eval()
    return mat, chk


def rel_err(a, b):
    num = (a - b).abs().max().item()
    den = b.abs().max().clamp(min=1e-30).item()
    return num, num / den


def run_forward(model, rho0, w0, x, t):
    rho_p, v_p, w_p, _ = model.forward_primitive(rho0, w0, x, t)
    return rho_p, v_p, w_p


# --------------------------------------------------------------------------- #
# 1. equivalence
# --------------------------------------------------------------------------- #
def check_equivalence(model_cls_name, kwargs, chunk_sizes, device, B, nx, nt):
    model_cls = _MODEL_CLASSES[model_cls_name]
    print(f"\n=== EQUIVALENCE [{model_cls_name}] (B={B}, nx={nx}, nt={nt}) ===")
    print(f"{'dtype':>7} {'chunk':>6} {'out|abs':>11} {'out|rel':>11} "
          f"{'grad|abs':>11} {'grad|rel':>11}  status")

    worst_rel = 0.0
    for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 2e-4)):
        rho0, w0, x, t = make_inputs(B, nx, nt, device, dtype)
        for cs in chunk_sizes:
            mat, chk = build_pair(model_cls, kwargs, cs, device, dtype)

            # outputs
            with torch.no_grad():
                rm, vm, wm = run_forward(mat, rho0, w0, x, t)
                rc, vc, wc = run_forward(chk, rho0, w0, x, t)
            oa = max(rel_err(rm, rc)[0], rel_err(vm, vc)[0], rel_err(wm, wc)[0])
            orr = max(rel_err(rm, rc)[1], rel_err(vm, vc)[1], rel_err(wm, wc)[1])

            # gradients w.r.t a scalar loss on a fixed shared target
            def grad_of(model):
                model.zero_grad(set_to_none=True)
                r, v, w = run_forward(model, rho0, w0, x, t)
                loss = (r.pow(2).mean() + v.pow(2).mean() + w.pow(2).mean())
                loss.backward()
                return torch.cat([p.grad.reshape(-1) for p in model.parameters()
                                  if p.grad is not None])

            gm = grad_of(mat)
            gc = grad_of(chk)
            ga, grr = rel_err(gm, gc)

            worst_rel = max(worst_rel, orr, grr)
            ok = (oa < tol and ga < tol) if dtype == torch.float64 else (orr < tol and grr < tol)
            print(f"{str(dtype).split('.')[-1]:>7} {cs:>6} {oa:>11.2e} {orr:>11.2e} "
                  f"{ga:>11.2e} {grr:>11.2e}  {'OK' if ok else 'FAIL'}")
            if dtype == torch.float64:
                assert oa < tol and ga < tol, (
                    f"fp64 equivalence FAILED for chunk={cs}: out|abs={oa:.2e}, "
                    f"grad|abs={ga:.2e} (tol {tol:.0e}). Chunked aggregation is "
                    f"NOT numerically equivalent to materialized."
                )
    print(f"worst working-precision rel error: {worst_rel:.2e}")
    return worst_rel


# --------------------------------------------------------------------------- #
# 2. benchmark
# --------------------------------------------------------------------------- #
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_model(model, rho0, w0, x, t, device, iters, warmup):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    def step(do_opt):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        opt.zero_grad(set_to_none=True)
        _sync(device); t0 = time.perf_counter()
        r, v, w = run_forward(model, rho0, w0, x, t)
        loss = r.pow(2).mean() + v.pow(2).mean() + w.pow(2).mean()
        _sync(device); t1 = time.perf_counter()
        loss.backward()
        _sync(device); t2 = time.perf_counter()
        if do_opt:
            opt.step()
        _sync(device); t3 = time.perf_counter()
        peak = (torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else float("nan"))
        return (t1 - t0), (t2 - t1), (t3 - t0), peak

    for _ in range(warmup):
        step(True)
    fwd = bwd = stp = 0.0
    peak = 0.0
    for _ in range(iters):
        f, b, s, p = step(True)
        fwd += f; bwd += b; stp += s; peak = max(peak, p)
    return fwd / iters, bwd / iters, stp / iters, peak


def benchmark(model_cls_name, kwargs, chunk_sizes, device, B, nx, nt, iters, warmup):
    model_cls = _MODEL_CLASSES[model_cls_name]
    print(f"\n=== BENCHMARK [{model_cls_name}] (B={B}, nx={nx}, nt={nt}, "
          f"iters={iters}, warmup={warmup}) ===")
    dtype = torch.float32
    rho0, w0, x, t = make_inputs(B, nx, nt, device, dtype)

    rows = []
    # materialized
    torch.manual_seed(0)
    mat = model_cls(aggregation_mode="materialized", **kwargs).to(device=device, dtype=dtype)
    f, b, s, p = time_model(mat, rho0, w0, x, t, device, iters, warmup)
    rows.append(("materialized", "-", f, b, s, p))
    del mat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for cs in chunk_sizes:
        torch.manual_seed(0)
        chk = model_cls(aggregation_mode="chunked", edge_chunk_size=cs,
                        **kwargs).to(device=device, dtype=dtype)
        try:
            f, b, s, p = time_model(chk, rho0, w0, x, t, device, iters, warmup)
            rows.append((f"chunked", cs, f, b, s, p))
        except torch.cuda.OutOfMemoryError:
            rows.append((f"chunked", cs, float("nan"), float("nan"),
                         float("nan"), float("nan")))
            torch.cuda.empty_cache()
        del chk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"{'mode':>13} {'chunk':>6} {'fwd(ms)':>9} {'bwd(ms)':>9} "
          f"{'step(ms)':>9} {'peak(MB)':>10}")
    for mode, cs, f, b, s, p in rows:
        print(f"{mode:>13} {str(cs):>6} {f*1e3:>9.2f} {b*1e3:>9.2f} "
              f"{s*1e3:>9.2f} {p:>10.1f}")

    # best tradeoff: among chunked rows that ran, the one minimizing
    # peak_mem * step_time (normalized to materialized).
    base = rows[0]
    base_mem, base_step = base[5], base[4]
    best = None
    for mode, cs, f, b, s, p in rows[1:]:
        if p != p:  # nan
            continue
        score = (p / base_mem) * (s / base_step) if device.type == "cuda" else s
        if best is None or score < best[0]:
            best = (score, cs, p, s)
    if best is not None and device.type == "cuda":
        _, cs, p, s = best
        print(f"\nbest mem*time tradeoff: chunk={cs}  "
              f"peak {p:.0f}MB ({p/base_mem:.2f}x materialized), "
              f"step {s*1e3:.2f}ms ({s/base_step:.2f}x materialized)")
    return rows


# --------------------------------------------------------------------------- #
# 3. OOM frontier
# --------------------------------------------------------------------------- #
def oom_frontier(model_cls_name, kwargs, device, chunk_size,
                 batch_list, nx, nt):
    if device.type != "cuda":
        print("\n(OOM frontier skipped: needs CUDA)")
        return
    model_cls = _MODEL_CLASSES[model_cls_name]
    print(f"\n=== OOM FRONTIER [{model_cls_name}] (nx={nx}, nt={nt}, "
          f"chunk={chunk_size}) ===")
    dtype = torch.float32
    for mode, ck in (("materialized", None), ("chunked", chunk_size)):
        largest = None
        for B in batch_list:
            torch.cuda.empty_cache()
            try:
                torch.manual_seed(0)
                mk = dict(kwargs)
                if mode == "chunked":
                    model = model_cls(aggregation_mode="chunked",
                                      edge_chunk_size=ck, **mk).to(device, dtype)
                else:
                    model = model_cls(aggregation_mode="materialized",
                                      **mk).to(device, dtype)
                rho0, w0, x, t = make_inputs(B, nx, nt, device, dtype)
                r, v, w, _ = model.forward_primitive(rho0, w0, x, t)
                (r.pow(2).mean() + v.pow(2).mean() + w.pow(2).mean()).backward()
                largest = B
                del model, r, v, w
            except torch.cuda.OutOfMemoryError:
                del model
                torch.cuda.empty_cache()
                break
        tag = f"chunk={chunk_size}" if mode == "chunked" else "materialized"
        print(f"  {tag:>16}: largest batch fitting = {largest}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="mark2", choices=list(_MODEL_CLASSES))
    ap.add_argument("--pressure-form", default="rho+rho2",
                    choices=["rho", "rho+rho2"])
    ap.add_argument("--chunk-sizes", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32, 64, 89])
    # equivalence uses a small model; benchmark uses a production-ish one.
    ap.add_argument("--equiv-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--k-x", type=int, default=7)
    ap.add_argument("--k-t", type=int, default=5)
    ap.add_argument("--d-latent", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--bench-batch", type=int, default=4)
    ap.add_argument("--bench-nx", type=int, default=128)
    ap.add_argument("--bench-nt", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--oom-chunk", type=int, default=16)
    ap.add_argument("--oom-batches", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64])
    args = ap.parse_args()

    _P_MOD.set_pressure_form(args.pressure_form)
    device = torch.device(args.device)
    print(f"device={device} torch={torch.__version__} "
          f"pressure_form={args.pressure_form} model={args.model}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    base_kwargs = dict(
        stencil_k_x=args.k_x, stencil_k_t=args.k_t,
        d_latent=args.d_latent, d_hidden=args.d_latent,
        causal_temporal=True, decoder_depth=3,
        adj_depth=3, nonadj_depth=3, update_depth=3,
        normalize_edge_offsets=True, use_checkpoint=False,
    )

    if not args.bench_only:
        # small, shallow model for a fast exact check
        equiv_kwargs = dict(base_kwargs)
        equiv_kwargs.update(d_latent=32, d_hidden=32, n_layers=2)
        check_equivalence(args.model, equiv_kwargs, args.chunk_sizes,
                          device, B=2, nx=24, nt=12)

    if not args.equiv_only:
        bench_kwargs = dict(base_kwargs)
        bench_kwargs.update(n_layers=args.n_layers)
        benchmark(args.model, bench_kwargs, args.chunk_sizes, device,
                  args.bench_batch, args.bench_nx, args.bench_nt,
                  args.iters, args.warmup)
        oom_frontier(args.model, bench_kwargs, device, args.oom_chunk,
                     args.oom_batches, args.bench_nx, args.bench_nt)


if __name__ == "__main__":
    main()
