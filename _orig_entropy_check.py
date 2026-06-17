import torch
from hyperbolic_pde.arz.model_arz_orig import HypNO_ARZ_Orig

kw = dict(stencil_k_x=2, stencil_k_t=2, d_latent=16, d_hidden=24, n_layers=2)

m_legacy = HypNO_ARZ_Orig(new_entropy=False, **kw)
m_new = HypNO_ARZ_Orig(new_entropy=True, **kw)

# Checkpoint compatibility: state_dict keys must be identical.
k_legacy = set(m_legacy.state_dict().keys())
k_new = set(m_new.state_dict().keys())
assert k_legacy == k_new, "state_dict keys differ -> checkpoint incompatibility!"
print("state_dict keys identical:", len(k_legacy), "tensors")

# A legacy checkpoint must load into a new_entropy=True model with no errors.
m_new.load_state_dict(m_legacy.state_dict(), strict=True)
print("legacy ckpt -> new_entropy model: strict load OK")

# Flags wired through.
assert m_legacy.new_entropy is False and m_new.new_entropy is True
assert m_legacy.lifting.new_entropy is False and m_new.lifting.new_entropy is True
assert all(not l.new_entropy for l in m_legacy.mp_layers)
assert all(l.new_entropy for l in m_new.mp_layers)
print("new_entropy threaded to lifting + all mp_layers")

# Forward pass in BOTH modes (exercises the new s_1 branch).
B, nt, nx = 2, 5, 12
rho0 = torch.rand(B, nt, nx) * 0.5 + 0.2
w0 = torch.rand(B, nt, nx) * 0.5 + 0.3
x = torch.linspace(0, 1, nx).view(1, 1, nx).expand(B, nt, nx).contiguous()
t = torch.linspace(0, 1, nt).view(1, nt, 1).expand(B, nt, nx).contiguous()

for name, m in [("legacy", m_legacy), ("new_entropy", m_new)]:
    m.eval()
    with torch.no_grad():
        rho_p, w_p, _ = m(rho0, w0, x, t)
    assert rho_p.shape == (B, nt, nx) and torch.isfinite(rho_p).all()
    assert w_p.shape == (B, nt, nx) and torch.isfinite(w_p).all()
    print(f"forward {name}: OK  rho mean={rho_p.mean():.4f} w mean={w_p.mean():.4f}")

print("ALL CHECKS PASSED")
