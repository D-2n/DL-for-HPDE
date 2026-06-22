import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--n", type=int, default=6)
parser.add_argument("--field", default="rho", choices=["rho", "w", "v"])
args = parser.parse_args()

d = np.load(args.data, allow_pickle=True)

print("Keys:", d.files)
print("rho shape:", d["rho"].shape)
print("p_form:", d["p_form"] if "p_form" in d else "missing")
print("tau:", d["tau"] if "tau" in d else "missing")
ic_types = d["ic_type"] if "ic_type" in d else ["?"] * d["rho"].shape[0]
print("IC type counts:")
for k in np.unique(ic_types):
    print(f"  {k}: {int((ic_types == k).sum())}")

field = d[args.field]
n = min(args.n, field.shape[0])
fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

for i in range(n):
    ax_top = axes[0, i]
    ax_bot = axes[1, i]

    im = ax_top.imshow(field[i], aspect="auto", origin="lower",
                       extent=[-1, 1, 0, 1], cmap="viridis")
    ax_top.set_title(f"{ic_types[i]} | segs={d['num_segments'][i] if 'num_segments' in d else '?'}", fontsize=7)
    ax_top.set_xlabel("x"); ax_top.set_ylabel("t")
    plt.colorbar(im, ax=ax_top)

    ax_bot.plot(d["rho"][i, 0], label="rho0")
    if "w0" in d:
        ax_bot.plot(d["w0"][i], label="w0")
    if "v0" in d:
        ax_bot.plot(d["v0"][i], label="v0")
    ax_bot.set_title(f"IC {i}", fontsize=7)
    ax_bot.legend(fontsize=6)

fig.suptitle(f"{args.data}  |  field={args.field}  |  N={field.shape[0]}", fontsize=9)
plt.tight_layout()
plt.savefig("dataset_plot.png", dpi=150)
print("saved dataset_plot.png")
plt.show()
