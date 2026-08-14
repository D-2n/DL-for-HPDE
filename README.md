# DL-for-HPDE

Neural operators for 1-D hyperbolic conservation laws (LWR traffic and the Aw–Rascle–Zhang system).

```bash
pip install -r requirements.txt
```

Run the commands below from the repository root.

## Datasets

LWR scripts write to the `path` field in `hyperbolic_pde/configs/hyperbolic_pde.yaml` (cluster scratch by default); change that to a local file first.

**LWR train**

```bash
python hyperbolic_pde/scripts/generate_data.py
```

**LWR eval**

```bash
python hyperbolic_pde/scripts/generate_ood_data.py --block paper_eval_data
```

**ARZ WFT train** (HypNO-ARZ train and HLL finetune)

```bash
python -m hyperbolic_pde.arz.datagen_arz \
  --out hyperbolic_pde/arz/data/arz_mixed_wft_prho_clean.npz \
  --fv-solver wft
```

**ARZ HLL pretrain**

```bash
python -m hyperbolic_pde.arz.datagen_arz \
  --out hyperbolic_pde/arz/data/arz_mixed_hll_prho.npz
```

**ARZ eval** (both ARZ models)

```bash
python -m hyperbolic_pde.arz.gen_evaluation_local
```
