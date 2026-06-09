# Stain-Scanner Aware Pathology Foundation Model

> **Stain-Scanner Robustness for Whole Slide Image Representation Learning**
> A two-phase framework that (1) encodes stain-scanner condition as a compact vector, and (2) injects it into a ViT-based pathology foundation model via adaptive conditioning.

---

## Overview

Pathology AI models trained at one institution often fail when applied to slides from different hospitals due to variations in staining protocols, reagent concentrations, tissue processing, and scanner hardware. These **stain-scanner variations** are style factors unrelated to tissue morphology, yet they are encoded together in the embedding space of existing foundation models (e.g., UNI, Virchow, CONCH), causing performance degradation on unseen domains.

This work addresses domain variation not by removing it as noise but by **explicitly representing it** as a structural conditioning signal within the model:

1. **Phase 1** learns a compact Stain-Scanner Vector from multi-stain, multi-scanner patches using supervised contrastive learning + ABMIL aggregation.
2. **Phase 2** injects the vector into a ViT backbone via three architectural conditioning strategies (AdaLN, Prompt Token, Cross-Attention), enabling morphology-style disentanglement at the representation level.

---

## Repository Structure

```
.
├── phase1_stain_encoder/       ← Stain-Scanner Vector learning (PLISM dataset)
│   ├── configs/                ← step1 & step2 YAML configs (SM / WSI)
│   ├── step1/                  ← Patch-level SupCon encoder (ViT-L)
│   ├── step2/                  ← Bag-level ABMIL aggregator
│   ├── eval/                   ← Clustering, kNN, linear probe, UMAP
│   ├── scripts/                ← Training & evaluation shell scripts
│   └── README.md
│
└── phase2_stain_aware_ssl/     ← Stain-Aware ViT + DINO SSL
    ├── configs/                ← 6 configs (3 conditioning × 2 scenarios)
    ├── ssl/                    ← Model, dataset, loss, augmentation, utils
    ├── eval/                   ← Evaluation entry point
    ├── scripts/                ← Training & evaluation shell scripts
    └── README.md
```

---

## Phase 1: Stain-Scanner Vector Learning

### Dataset: PLISM

Multi-stain, multi-scanner pathology benchmark:
- **7 scanners** (AT2, GT450, P, S210, S360, S60, SQ)
- **13 stains** (GIV, GIVH, GM, GMH, GV, GVH, HR, HRH, KR, KRH, LM, LMH, MY)
- **46 tissue types**, **310,947 patches** (perfectly balanced: 3,417 per scanner×stain pair)

### Step 1 — Patch-Level Supervised Contrastive Learning

ViT-L/16 backbone trained with supervised contrastive loss, where the positive key is `device||stain` (e.g., `GT450||GIV`). Two augmented views of patches from the same scanner-stain combination are pulled together in embedding space.

```
Patch [256×256×3]  →  ViT-L backbone  →  MLP projector  →  L2-normalized [1024-d]
```

| Hyperparameter | Value |
|---|---|
| Backbone | `vit_large_patch16_224.augreg_in21k` |
| Embedding dim | 1024 |
| Batch size | 256 patches × 2 views |
| Optimizer | AdamW, lr=1e-4, cosine+warmup |
| Epochs | 100 |
| AMP | bfloat16 |
| Augmentation | 4-stage (geometric → distortion → blur/noise → color) |

### Step 2 — Bag-Level ABMIL Aggregator

Stage 1 backbone is frozen. An ABMIL aggregator is trained on top of pre-extracted features (bags of 32 patches) to produce a slide/condition-level 256-d Stain-Scanner Vector.

```
Features [N×1024]  →  Bag sample [32×1024]  →  ABMIL  →  MLP projector  →  L2-normalized [256-d]
```

| Aggregator | Params | MFLOPs/bag | Notes |
|---|---|---|---|
| **ABMIL** (proposed) | 527K | 34 | Gated attention, selected for deployment |
| CrossAttention | 4.2M | 138 | 8× heavier, marginal gain |
| MeanPool | 264K | 0.5 | Baseline |

### Evaluation Results (PLISM val/test)

| Metric | Value |
|---|---|
| pos_key silhouette | ~0.994 |
| Device linear probe accuracy | 1.00 |
| Stain silhouette | ~0.19 |

### Quick Start

```bash
cd phase1_stain_encoder

# Step 1: Train patch encoder
bash scripts/train_step1_sm.sh

# Extract features (backbone only, projector dropped)
bash scripts/extract_features.sh <step1_ckpt_sm> <step1_ckpt_wsi>

# Step 2: Train ABMIL aggregator (ablation: 6 configs)
bash scripts/train_step2_ablation.sh both

# Evaluate
bash scripts/eval.sh runs/step2_sm_abmil_*/stainvecs output/eval_sm
```

---

## Phase 2: Stain-Aware ViT + DINO SSL

Three architectural strategies for injecting the Stain-Scanner Vector into a ViT:

| Strategy | Mechanism | Notes |
|---|---|---|
| **AdaLN** | Generates scale/shift for each transformer block's LayerNorm | Strongest conditioning signal |
| **Prompt Token (VPT)** | Converts vector to prompt tokens prepended to patch sequence | Minimal backbone modification |
| **Cross-Attention** | Patch tokens attend to stain context via cross-attention layer | Best morphology-style separation |

Two training scenarios:
- **`scratch`**: Full DINO SSL from random init with stain conditioning
- **`uni_frozen`**: UNI backbone frozen; only conditioning modules + DINO head trained

### Quick Start

```bash
cd phase2_stain_aware_ssl

# Single config
bash scripts/train.sh configs/tcga_scratch_adaln.yaml

# All 6 configs
bash scripts/train_all_6_configs.sh

# Evaluate
bash scripts/eval.sh configs/tcga_scratch_adaln.yaml \
  runs/tcga_scratch_adaln_v1/best.pth \
  runs/tcga_scratch_adaln_v1/eval
```

> **Before running:** update `DATA.TRAIN_DIR`, `DATA.VAL_DIR`, and `DATA.STAIN_VECTOR_DIR` in the config files.

---

## End-to-End Inference (Phase 1)

A self-contained inference pipeline is available under `stain_vector/` (bundled weights included):

```python
from stain_vector.pipeline import StainVectorPipeline
import numpy as np

pipe = StainVectorPipeline(domain="sm", device="cuda:0")   # or domain="wsi"
patches = np.load("patches.npy")   # [N, 256, 256, 3] uint8
vec = pipe.infer(patches)          # → np.ndarray [256], L2-normalized
```

---

## Dependencies

```
torch >= 2.0
timm
numpy
pandas
pillow
scikit-learn
pyyaml
tqdm
seaborn        (evaluation visualizations)
matplotlib
umap-learn     (optional, for UMAP plots)
```

---

## Technical Highlights

- **Stain-Scanner Vector**: A 256-d L2-normalized bag-level descriptor capturing scanner + staining condition beyond simple color statistics, learned via supervised contrastive loss over 91 device×stain categories.
- **Morphology-Style Disentanglement**: Explicit structural separation between tissue morphology and stain/scanner style at the architecture level, in contrast to augmentation-based or normalization-based approaches.
- **Multi-Scale Aggregation**: ABMIL attention weights allow the model to focus on the most style-representative patches within a bag while ignoring uninformative (e.g., background) tiles.
- **Parameter-Efficient Adaptation**: The `uni_frozen` scenario adapts existing pathology foundation models (UNI) to domain robustness without retraining the full backbone.

---

## Citation

> *Work in progress. Citation information will be added upon publication.*

---

## License

Code: MIT  
PLISM dataset: see original dataset license.
