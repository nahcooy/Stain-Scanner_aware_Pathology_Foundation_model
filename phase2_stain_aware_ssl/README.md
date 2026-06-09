#!/usr/bin/env markdown
# Phase 2: Stain-Aware ViT with DINO SSL (TCGA PNG Patches)

본 디렉토리는 `phase1_stain_encoder` 스타일을 유지하면서, Phase 2 목표인
Stain-Scanner conditioning 비교(AdaLN / Prompt / Cross-Attention)와
`scratch` vs `UNI frozen` 시나리오를 통합한 학습/평가 코드를 제공합니다.

## 핵심 구성

- **SSL 목표**: TCGA patch PNG 기반 DINO self-supervised representation learning
- **Conditioning 방법 3종**
  - `adaln`
  - `prompt` (Stain Prompt Token / VPT-deep 지원)
  - `cross_attention` (patch tokens → stain context token)
- **학습 시나리오 2종**
  - `scratch`
  - `uni_frozen` (backbone freeze + conditioning/head 학습)
- **총 6개 config 제공**: `configs/*.yaml`

## 디렉토리 구조

```text
phase2_stain_aware_ssl/
├── configs/
│   ├── tcga_scratch_adaln.yaml
│   ├── tcga_scratch_prompt.yaml
│   ├── tcga_scratch_cross_attention.yaml
│   ├── tcga_uni_frozen_adaln.yaml
│   ├── tcga_uni_frozen_prompt.yaml
│   └── tcga_uni_frozen_cross_attention.yaml
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── train_all_6_configs.sh
├── ssl/
│   ├── augmentations.py
│   ├── dataset.py
│   ├── loss.py
│   ├── model.py
│   ├── train.py
│   ├── run_eval.py
│   └── utils.py
└── eval/
    └── run_eval.py
```

## 데이터 포맷

### 1) Patch 디렉토리

`TRAIN_DIR`, `VAL_DIR` 아래 PNG를 재귀적으로 스캔합니다.

```text
train/
  BRCA/
    slide_a_patch_0001.png
    ...
  LUAD/
    ...
val/
  BRCA/
  LUAD/
```

### 2) Stain Vector 매핑

아래 두 방식 중 하나를 사용합니다.

1. **Mirrored directory 방식** (`STAIN_VECTOR_DIR`)
   - patch 상대경로를 그대로 따라가며 확장자만 `.npy`로 변경
   - 예: `train/BRCA/a.png` → `stain_vectors/train/BRCA/a.npy`
2. **CSV index 방식** (`STAIN_INDEX_CSV`)
   - 헤더에 이미지 경로 컬럼(`image_path`/`patch_path`/`path`/`relative_path`)
   - 벡터 경로 컬럼(`stain_vector_path`/`vector_path`/`stain_path`)

`STAIN_DIM`은 기본 256입니다.

## 학습 실행

```bash
cd /home/work/workspace/nahcooy/stain/phase2_stain_aware_ssl
bash scripts/train.sh configs/tcga_scratch_adaln.yaml
```

6개 전부 순차 실행:

```bash
bash scripts/train_all_6_configs.sh
```

## 평가 실행

```bash
bash scripts/eval.sh \
  configs/tcga_scratch_adaln.yaml \
  runs/tcga_scratch_adaln_v1/best.pth \
  runs/tcga_scratch_adaln_v1/eval
```

평가 결과:
- `embeddings_train.npz`
- `embeddings_val.npz`
- `metrics.json` (kNN / linear probe)

## 의존성

- Python >= 3.10
- torch >= 2.0
- torchvision
- timm
- pyyaml
- tqdm
- pillow
- numpy
- scikit-learn (평가 지표용)

## 구현 메모

- 현재 기본 config는 안정성을 위해 `N_LOCAL_CROPS: 0`으로 설정되어 있습니다.
- local crops를 사용하려면 `AUG.N_LOCAL_CROPS`를 늘리고 batch/GPU 메모리에 맞게 조정하십시오.
- `uni_frozen` 설정에서는 `MODEL.BACKBONE.UNI_SOURCE`를 실제 checkpoint 또는
  `hf-hub:` source로 바꿔서 사용하십시오.

