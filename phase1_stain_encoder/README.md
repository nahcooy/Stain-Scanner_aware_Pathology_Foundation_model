# Phase 1: Stain-Scanner Vector Generation via Supervised Contrastive Learning

> **Stain-Aware Pathology Foundation Model — Phase 1**
> ViT-L patch encoder + ABMIL aggregator trained to produce a compact 256-dim
> stain-scanner fingerprint that fully separates all 91 device × stain
> combinations present in the PLISM dataset.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1 PIPELINE                                   │
│                                                                             │
│   PLISM Patches          Step 1 — Patch Encoder                             │
│   ┌──────────┐           ┌──────────────────────────────────────────────┐   │
│   │  256×256 │──────────▶│  ViT-L/16 (ImageNet-21k pretrained)         │   │
│   │  patches │           │  Supervised Contrastive Loss                 │   │
│   │  SM/WSI  │           │  pos_key = device ║ stain                   │   │
│   └──────────┘           │  Output: 1024-dim patch features            │   │
│                          └──────────────────┬───────────────────────────┘   │
│                                             │  features/{sm,wsi}/{split}/   │
│                                             │  {pos_key}.npy  [N, 1024]     │
│                                             ▼                               │
│                          Step 2 — ABMIL Aggregator                          │
│                          ┌──────────────────────────────────────────────┐   │
│                          │  Bag: 32 patches from same pos_key           │   │
│                          │  ABMIL / CrossAttn / MeanPool (ablations)   │   │
│                          │  Supervised Contrastive Loss                 │   │
│                          │  Output: 256-dim L2-normalised stain vector │   │
│                          └──────────────────┬───────────────────────────┘   │
│                                             ▼                               │
│                          Stain Vector  [256-dim, L2-normalised]             │
│                          → 91 tight clusters (one per device×stain)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

Phase 1의 목표는 조직 병리 이미지에서 스캐너-염색 조합의 도메인 정보를 인코딩하는 **256차원 Stain Vector**를 학습하는 것입니다. 이 벡터는 이후 Phase 2의 도메인 적응(domain adaptation)에 활용됩니다.

---

## Dataset — PLISM

PLISM(Pathology Laboratory Image Scanning Metrics) 데이터셋은 7종의 스캐너와 13종의 염색법을 포함하는 대규모 병리 이미지 벤치마크입니다.

| Property          | Value                         |
|-------------------|-------------------------------|
| Scanners          | AT2, GT450, P, S210, S360, S60, SQ (7 total) |
| Stains            | HE, PAS, PSR, TRI, ... (13 total)            |
| Tissue types      | 46                                            |
| Total patches     | 310,947                                       |
| Unique pos_keys   | 91 (all scanner × stain combinations)         |
| Patches per key   | 3,417 (perfectly balanced)                    |
| Patch size        | 256 × 256 px                                  |
| Domains           | SM (Scanning Microscope), WSI (Whole-Slide)   |

**데이터 분할(Splits)**

| Split           | 설명                                     |
|-----------------|------------------------------------------|
| `train`         | 학습용                                   |
| `val`           | 검증용 (하이퍼파라미터 선택)               |
| `internal_test` | 내부 테스트 (seen scanner × stain)        |
| `unseen_test`   | 외부 일반화 테스트 (unseen combinations)  |

---

## Method

### Step 1 — Patch-Level Contrastive Encoder

ViT-L/16을 backbone으로 사용하며 `augreg_in21k` 사전학습 가중치에서 파인튜닝합니다. Supervised Contrastive Loss를 이용해 같은 `pos_key` (device||stain)에 속하는 패치를 가깝게, 다른 `pos_key`를 멀게 학습합니다.

- **입력**: 256×256 패치 (SM: float32 numpy arrays, WSI: uint8)
- **출력**: 1024-dim L2 정규화 특징 벡터
- **저장 경로**: `features/{sm,wsi}/{split}/{pos_key}.npy`  shape `[N, 1024]`
- **학습 설정**: cosine LR decay (warmup 5 epoch), AdamW, AMP(bf16), `torch.compile`

### Step 2 — Bag-Level Stain Vector Aggregator

Step 1에서 추출한 패치 특징을 "bag" 단위로 집계하여 최종 256-dim stain vector를 생성합니다.
하나의 bag = 같은 `pos_key`에서 무작위로 추출한 32개 패치.

| Aggregator        | Description                                    | Role     |
|-------------------|------------------------------------------------|----------|
| **ABMIL**         | Attention-Based MIL (2-layer MLP attention)    | Proposed |
| Cross-Attention   | Multi-head cross-attention pooling             | Ablation |
| Mean-Pool         | Simple mean aggregation                        | Ablation |

- **학습 설정**: 500 epoch, cosine LR (warmup 30 epoch), AdamW, AMP(bf16)
- **배치 구성**: 64 keys × 16 bags/key × 32 patches/bag = 32,768 patches/step

---

## Requirements

```
python >= 3.10
torch  >= 2.0
torchvision
timm >= 0.9
albumentations >= 1.3
scikit-learn >= 1.3
umap-learn >= 0.5
matplotlib >= 3.7
seaborn >= 0.12
numpy >= 1.24
pyyaml
tqdm
```

---

## Installation

```bash
# 1. 저장소 클론
git clone <repository_url>
cd phase1_stain_encoder

# 2. 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations scikit-learn umap-learn matplotlib seaborn pyyaml tqdm
```

---

## Quickstart

### 1. Step 1 — Patch Encoder 학습

```bash
# SM domain (cuda:3)
bash scripts/train_step1_sm.sh

# WSI domain (cuda:2) — 별도 터미널에서 병렬 실행 가능
bash scripts/train_step1_wsi.sh
```

### 2. Feature Extraction

```bash
bash scripts/extract_features.sh \
    runs/step1_sm/best.pth \
    runs/step1_wsi/best.pth
```

### 3. Step 2 — ABMIL Aggregator 학습 (ablation 포함)

```bash
# 3가지 aggregator를 서로 다른 GPU에서 병렬 실행
bash scripts/train_step2_ablation.sh both

# SM만 실행
bash scripts/train_step2_ablation.sh sm
```

### 4. Evaluation

```bash
bash scripts/eval.sh \
    runs/step2_sm_abmil/stainvecs \
    runs/step2_sm_abmil/eval
```

또는 직접 Python 호출:

```bash
python eval/run_eval.py \
    --stainvec_dir  runs/step2_sm_abmil/stainvecs \
    --output_dir    runs/step2_sm_abmil/eval \
    --splits        val internal_test unseen_test \
    --eval_pairs    val:internal_test val:unseen_test internal_test:unseen_test
```

---

## Directory Structure

```
phase1_stain_encoder/
├── configs/
│   ├── step1_sm.yaml          # Step 1 SM 학습 설정
│   ├── step1_wsi.yaml         # Step 1 WSI 학습 설정
│   ├── step2_sm.yaml          # Step 2 SM 학습 설정
│   └── step2_wsi.yaml         # Step 2 WSI 학습 설정
├── eval/
│   ├── metrics.py             # clustering / kNN / linear probe / retrieval
│   ├── visualize.py           # UMAP, t-SNE, scatter plots
│   └── run_eval.py            # 통합 평가 엔트리포인트
├── scripts/
│   ├── train_step1_sm.sh
│   ├── train_step1_wsi.sh
│   ├── extract_features.sh
│   ├── train_step2_ablation.sh
│   └── eval.sh
├── step1/                     # Step 1 학습/추론 코드 (별도 구현)
│   ├── train.py
│   ├── extract_features.py
│   ├── dataset.py
│   └── model.py
├── step2/                     # Step 2 학습/추론 코드 (별도 구현)
│   ├── train.py
│   ├── model.py
│   └── dataset.py
├── features/                  # Step 1 추출 특징 (자동 생성)
│   ├── sm/{split}/{pos_key}.npy
│   └── wsi/{split}/{pos_key}.npy
├── runs/                      # 학습 결과 체크포인트 (자동 생성)
└── README.md
```

---

## Configuration Reference

### Step 1 (`step1_{sm,wsi}.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `SEED` | int | 재현성을 위한 전역 시드 |
| `DOMAIN` | str | `sm` 또는 `wsi` |
| `PROCESSED_DIR` | str | PLISM 전처리 데이터 경로 |
| `IMAGE_SIZE` | int | 패치 크기 (256) |
| `TRAIN_BATCH_SIZE` | int | SM: 256, WSI: 128 (메모리 차이) |
| `AUG.STAGE` | int | 데이터 증강 강도 단계 (1–4) |
| `MODEL.TIMM_NAME` | str | timm 모델 식별자 |
| `MODEL.EMBED_DIM` | int | backbone 출력 차원 (1024) |
| `MODEL.PROJ_HIDDEN_DIM` | int | projection head 히든 차원 |
| `TRAIN.WARMUP_EPOCHS` | int | cosine LR warmup 에폭 수 |
| `TRAIN.MIN_LR_RATIO` | float | 최솟값 = BASE_LR × MIN_LR_RATIO |
| `LOSS.TEMPERATURE` | float | SupCon temperature τ |
| `LOSS.POSITIVE_LOSS_WEIGHT` | float | positive pair 추가 가중치 |
| `AMP.DTYPE` | str | `bf16` 또는 `fp16` |
| `MODEL.TORCH_COMPILE` | bool | `torch.compile` 활성화 여부 |

### Step 2 (`step2_{sm,wsi}.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `FEAT_DIR` | str | Step 1 특징 디렉터리 |
| `BAG.BAG_SIZE` | int | bag당 패치 수 (32) |
| `BAG.BATCH_KEYS` | int | 배치당 pos_key 수 (64) |
| `BAG.BAGS_PER_KEY` | int | key당 bag 수 (16) |
| `BAG.STEPS_PER_EPOCH` | int | epoch당 학습 스텝 수 |
| `MODEL.AGG_TYPE` | str | `abmil` / `cross_attention` / `mean_pool` |
| `MODEL.EMBED_DIM` | int | 출력 stain vector 차원 (256) |
| `MODEL.AGG_HIDDEN_DIM` | int | ABMIL attention 히든 차원 |
| `MODEL.AGG_HEADS` | int | CrossAttn head 수 |
| `TRAIN.EPOCHS` | int | 총 학습 에폭 (500) |
| `TRAIN.WARMUP_EPOCHS` | int | cosine LR warmup (30) |

---

## Evaluation Metrics

평가 지표는 `eval/metrics.py`에 구현되어 있으며, 모든 함수는 내부적으로 L2 정규화를 적용합니다.

| Metric | 함수 | 설명 |
|--------|------|------|
| **Silhouette Score** | `clustering_metrics` | 클러스터 내 응집도 vs 분리도. 1에 가까울수록 좋음 |
| **Davies-Bouldin Index** | `clustering_metrics` | 클러스터 간 거리 대비 내부 산포. 낮을수록 좋음 |
| **Calinski-Harabasz Index** | `clustering_metrics` | 클러스터 간 산포 vs 내부 산포 비율. 높을수록 좋음 |
| **KMeans ARI** | `clustering_metrics` | KMeans 예측 vs 실제 레이블 Adjusted Rand Index |
| **KMeans NMI** | `clustering_metrics` | KMeans 예측 vs 실제 레이블 Normalized Mutual Info |
| **kNN Purity@k** | `knn_purity` | 가장 가까운 k개 이웃 중 같은 레이블의 비율 |
| **Linear Probe Acc.** | `linear_probe` | LogReg 선형 분류 정확도 (scanner / stain 분리 능력) |
| **Linear Probe F1** | `linear_probe` | Macro/Weighted F1 |
| **Retrieval Recall@k** | `retrieval_metrics` | Cross-split: 쿼리의 top-k 검색 결과 중 정답 비율 |

---

## Results

### Aggregator Ablation (SM domain, val split)

| Model | pos_key Silhouette ↑ | Device Linear Probe ↑ | Stain Linear Probe ↑ |
|---|---|---|---|
| **ABMIL (proposed)** | ~0.994 | 1.000 | ~0.85 |
| Cross-Attention | ~0.994 | 1.000 | ~0.85 |
| Mean-Pool | ~0.980 | 0.998 | ~0.82 |

**주요 결과 해석:**

- **ABMIL**과 **Cross-Attention** 모두 pos_key silhouette ~0.994를 달성하며 91개 scanner×stain 조합을 거의 완벽하게 분리합니다.
- **Device Linear Probe 1.000**은 스캐너 정보가 stain vector에 완전히 인코딩됨을 의미합니다.
- **Mean-Pool** 기준선도 강력한 성능을 보이나 stain 분류에서 ~3%p 하락, silhouette에서 약간의 저하가 관찰됩니다.
- 계산 비용 대비 성능을 고려할 때 **ABMIL이 최적 선택**입니다.

---

## Citation

본 연구가 도움이 되었다면 아래 형식으로 인용해 주세요 (출판 후 업데이트 예정):

```bibtex
@article{stain_aware_pathology_2025,
  title   = {Stain-Aware Pathology Foundation Model},
  author  = {[Authors]},
  journal = {[Journal/Conference]},
  year    = {2025},
  note    = {Preprint}
}
```

---

## License

본 코드는 연구 목적으로 공개되었습니다. 상업적 사용 전 별도 문의 바랍니다.
