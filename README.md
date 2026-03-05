# 2025 자율주행 AI 챌린지 - Semantic Segmentation (Camera)

2025 제2회 자율주행 AI챌린지(과학기술정보통신부)  
카메라 기반 자율주행 환경에서 **객체를 픽셀 단위로 구분하는 Semantic Segmentation**을 수행하기 위한 학습/평가 + 데이터 전처리(oversampling/증강) 코드입니다.

---

## 핵심 아이디어
- **DDRNet 기반 세그멘테이션 학습 + Teacher-Student Knowledge Distillation**
- **normal 이미지에 degradation 증강을 “체인(연속 적용)”으로 확률적으로 주입**
- **희소 클래스(rare class) 픽셀 비율 기반 oversampling CSV로 학습 분포 보정**

---

## 코드 구성 (주요 파일)
### 학습/평가
- `train_kd.py` : DDP 기반 KD 학습(Teacher: DualResNet, Student: DDRNet)

### 모델/로스
- `DDRNet.py` : Student 모델
- `DDRNet_39.py` : Teacher(DualResNet) 관련
- `kd_losses/` : DCSF Feature KD / Output KD 등

### 전처리 / 샘플링 / 증강
- `over_csv.py` : rare class 기반 oversampling CSV 생성
- `over_csv_check.py` : 생성된 CSV 점검
- `label_alignment_for_oversampled.py` : oversampling 후 라벨 정합/정리
- `degra_for_aug.py` : 학습용 degradation 증강(transform) 구현
- `save_augmented.py` : 증강 결과 저장/시각화

---

## 데이터셋 폴더 구조 (중요)
`SegmentationDataset`은 아래 구조를 전제로 이미지/라벨을 매핑합니다.

```
<DATASET_ROOT>/
  train/
    normal/       (또는 low_light / overbright / degradation)
      image/...
      labelmap/...
  val/
    normal/       (또는 low_light / overbright / degradation)
      image/...
      labelmap/...
```

- `image/` : RGB 이미지
- `labelmap/` : 라벨 PNG(그레이스케일, 픽셀 값이 class id)
- ignore label: `255` (`ignore_index=255`)

---

## Degradation 증강 방식 (체인)
학습 시 **tag가 normal인 샘플에만**, 아래 체인 중 하나를 확률적으로 골라 **연속 적용**합니다.

- `("rain", "raindrop", "low_light")`
- `("rain", "raindrop")`
- `("rain", "raindrop", "haze")`
- `("low_light",)`
- `("haze",)`

동작 흐름:
1) 배치 로드  
2) Train이면 기하 증강(스케일/랜덤크롭/좌우반전)  
3) `tag == "normal"`인 경우 `normal_aug_prob` 확률로 체인 1개 선택  
4) 체인 내부 각 증강은 severity를 `[severity_min, severity_max]`에서 랜덤으로 적용  
5) 적용된 (증강명, severity)는 meta에 기록되어 epoch 단위 카운트 가능  

---

## 빠른 실행 가이드

### 1) 환경
- Python + PyTorch(CUDA) + torchvision
- OpenCV, Pillow, numpy, tqdm
- tensorboardX

### 2) Oversampling CSV 생성
```bash
python over_csv.py
python over_csv_check.py
```

### 3) KD 학습 (DDP)
```
torchrun --nproc_per_node=2 train_kd.py \
  --dataset_dir ./dataset/SemanticDataset/ \
  --loadpath ./DDRNet23s_imagenet.pth \
  --teacher_loadpath ./teacher/teacher_weights_only.pth \
  --result_dir ./pths/DDRNet_kd_run01 \
  --epochs 500 \
  --batch_size 8 \
  --lr 5e-4 \
  --crop_size [1024,1024] \
  --scale_range [0.75,1.25] \
  --normal_aug_prob 0.8 \
  --severity_min 1 \
  --severity_max 5
```
