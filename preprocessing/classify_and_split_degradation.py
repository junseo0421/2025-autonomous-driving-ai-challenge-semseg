# classify_and_split_degradation_multi_relaxed.py
import cv2, os, json, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

# ---- 분류 대상: low_light / degradation / overbright ----
CATEGORIES = ["low_light", "degradation", "overbright"]

LOW_LIGHT_CFG = dict(
    mean_Y_max     = 0.33,   # 0.34 → 0.33  (평균 밝기 0.339, 0.335 컷)
    dark_ratio_min = 0.43,   # 0.32 → 0.43  (dark_ratio 0.418 컷)
    alt_mean_Y_max = 0.31,   # 0.32 → 0.31  (보조분기 타이트)
    std_Y_max      = 0.205   # 0.21 → 0.205 (보조분기 타이트)
)

# (구) haze 임계값 → veiling-type degradation 프록시 임계값
DEGR_CFG = dict(
    dcp_mean_min = 0.285,  # 0.325 → 0.285
    std_Y_max    = 0.285   # 0.280 → 0.285
)

# 간단 보강 게이트(선택)
DEGR_ENHANCE = dict(
    use_blur_gate = True,      # Laplacian 분산 기반 블러 게이트 사용
    lap_var_max   = 0.0015,    # gray(0~1 정규화)에서 Laplacian 분산 임계(데이터로 튜닝 권장)
    use_sat_gate  = False,     # HSV S 채널 기반 채도 게이트 사용 여부
    mean_S_max    = 0.30       # S 평균 임계(0~1), 낮을수록 베일/수막 가능성↑
)

OVERBRIGHT_CFG = dict(
    mean_Y_min       = 0.59,  # 0.52 → 0.59  (평균밝기만 높은 케이스 과검출 억제)
    bright_ratio_min = 0.17   # 0.09 → 0.17  (밝은 픽셀 비율만 높은 케이스 과검출 억제)
)

# ROI 컷 설정 (현재 로직에서는 사용하지 않지만, 향후 확장 대비 유지 가능)
ROI_TOP_CUT = 0.18
ROI_BOTTOM_CUT = 0.10

def ensure_dirs(base: Path, make_combo: bool = False):
    (base / "normal").mkdir(parents=True, exist_ok=True)
    for t in CATEGORIES:
        (base / t).mkdir(parents=True, exist_ok=True)
    if make_combo:
        # 단일 라벨이므로 실제로는 사용되지 않지만 옵션 유지
        for r in range(2, len(CATEGORIES) + 1):
            for comb in combinations(CATEGORIES, r):
                (base / ("+".join(sorted(comb)))).mkdir(parents=True, exist_ok=True)

def place(dst_path: Path, src_path: Path, mode: str = "hardlink"):
    if mode == "copy":
        shutil.copy2(src_path, dst_path)
        return
    try:
        if mode == "hardlink":
            os.link(src_path, dst_path)
        elif mode == "symlink":
            os.symlink(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    except Exception:
        shutil.copy2(src_path, dst_path)

def compute_metrics_bgr(img_bgr: np.ndarray):
    # YUV 기반 기본 지표
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    Y = yuv[..., 0].astype(np.float32) / 255.0
    mean_Y = float(Y.mean())
    std_Y = float(Y.std())
    dark_ratio   = float((Y < 0.20).mean())
    bright_ratio = float((Y > 0.95).mean())

    # DCP 평균
    b, g, r = cv2.split(img_bgr.astype(np.float32) / 255.0)
    dark = np.minimum(np.minimum(r, g), b)
    dark = cv2.erode(dark, np.ones((15, 15), np.uint8))
    dcp_mean = float(dark.mean())

    # --- 간단 보강용 지표 ---
    # 1) Laplacian 분산(blur 게이트) - gray(0~1 정규화)에서 계산
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # 2) 채도 평균(HSV S 채널)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    mean_S = float(hsv[..., 1].mean())

    # 필요 시 디버그
    # print(f"[DEBUG][metrics] mean_Y={mean_Y:.3f}, std_Y={std_Y:.3f}, "
    #       f"dark_ratio={dark_ratio:.3f}, bright_ratio={bright_ratio:.3f}, dcp_mean={dcp_mean:.3f}, "
    #       f"lap_var={lap_var:.5f}, mean_S={mean_S:.3f}")

    return dict(mean_Y=mean_Y, std_Y=std_Y, dark_ratio=dark_ratio,
                bright_ratio=bright_ratio, dcp_mean=dcp_mean,
                lap_var=lap_var, mean_S=mean_S)

def classify_single(metrics):
    """
    단일 라벨 반환: 우선순위 low_light > overbright > degradation > normal
    (기존 AND/OR 조건 논리는 그대로 유지, degradation에 보강 게이트를 OR로 추가)
    """
    # low_light 조건 (원래 멀티라벨에서 사용하던 and/or 그대로)
    is_low_light = (
        (metrics["mean_Y"] < LOW_LIGHT_CFG["mean_Y_max"] and metrics["dark_ratio"] > LOW_LIGHT_CFG["dark_ratio_min"]) or
        (metrics["mean_Y"] < LOW_LIGHT_CFG["alt_mean_Y_max"] and metrics["std_Y"]   < LOW_LIGHT_CFG["std_Y_max"])
    )

    # overbright 조건 (기존 로직 유지: OR)
    is_overbright = (
        metrics["bright_ratio"] > OVERBRIGHT_CFG["bright_ratio_min"] or
        metrics["mean_Y"]       > OVERBRIGHT_CFG["mean_Y_min"]
    )

    # degradation(구 haze) 기본 프록시: DCP↑ & std_Y↓
    is_degr_veiling = (
        metrics["dcp_mean"] > DEGR_CFG["dcp_mean_min"] and
        metrics["std_Y"]    < DEGR_CFG["std_Y_max"]
    )

    # 간단 보강: blur / sat 게이트 (선택 적용, 기본은 blur만 사용)
    is_degr_blur = False
    if DEGR_ENHANCE["use_blur_gate"]:
        is_degr_blur = (metrics.get("lap_var", 1.0) < DEGR_ENHANCE["lap_var_max"])

    is_degr_sat = False
    if DEGR_ENHANCE["use_sat_gate"]:
        is_degr_sat = (metrics.get("mean_S", 1.0) < DEGR_ENHANCE["mean_S_max"])

    # 최종 degradation 판정: veiling 프록시 OR (blur/sat 보강)
    is_degradation = is_degr_veiling or is_degr_blur or is_degr_sat

    # 우선순위 적용
    if is_low_light:
        return ["low_light"]
    elif is_overbright:
        return ["overbright"]
    elif is_degradation:
        return ["degradation"]
    else:
        return ["normal"]

def split_and_save_multi(
    src_dir,
    out_dir,
    link_mode: str = "hardlink",
    make_combo: bool = False,  # 단일 라벨이므로 기본 False
    exts=(".jpg", ".jpeg", ".png"),
    save_json="degradation_tags_test.json"
):
    """
    단일 라벨 분류 파이프라인 (함수명은 호환성 유지를 위해 유지)
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_dir, make_combo=make_combo)

    img_paths = sorted([p for p in src_dir.rglob("*.*") if p.suffix.lower() in exts])
    results = []

    for p in tqdm(img_paths, desc="Classifying (low_light / degradation / overbright → single)"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        m = compute_metrics_bgr(img)
        tags = classify_single(m)  # 항상 길이 1의 리스트

        label = tags[0]
        dst = out_dir / label / p.name
        place(dst, p, link_mode)

        # (단일 라벨이므로 combo는 생성되지 않음. make_combo=True여도 사용되지 않음)
        rec = {"path": str(p), "metrics": m, "tags": tags}
        results.append(rec)

    with open(out_dir / save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {out_dir}")
    print("생성 폴더:", ", ".join(sorted([d.name for d in out_dir.iterdir() if d.is_dir()])))

if __name__ == "__main__":
    # 사용 예시
    # split_and_save_multi(
    #     r"C:\Users\8138\Desktop\2025_ai_contest\SemanticSeg\dataset\debug\image\test",
    #     r"D:\2025_ai_contest\DEBUG",
    #     link_mode="copy",  # symlink : 바로가기
    #     make_combo=False
    # )

    # 다른 사용 예시
    split_and_save_multi(
        r"C:\Users\8138\Desktop\2025_ai_contest\SemanticSeg\dataset\SemanticDataset_final\image\train",
        r"D:\2025_ai_contest\preprocessing7_degra",
        link_mode="copy",
        make_combo=False
    )

    # split_and_save_multi(
    #     r"C:\Users\8138\Desktop\2025_ai_contest\SemanticSeg\dataset\SemanticDatasetTest\image\test",
    #     r"D:\2025_ai_contest\preprocessing7_degra_test",
    #     link_mode="copy",  # symlink : 바로가기
    #     make_combo=False
    # )
