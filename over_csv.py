#!/usr/bin/env python3
# build_oversampling_csv_expand.py
import os, csv, json, math, random, shutil
from glob import glob
from collections import Counter, defaultdict
import numpy as np
import cv2
from tqdm import tqdm

# ======== 설정 ========
DATASET_DIR = "./dataset/SemanticDataset_final"
IMAGE_DIR   = os.path.join(DATASET_DIR, "image")
LABEL_DIR   = os.path.join(DATASET_DIR, "labelmap")

OUT_CSV   = os.path.join(DATASET_DIR, "train_oversampled_expand.csv")
OUT_STATS = os.path.join(DATASET_DIR, "train_oversampled_expand_stats.json")

# ---- 복사(분리 저장) 설정 ----
COPY_ROOT        = os.path.join(r"D:\2025_ai_contest\dataset\SemanticDataset_final_copy_split_v2")
DEST_OS          = os.path.join(COPY_ROOT, "oversampled")             # oversampling 복사본들
DEST_OS_ORI      = os.path.join(COPY_ROOT, "oversampled_originals")   # oversampling에 선정된 원본
DEST_ORIG_REST   = os.path.join(COPY_ROOT, "originals_remaining")     # 나머지 원본
PREFIX_OS        = ""       # oversampled 복사본 프리픽스
PREFIX_OSORI     = ""    # oversampled에 선정된 원본 프리픽스
PREFIX_ORIG_REST = ""     # 나머지 원본 프리픽스
COPY_LABELS      = True        # 라벨도 함께 복사

RARE = [9, 12, 13, 16]

# 추가 샘플(=원본 포함 이후 남는 수) 쿼터 비율
SOLO_RATIO     = 0.25
SOLO_QUOTA     = {9:0.06, 12:0.03, 13:0.11, 16:0.05}  # 합=0.25
COMBO12_RATIO  = 0.25
COMBOX_RATIO   = 0.20
NONRARE_RATIO  = 0.30

# 버킷 내부 가중치(rare 픽셀 비율 기반)
ALPHA = {9:0.6, 12:1.0, 13:0.4, 16:0.5}
TAU, GAMMA = 0.005, 0.7
BONUS_12_COMBO = 1.3

# 이미지별 추가 중복 상한(“추가 샘플”에만 적용; 원본 1회는 별도)
CAP_DEFAULT = 3

# 총 샘플 수 설정: expansion_factor 또는 target_samples 중 하나 사용
EXPANSION_FACTOR = 1.4   # 총 샘플 수 = 원본 * 1.4
TARGET_SAMPLES   = None  # 예: 25000 (설정 시 EXPANSION_FACTOR 무시)

SEED = 20250918
random.seed(SEED); np.random.seed(SEED)

# ======== 유틸 ========
def list_images(root):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp")
    paths = []
    for e in exts: paths += glob(os.path.join(root,"**",e), recursive=True)
    return sorted(paths)

def map_label(image_path):
    rel = os.path.relpath(image_path, IMAGE_DIR)
    d, name = os.path.split(rel)
    base, ext = os.path.splitext(name)
    if name.endswith("_leftImg8bit.png"):
        new_name = base.replace("_leftImg8bit","_gtFine_CategoryId") + ".png"
    else:
        new_name = base + "_CategoryId.png"
    return os.path.join(LABEL_DIR, d, new_name)

def read_label(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None: return None
    if arr.ndim == 3: arr = arr[...,0]
    if arr.dtype != np.uint8: arr = arr.astype(np.uint8)
    return arr

def scan_stats():
    stats = []
    for ip in tqdm(list_images(IMAGE_DIR), desc="Scanning", ncols=100):
        lp = map_label(ip)
        if not os.path.exists(lp): continue
        lab = read_label(lp)
        if lab is None: continue
        h = np.bincount(lab.ravel(), minlength=256)
        total = int((lab != 255).sum()); total = total if total>0 else lab.size
        rare_pix = {c:int(h[c]) for c in RARE}
        present = sorted([c for c in RARE if rare_pix[c] > 0])
        stats.append({
            "image_path": ip, "label_path": lp, "total_pixels": total,
            "rare_pixels": rare_pix, "present": present
        })
    return stats

def weight_of(st):
    tot = max(st["total_pixels"],1)
    base = 0.0
    for c in RARE:
        rc = st["rare_pixels"].get(c,0)
        if rc>0:
            r = rc/tot
            base += ALPHA[c]*((r/TAU)**GAMMA)
    if len(st["present"])>=2 and 12 in st["present"]:
        base *= BONUS_12_COMBO
    return max(base, 1e-6)

def make_buckets(stats):
    buckets = {"S9":[], "S12":[], "S13":[], "S16":[], "C12":[], "Cx":[], "N":[]}
    weights = {}
    for i, st in enumerate(stats):
        pres = st["present"]
        if len(pres)==0:
            buckets["N"].append(i); weights[i]=1.0
        elif len(pres)==1:
            key = f"S{pres[0]}" if f"S{pres[0]}" in buckets else "N"
            buckets[key].append(i); weights[i]=weight_of(st)
        else:
            key = "C12" if 12 in pres else "Cx"
            buckets[key].append(i); weights[i]=weight_of(st)
    return buckets, weights

def safe_draw_with_cap(pool, want_k, weights_map, cap_per_img, already_count):
    if not pool or want_k<=0: return []
    capacity = sum( max(cap_per_img - already_count.get(i,0), 0) for i in pool )
    k = min(want_k, capacity)
    if k <= 0: return []

    remaining = {i for i in pool if (cap_per_img - already_count.get(i,0)) > 0}
    out = []

    while len(out) < k and remaining:
        rem_list = list(remaining)
        rem_w = np.array([weights_map[i] for i in rem_list], dtype=np.float64)
        rem_w = rem_w/rem_w.sum() if rem_w.sum()>0 else np.ones_like(rem_w)/len(rem_w)
        pick = np.random.choice(rem_list, p=rem_w)
        if already_count.get(pick,0) < cap_per_img:
            out.append(pick)
            already_count[pick] = already_count.get(pick,0) + 1
            if already_count[pick] >= cap_per_img:
                remaining.discard(pick)
        else:
            remaining.discard(pick)
    return out

def redistribute_shortfall(plan, got, limit_factor=3.0):
    need_total = 0
    for k,v in plan.items():
        lack = max(v - got.get(k,0), 0)
        need_total += lack
    if need_total == 0: return plan

    candidate = {k:v for k,v in plan.items() if got.get(k,0) < plan[k]*limit_factor}
    tot = sum(candidate.values())
    if tot == 0: return plan
    plan2 = dict(plan)
    for k,v in candidate.items():
        add = int(round(need_total * (v/tot)))
        plan2[k] += add
    return plan2

# ---------- 파일 복사 유틸 ----------
def _prefixed(name, prefix, rep_suffix=None):
    base, ext = os.path.splitext(name)
    if rep_suffix:
        return f"{prefix}{base}_{rep_suffix}{ext}"
    return f"{prefix}{base}{ext}"

def _copy_pair(img_path, lab_path, dest_root, prefix, rep_suffix=None):
    # image 경로
    rel_img = os.path.relpath(img_path, IMAGE_DIR)
    img_dir = os.path.dirname(rel_img)
    img_name = os.path.basename(img_path)
    out_img_dir = os.path.join(dest_root, "image", img_dir)
    os.makedirs(out_img_dir, exist_ok=True)
    out_img_path = os.path.join(out_img_dir, _prefixed(img_name, prefix, rep_suffix))
    shutil.copy2(img_path, out_img_path)

    if COPY_LABELS and lab_path and os.path.exists(lab_path):
        rel_lab = os.path.relpath(lab_path, LABEL_DIR)
        lab_dir = os.path.dirname(rel_lab)
        lab_name = os.path.basename(lab_path)
        out_lab_dir = os.path.join(dest_root, "labelmap", lab_dir)
        os.makedirs(out_lab_dir, exist_ok=True)
        out_lab_path = os.path.join(out_lab_dir, _prefixed(lab_name, prefix, rep_suffix))
        shutil.copy2(lab_path, out_lab_path)

def perform_copy(stats, extra_picks):
    # extra_picks: oversampling으로 추가된 인덱스들의 리스트(중복 포함)
    rep_counts = Counter(extra_picks)            # 각 인덱스 별 복제 횟수
    os_unique_idxs = set(rep_counts.keys())      # oversampled에 선정된 원본 세트
    all_idxs = set(range(len(stats)))
    rest_orig_idxs = sorted(all_idxs - os_unique_idxs)

    # 1) oversampled 복사본들: 중복 수만큼 모두 복사 (rep001, rep002, ...)
    n_os_copies = 0
    for idx, cnt in rep_counts.items():
        st = stats[idx]
        for k in range(1, cnt+1):
            _copy_pair(st["image_path"], st["label_path"], DEST_OS, PREFIX_OS, rep_suffix=f"rep{k:03d}")
            n_os_copies += 1

    # 2) oversampled에 선정된 원본: 각 인덱스당 1장
    for idx in sorted(os_unique_idxs):
        st = stats[idx]
        _copy_pair(st["image_path"], st["label_path"], DEST_OS_ORI, PREFIX_OSORI)

    # 3) 나머지 원본
    for idx in rest_orig_idxs:
        st = stats[idx]
        _copy_pair(st["image_path"], st["label_path"], DEST_ORIG_REST, PREFIX_ORIG_REST)

    print("[COPY] oversampled copies     :", n_os_copies)
    print("[COPY] oversampled originals  :", len(os_unique_idxs))
    print("[COPY] remaining originals    :", len(rest_orig_idxs))
    print("[COPY] Dest root:", COPY_ROOT)

def main():
    try: cv2.setNumThreads(1)
    except: pass

    stats = scan_stats()
    if not stats:
        print("[ERROR] No images found."); return

    # ===== 1) 원본 전부 1회 포함 =====
    base_rows = []
    for st in stats:
        pres = "-".join(map(str, st["present"])) if st["present"] else "none"
        if len(st["present"])==0: bucket="N"
        elif len(st["present"])==1: bucket=f"S{st['present'][0]}"
        else: bucket="C12" if 12 in st["present"] else "Cx"
        base_rows.append((st["image_path"], st["label_path"], bucket, pres))

    original_count = len(base_rows)

    # 총 샘플 수 결정
    if TARGET_SAMPLES is not None and TARGET_SAMPLES > original_count:
        total_target = TARGET_SAMPLES
    else:
        total_target = int(round(original_count * EXPANSION_FACTOR))
        if total_target <= original_count:
            total_target = original_count  # 최소: 원본 그대로

    extra_needed = total_target - original_count
    print(f"[INFO] Original unique images: {original_count}")
    print(f"[INFO] Target total samples  : {total_target} (extra {extra_needed})")

    extra_picks = []

    if extra_needed > 0:
        # ===== 2) 추가분 샘플링 =====
        buckets, weights = make_buckets(stats)

        # 추가분 계획 수량(비율은 '추가분'에 대해 적용)
        plan = {
            "S9":  int(round(extra_needed * SOLO_QUOTA[9])),
            "S12": int(round(extra_needed * SOLO_QUOTA[12])),
            "S13": int(round(extra_needed * SOLO_QUOTA[13])),
            "S16": int(round(extra_needed * SOLO_QUOTA[16])),
            "C12": int(round(extra_needed * COMBO12_RATIO)),
            "Cx":  int(round(extra_needed * COMBOX_RATIO)),
            "N":   extra_needed - int(round(extra_needed*(SOLO_RATIO + COMBO12_RATIO + COMBOX_RATIO)))
        }

        # 버킷 cap 설정(추가분 한정; 원본 1회는 이미 포함)
        cap = {
            "S12": 10 if len(buckets["S12"])<=20 else 5,
            "S9": CAP_DEFAULT, "S13": CAP_DEFAULT, "S16": CAP_DEFAULT,
            "C12": CAP_DEFAULT, "Cx": CAP_DEFAULT, "N": CAP_DEFAULT
        }

        # 이미지별 추가분 사용 횟수
        extra_used = Counter()

        # 1차 드로우
        got = {}
        for name in ["S9","S12","S13","S16","C12","Cx","N"]:
            drawn = safe_draw_with_cap(buckets[name], plan[name], weights, cap[name], extra_used)
            extra_picks += drawn
            got[name] = len(drawn)

        # 못 채운 추가분 재분배 후 2차 보충
        short = extra_needed - len(extra_picks)
        if short > 0:
            plan2 = redistribute_shortfall(plan, got)
            for name in ["S9","S12","S13","S16","C12","Cx","N"]:
                more = max(plan2[name] - got.get(name,0), 0)
                if more > 0:
                    drawn = safe_draw_with_cap(buckets[name], more, weights, cap[name], extra_used)
                    extra_picks += drawn
                    got[name] += len(drawn)

        # 초과 시 잘라냄
        if len(extra_picks) > extra_needed:
            extra_picks = extra_picks[:extra_needed]

    # ===== 3) CSV 저장 (원본 + 추가분) =====
    rows_extra = []
    for idx in extra_picks:
        st = stats[idx]
        pres = "-".join(map(str, st["present"])) if st["present"] else "none"
        if len(st["present"])==0: bucket="N"
        elif len(st["present"])==1: bucket=f"S{st['present'][0]}"
        else: bucket="C12" if 12 in st["present"] else "Cx"
        rows_extra.append((st["image_path"], st["label_path"], bucket, pres))

    all_rows = base_rows + rows_extra
    random.shuffle(all_rows)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_path","label_path","bucket","present"])
        w.writerows(all_rows)

    cnt_bucket = Counter()
    for row in rows_extra:
        cnt_bucket[row[2]] += 1

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump({
            "mode":"expand",
            "original_unique_images": original_count,
            "total_samples": len(all_rows),
            "extra_added": len(rows_extra),
            "extra_bucket_counts": dict(cnt_bucket),
            "cap_default": CAP_DEFAULT,
            "seed": SEED
        }, f, indent=2, ensure_ascii=False)

    print(f"[OK] CSV:   {OUT_CSV}")
    print(f"[OK] Stats: {OUT_STATS}")
    print("Original unique images:", original_count)
    print("Total samples (rows):  ", len(all_rows))
    print("Extra added (rows):    ", len(rows_extra))
    print("Extra by bucket:       ", dict(cnt_bucket))

    # ===== 4) 파일 복사(분리 저장) =====
    perform_copy(stats, extra_picks)

if __name__ == "__main__":
    main()
