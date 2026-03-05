#!/usr/bin/env python3
import os
import csv
from collections import Counter, defaultdict
from itertools import combinations

# === 설정 ===
CSV_PATH = "/workspace/SemanticSeg/dataset/train_oversampled.csv"
RARE = [9, 12, 13, 16]

def parse_present(s: str):
    s = (s or "").strip()
    if s == "" or s.lower() == "none":
        return tuple()
    parts = [p for p in s.split("-") if p != ""]
    try:
        ints = sorted(set(int(p) for p in parts))
    except ValueError:
        ints = []
    return tuple(ints)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        return

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            present = parse_present(row.get("present", ""))
            image_path = row["image_path"]
            rows.append({"image_path": image_path, "present": present})

    # ---- 전체 수량 ----
    total_rows = len(rows)                         # 샘플(행) 수 → 오버샘플링 반영
    unique_images = len(set([r["image_path"] for r in rows]))  # 고유 이미지 수(원본 분포 관점)

    # ---- 유니크 이미지 기준(원본 분포로 간주) 집계 ----
    img_to_present = {}
    for r in rows:
        ip = r["image_path"]
        pr = r["present"]
        # 동일 이미지가 여러 행에 있을 때, 보다 풍부한 집합을 선택
        if ip not in img_to_present or len(pr) > len(img_to_present[ip]):
            img_to_present[ip] = pr

    base_single_imgs = Counter()   # 클래스 단독(원본, 유니크 이미지 기준)
    base_combo_imgs = Counter()    # 콤보(원본, 유니크 이미지 기준)
    base_any_imgs = Counter()      # 클래스 '존재'(원본, 유니크 이미지 기준)

    for pr in img_to_present.values():
        if len(pr) == 0:
            continue
        # any presence per class
        for c in RARE:
            if c in pr:
                base_any_imgs[c] += 1
        # single-only
        if len(pr) == 1 and pr[0] in RARE:
            base_single_imgs[pr[0]] += 1
        # combos
        if len(pr) >= 2:
            for k in range(2, len(pr) + 1):
                for combo in combinations(pr, k):
                    if all(c in RARE for c in combo):
                        base_combo_imgs[combo] += 1

    # ---- 행 기준(오버샘플링 결과) 집계 ----
    over_single_rows = Counter()   # 클래스 단독(행 기준)
    over_combo_rows = Counter()    # 콤보(행 기준)
    over_any_rows = Counter()      # 클래스 '존재'(행 기준)

    for r in rows:
        pr = r["present"]
        if len(pr) == 0:
            continue
        # any presence per class
        for c in RARE:
            if c in pr:
                over_any_rows[c] += 1
        # single-only
        if len(pr) == 1 and pr[0] in RARE:
            over_single_rows[pr[0]] += 1
        # combos
        if len(pr) >= 2:
            for k in range(2, len(pr) + 1):
                for combo in combinations(pr, k):
                    if all(c in RARE for c in combo):
                        over_combo_rows[combo] += 1

    # ---- 출력 ----
    print("===== 전체 =====")
    print(f"총 샘플 수(행 수, oversampled): {total_rows}")
    print(f"유니크 이미지 수(원본 분포):    {unique_images}")

    # 1) 클래스 '존재' 기준 (any) — 참고용
    print("\n===== Rare Class 존재(any) 기준 =====")
    for c in RARE:
        base = base_any_imgs[c]
        over = over_any_rows[c]
        inc = over - base
        print(f"Class {c}: base={base}  →  over(rows)={over}  |  +{inc}")

    # 2) 단독(single-only)
    print("\n===== Rare Class 단독(single-only) =====")
    for c in RARE:
        base = base_single_imgs[c]
        over = over_single_rows[c]
        inc = over - base
        print(f"Class {c} only: base={base}  →  over(rows)={over}  |  +{inc}")

    # 3) 콤비네이션(combinations)
    print("\n===== Rare Class 콤비네이션(combination) =====")
    all_combos = set(base_combo_imgs.keys()) | set(over_combo_rows.keys())
    for combo in sorted(all_combos):
        base = base_combo_imgs[combo]
        over = over_combo_rows[combo]
        inc = over - base
        print(f"Classes {combo}: base={base}  →  over(rows)={over}  |  +{inc}")

if __name__ == "__main__":
    main()
