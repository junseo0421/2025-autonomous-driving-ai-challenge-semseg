#!/usr/bin/env python3
# align_labels_by_loader_rule.py
# - split 이미지가 있던 "leaf" 폴더 안에 image/ labelmap/ colormap/ 생성
# - 이미지 이동(또는 복사) + 라벨/컬러 복사
# - 접두사(os_/osori_/orig_)와 _rep#### 제거해 만든 "기본 스템"으로 원본 라벨/컬러를 찾고
#   코어별로 1번만 고른 뒤(캐시) 모든 rep에 동일 소스를 사용

import argparse, shutil, sys, re
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# ALLOWED_TOP = {"oversampled", "oversampled_originals", "originals_remaining"}
ALLOWED_TOP = {"train", "val"}
PREFIXES = ("os_", "osori_", "osorig_", "orig_")
REP_SUFFIX_RE = re.compile(r"_rep(\d{1,4})$", re.IGNORECASE)
SKIP_BASENAMES = {"oversampled.png","oversampled_originals.png","originals_remaining.png","oiriginal_remaining.png"}

IMG_SUFFIX = "_leftImg8bit"
LABEL_SUFFIX = "_gtFine_CategoryId"
COLOR_SUFFIX = "_gtFine_color"
FALLBACK_LABEL_SUFFIXES = ["_CategoryId"]
FALLBACK_COLOR_SUFFIXES = ["._color", "_color"]

DEST_SUBFOLDERS = {"image","labelmap","colormap"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def strip_prefix_and_rep(stem: str) -> str:
    s = stem
    s_lower = s.lower()
    for pf in PREFIXES:
        if s_lower.startswith(pf):
            s = s[len(pf):]
            break
    return REP_SUFFIX_RE.sub("", s)

def derive_cores(base_img_stem: str) -> tuple[str, str | None]:
    if base_img_stem.endswith(IMG_SUFFIX):
        core_full = base_img_stem[:-len(IMG_SUFFIX)]
    else:
        core_full = base_img_stem
    m = re.search(r"(\d+)$", core_full)
    core_num = m.group(1) if m else None
    return core_full, core_num

def label_stem_candidates(base_img_stem: str) -> list[str]:
    cands = []
    core_full, core_num = derive_cores(base_img_stem)
    cores = [core_full] + ([core_num] if (core_num and core_num != core_full) else [])
    for core in cores:
        if base_img_stem.endswith(IMG_SUFFIX):
            cands.append(core + LABEL_SUFFIX)
        for suf in FALLBACK_LABEL_SUFFIXES:
            cands.append(core + suf)
    seen, out = set(), []
    for s in cands:
        k = s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

def color_stem_candidates(base_img_stem: str) -> list[str]:
    cands = []
    core_full, core_num = derive_cores(base_img_stem)
    cores = [core_full] + ([core_num] if (core_num and core_num != core_full) else [])
    for core in cores:
        if base_img_stem.endswith(IMG_SUFFIX):
            cands.append(core + COLOR_SUFFIX)
        for suf in FALLBACK_COLOR_SUFFIXES:
            cands.append(core + suf)
    seen, out = set(), []
    for s in cands:
        k = s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

def index_by_stem(root: Path) -> dict[str, list[Path]]:
    idx = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file():
            idx[p.stem.lower()].append(p)
    return idx

def gather_candidates(index: dict[str, list[Path]], stem_candidates: list[str]) -> list[Path]:
    out, seen = [], set()
    for stem in stem_candidates:
        for p in index.get(stem.lower(), []):
            key = p.as_posix().lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out

def pick_best(cands: list[Path], core_full: str, prefer_set: str | None = None) -> Path | None:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    pngs = [p for p in cands if p.suffix.lower() == ".png"]
    if pngs:
        cands = pngs
        if len(cands) == 1:
            return cands[0]
    scene_token = core_full.split("_", 1)[0].lower() if "_" in core_full else None
    if scene_token:
        filt = [p for p in cands if scene_token in p.as_posix().lower()]
        if filt:
            cands = filt
            if len(cands) == 1:
                return cands[0]
    if prefer_set:
        ps = prefer_set.lower()
        filt = [p for p in cands if f"/{ps}/" in p.as_posix().lower() or f"\\{ps}\\" in p.as_posix().lower()]
        if filt:
            cands = filt
            if len(cands) == 1:
                return cands[0]
    cands = sorted(cands, key=lambda p: (len(p.as_posix()), p.as_posix().lower()))
    return cands[0]

def main():
    ap = argparse.ArgumentParser(description="split leaf 폴더 내 image/ labelmap/ colormap 구성하여 정렬 복사")
    ap.add_argument("--split_root", required=True, type=Path)
    ap.add_argument("--orig_labelmap_root", required=True, type=Path)
    ap.add_argument("--orig_colormap_root", required=True, type=Path)
    ap.add_argument("--prefer_set", type=str, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--move_split_images", action="store_true",
                    help="원래 split 이미지를 leaf/image/로 이동(기본은 복사)")
    args = ap.parse_args()

    sr = args.split_root.resolve()
    lbl_root = args.orig_labelmap_root.resolve()
    col_root = args.orig_colormap_root.resolve()

    if not sr.exists():       print(f"[!] split_root 없음: {sr}", file=sys.stderr); sys.exit(1)
    if not lbl_root.exists(): print(f"[!] orig_labelmap_root 없음: {lbl_root}", file=sys.stderr); sys.exit(1)
    if not col_root.exists(): print(f"[!] orig_colormap_root 없음: {col_root}", file=sys.stderr); sys.exit(1)

    print("[*] 원본 labelmap 인덱싱 중...")
    label_idx = index_by_stem(lbl_root)
    print(f"    고유 key 수(라벨): {len(label_idx)}")

    print("[*] 원본 colormap 인덱싱 중...")
    color_idx = index_by_stem(col_root)
    print(f"    고유 key 수(컬러): {len(color_idx)}")

    # split 이미지 수집 (이미 leaf/image 등으로 들어간 것은 제외)
    images = []
    for top in ALLOWED_TOP:
        d = sr / top
        if d.exists():
            for p in d.rglob("*"):
                if is_image(p) and p.name not in SKIP_BASENAMES and p.parent.name not in DEST_SUBFOLDERS:
                    images.append(p)
    print(f"[*] 대상 이미지: {len(images)}개")

    # 코어별 고정 매핑 캐시
    lbl_cache: dict[str, Path] = {}
    col_cache: dict[str, Path] = {}

    total = len(images)
    ok_lbl = ok_col = 0
    miss_lbl = miss_col = 0
    cached_lbl = cached_col = 0
    miss_lbl_samples, miss_col_samples = [], []

    for i, src_img in enumerate(images, 1):
        # leaf 폴더(= split 이미지가 있던 폴더)
        leaf = src_img.parent
        leaf_img_dir = leaf / "image"
        leaf_lbl_dir = leaf / "labelmap"
        leaf_col_dir = leaf / "colormap"

        # 목적지 파일 경로(파일명은 split 이미지 그대로, 라벨/컬러는 .png)
        dst_img = leaf_img_dir / src_img.name
        dst_lbl = (leaf_lbl_dir / src_img.name).with_suffix(".png")
        dst_col = (leaf_col_dir / src_img.name).with_suffix(".png")

        # 1) 매칭용 스템 생성
        base_img_stem = strip_prefix_and_rep(src_img.stem)
        core_full, core_num = derive_cores(base_img_stem)
        group_key = core_full or (core_num or base_img_stem)

        # 2) 후보 수집 및 소스 결정(캐시 사용)
        lbl_src = lbl_cache.get(group_key)
        if lbl_src is None:
            lbl_src = pick_best(gather_candidates(label_idx, label_stem_candidates(base_img_stem)),
                                core_full, args.prefer_set)
            if lbl_src:
                lbl_cache[group_key] = lbl_src
        else:
            cached_lbl += 1

        col_src = col_cache.get(group_key)
        if col_src is None:
            col_src = pick_best(gather_candidates(color_idx, color_stem_candidates(base_img_stem)),
                                core_full, args.prefer_set)
            if col_src:
                col_cache[group_key] = col_src
        else:
            cached_col += 1

        # 3) 디렉토리 생성
        if not args.dry_run:
            leaf_img_dir.mkdir(parents=True, exist_ok=True)
            leaf_lbl_dir.mkdir(parents=True, exist_ok=True)
            leaf_col_dir.mkdir(parents=True, exist_ok=True)

        # 4) 이미지 이동(또는 복사)
        if not args.dry_run:
            if args.move_split_images:
                # 같은 경로가 아니면 이동
                if src_img.resolve() != dst_img.resolve():
                    shutil.move(str(src_img), str(dst_img))
            else:
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

        # 5) 라벨/컬러 복사
        if lbl_src:
            ok_lbl += 1
            if not args.dry_run:
                shutil.copy2(lbl_src, dst_lbl)
        else:
            miss_lbl += 1
            if len(miss_lbl_samples) < 10:
                miss_lbl_samples.append(str(src_img.relative_to(sr)))

        if col_src:
            ok_col += 1
            if not args.dry_run:
                shutil.copy2(col_src, dst_col)
        else:
            miss_col += 1
            if len(miss_col_samples) < 10:
                miss_col_samples.append(str(src_img.relative_to(sr)))

        if i % 2000 == 0:
            print(f"    - 진행 {i}/{total}  (LBL ok:{ok_lbl} cached:{cached_lbl}, COL ok:{ok_col} cached:{cached_col})")

    print("\n========== 요약 ==========")
    print(f"총 이미지 수                 : {total}")
    print(f"labelmap  매칭 성공/실패     : {ok_lbl} / {miss_lbl}   (코어 캐시 적중 {cached_lbl})")
    print(f"colormap  매칭 성공/실패     : {ok_col} / {miss_col}   (코어 캐시 적중 {cached_col})")

    if miss_lbl_samples:
        print("\n[라벨 못 찾은 예시(최대 10)]")
        for s in miss_lbl_samples: print(" -", s)
    if miss_col_samples:
        print("\n[컬러 못 찾은 예시(최대 10)]")
        for s in miss_col_samples: print(" -", s)
    print("\n완료.")

if __name__ == "__main__":
    main()
