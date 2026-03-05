# summarize_degradation_tags_console.py
import json
from collections import Counter
from itertools import combinations

# ✅ 단일 라벨 체계: 우선순위 = low_light > overbright > degradation > normal
TAGS = ["low_light", "overbright", "degradation", "normal"]
_ALLOWED = set(TAGS)
_PRIORITY_RANK = {t: i for i, t in enumerate(TAGS)}  # 낮을수록 우선

# ✅ 레거시 호환: haze → degradation 매핑
_LEGACY_MAP = {"haze": "degradation"}

def _pick_single_label(filtered_tags):
    """
    주어진 태그 리스트에서 우선순위(low_light > overbright > degradation > normal)로 하나만 선택
    filtered_tags는 이미 _ALLOWED로 필터링된 상태라고 가정
    """
    if not filtered_tags:
        return "normal"
    # 우선순위가 가장 높은 태그를 1개 선택
    return min(filtered_tags, key=lambda t: _PRIORITY_RANK.get(t, 1_000))

def read_tags(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    items = []
    for r in recs:
        raw = list(r.get("tags", [])) or ["normal"]
        # ✅ 레거시 태그 매핑 적용 (haze → degradation)
        raw = [_LEGACY_MAP.get(t, t) for t in raw]
        # ✅ 허용된 태그만 남김
        filtered = [t for t in raw if t in _ALLOWED]
        # ✅ 단일 라벨로 강제 선택 (우선순위 적용)
        single = _pick_single_label(filtered)
        # 호환성 유지를 위해 여전히 리스트 형태로 저장하지만 길이는 항상 1
        items.append((r["path"], [single]))
    return items

def summarize(items):
    N = len(items)
    tag_counter = Counter()
    nlabels_counter = Counter()
    combo_counter = Counter()
    cooc = {a: Counter() for a in TAGS}

    for _, tags in items:
        tags = tags if tags else ["normal"]
        # 단일 라벨 강제 후이므로 uniq 길이는 항상 1
        uniq = sorted(set(tags))

        tag_counter.update(uniq)
        nlabels_counter[len(uniq)] += 1

        # 단일 라벨이므로 조합/공동발생은 사실상 발생하지 않음(대각선만 증가)
        if len(uniq) >= 2:
            combo_key = "+".join(uniq)
            combo_counter[combo_key] += 1
            for a, b in combinations(uniq, 2):
                cooc[a][b] += 1
                cooc[b][a] += 1

        # 대각선(자기 자신) 카운트
        for a in uniq:
            cooc[a][a] += 1

    return dict(
        N=N,
        tag_counter=tag_counter,
        nlabels_counter=nlabels_counter,
        combo_counter=combo_counter,
        cooc=cooc
    )

def print_summary(S):
    N = S["N"]
    print(f"\n총 이미지 수: {N}\n")

    print("== 태그별 분포 (단일 라벨) ==")
    for t in TAGS:
        c = S["tag_counter"].get(t, 0)
        r = (c / N * 100) if N else 0.0
        print(f"  {t:11s}: {c:5d}  ({r:6.2f}%)")

    print("\n== 이미지당 태그 개수 분포 (단일 라벨 적용 후) ==")
    for k in sorted(S["nlabels_counter"]):
        c = S["nlabels_counter"][k]
        r = (c / N * 100) if N else 0.0
        print(f"  {k}개 태그: {c:5d}  ({r:6.2f}%)")

    # 단일 라벨에서는 조합이 존재하지 않지만, 과거 JSON 호환을 위해 출력 분기 유지
    if S["combo_counter"]:
        print("\n== 태그 조합 분포 (2개 이상) ==")
        for combo, c in S["combo_counter"].most_common():
            r = (c / N * 100) if N else 0.0
            print(f"  {combo:20s}: {c:5d}  ({r:6.2f}%)")
    else:
        print("\n== 태그 조합 분포: 없음 ==")

    print("\n== 공동발생 매트릭스 (counts) ==")
    header = [""] + TAGS
    row_fmt = "{:>12}" * len(header)
    print(row_fmt.format(*header))
    for a in TAGS:
        row = [a] + [S["cooc"].get(a, {}).get(b, 0) for b in TAGS]
        print(row_fmt.format(*row))

if __name__ == "__main__":
    # 경로
    json_path = r"D:\2025_ai_contest\SemanticDataset_final_copy_split_degra_v2\oversampled_originals\degradation_tags_test.json"
    # json_path = r"D:\2025_ai_contest\SemanticDataset_final_copy_split_degra_v2\oversampled\degradation_tags_test.json"
    # json_path = r"D:\2025_ai_contest\SemanticDataset_final_copy_split_degra_v2\originals_remaining\degradation_tags_test.json"

    items = read_tags(json_path)
    S = summarize(items)
    print_summary(S)
