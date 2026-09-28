"""
대응표(pill_master.json) + 변경/삭제 CSV + 각도 규칙으로 최종 COCO(train_coco_final.json)를 만든다.
- 폴더를 훑지 않아서 몇 초면 끝난다.
- CSV를 고칠 때마다 이것만 다시 실행하면 된다. 순서/횟수와 상관없이 결과가 같다.
- CSV 행은 "원래 클래스 + uid"가 실제와 맞을 때만 적용하고, 안 맞는 행은 적용하지 않고 알려준다.
- train/val 은 "조합" 단위로 나눈다. 같은 조합의 다른 각도 사진이 양쪽에 갈라져 들어가지 않게 하려는 것.
  나눈 결과는 사진마다 "split" 필드에 들어가서, 이 파일만 받으면 누구나 같은 분할을 쓴다.
- 클래스마다 대회 category_id(제출할 때 쓰는 번호)를 "code" 필드로 함께 넣는다.

사용법:
    python src/make_dataset.py
"""
import json
import random
from collections import defaultdict

from pill_common import (BUILD_LOG_PATH, CHANGES_CSV, DELETES_CSV, FINAL_COCO_PATH,
                         load_master, load_rules, select_by_angle)

USE_ANGLE_RULE = True   # False 로 하면 각도 규칙 없이 모든 각도의 사진을 쓴다
VAL_RATIO = 0.1         # 검증용으로 떼어 둘 조합의 비율
SPLIT_SEED = 42         # 분할을 고정하는 값


def split_by_combo(stems, images):
    """{사진 이름: "train" 또는 "val"}. 같은 조합의 사진은 항상 같은 쪽으로 간다."""
    groups = defaultdict(list)
    for s in stems:
        groups[images[s]["combo"] or s].append(s)     # 조합 패턴이 없는 사진은 혼자 한 묶음
    keys = sorted(groups)
    random.Random(SPLIT_SEED).shuffle(keys)
    n_val = max(1, int(len(keys) * VAL_RATIO)) if keys else 0
    val_keys = set(keys[:n_val])
    return {s: ("val" if k in val_keys else "train") for k, ss in groups.items() for s in ss}


def main():
    master = load_master()
    images = master["images"]
    changes, deletes, problems, warnings = load_rules(master)

    stems = [s for s in sorted(images) if not images[s]["excluded"]]
    n_after_exclude = len(stems)
    if USE_ANGLE_RULE:
        keep = select_by_angle(stems)
        stems = [s for s in stems if s in keep]
    split_of = split_by_combo(stems, images)

    pills_by_stem = defaultdict(list)
    for p in master["pills"]:
        pills_by_stem[p["stem"]].append(p)

    images_out, anns_out = [], []
    n_deleted = n_moved = n_bad_bbox = 0
    split_imgs = {"train": 0, "val": 0}
    split_anns = {"train": 0, "val": 0}
    for img_id, s in enumerate(stems, start=1):
        im = images[s]
        sp = split_of[s]
        images_out.append({"id": img_id, "file_name": im["file_name"], "path": im["path"],
                           "width": im["width"], "height": im["height"], "split": sp})
        split_imgs[sp] += 1
        for p in sorted(pills_by_stem.get(s, []), key=lambda p: (p["rank"], p["k"])):
            if not p["bbox_ok"]:
                n_bad_bbox += 1
                continue
            if p["uid"] in deletes:
                n_deleted += 1
                continue
            cls = p["class_id"]
            if p["uid"] in changes:
                cls = changes[p["uid"]]
                n_moved += 1
            x, y, w, h = p["bbox"]
            anns_out.append({"id": len(anns_out) + 1, "image_id": img_id, "category_id": cls,
                             "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "uid": p["uid"]})
            split_anns[sp] += 1

    categories = [{"id": c["id"], "name": c["name"], "code": c.get("code")}
                  for c in sorted(master["classes"], key=lambda c: c["id"])]
    used = {a["category_id"] for a in anns_out}
    empty_classes = [c["id"] for c in categories if c["id"] not in used]
    val_ids = {i["id"] for i in images_out if i["split"] == "val"}
    val_class_set = {a["category_id"] for a in anns_out if a["image_id"] in val_ids}
    not_in_val = sorted(used - val_class_set)

    FINAL_COCO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_COCO_PATH, "w", encoding="utf-8") as f:
        json.dump({"images": images_out, "annotations": anns_out, "categories": categories},
                  f, ensure_ascii=False)

    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({"problems": problems, "warnings": warnings}, f, ensure_ascii=False, indent=2)

    print(f"제외된 사진 {sum(1 for im in images.values() if im['excluded'])}장, "
          f"각도 규칙 적용 전 {n_after_exclude}장 -> 후 {len(stems)}장 (각도 규칙 {'사용' if USE_ANGLE_RULE else '안 씀'})")
    print(f"정상 bbox가 아니라 뺀 알약 {n_bad_bbox}개")
    print(f"CSV: 변경 {n_moved}개 적용, 삭제 {n_deleted}개 적용 (CSV: {CHANGES_CSV.name}, {DELETES_CSV.name})")
    print(f"적용 안 된 CSV 행 {len(problems)}개 / 참고 {len(warnings)}개")
    for pr in problems[:10]:
        where = f" {pr['line']}번째 줄" if pr.get("line") else ""
        print(f"  [안 됨] {pr['file']}{where}: {pr['reason']}")
    if len(problems) > 10:
        print(f"  ... 나머지는 {BUILD_LOG_PATH} 에 있음")
    print(f"학습/검증 나누기(조합 단위, 검증 {VAL_RATIO:.0%}, seed {SPLIT_SEED}): "
          f"사진 {split_imgs['train']}/{split_imgs['val']}장, 알약 {split_anns['train']}/{split_anns['val']}개")
    if not_in_val:
        print(f"  검증용에 알약이 하나도 없는 클래스 {len(not_in_val)}종 (검증 점수에 안 잡힘): {not_in_val[:10]}")
    multi = [c for c in master["classes"] if c.get("codes")]
    no_code = [c["id"] for c in categories if c["code"] is None]
    if multi:
        print(f"[주의] 코드가 2개 이상인 클래스 {len(multi)}종 -> 제출할 때 가장 많은 코드 하나로만 나감 (build_master.py 출력 참고)")
    if no_code:
        print(f"[주의] 대회 코드가 없는 클래스: {no_code}")
    print(f"\n최종: 이미지 {len(images_out)}장, annotation {len(anns_out)}건, 클래스 {len(categories)}종"
          f" (알약이 하나도 없는 클래스 {len(empty_classes)}종) -> {FINAL_COCO_PATH}")


if __name__ == "__main__":
    main()