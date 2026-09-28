"""알약 데이터 파이프라인 공통 설정과 함수. 다른 스크립트가 import 해서 쓴다."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# ── 설정 ────────────────────────────────────────────────
# (이미지 폴더, 라벨 json 폴더). AI Hub val 도 쓰고 싶으면 아래 주석을 풀고
# python src/build_master.py 를 다시 실행하면 된다. (기존 고유번호는 그대로 유지됨)
DATA_ROOTS = [
    ("data/raw/sprint_ai_project1_data/train_images", "data/raw/sprint_ai_project1_data/train_annotations"),
    ("data/raw/aihub_pill_data/train_images_aihub", "data/raw/aihub_pill_data/train_annotations_aihub"),
    # ("data/raw/aihub_pill_data/val_images_aihub", "data/raw/aihub_pill_data/val_annotations_aihub"),
]

MASTER_PATH = Path("data/processed/pill_master.json")
CHANGES_CSV = Path("notebooks/pill_changes.csv")   # 헤더: 원본id,uid,수정id
DELETES_CSV = Path("notebooks/pill_deletes.csv")   # 헤더: id,uid
FINAL_COCO_PATH = Path("data/processed/train_coco_final.json")
BUILD_LOG_PATH = Path("outputs/dataset_build_log.json")

MIN_VALID_PILLS_PER_IMAGE = 3   # 사진 한 장에 정상 bbox가 이보다 적으면 사진을 통째로 제외
START_UID = 10001               # 알약 고유번호 시작값
ANGLES = ("70", "75", "90")


def norm(name):
    """공백을 없앤 비교용 이름."""
    return re.sub(r"\s+", "", name or "").strip()


def split_combo_angle(stem):
    """사진 이름에서 (조합 이름, 각도)를 뽑는다. 패턴이 없으면 (None, None)."""
    for angle in ANGLES:
        marker = f"_0_2_0_2_{angle}_000_200"
        if marker in stem:
            return stem.split(marker)[0], angle
    return None, None


def select_by_angle(stems):
    """조합 단위 각도 규칙. 70+90 둘 다 있으면 둘 다, 하나만 있으면 그것, 둘 다 없으면 75.
    패턴이 없는 사진은 무조건 남긴다. 남길 사진 이름의 집합을 돌려준다."""
    combo_angles = defaultdict(dict)
    keep = set()
    for s in stems:
        combo, angle = split_combo_angle(s)
        if combo is None:
            keep.add(s)
        else:
            combo_angles[combo][angle] = s
    for angle_stem in combo_angles.values():
        has70, has75, has90 = "70" in angle_stem, "75" in angle_stem, "90" in angle_stem
        if has70 and has90:
            keep.add(angle_stem["70"])
            keep.add(angle_stem["90"])
        elif has70 or has90:
            keep.add(angle_stem["70"] if has70 else angle_stem["90"])
        elif has75:
            keep.add(angle_stem["75"])
    return keep


def load_master():
    if not MASTER_PATH.exists():
        raise SystemExit(f"{MASTER_PATH} 가 없음. 먼저 python src/build_master.py 를 실행해줘.")
    with open(MASTER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_rows(path, ncols):
    """CSV를 읽어서 [(줄번호, (정수, ...))], [(줄번호, 원문)] 을 돌려준다. 첫 줄은 헤더."""
    rows, bad = [], []
    path = Path(path)
    if not path.exists():
        return rows, bad
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for lineno, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue
            try:
                vals = tuple(int(c.strip()) for c in row[:ncols])
                if len(vals) < ncols:
                    raise ValueError
                rows.append((lineno, vals))
            except ValueError:
                bad.append((lineno, ",".join(row)))
    return rows, bad


def load_rules(master):
    """변경/삭제 CSV를 읽고 검증한다.
    반환: changes {uid: 새클래스}, deletes {uid}, problems [적용 안 된 행], warnings [참고]"""
    pill_by_uid = {p["uid"]: p for p in master["pills"]}
    images = master["images"]
    class_ids = {c["id"] for c in master["classes"]}
    problems, warnings = [], []

    ch_rows, ch_bad = read_rows(CHANGES_CSV, 3)
    de_rows, de_bad = read_rows(DELETES_CSV, 2)
    for lineno, raw in ch_bad:
        problems.append({"file": "변경", "line": lineno, "reason": f"숫자로 읽을 수 없음: {raw}"})
    for lineno, raw in de_bad:
        problems.append({"file": "삭제", "line": lineno, "reason": f"숫자로 읽을 수 없음: {raw}"})

    def note_no_effect(kind, lineno, p):
        im = images.get(p["stem"], {})
        if im.get("excluded") or not p["bbox_ok"]:
            warnings.append({"file": kind, "line": lineno, "uid": p["uid"],
                             "reason": "이미 데이터에서 빠지는 사진/알약이라 영향 없음"})

    ch = {}
    ch_conflict = set()
    for lineno, (src, uid, dst) in ch_rows:
        p = pill_by_uid.get(uid)
        if p is None:
            problems.append({"file": "변경", "line": lineno, "uid": uid, "reason": "그런 uid가 없음"})
            continue
        if p["class_id"] != src:
            problems.append({"file": "변경", "line": lineno, "uid": uid,
                             "reason": f"uid {uid}의 원래 클래스는 {p['class_id']}인데 CSV에는 {src}"})
            continue
        if dst not in class_ids:
            problems.append({"file": "변경", "line": lineno, "uid": uid, "reason": f"수정id {dst} 클래스가 없음"})
            continue
        if uid in ch:
            if ch[uid][0] == dst:
                warnings.append({"file": "변경", "line": lineno, "uid": uid, "reason": "같은 행이 중복됨(하나만 적용)"})
            else:
                ch_conflict.add(uid)
            continue
        ch[uid] = (dst, lineno)
    for uid in ch_conflict:
        problems.append({"file": "변경", "line": ch[uid][1], "uid": uid,
                         "reason": "같은 uid에 서로 다른 수정id가 여러 개 있어 모두 적용 안 함"})
        del ch[uid]

    de = {}
    for lineno, (cls, uid) in de_rows:
        p = pill_by_uid.get(uid)
        if p is None:
            problems.append({"file": "삭제", "line": lineno, "uid": uid, "reason": "그런 uid가 없음"})
            continue
        if p["class_id"] != cls:
            problems.append({"file": "삭제", "line": lineno, "uid": uid,
                             "reason": f"uid {uid}의 원래 클래스는 {p['class_id']}인데 CSV에는 {cls}"})
            continue
        if uid in de:
            warnings.append({"file": "삭제", "line": lineno, "uid": uid, "reason": "같은 행이 중복됨(하나만 적용)"})
            continue
        de[uid] = lineno

    for uid in sorted(set(ch) & set(de)):
        problems.append({"file": "변경/삭제", "line": None, "uid": uid,
                         "reason": f"uid {uid}가 변경과 삭제에 동시에 있어 둘 다 적용 안 함"})
        del ch[uid]
        del de[uid]

    changes = {}
    for uid, (dst, lineno) in ch.items():
        p = pill_by_uid[uid]
        if dst == p["class_id"]:
            warnings.append({"file": "변경", "line": lineno, "uid": uid, "reason": "원래 클래스와 같아 변경 없음"})
            continue
        note_no_effect("변경", lineno, p)
        changes[uid] = dst
    for uid, lineno in de.items():
        note_no_effect("삭제", lineno, pill_by_uid[uid])

    return changes, set(de), problems, warnings


def load_train_samples(subset=None):
    """train_coco_final.json 을 읽어서 학습/검증용 샘플 목록을 만든다. (torch 없이 동작)
    subset: None(전체) / "train" / "val"
    반환: samples, label_to_raw_id {모델 라벨(1~N): 대회 category_id}, categories {대회 category_id: 이름},
          raw_id_to_label
    샘플 하나: {"path", "image_id", "boxes"(xyxy 목록), "labels"(모델 라벨 목록)}"""
    if subset not in (None, "train", "val"):
        raise ValueError(f"subset은 None/'train'/'val' 중 하나여야 합니다: {subset}")
    if not FINAL_COCO_PATH.exists():
        raise SystemExit(f"{FINAL_COCO_PATH} 가 없음. 먼저 python src/make_dataset.py 를 실행해줘.")
    with open(FINAL_COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    class_to_label = {c["id"]: i + 1 for i, c in enumerate(cats)}
    label_to_raw_id, categories = {}, {}
    for c in cats:
        if c.get("code") is None:
            raise SystemExit(f"클래스 {c['id']}({c['name']})에 대회 코드(code)가 없음. build_master.py 를 다시 실행해줘.")
        label_to_raw_id[class_to_label[c["id"]]] = int(c["code"])
        categories[int(c["code"])] = c["name"]
    raw_id_to_label = {v: k for k, v in label_to_raw_id.items()}

    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)

    samples = []
    for im in coco["images"]:
        if subset is not None and im["split"] != subset:
            continue
        boxes, labels = [], []
        for a in by_image.get(im["id"], []):
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(class_to_label[a["category_id"]])
        samples.append({"path": im["path"], "image_id": im["id"], "boxes": boxes, "labels": labels})
    return samples, label_to_raw_id, categories, raw_id_to_label