"""
모든 알약(라벨 json 하나 = 알약 하나)에 고유번호(uid)를 붙여 대응표(pill_master.json)를 만든다.
- 원본 json 은 절대 수정하지 않는다.
- 다시 실행해도 이미 받은 uid 와 클래스 번호는 그대로 유지되고, 새 알약에만 새 uid 가 붙는다.
- 사진 제외 규칙은 기존과 같다: 첫 json 이 깨졌거나, 정상 bbox 가 3개 미만이면 사진을 통째로 제외.

사용법:
    python src/build_master.py
"""
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from pill_common import (DATA_ROOTS, MASTER_PATH, MIN_VALID_PILLS_PER_IMAGE, START_UID,
                         norm, split_combo_angle)


def posix(p):
    return Path(p).as_posix()


CODE_RE = re.compile(r"K-(\d+)")


def parse_code(*candidates):
    """'K-016548' 같은 문자열에서 숫자 코드(16548)를 뽑는다. 없으면 None."""
    for c in candidates:
        if c:
            m = CODE_RE.search(str(c))
            if m:
                return int(m.group(1))
    return None


def load_json(path):
    """정상이고 dl_name 이 있는 json 만 돌려준다. 아니면 None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data["images"][0].get("dl_name"):
            return None
        return data
    except Exception:
        return None


def scan_roots():
    img_paths, json_paths = {}, {}
    for img_dir, annot_dir in DATA_ROOTS:
        for d in (img_dir, annot_dir):
            if not Path(d).exists():
                print(f"[경고] 폴더가 없음: {d}")
        imgs, jsons = {}, defaultdict(list)
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith(".png"):
                    imgs[os.path.splitext(file)[0]] = os.path.join(root, file)
        for root, _, files in os.walk(annot_dir):
            for file in files:
                if file.endswith(".json"):
                    jsons[os.path.splitext(file)[0]].append(os.path.join(root, file))
        img_paths.update(imgs)
        json_paths.update(jsons)
    stems = sorted(set(img_paths) & set(json_paths))
    return img_paths, json_paths, stems


def main():
    old = None
    if MASTER_PATH.exists():
        with open(MASTER_PATH, "r", encoding="utf-8") as f:
            old = json.load(f)
    old_uid = {p["key"]: p["uid"] for p in old["pills"]} if old else {}
    next_uid = old["next_uid"] if old else START_UID

    print("폴더를 훑는 중... (처음 한 번만 오래 걸림)")
    img_paths, json_paths, stems = scan_roots()
    print(f"사진+라벨이 둘 다 있는 사진: {len(stems)}장")

    images, pills, names_by_image = {}, [], {}
    for n, stem in enumerate(stems, start=1):
        if n % 2000 == 0:
            print(f"  {n}/{len(stems)}")
        paths = json_paths[stem]
        first = load_json(paths[0])
        w = h = None
        first_ok = False
        if first is not None:
            try:
                w, h = first["images"][0]["width"], first["images"][0]["height"]
                first_ok = w is not None and h is not None
            except Exception:
                first_ok = False

        entries, names = [], []
        for rank, jp in enumerate(paths):
            data = first if rank == 0 else load_json(jp)
            if data is None:
                continue
            name = data["images"][0]["dl_name"]
            names.append(name)
            meta = data["images"][0]
            code_meta = parse_code(meta.get("drug_N"), meta.get("dl_mapping_code"))
            code_folder = parse_code(Path(jp).parent.name)
            code = code_meta if code_meta is not None else code_folder
            for k, anno in enumerate(data.get("annotations", [])):
                bbox = anno.get("bbox")
                ok = False
                if first_ok and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, bw, bh = bbox
                    ok = not (x < 0 or y < 0 or (x + bw) > w or (y + bh) > h)
                entries.append({
                    "key": f"{posix(jp)}#{k}", "stem": stem, "rank": rank, "k": k,
                    "json": posix(jp), "orig_name": name,
                    "code": code, "code_folder": code_folder,
                    "raw_cat": anno.get("category_id"),
                    "bbox": list(bbox) if isinstance(bbox, (list, tuple)) else None,
                    "bbox_ok": ok,
                })

        n_valid = sum(1 for e in entries if e["bbox_ok"])
        excluded = (not first_ok) or n_valid < MIN_VALID_PILLS_PER_IMAGE
        if not first_ok:
            reason = "첫 json 손상"
        elif excluded:
            reason = f"정상 bbox {n_valid}개"
        else:
            reason = ""
        combo, angle = split_combo_angle(stem)
        images[stem] = {
            "file_name": stem + ".png", "path": posix(img_paths[stem]),
            "width": w, "height": h, "combo": combo, "angle": angle,
            "n_valid": n_valid, "excluded": excluded, "reason": reason,
        }
        pills.extend(entries)
        if not excluded:
            names_by_image[stem] = names

    # 클래스: 처음엔 이름(공백 제거) 가나다순 번호, 다시 실행하면 기존 번호 유지
    variants = defaultdict(set)
    for names in names_by_image.values():
        for nm in names:
            variants[norm(nm)].add(nm)
    classes = list(old["classes"]) if old else []
    known = {c["norm"]: c for c in classes}
    max_id = max((c["id"] for c in classes), default=0)
    new_classes = 0
    for nk in sorted(k for k in variants if k not in known):
        max_id += 1
        c = {"id": max_id, "norm": nk, "name": sorted(variants[nk])[0]}
        classes.append(c)
        known[nk] = c
        new_classes += 1

    pills.sort(key=lambda p: (p["stem"], p["rank"], p["k"]))
    new_uids = 0
    for p in pills:
        uid = old_uid.get(p["key"])
        if uid is None:
            uid = next_uid
            next_uid += 1
            new_uids += 1
        p["uid"] = uid
        c = known.get(norm(p["orig_name"]))
        p["class_id"] = c["id"] if c else None

    # 클래스 <-> 대회 코드(category_id) 대응. 제출할 때 필요하다.
    kept_stems = {s for s, im in images.items() if not im["excluded"]}
    code_counts = defaultdict(Counter)      # 클래스 -> {코드: 개수}
    code_classes = defaultdict(set)         # 코드 -> {클래스}
    for p in pills:
        if p["class_id"] is not None and p["stem"] in kept_stems and p["code"] is not None:
            code_counts[p["class_id"]][p["code"]] += 1
            code_classes[p["code"]].add(p["class_id"])
    for c in classes:
        cnt = code_counts.get(c["id"])
        c["code"] = cnt.most_common(1)[0][0] if cnt else None
        c["codes"] = [[k, v] for k, v in cnt.most_common()] if cnt and len(cnt) > 1 else None

    multi_code = [c for c in classes if c["codes"]]
    shared_code = {k: sorted(v) for k, v in code_classes.items() if len(v) > 1}
    no_code = [c for c in classes if c["code"] is None]
    kaggle_bad = [p for p in pills if p["raw_cat"] not in (None, 1) and p["code"] is not None
                  and p["raw_cat"] != p["code"]]
    folder_bad = sum(1 for p in pills if p["code_folder"] is not None and p["code"] is not None
                     and p["code_folder"] != p["code"])

    master = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "roots": DATA_ROOTS, "next_uid": next_uid,
        "classes": classes, "images": images, "pills": pills,
    }
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MASTER_PATH, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, separators=(",", ":"))

    kept = [s for s, im in images.items() if not im["excluded"]]
    kept_set = set(kept)
    n_ann = sum(1 for p in pills if p["stem"] in kept_set and p["bbox_ok"])
    print("\n=== 대응표 생성 완료 ===")
    print(f"파일: {MASTER_PATH}")
    print(f"사진 {len(images)}장 중 제외 {len(images) - len(kept)}장, 남는 사진 {len(kept)}장")
    print(f"알약(라벨) {len(pills)}개 (새로 붙은 uid {new_uids}개), 남는 사진의 정상 알약 {n_ann}개")
    print(f"클래스 {len(classes)}종 (새로 생긴 클래스 {new_classes}종)")

    print("\n=== 클래스 <-> 대회 코드 점검 (제출할 때 이 코드로 바꿔서 냄) ===")
    print(f"코드를 못 찾은 클래스: {len(no_code)}종")
    print(f"이름은 하나인데 코드가 2개 이상인 클래스: {len(multi_code)}종")
    for c in multi_code[:10]:
        print(f"  클래스 {c['id']} ({c['name']}): 코드 {c['codes']}  -> 가장 많은 {c['code']} 로 제출됨")
    print(f"같은 코드인데 이름이 다른 클래스로 나뉜 경우: {len(shared_code)}건")
    for code, cids in list(shared_code.items())[:10]:
        print(f"  코드 {code}: 클래스 {cids}")
    print(f"Kaggle 원본의 category_id 와 코드가 다른 알약: {len(kaggle_bad)}개 (0이어야 '코드=대회 번호'가 확인된 것)")
    for p in kaggle_bad[:3]:
        print(f"  {p['json']}: category_id={p['raw_cat']}, 코드={p['code']}")
    print(f"폴더 이름의 코드와 json 속 코드가 다른 알약: {folder_bad}개")


if __name__ == "__main__":
    main()