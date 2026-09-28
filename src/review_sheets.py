"""
확인용 그림을 만든다. 사진 위 번호는 알약 고유번호(uid)라서, 이 번호를 그대로 CSV에 적으면 된다.

사용법:
    python src/review_sheets.py              # 변경/삭제한 알약만 모아서 (변경은 옮겨간 클래스의 기준 알약과 나란히)
    python src/review_sheets.py --class 5    # 클래스 5번(최종 데이터 기준) 전체
    python src/review_sheets.py --all        # 모든 클래스 (오래 걸림)
"""
import argparse
import json
import time
from collections import defaultdict

import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from pill_common import FINAL_COCO_PATH, load_master, load_rules

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("outputs/review")
PAGE_SIZE = 80      # 클래스 그림 한 장에 넣는 알약 수
COLS = 10
PAIRS_PER_PAGE = 24  # 변경 그림 한 장에 넣는 (옮긴 알약 | 기준 알약) 쌍의 수
SINGLES_PER_PAGE = 40


def short(name, n=9):
    name = name or "?"
    return name if len(name) <= n else name[:n] + "…"


def placeholder():
    ph = np.full((100, 100, 3), 230, dtype=np.uint8)
    cv2.line(ph, (10, 10), (90, 90), (255, 0, 0), 4)
    cv2.line(ph, (90, 10), (10, 90), (255, 0, 0), 4)
    return ph


def crop_pill(pill, images, context, box_color=None):
    """알약 사진을 잘라서 (RGB 배열, 상태)를 돌려준다. context=True 면 주변 여백과 박스를 함께 보여준다."""
    im = images.get(pill["stem"])
    if im is None:
        return None, "사진 정보 없음"
    img = cv2.imread(im["path"])
    if img is None:
        return None, "사진 못 읽음"
    x, y, w, h = [int(v) for v in pill["bbox"]]
    pad = max(20, int(0.25 * max(w, h))) if context else 0
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    view = img[y0:y1, x0:x1].copy()
    if view.size == 0:
        return None, "빈 영역"
    if context and box_color is not None:
        cv2.rectangle(view, (x - x0, y - y0), (x - x0 + w, y - y0 + h), box_color, 3)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB), "ok"


def clear(patterns):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for p in OUTPUT_DIR.glob(pat):
            try:
                p.unlink()
            except OSError as e:
                print(f"[경고] 지우지 못함(열려 있는 파일?): {p} - {e}")


def load_final():
    if not FINAL_COCO_PATH.exists():
        raise SystemExit(f"{FINAL_COCO_PATH} 가 없음. 먼저 python src/make_dataset.py 를 실행해줘.")
    with open(FINAL_COCO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, name):
    path = OUTPUT_DIR / name
    try:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    except OSError:
        fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)


def sheet_changes(master, final, changes):
    if not changes:
        print("변경: 적용된 항목 없음")
        return
    pill_by_uid = {p["uid"]: p for p in master["pills"]}
    cls_name = {c["id"]: c["name"] for c in master["classes"]}
    images = master["images"]

    ref = {}   # 클래스별 기준 알약: 변경 대상이 아닌, 그 클래스의 첫 알약
    for a in final["annotations"]:
        if a["uid"] not in changes and a["category_id"] not in ref:
            ref[a["category_id"]] = pill_by_uid[a["uid"]]

    items = sorted(changes.items(), key=lambda kv: (kv[1], pill_by_uid[kv[0]]["class_id"], kv[0]))
    n_pages = (len(items) + PAIRS_PER_PAGE - 1) // PAIRS_PER_PAGE
    stamp = time.strftime("%m-%d %H:%M")
    fails = 0
    for page in range(n_pages):
        chunk = items[page * PAIRS_PER_PAGE:(page + 1) * PAIRS_PER_PAGE]
        rows = (len(chunk) + 3) // 4
        fig, axes = plt.subplots(rows, 8, figsize=(8 * 2.1, rows * 2.6), gridspec_kw={"hspace": 0.7})
        axes = np.atleast_1d(axes).reshape(rows, 8)
        for i, (uid, dst) in enumerate(chunk):
            p = pill_by_uid[uid]
            r, c = divmod(i, 4)
            ax_l, ax_r = axes[r, c * 2], axes[r, c * 2 + 1]
            view, st = crop_pill(p, images, True, (0, 200, 0))
            fails += st != "ok"
            ax_l.imshow(view if view is not None else placeholder())
            ax_l.set_title(f"{uid}\n{p['class_id']}→{dst}", fontsize=8, color="black" if st == "ok" else "red")
            rp = ref.get(dst)
            rview, rst = (crop_pill(rp, images, True, (200, 120, 0)) if rp else (None, "기준 없음"))
            ax_r.imshow(rview if rview is not None else placeholder())
            ax_r.set_title(f"기준 {dst}\n{short(cls_name.get(dst))}", fontsize=8, color="gray")
        for ax in axes.flatten():
            ax.axis("off")
        fig.suptitle(f"변경 {len(items)}건 | {page + 1}/{n_pages}페이지 | 왼쪽=옮긴 알약(초록 박스), 오른쪽=옮겨간 클래스의 기준 알약 | 생성 {stamp}",
                     fontsize=10)
        save_fig(fig, f"changes_p{page + 1}.png")
    print(f"변경: {len(items)}건 -> {n_pages}페이지 (사진 못 읽음 {fails}건)")


def sheet_deletes(master, deletes):
    if not deletes:
        print("삭제: 적용된 항목 없음")
        return
    pill_by_uid = {p["uid"]: p for p in master["pills"]}
    cls_name = {c["id"]: c["name"] for c in master["classes"]}
    images = master["images"]
    uids = sorted(deletes, key=lambda u: (pill_by_uid[u]["class_id"], u))
    n_pages = (len(uids) + SINGLES_PER_PAGE - 1) // SINGLES_PER_PAGE
    stamp = time.strftime("%m-%d %H:%M")
    fails = 0
    for page in range(n_pages):
        chunk = uids[page * SINGLES_PER_PAGE:(page + 1) * SINGLES_PER_PAGE]
        cols = min(len(chunk), 8)
        rows = (len(chunk) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.6), gridspec_kw={"hspace": 0.7})
        axes = np.atleast_1d(axes).flatten()
        for ax, uid in zip(axes, chunk):
            p = pill_by_uid[uid]
            view, st = crop_pill(p, images, True, (255, 0, 0))
            fails += st != "ok"
            ax.imshow(view if view is not None else placeholder())
            ax.set_title(f"{uid}\n{p['class_id']} {short(cls_name.get(p['class_id']))}", fontsize=8,
                         color="black" if st == "ok" else "red")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"삭제 {len(uids)}건 | {page + 1}/{n_pages}페이지 | 빨간 박스가 지워진 알약(박스만 지워지고 사진은 남음) | 생성 {stamp}",
                     fontsize=10)
        save_fig(fig, f"deletes_p{page + 1}.png")
    print(f"삭제: {len(uids)}건 -> {n_pages}페이지 (사진 못 읽음 {fails}건)")


def sheet_class(master, final, class_id, by_class):
    pill_by_uid = {p["uid"]: p for p in master["pills"]}
    cls_name = {c["id"]: c["name"] for c in master["classes"]}
    images = master["images"]
    anns = by_class.get(class_id, [])
    if not anns:
        print(f"class {class_id}: 알약 없음")
        return
    clear([f"class_{class_id}_p*.png"])
    n_total = len(anns)
    n_pages = (n_total + PAGE_SIZE - 1) // PAGE_SIZE
    stamp = time.strftime("%m-%d %H:%M")
    fails = 0
    for page in range(n_pages):
        chunk = anns[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
        cols = min(len(chunk), COLS)
        rows = (len(chunk) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.4), gridspec_kw={"hspace": 0.5})
        axes = np.atleast_1d(axes).flatten()
        for ax, a in zip(axes, chunk):
            p = pill_by_uid[a["uid"]]
            view, st = crop_pill(p, images, False)
            fails += st != "ok"
            ax.imshow(view if view is not None else placeholder())
            label = f"{a['uid']}"
            if p["class_id"] != class_id:
                label += f"\n(원래 {p['class_id']})"
            ax.set_title(label, fontsize=8, color="black" if st == "ok" else "red")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"class {class_id} ({cls_name.get(class_id)}) | 전체 {n_total}개 | {page + 1}/{n_pages}페이지 | "
                     f"사진 위 번호=uid | 생성 {stamp}", fontsize=10)
        save_fig(fig, f"class_{class_id}_p{page + 1}.png")
    print(f"class {class_id}: {n_total}개 -> {n_pages}페이지 (사진 못 읽음 {fails}건)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="class_id", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    master = load_master()
    if args.class_id is None and not args.all:
        changes, deletes, _problems, _warnings = load_rules(master)
        final = load_final()
        clear(["changes_p*.png", "deletes_p*.png"])
        sheet_changes(master, final, changes)
        sheet_deletes(master, deletes)
        print(f"\n결과 폴더: {OUTPUT_DIR}")
        return

    final = load_final()
    by_class = defaultdict(list)
    for a in final["annotations"]:
        by_class[a["category_id"]].append(a)
    if args.all:
        clear(["class_*_p*.png"])
        for cid in sorted(by_class):
            sheet_class(master, final, cid, by_class)
    else:
        sheet_class(master, final, args.class_id, by_class)
    print(f"\n결과 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()