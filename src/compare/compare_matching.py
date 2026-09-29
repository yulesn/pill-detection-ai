import pandas as pd
import numpy as np

import numpy as np

A_FILE = "exp_yolo11n_augmented.csv"
B_FILE = "submission_100epoch_conf025.csv"

IOU_THRESHOLD = 0.5

a = pd.read_csv(A_FILE)
b = pd.read_csv(B_FILE)


def iou(box1, box2):
    ax1, ay1, aw, ah = box1
    bx1, by1, bw, bh = box2

    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


matched_same = []
matched_diff = []
only_a = []
only_b = []

image_ids = sorted(set(a["image_id"]) | set(b["image_id"]))

for image_id in image_ids:

    aa = a[a["image_id"] == image_id]
    bb = b[b["image_id"] == image_id]

    used_b = set()

    for _, ar in aa.iterrows():

        box_a = [
            ar["bbox_x"],
            ar["bbox_y"],
            ar["bbox_w"],
            ar["bbox_h"]
        ]

        candidates = []

        for bi, br in bb.iterrows():

            if bi in used_b:
                continue

            box_b = [
                br["bbox_x"],
                br["bbox_y"],
                br["bbox_w"],
                br["bbox_h"]
            ]

            score = iou(box_a, box_b)
            candidates.append((score, bi))

        if not candidates:
            only_a.append(ar.to_dict())
            continue

        best_iou, best_bi = max(candidates)

        if best_iou >= IOU_THRESHOLD:

            br = bb.loc[best_bi]
            used_b.add(best_bi)

            row = {
                "image_id": image_id,
                "iou": best_iou,
                "A_category": int(ar["category_id"]),
                "B_category": int(br["category_id"]),
                "A_score": float(ar["score"]),
                "B_score": float(br["score"]),
                "A_x": float(ar["bbox_x"]),
                "A_y": float(ar["bbox_y"]),
                "A_w": float(ar["bbox_w"]),
                "A_h": float(ar["bbox_h"]),
                "B_x": float(br["bbox_x"]),
                "B_y": float(br["bbox_y"]),
                "B_w": float(br["bbox_w"]),
                "B_h": float(br["bbox_h"]),
            }

            if int(ar["category_id"]) == int(br["category_id"]):
                matched_same.append(row)
            else:
                matched_diff.append(row)

        else:
            only_a.append(ar.to_dict())

    for bi, br in bb.iterrows():
        if bi not in used_b:
            only_b.append(br.to_dict())


same = pd.DataFrame(matched_same)
diff = pd.DataFrame(matched_diff)
only_a_df = pd.DataFrame(only_a)
only_b_df = pd.DataFrame(only_b)


print("=" * 90)
print("A vs B BBOX MATCHING ANALYSIS")
print("=" * 90)

print(f"\nIoU threshold: {IOU_THRESHOLD}")

print("\n" + "=" * 90)
print("1. TOTAL")
print("=" * 90)

print(f"A predictions : {len(a):,}")
print(f"B predictions : {len(b):,}")
print(f"Matched - same category : {len(same):,}")
print(f"Matched - different category : {len(diff):,}")
print(f"A only : {len(only_a_df):,}")
print(f"B only : {len(only_b_df):,}")


print("\n" + "=" * 90)
print("2. MATCH RATIO")
print("=" * 90)

total_matched = len(same) + len(diff)

if total_matched > 0:
    print(f"Total matched : {total_matched:,}")
    print(
        f"Same category : "
        f"{len(same):,} "
        f"({len(same) / total_matched * 100:.2f}%)"
    )
    print(
        f"Different category : "
        f"{len(diff):,} "
        f"({len(diff) / total_matched * 100:.2f}%)"
    )


print("\n" + "=" * 90)
print("3. CATEGORY MISMATCHES")
print("=" * 90)

if len(diff) > 0:

    category_changes = (
        diff
        .groupby(["A_category", "B_category"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(category_changes.head(50).to_string(index=False))

else:
    print("No category mismatches.")


print("\n" + "=" * 90)
print("4. B ONLY PREDICTIONS")
print("=" * 90)

if len(only_b_df) > 0:

    print("Category counts:")
    print(
        only_b_df["category_id"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\nScore statistics:")
    print(
        only_b_df["score"]
        .describe()
        .to_string()
    )

else:
    print("No B-only predictions.")


print("\n" + "=" * 90)
print("5. A ONLY PREDICTIONS")
print("=" * 90)

if len(only_a_df) > 0:

    print("Category counts:")
    print(
        only_a_df["category_id"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\nScore statistics:")
    print(
        only_a_df["score"]
        .describe()
        .to_string()
    )

else:
    print("No A-only predictions.")


print("\n" + "=" * 90)
print("6. TOP CATEGORY MISMATCH EXAMPLES")
print("=" * 90)

if len(diff) > 0:

    print(
        diff[
            [
                "image_id",
                "iou",
                "A_category",
                "B_category",
                "A_score",
                "B_score"
            ]
        ]
        .sort_values("iou", ascending=False)
        .head(50)
        .to_string(index=False)
    )

else:
    print("No mismatches.")


print("\n" + "=" * 90)
print("7. SCORE DIFFERENCE - SAME CATEGORY")
print("=" * 90)

if len(same) > 0:

    same = same.copy()
    same["score_diff"] = same["B_score"] - same["A_score"]

    print(
        same[
            [
                "image_id",
                "iou",
                "A_category",
                "A_score",
                "B_score",
                "score_diff"
            ]
        ]
        .assign(abs_diff=lambda x: x["score_diff"].abs())
        .sort_values("abs_diff", ascending=False)
        .drop(columns=["abs_diff"])
        .head(50)
        .to_string(index=False)
    )

else:
    print("No same-category matches.")


print("\n" + "=" * 90)
print("8. IOU DISTRIBUTION")
print("=" * 90)

all_matched = pd.concat(
    [same, diff],
    ignore_index=True
)

if len(all_matched) > 0:

    bins = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        1.01
    ]

    distribution = (
        pd.cut(
            all_matched["iou"],
            bins=bins,
            include_lowest=True
        )
        .value_counts()
        .sort_index()
    )

    print(distribution.to_string())

else:
    print("No matched boxes.")


print("\n" + "=" * 90)
print("9. SAVING RESULTS")
print("=" * 90)

same.to_csv("match_same_category.csv", index=False)
diff.to_csv("match_different_category.csv", index=False)
only_a_df.to_csv("only_a.csv", index=False)
only_b_df.to_csv("only_b.csv", index=False)

print("Saved:")
print("  match_same_category.csv")
print("  match_different_category.csv")
print("  only_a.csv")
print("  only_b.csv")

print("\n" + "=" * 90)
print("DONE")
