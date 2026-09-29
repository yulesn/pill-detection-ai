import json
from pathlib import Path
import pandas as pd

A_FILE = Path("exp_yolo11n_augmented.csv")
B_FILE = Path("submission_100epoch_conf025.csv")
TRAIN_ANN = Path("data/raw/sprint_ai_project1_data/train_annotations")


def collect_categories(root):
    result = {}

    files = list(root.rglob("*.json"))
    print(f"JSON files found: {len(files):,}")

    for i, path in enumerate(files, 1):

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception:
            try:
                with open(path, "r", encoding="cp949") as f:
                    data = json.load(f)
            except Exception:
                continue

        categories = data.get("categories", [])

        for cat in categories:
            if "id" in cat:
                cid = int(cat["id"])
                name = cat.get("name", "")
                result[cid] = name

        if i % 500 == 0:
            print(f"Processed: {i:,}/{len(files):,}")

    return result


print("=" * 90)
print("CATEGORY SET COMPARISON")
print("=" * 90)

a = pd.read_csv(A_FILE)
b = pd.read_csv(B_FILE)

a_cats = set(a["category_id"].astype(int).unique())
b_cats = set(b["category_id"].astype(int).unique())

print(f"\nA CSV categories: {len(a_cats)}")
print(f"B CSV categories: {len(b_cats)}")

print("\nA only:")
print(sorted(a_cats - b_cats))

print("\nB only:")
print(sorted(b_cats - a_cats))

print("\nIntersection:")
print(len(a_cats & b_cats))


print("\n" + "=" * 90)
print("ORIGINAL TRAIN CATEGORY SET")
print("=" * 90)

original = collect_categories(TRAIN_ANN)

print(f"\nOriginal unique category IDs: {len(original)}")

original_cats = set(original.keys())

print("\nA categories that exist in original train:")
a_in_original = sorted(a_cats & original_cats)
print(a_in_original)
print(f"Count: {len(a_in_original)}")

print("\nA categories NOT found in original train:")
a_not_original = sorted(a_cats - original_cats)
print(a_not_original)
print(f"Count: {len(a_not_original)}")

print("\nB categories that exist in original train:")
b_in_original = sorted(b_cats & original_cats)
print(b_in_original)
print(f"Count: {len(b_in_original)}")

print("\nB categories NOT found in original train:")
b_not_original = sorted(b_cats - original_cats)
print(b_not_original)
print(f"Count: {len(b_not_original)}")


print("\n" + "=" * 90)
print("A-ONLY CATEGORY NAMES")
print("=" * 90)

for cid in sorted(a_cats - b_cats):
    print(f"{cid}: {original.get(cid, 'NOT FOUND')}")


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
