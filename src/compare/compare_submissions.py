import pandas as pd
import numpy as np

A_FILE = "exp_yolo11n_augmented.csv"
B_FILE = "submission_100epoch_conf025.csv"

a = pd.read_csv(A_FILE)
b = pd.read_csv(B_FILE)

print("=" * 80)
print("KAGGLE SUBMISSION 비교")
print("=" * 80)

print(f"\nA = {A_FILE}  [Kaggle ~0.60]")
print(f"B = {B_FILE}  [Kaggle ~0.45]")


# ============================================================
# 1. 전체 규모
# ============================================================

print("\n" + "=" * 80)
print("1. 전체 Prediction 규모")
print("=" * 80)

print(f"A rows : {len(a):,}")
print(f"B rows : {len(b):,}")

print(f"A images : {a['image_id'].nunique():,}")
print(f"B images : {b['image_id'].nunique():,}")

print(f"A categories : {a['category_id'].nunique():,}")
print(f"B categories : {b['category_id'].nunique():,}")


# ============================================================
# 2. 이미지당 detection 수
# ============================================================

print("\n" + "=" * 80)
print("2. 이미지당 Detection 개수")
print("=" * 80)

a_count = a.groupby("image_id").size()
b_count = b.groupby("image_id").size()

print("\n[A ~0.60]")
print(a_count.value_counts().sort_index())

print("\n[B ~0.45]")
print(b_count.value_counts().sort_index())

print("\n평균 Detection / Image")
print(f"A : {a_count.mean():.3f}")
print(f"B : {b_count.mean():.3f}")


# ============================================================
# 3. 이미지별 detection 개수 차이
# ============================================================

print("\n" + "=" * 80)
print("3. Detection 개수가 다른 이미지")
print("=" * 80)

counts = pd.DataFrame({
    "A": a_count,
    "B": b_count
}).fillna(0).astype(int)

counts["diff_B_minus_A"] = counts["B"] - counts["A"]
counts["abs_diff"] = counts["diff_B_minus_A"].abs()

print(f"차이가 있는 이미지 : {(counts['A'] != counts['B']).sum():,}")

print("\n가장 차이가 큰 이미지 30개:")

print(
    counts
    .sort_values("abs_diff", ascending=False)
    .head(30)
    .to_string()
)


# ============================================================
# 4. Category 분포
# ============================================================

print("\n" + "=" * 80)
print("4. Category ID 분포")
print("=" * 80)

cat_a = a["category_id"].value_counts().sort_index()
cat_b = b["category_id"].value_counts().sort_index()

cat = pd.DataFrame({
    "A_0.60": cat_a,
    "B_0.45": cat_b
}).fillna(0).astype(int)

cat["B_minus_A"] = cat["B_0.45"] - cat["A_0.60"]

print(cat.to_string())


# ============================================================
# 5. Category ID 자체 비교
# ============================================================

print("\n" + "=" * 80)
print("5. Category ID 자체 비교")
print("=" * 80)

cats_a = set(a["category_id"])
cats_b = set(b["category_id"])

print(f"A category 수 : {len(cats_a)}")
print(f"B category 수 : {len(cats_b)}")

print("\nA에만 있는 category:")
print(sorted(cats_a - cats_b))

print("\nB에만 있는 category:")
print(sorted(cats_b - cats_a))


# ============================================================
# 6. Score 비교
# ============================================================

print("\n" + "=" * 80)
print("6. Score 비교")
print("=" * 80)

for name, df in [
    ("A ~0.60", a),
    ("B ~0.45", b)
]:
    print(f"\n{name}")

    print(f"mean   : {df['score'].mean():.4f}")
    print(f"median : {df['score'].median():.4f}")
    print(f"min    : {df['score'].min():.4f}")
    print(f"max    : {df['score'].max():.4f}")

    print("\nScore 구간:")

    bins = [
        0,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
        1.00
    ]

    print(
        pd.cut(
            df["score"],
            bins=bins,
            include_lowest=True
        )
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# 7. BBox 통계
# ============================================================

print("\n" + "=" * 80)
print("7. BBox 통계")
print("=" * 80)

for name, df in [
    ("A ~0.60", a),
    ("B ~0.45", b)
]:

    print(f"\n{name}")

    for col in [
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h"
    ]:
        print(
            f"{col:8s} "
            f"mean={df[col].mean():8.2f} "
            f"median={df[col].median():8.2f} "
            f"min={df[col].min():8.2f} "
            f"max={df[col].max():8.2f}"
        )


# ============================================================
# 8. BBox 이상값
# ============================================================

print("\n" + "=" * 80)
print("8. BBox 이상값")
print("=" * 80)

for name, df in [
    ("A ~0.60", a),
    ("B ~0.45", b)
]:

    invalid = (
        (df["bbox_x"] < 0) |
        (df["bbox_y"] < 0) |
        (df["bbox_w"] <= 0) |
        (df["bbox_h"] <= 0) |
        (df["bbox_x"] + df["bbox_w"] > 976) |
        (df["bbox_y"] + df["bbox_h"] > 1280)
    )

    print(f"{name}: {invalid.sum():,}")


# ============================================================
# 9. Image + Category 조합 비교
# ============================================================

print("\n" + "=" * 80)
print("9. Image + Category별 Prediction 개수")
print("=" * 80)

pair_a = a.groupby(
    ["image_id", "category_id"]
).size()

pair_b = b.groupby(
    ["image_id", "category_id"]
).size()

pairs = pd.DataFrame({
    "A": pair_a,
    "B": pair_b
}).fillna(0).astype(int)

pairs["diff"] = pairs["B"] - pairs["A"]

different_pairs = pairs[pairs["A"] != pairs["B"]]

print(
    f"차이가 있는 image/category 조합 : "
    f"{len(different_pairs):,}"
)

print("\n차이가 큰 조합 50개:")

print(
    different_pairs
    .sort_values(
        "diff",
        key=lambda x: x.abs(),
        ascending=False
    )
    .head(50)
    .to_string()
)


# ============================================================
# 10. Annotation ID
# ============================================================

print("\n" + "=" * 80)
print("10. Annotation ID")
print("=" * 80)

for name, df in [
    ("A ~0.60", a),
    ("B ~0.45", b)
]:

    expected = np.arange(1, len(df) + 1)
    actual = df["annotation_id"].to_numpy()

    print(f"\n{name}")
    print(f"첫 번째 : {df['annotation_id'].iloc[0]}")
    print(f"마지막  : {df['annotation_id'].iloc[-1]}")
    print(f"연속    : {np.array_equal(actual, expected)}")
    print(f"중복    : {df['annotation_id'].duplicated().sum()}")


# ============================================================
# 11. 소수점 bbox 여부
# ============================================================

print("\n" + "=" * 80)
print("11. BBox 소수점 여부")
print("=" * 80)

for name, df in [
    ("A ~0.60", a),
    ("B ~0.45", b)
]:

    cols = [
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h"
    ]

    decimal_count = sum(
        (df[col] % 1 != 0).sum()
        for col in cols
    )

    total = len(df) * 4

    print(
        f"{name}: "
        f"{decimal_count:,} / {total:,} "
        f"({decimal_count / total * 100:.2f}%)"
    )


# ============================================================
# 12. 샘플
# ============================================================

print("\n" + "=" * 80)
print("12. A 첫 20개")
print("=" * 80)

print(a.head(20).to_string(index=False))

print("\n" + "=" * 80)
print("12. B 첫 20개")
print("=" * 80)

print(b.head(20).to_string(index=False))


print("\n" + "=" * 80)
print("비교 완료")
print("=" * 80)
  