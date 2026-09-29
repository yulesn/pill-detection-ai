import json
from pathlib import Path
from collections import defaultdict

TARGETS = {
    4378, 5094, 5886, 6192, 6563, 10221, 10224,
    12420, 18110, 21026, 22627, 23203, 23223,
    27653, 29871, 31705, 33878, 44199
}

ROOT = Path("data/aihub/labels")

result = defaultdict(lambda: {
    "count": 0,
    "names": set()
})

files = list(ROOT.rglob("*.json"))

print("=" * 90)
print("AI-HUB CATEGORY CHECK")
print("=" * 90)
print(f"JSON files: {len(files):,}")

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

    # categories 확인
    for cat in data.get("categories", []):
        cid = cat.get("id")

        try:
            cid = int(cid)
        except (TypeError, ValueError):
            continue

        if cid in TARGETS:
            result[cid]["count"] += 1

            name = cat.get("name")
            if name:
                result[cid]["names"].add(str(name))

    # annotations 안의 drug_N 확인
    for ann in data.get("annotations", []):
        drug_n = ann.get("drug_N")

        try:
            drug_n = int(drug_n)
        except (TypeError, ValueError):
            continue

        if drug_n in TARGETS:
            result[drug_n]["count"] += 1

    if i % 5000 == 0:
        print(f"Processed: {i:,}/{len(files):,}")

print("\n" + "=" * 90)
print("RESULT")
print("=" * 90)

for cid in sorted(TARGETS):
    count = result[cid]["count"]
    names = sorted(result[cid]["names"])

    print(f"\n{cid}")
    print(f"  occurrences: {count}")

    if names:
        print(f"  names: {', '.join(names[:10])}")
    else:
        print("  names: NONE")

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
