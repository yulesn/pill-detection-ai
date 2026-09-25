import json
import glob
from collections import Counter

files = glob.glob("data/aihub/labels/**/*.json", recursive=True)

print("전체 JSON:", len(files))

drugs = Counter()
bad_files = []

for i, file in enumerate(files, 1):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for img in data.get("images", []):
            drug_code = img.get("drug_N")
            drug_name = img.get("dl_name_en")

            if drug_code:
                drugs[(drug_code, drug_name)] += 1

    except Exception:
        bad_files.append(file)

    if i % 5000 == 0:
        print(f"{i:,} / {len(files):,} 처리")

print()
print("정상 JSON:", len(files) - len(bad_files))
print("오류 JSON:", len(bad_files))
print("약품 종류:", len(drugs))

print("\n=== 약품 샘플 30개 ===")

for (code, name), count in drugs.most_common(30):
    print(f"{code} | {name} | {count}개")

print("\n=== 오류 파일 샘플 ===")

for file in bad_files[:10]:
    print(file)