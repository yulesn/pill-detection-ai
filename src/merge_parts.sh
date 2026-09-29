#!/bin/bash
# *.zip.part* 조각들을 원래 파일명 기준으로 찾아서 하나의 zip으로 합침
# 사용법: bash src/merge_parts.sh <폴더경로>

SRC_DIR="$1"

if [ -z "$SRC_DIR" ]; then
  echo "사용법: bash merge_parts.sh <폴더 경로>"
  exit 1
fi

find "$SRC_DIR" -name "*.zip.part*" | sed -E 's/\.zip\.part[0-9]*$/.zip/' | sort -u | while IFS= read -r zipname; do
  dir=$(dirname "$zipname")
  base=$(basename "$zipname")
  echo "[병합 중] $zipname"

  find "$dir" -maxdepth 1 -name "${base}.part*" -print0 \
    | sort -zt'.' -k2V \
    | xargs -0 cat > "$zipname"

  find "$dir" -maxdepth 1 -name "${base}.part*" -delete

  echo "  -> 완료 ($(du -h "$zipname" | cut -f1))"
done

echo ""
echo "===== 병합 완료 ====="