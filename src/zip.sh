#!/bin/bash
# 지정한 폴더 안의 모든 zip(.zip, .zip.part0 등)을 재귀적으로 끝까지 다 풀어줌
# 한글 파일명 인코딩(cp949) 문제 대응 + 실패한 파일은 재시도 안 하고 넘어감
# 사용법: bash src/zip.sh <압축 풀 대상 폴더>

SRC_DIR="$1"

if [ -z "$SRC_DIR" ]; then
  echo "사용법: bash zip.sh <폴더 경로>"
  exit 1
fi

FAILED_LOG="$SRC_DIR/_실패목록.txt"
> "$FAILED_LOG"

while true; do
  zips=$(find "$SRC_DIR" \( -iname "*.zip" -o -iname "*.zip.part*" \) ! -name "*.failed")
  if [ -z "$zips" ]; then
    echo ""
    echo "===== 완료 ====="
    if [ -s "$FAILED_LOG" ]; then
      echo "실패한 파일 목록 (${FAILED_LOG}):"
      cat "$FAILED_LOG"
    else
      echo "실패 없이 전부 성공!"
    fi
    break
  fi

  echo "$zips" | while IFS= read -r zipfile; do
    dir=$(dirname "$zipfile")
    echo "[압축 해제 중] $zipfile"
    if unzip -O cp949 -o "$zipfile" -d "$dir" > /dev/null 2>&1; then
      rm "$zipfile"
    else
      echo "  → 실패, 건너뜀 (다시 시도 안 함)"
      echo "$zipfile" >> "$FAILED_LOG"
      mv "$zipfile" "${zipfile}.failed"
    fi
  done
done