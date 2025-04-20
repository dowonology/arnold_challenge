#!/bin/bash

echo "🔄 공유할 파일 복사 + Git add 시작..."

# 공유할 외부 파일 → shared/ 내부로 복사
# 포맷: [원본경로]="복사할위치"
declare -A FILES_TO_COPY=(
  ["/root/arnold/eval.py"]="eval.py"
  ["/root/arnold/tasks/base_task.py"]="tasks/base_task.py"
)

for SRC in "${!FILES_TO_COPY[@]}"; do
  DST="${FILES_TO_COPY[$SRC]}"

  if [ -f "$SRC" ]; then
    # 필요한 경우 서브디렉토리 생성
    mkdir -p "$(dirname "$DST")"
    cp -i "$SRC" "$DST"
    git add "$DST"
    echo "✅ 복사 & add: $SRC → $DST"
  else
    echo "⚠️  파일 없음: $SRC"
  fi
done

git add .

echo "✅ 완료! 이제 commit 하세요."
