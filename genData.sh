#!/bin/bash

# Game of Life 데이터 자동 생성 스크립트
# 생존 비율 0.01~0.99까지 99개 데이터셋 생성

echo "🚀 GOL 데이터셋 자동 생성 시작..."
echo "📊 생존 비율: 0.01 ~ 0.99 (99개 파일)"
echo "📁 각 파일: 1000개 샘플"
echo ""

# 진행 상황 추적
total_files=99
current=0

# 데이터 생성 (0.01부터 0.99까지)
for ratio in $(seq -f "%.2f" 0.01 0.01 0.99); do
    current=$((current + 1))
    progress=$((current * 100 / total_files))
    
    echo "[$current/$total_files] ($progress%) 생성 중: database-54321_1000_${ratio}.txt"
    
    # GPU가 있으면 GPU 모드, 없으면 CPU 모드 자동 선택
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        python3 new_project/datagen.py 54321 1000 $ratio --one_file
    else
        echo "  ⚠️  GPU 미감지, CPU 모드로 실행..."
        python3 new_project/datagen.py 54321 1000 $ratio --one_file --cpu
    fi
    
    # 에러 체크
    if [ $? -ne 0 ]; then
        echo "❌ 오류 발생: database-54321_1000_${ratio}.txt 생성 실패"
        echo "💡 해결 방법:"
        echo "   1. 빌드 확인: cd build && make"
        echo "   2. CPU 모드: --cpu 옵션 추가"
        exit 1
    fi
done

echo ""
echo "✅ 모든 데이터셋 생성 완료!"
echo "📊 총 파일: $total_files개"
echo "📁 저장 위치: train_data/"
echo "💾 총 샘플: $((total_files * 1000))개"