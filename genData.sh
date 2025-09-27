#!/bin/bash

# 데이터 생성
for ratio in $(seq -f "%.2f" 0.01 0.01 0.99); do
    echo "생성 중: database-54321_1000_${ratio}.txt"
    python3 new_project/datagen.py GOLdatagen 54321 1000 $ratio --one_file=True
done

echo "모든 데이터셋 생성 완료!"