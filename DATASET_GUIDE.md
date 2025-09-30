# 📊 Dataset Configuration Guide

`dataset_config.json` 파일을 통해 자신만의 데이터셋을 정의하고 실험할 수 있습니다.

## 🎯 **기본 사용법**

### 1️⃣ **데이터 생성하기**
```bash
# 기본 99개 데이터셋 생성
./genData.sh                    # Linux/macOS
.\genData.ps1                   # Windows

# 커스텀 데이터 생성 (예시)
python3 new_project/datagen.py 12345 500 0.7 --one_file --cpu
python3 new_project/datagen.py 99999 1000 0.8 --one_file --cpu
```

### 2️⃣ **dataset_config.json에 추가하기**
```json
{
  "datasets": {
    "my_experiment": {
      "name": "내 실험용 데이터셋",
      "type": "simulation_files",
      "paths": [
        "database-12345_500_0.700000.txt",
        "database-99999_1000_0.800000.txt"
      ],
      "description": "고밀도 패턴 분석 실험",
      "expected_samples": 1500,
      "grid_size": [50, 50]
    }
  }
}
```

### 3️⃣ **훈련 실행하기**
```bash
cd new_project
python3 train.py --dataset my_experiment --epochs 50

# 모델 설정도 함께 사용
python3 train.py --dataset my_experiment --model-config small_cnn
python3 train.py --dataset my_experiment --model-config large_cnn --epochs 30
```

## 🛠️ **설정 옵션 설명**

### **데이터셋 블록 구조**
```json
"dataset_name": {
  "name": "사용자에게 표시될 이름",
  "type": "simulation_files",           // 고정값
  "paths": ["파일1.txt", "파일2.txt"],    // 데이터 파일 목록
  "description": "데이터셋 설명",
  "expected_samples": 총샘플수,          // 예상 총 샘플 개수
  "grid_size": [50, 50]                // 입력 그리드 크기 (고정)
}
```

### **필드별 세부 설명**

| 필드 | 설명 | 예시 |
|------|------|------|
| `name` | UI에 표시되는 데이터셋 이름 | `"My Custom Dataset"` |
| `type` | 데이터 타입 (항상 `"simulation_files"`) | `"simulation_files"` |
| `paths` | 데이터 파일 경로 배열 | `["file1.txt", "file2.txt"]` |
| `description` | 데이터셋 설명 및 용도 | `"고밀도 패턴 실험용"` |
| `expected_samples` | 예상 총 샘플 수 | `2000` |
| `grid_size` | 입력 그리드 크기 (50x50 고정) | `[50, 50]` |

## 🎮 **실전 예시**

### **예시 1: 빠른 테스트용 소규모 데이터셋**
```json
"quick_test": {
  "name": "Quick Test Dataset",
  "type": "simulation_files",
  "paths": [
    "database-12345_10_0.300000.txt"
  ],
  "description": "빠른 모델 테스트용 (10개 샘플)",
  "expected_samples": 10,
  "grid_size": [50, 50]
}
```

```bash
# 데이터 생성
python3 new_project/datagen.py 12345 10 0.3 --one_file --cpu

# 훈련 (빠른 테스트)
python3 train.py --dataset quick_test --epochs 5
```

### **예시 2: 특정 생존율 범위 실험**
```json
"high_survival": {
  "name": "High Survival Rate Experiment",
  "type": "simulation_files", 
  "paths": [
    "database-77777_500_0.700000.txt",
    "database-77777_500_0.800000.txt",
    "database-77777_500_0.900000.txt"
  ],
  "description": "고생존율 패턴 분석 (0.7-0.9)",
  "expected_samples": 1500,
  "grid_size": [50, 50]
}
```

```bash
# 데이터 생성
python3 new_project/datagen.py 77777 500 0.7 --one_file --cpu
python3 new_project/datagen.py 77777 500 0.8 --one_file --cpu  
python3 new_project/datagen.py 77777 500 0.9 --one_file --cpu

# 훈련
python3 train.py --dataset high_survival --epochs 30
```

### **예시 3: 시드별 비교 실험**
```json
"seed_comparison": {
  "name": "Seed Comparison Experiment",
  "type": "simulation_files",
  "paths": [
    "database-11111_1000_0.500000.txt",
    "database-22222_1000_0.500000.txt", 
    "database-33333_1000_0.500000.txt"
  ],
  "description": "동일 생존율, 다른 시드 비교 실험",
  "expected_samples": 3000,
  "grid_size": [50, 50]
}
```

```bash
# 데이터 생성 (같은 생존율, 다른 시드)
python3 new_project/datagen.py 11111 1000 0.5 --one_file --cpu
python3 new_project/datagen.py 22222 1000 0.5 --one_file --cpu
python3 new_project/datagen.py 33333 1000 0.5 --one_file --cpu

# 훈련
python3 train.py --dataset seed_comparison --epochs 40
```

## 📁 **파일 경로 규칙**

데이터 파일은 다음 위치에서 자동으로 찾습니다:
1. `../train_data/` (프로젝트 루트의 train_data)
2. `train_data/` (현재 디렉토리의 train_data)  
3. `../../train_data/` (상위 디렉토리의 train_data)
4. `./train_data/` (현재 디렉토리의 train_data)

## 🚀 **고급 활용**

### **배치 데이터 생성 스크립트**
```bash
#!/bin/bash
# custom_data_gen.sh

echo "🎯 커스텀 실험 데이터 생성 중..."

# 고밀도 패턴
for ratio in 0.7 0.8 0.9; do
    python3 new_project/datagen.py 77777 500 $ratio --one_file --cpu
done

# 저밀도 패턴  
for ratio in 0.1 0.2 0.3; do
    python3 new_project/datagen.py 11111 1000 $ratio --one_file --cpu
done

echo "✅ 커스텀 데이터 생성 완료!"
```

### **자동 실험 스크립트**
```bash
#!/bin/bash
# auto_experiment.sh

datasets=("quick_test" "high_survival" "seed_comparison")

for dataset in "${datasets[@]}"; do
    echo "🚀 실험 시작: $dataset"
    python3 train.py --dataset $dataset --epochs 50 --output "models/${dataset}_model.pth"
    echo "✅ 완료: $dataset"
done
```

## 💡 **팁과 요령**

1. **빠른 테스트**: 작은 샘플 수로 먼저 테스트
2. **의미있는 이름**: 데이터셋 이름에 실험 목적 포함
3. **단계별 접근**: 작은 실험부터 시작해서 점진적으로 확장
4. **시드 관리**: 재현 가능한 실험을 위해 시드 값 기록
5. **백업**: 중요한 실험 데이터는 별도 백업

## ❓ **문제 해결**

### **파일을 찾을 수 없습니다**
```
❌ 파일을 찾을 수 없습니다: database-12345_100_0.300000.txt
```
**해결책**: 
1. 파일명이 정확한지 확인
2. 데이터가 생성되었는지 확인: `ls train_data/`
3. 경로 설정 확인

### **샘플 수가 맞지 않습니다**
```
📊 예상 샘플: 1000개, 실제: 800개
```
**해결책**:
1. `expected_samples` 값을 실제 파일의 샘플 수로 수정
2. 데이터 생성 시 샘플 수 확인

### **메모리 부족**
**해결책**:
1. `batch_size` 줄이기: `--batch-size 16`
2. `num_workers` 줄이기: 설정 파일에서 수정
3. 더 작은 데이터셋으로 시작

## 🧠 **모델 설정 가이드**

데이터셋뿐만 아니라 모델 구조도 JSON으로 관리할 수 있습니다!

### **🎯 모델 설정 사용법**

```bash
# 사용 가능한 모델 확인
python3 train.py --list-models

# 특정 모델로 훈련
python3 train.py --dataset custom_experiment_1 --model-config small_cnn
python3 train.py --dataset custom_high_density --model-config large_cnn
```

### **📊 기본 제공 모델들**

| 모델 | 크기 | 활성화 | 설명 | 추천 용도 |
|------|------|--------|------|-----------|
| `small_cnn` | 16→32 | ReLU | 빠른 테스트용 | 프로토타이핑 |
| `standard_cnn` | 32→64 | Swish | 균형잡힌 성능 | 일반적인 실험 |
| `large_cnn` | 64→128 | Swish | 높은 정확도 | 정밀한 분석 |
| `experimental_relu` | 48→96 | ReLU | ReLU + bias 실험 | 활성화 함수 연구 |
| `lightweight` | 8→16 | ReLU | 극경량 모델 | 빠른 반복 실험 |

### **🛠️ 커스텀 모델 추가**

`model_hyper.json`의 `model_configs`에 새 모델을 추가하세요:

```json
"my_custom_model": {
  "name": "My Custom CNN",
  "description": "나만의 실험용 모델",
  "structure": {
    "hidden1": 24,
    "hidden2": 48,
    "activation": "swish",
    "use_bias": true,
    "stride": 1
  },
  "training": {
    "recommended_epochs": 40,
    "recommended_lr": 0.0015,
    "recommended_batch_size": 24
  }
}
```

### **🧪 실험 설정 (NEW!)**

이제 데이터셋과 모델을 함께 묶은 실험 설정을 사용할 수 있습니다!

```bash
# 사용 가능한 실험 확인
python3 train.py --list-experiments

# 실험 실행 (모든 설정이 자동으로!)
python3 train.py --experiment quick_test
python3 train.py --experiment high_accuracy_experiment
```

### **🔧 커스텀 실험 추가**

`dataset_config.json`의 `custom_experiments`에 새 실험을 추가하세요:

```json
"my_experiment": {
  "name": "My Complete Experiment",
  "description": "데이터셋과 모델을 함께 설정한 실험",
  "dataset_key": "custom_small", 
  "model_key": "standard_cnn",
  "training_overrides": {
    "epochs": 50,
    "lr": 0.002,
    "batch_size": 32
  },
  "notes": "특별한 용도의 실험 설정"
}
```

### **🚀 실전 조합 예시**

```bash
# === 새로운 방법: 실험 설정 사용 (권장!) ===
python3 train.py --experiment quick_test           # 빠른 테스트
python3 train.py --experiment high_accuracy_experiment  # 고정밀도 분석
python3 train.py --experiment my_custom_setup      # 내 맞춤 설정

# === 기존 방법: 개별 설정 ===
python3 train.py --dataset custom_experiment_1 --model-config small_cnn --epochs 10
python3 train.py --dataset custom_high_density --model-config large_cnn --epochs 50
python3 train.py --dataset full_simulation --model-config lightweight --epochs 30
python3 train.py --dataset custom_mid_range --model-config my_custom_model
```

### **💡 모델 선택 가이드**

- **빠른 테스트**: `small_cnn` + `custom_experiment_1`
- **균형잡힌 실험**: `standard_cnn` + `custom_mid_range`
- **고정밀 분석**: `large_cnn` + `custom_high_density`
- **대용량 처리**: `lightweight` + `full_simulation`
- **ReLU 연구**: `experimental_relu` + 모든 데이터셋

이제 나만의 실험을 시작해보세요! 🎉