# GOLexp_pytorch

Game of Life 패턴을 PyTorch로 학습하는 복잡계 AI 실험 프로젝트

## 📋 필요한 것

- cmake
- make  
- python3.12+
- CUDA (GPU 가속용)
- PyTorch, numpy, pygame

## 🚀 빠른 시작

```bash
# 1. 저장소 복제
git clone https://github.com/sungjeahyun100/GOLexp_pytroch.git
cd GOLexp_pytroch

# 2. Python 가상환경 설정
python3 -m venv myexp
source ./myexp/bin/activate

# 3. PyTorch 및 필수 라이브러리 설치
pip3 install torch pygame numpy

# 4. C++ 빌드 환경 설정
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..

# 5. 데이터 생성
mkdir train_data
./genData.sh

# 6. 모델 훈련 시작
cd new_project
python3 train.py --dataset small_simulation --epochs 50
```

## 📊 데이터셋 구조

- **생존 비율**: 0.01 ~ 0.99 (99개 파일)
- **각 파일**: 1000개 샘플
- **총 샘플**: 99,000개
- **입력**: 10x10 이진 그리드
- **출력**: 10bit 패턴 분류

## 🎮 사용법

### 모델 훈련

**JSON 구성 파일 사용** (권장):
```bash
cd new_project
python3 train.py --dataset small_simulation --epochs 50
python3 train.py --dataset full_simulation --epochs 100
```

**직접 파일 지정**:
```bash
python3 train.py --files ../train_data/database-54321_1000_0.300000.txt --epochs 30
```

**훈련 옵션**:
- `--dataset`: JSON 구성 파일에서 데이터셋 선택
- `--files`: 데이터 파일 직접 지정
- `--epochs`: 에폭 수 (기본값: 50)
- `--lr`: 학습률 (기본값: 0.0001)
- `--batch_size`: 배치 크기 (기본값: 512)

### 시각화 및 테스트
```bash
python3 interface.py
```
### 데이터셋 구성

`dataset_config.json` 파일에서 데이터셋을 관리합니다:

```json
{
  "small_simulation": {
    "files": ["../train_data/database-54321_1000_0.300000.txt"],
    "description": "Small dataset for testing (20K samples)"
  },
  "full_simulation": {
    "files": ["../train_data/database-12345_600000_0.300000.txt"],
    "description": "Full dataset for production training (99K samples)"
  }
}
```

### 새 데이터 생성
```bash
python3 datagen.py --one_file=True <시드> <데이터량> <생존비율>
```
**파라미터 설명**:
- 생존비율: 10×10 셀 중 살아있는 셀의 비율 (0.0-1.0)
- 데이터량: 생성할 샘플 수
- 시드: 재현 가능한 랜덤 생성을 위한 시드값

## 🧠 모델 아키텍처

- **CNN 레이어**: 3개 (32→64→128 채널)
- **FC 레이어**: 5개 (1024→512→256→128→10)
- **활성화 함수**: Swish/SiLU
- **정규화**: BatchNorm (affine=False)
- **출력**: 10bit 이진 분류

## 📈 성능 지표

- **데이터셋별 예상 Loss**:
  - `small_simulation` (20K): ~0.120
  - `full_simulation` (99K): ~0.090
  - 이론적 하한: ~0.008
- **훈련 시간**: RTX 4060 기준 ~10분/에폭 (full dataset)
- **메모리 사용량**: ~4GB VRAM (batch_size=512)

## 🔬 실험 목표

복잡계(Game of Life)의 패턴을 AI가 어떻게 학습하는지 관찰:
- 정보 손실 없는 순수한 학습
- 카오스 이론과 나비 효과의 영향
- 스케일링 법칙 검증

## 📁 프로젝트 구조

```
GOLexp_pytorch/
├── build/                      # C++ 빌드 결과
├── train_data/                 # 훈련 데이터 (다양한 생존비율 데이터)
├── new_project/                # Python 모델 프로젝트
│   ├── src/                    # 모듈형 소스 코드
│   │   ├── data_loader.py      # 데이터 로딩 및 관리
│   │   └── model.py            # CNN 모델 정의
│   ├── train.py                # 메인 훈련 스크립트
│   ├── interface.py            # 시각화 및 테스트 인터페이스
│   ├── datagen.py              # 데이터 생성 유틸리티
│   ├── dataset_config.json     # 데이터셋 구성 파일
│   └── saved_models/           # 훈련된 모델 저장소
├── myexp/                      # Python 가상환경
├── genData.sh                  # 데이터 생성 스크립트
└── CMakeLists.txt              # C++ 빌드 설정
```




