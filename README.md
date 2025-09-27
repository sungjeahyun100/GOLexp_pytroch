# GOLexp_pytorch

Game of Life 패턴을 PyTorch로 학습하는 복잡계 AI 실험 프로젝트

## 📋 필요한 것

- cmake
- make  
- python3.12
- CUDA (GPU 가속용, 선택사항)

## 🚀 설정 스크립트

```bash
# 1. 데이터 디렉토리 생성
mkdir train_data

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
./genData.sh

# 6. 모델 훈련 시작
cd new_project
python3 model.py
```

## 📊 데이터셋 구조

- **생존 비율**: 0.01 ~ 0.99 (99개 파일)
- **각 파일**: 1000개 샘플
- **총 샘플**: 99,000개
- **입력**: 10x10 이진 그리드
- **출력**: 10bit 패턴 분류

## 🎮 사용법

### 모델 훈련
```bash
cd new_project
python3 model.py
```

**훈련 옵션**:
1. 새 모델 훈련 (소형/전체 데이터셋 선택)
2. 기존 모델 추가 훈련

### 인터렉티브 테스트
```bash
python3 interface.py
```

## 🧠 모델 아키텍처

- **CNN 레이어**: 3개 (32→64→128 채널)
- **FC 레이어**: 5개 (1024→512→256→128→10)
- **활성화 함수**: Swish/SiLU
- **정규화**: BatchNorm (affine=False)
- **출력**: 10bit 이진 분류

## 📈 성능 지표

- **데이터 크기별 예상 Loss**:
  - 20K 샘플: ~0.120
  - 99K 샘플: ~0.090
  - 이론적 하한: ~0.008

## 🔬 실험 목표

복잡계(Game of Life)의 패턴을 AI가 어떻게 학습하는지 관찰:
- 정보 손실 없는 순수한 학습
- 카오스 이론과 나비 효과의 영향
- 스케일링 법칙 검증

## 📁 프로젝트 구조

```
GOLexp_pytorch/
├── build/                 # C++ 빌드 결과
├── train_data/            # 훈련 데이터 (99K 샘플)
├── new_project/           # Python 모델 코드
│   ├── model.py          # 메인 훈련 스크립트
│   ├── interface.py      # 인터렉티브 테스트
│   └── saved_models/     # 훈련된 모델들
├── myexp/                # Python 가상환경
├── genData.sh           # 데이터 생성 스크립트
└── CMakeLists.txt       # C++ 빌드 설정
```




