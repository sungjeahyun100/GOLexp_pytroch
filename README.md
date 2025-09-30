# GOLexp_pytorch

Game of Life 패턴을 PyTorch로 학습하는 복잡계 AI 실험 프로젝트

## 🐳 Docker로 빠른 시작 (권장)

**가장 쉬운 방법**: Docker를 사용하면 환경 설정 없이 바로 실행 가능!
- **CPU 컨테이너**: 모든 환경에서 동작 (GOLdatagen_cpu.so)
- **GPU 컨테이너**: NVIDIA GPU + Docker Runtime 필요 (GOLdatagen_gpu.so)

```bash
# 1. 프로젝트 클론
git clone https://github.com/sungjeahyun100/GOLexp_pytroch.git
cd GOLexp_pytroch

# 2. Docker 환경 자동 구축
./docker-setup.sh          # Linux/macOS
# 또는
.\docker-setup.ps1          # Windows (PowerShell)

# 3. 컨테이너 실행 (Docker Compose v2)
docker compose up -d golexp-gpu  # GPU가 있는 경우
# 또는
docker compose up -d golexp-cpu  # GPU가 없는 경우

# 4. 컨테이너 접속 및 훈련 시작
docker exec -it golexp-gpu bash  # GPU 컨테이너
# 또는  
docker exec -it golexp-cpu bash  # CPU 컨테이너

# 5. 바로 훈련 시작! (데이터가 없는 경우 먼저 생성)
./genData.sh              # 99개 데이터셋 자동 생성 (컨테이너 내부)
cd new_project
python3 train.py --dataset small_simulation --epochs 50
```

📖 **자세한 Docker 사용법**: [DOCKER.md](DOCKER.md) 참조

---

## 📋 수동 설치 (필요한 것)

### 필수 요구사항
- **cmake** (3.18+)
- **make** 
- **python3.12+**
- **C++ 컴파일러** (GCC 12+ 권장)

### 선택 사항
- **CUDA Toolkit 12.1+** (GPU 가속용, GOLdatagen_gpu.so)
- **NVIDIA GPU** (RTX 시리즈 권장)

### Python 패키지
- **PyTorch** (torch)
- **numpy**
- **pygame** (시각화용)

## 🚀 필수 설정 스크립트

```bash
# 1. 저장소 복제
git clone https://github.com/sungjeahyun100/GOLexp_pytroch.git
cd GOLexp_pytroch

# 2. Python 가상환경 설정
python3 -m venv myexp
source ./myexp/bin/activate    # Linux/macOS
# 또는
.\myexp\Scripts\Activate.ps1   # Windows PowerShell

# 3. PyTorch 및 필수 라이브러리 설치
pip3 install torch pygame numpy

# 4. C++ 빌드 환경 설정
mkdir build
cd build
cmake ..              # CPU 전용 빌드
# 또는
cmake .. -DCUDA_ENABLED=ON  # GPU 지원 빌드 (CUDA 필요)

# Linux/macOS
make -j$(nproc)
# Windows
cmake --build . --config Release -j

cd ..

# 5. 데이터 생성
mkdir train_data
./genData.sh          # Linux/macOS
.\genData.ps1         # Windows (PowerShell)
# 또는 Python으로 직접 실행:
# cd new_project && python datagen.py 54321 1000 0.3 --cpu

# 6. 모델 훈련 시작
cd new_project
python3 train.py --dataset small_simulation --epochs 50
```

## 📊 데이터셋 구조

- **생존 비율**: 0.01 ~ 0.99 (99개 파일)
- **각 파일**: 1000개 샘플 (기본값)
- **총 샘플**: 99,000개 (전체 데이터셋)
- **입력**: 10×10 이진 그리드 (패턴)
- **시뮬레이션**: 100×100 보드에서 최대 2500세대 진화
- **출력**: 최종 생존 셀 수 (레이블)

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

**GPU 모드** (기본값, CUDA 가속):
```bash
cd new_project
python3 datagen.py 12345 1000 0.3 --verbose
# GOLdatagen_gpu.so 라이브러리 사용
```

**CPU 모드** (GPU 없는 환경, 메모리 최적화):
```bash
cd new_project  
python3 datagen.py 12345 1000 0.3 --cpu --verbose
# GOLdatagen_cpu.so 라이브러리 사용
```

**단일 파일 모드** (대용량 데이터를 하나의 파일로):
```bash
# GPU 단일파일
python3 datagen.py 54321 10000 0.25 --one_file --verbose
# CPU 단일파일  
python3 datagen.py 54321 10000 0.25 --one_file --cpu --verbose
```

**자동 데이터셋 생성** (99개 파일):
```bash
./genData.sh  # GPU/CPU 자동 선택, 0.01~0.99 비율
```

**파라미터 설명**:
- `시드`: 재현 가능한 랜덤 생성을 위한 시드값 (uint32)
- `데이터량`: 생성할 샘플 수 (uint32)
- `생존비율`: 10×10 패턴 셀 중 살아있는 셀의 비율 (0.0-1.0)
- `--cpu`: CPU 전용 라이브러리 사용 (GOLdatagen_cpu.so, 메모리 최적화)
- `--one_file`: 모든 데이터를 단일 텍스트 파일에 저장
- `--verbose`: 상세한 진행 상황 및 성능 정보 출력

**라이브러리 자동 선택**:
- **GPU 모드**: `GOLdatagen_gpu.so` (CUDA 가속, ~7.8MB)
- **CPU 모드**: `GOLdatagen_cpu.so` (벡터 최적화, ~839KB)

## 🧠 모델 아키텍처

- **CNN 레이어**: 3개 (32→64→128 채널)
- **FC 레이어**: 5개 (1024→512→256→128→10)
- **활성화 함수**: Swish/SiLU
- **정규화**: BatchNorm (affine=False)
- **출력**: 10bit 이진 분류

## 📈 성능 지표

### 데이터 생성 성능 (1000 샘플, 100×100 보드)
- **GPU 모드**: ~3-9초 (생존비율에 따라 변동)
- **CPU 모드**: ~5-9초 (메모리 최적화로 GPU와 유사한 성능)
- **메모리 사용량**: CPU 78KB, GPU ~7.8MB (라이브러리 포함)

### 모델 훈련 성능
- **데이터셋별 예상 Loss**:
  - `small_simulation` (20K): ~0.120
  - `full_simulation` (99K): ~0.090
  - 이론적 하한: ~0.008
- **훈련 시간**: RTX 4060 기준 ~10분/에폭 (full dataset)
- **메모리 사용량**: ~4GB VRAM (batch_size=512)

## 📁 프로젝트 구조

```
GOLexp_pytorch/
├── build/                      # C++ 빌드 결과
│   ├── GOLdatagen_cpu.so       # CPU 전용 라이브러리 (839KB)
│   ├── GOLdatagen_gpu.so       # GPU 가속 라이브러리 (7.8MB)
│   └── libexp_GOLdatagen_dependency.a  # CUDA 의존성
├── CUDAcode/                   # C++/CUDA 소스 코드
│   ├── GOLdatagen.cpp          # CPU 래퍼 함수
│   ├── GOLdatagen.cu           # GPU 래퍼 함수  
│   ├── GOLdatabase_host.cpp    # CPU 최적화 구현
│   ├── GOLdatabase_2.cu        # GPU CUDA 구현
│   └── d_matrix_2.cu           # GPU 메모리 관리
├── train_data/                 # 훈련 데이터 (다양한 생존비율)
├── new_project/                # Python 모델 프로젝트
│   ├── src/                    # 모듈형 소스 코드
│   │   ├── data_loader.py      # 데이터 로딩 및 관리
│   │   └── model.py            # CNN 모델 정의
│   ├── train.py                # 메인 훈련 스크립트
│   ├── interface.py            # 시각화 및 테스트 인터페이스
│   ├── datagen.py              # 데이터 생성 (CPU/GPU 자동 선택)
│   ├── dataset_config.json     # 데이터셋 구성 파일
│   └── saved_models/           # 훈련된 모델 저장소
├── docker-setup.sh             # Docker 환경 자동 구축
├── genData.sh                  # 데이터 생성 스크립트 (GPU/CPU 자동)
├── docker-compose.yml          # Docker 컨테이너 설정
├── Dockerfile.cpu              # CPU 전용 컨테이너
├── Dockerfile.gpu              # GPU 지원 컨테이너
└── CMakeLists.txt              # C++ 빌드 설정 (CPU/GPU 분리)
```




