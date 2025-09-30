# 🐳 Docker 사용 가이드

Game of Life 실험 프로젝트를 Docker로 실행하는 방법입니다.

## 📋 사전 요구사항

### 필수 설치
- Docker Engine (20.10.0+)
- Docker Compose (2.0.0+)

### GPU 지원 (선택사항)
- NVIDIA Docker (nvidia-docker2)
- NVIDIA 드라이버 (470.57.02+)
- CUDA 호환 GPU

## 🚀 빠른 시작

### 1. GPU 지원 환경에서 실행
```bash
# 프로젝트 클론
git clone <repository-url>
cd GOLexp_pytroch

# GPU 컨테이너 실행
docker compose up -d golexp-gpu

# 컨테이너에 접속
docker compose exec golexp-gpu bash

# 훈련 실행
cd new_project
python3 train.py --dataset small_simulation --epochs 50
```

### 2. CPU 전용 환경에서 실행
```bash
# CPU 컨테이너 실행
docker compose up -d golexp-cpu

# 컨테이너에 접속
docker compose exec golexp-cpu bash

# 훈련 실행
cd new_project
python3 train.py --dataset small_simulation --epochs 50
```

### 3. 개발 환경에서 실행
```bash
# 개발용 컨테이너 실행 (실시간 코드 변경 반영)
docker compose up -d golexp-dev

# 컨테이너에 접속
docker compose exec golexp-dev bash
```

## 🔧 상세 사용법

### Dockerfile 종류별 빌드

#### GPU 버전 빌드
```bash
docker build -f Dockerfile.gpu -t golexp:gpu .
```

#### CPU 버전 빌드
```bash
docker build -f Dockerfile.cpu -t golexp:cpu .
```

#### 기본 멀티스테이지 빌드
```bash
docker build -t golexp:latest .
```

### 직접 컨테이너 실행

#### GPU 지원 실행
```bash
docker run --gpus all -it \
  -v $(pwd):/app \
  -v $(pwd)/train_data:/app/train_data \
  golexp:gpu bash
```

#### CPU 전용 실행
```bash
docker run -it \
  -v $(pwd):/app \
  -v $(pwd)/train_data:/app/train_data \
  golexp:cpu bash
```

## 📊 데이터 생성 및 훈련

### 컨테이너 내에서 데이터 생성
```bash
# 데이터 생성 (CUDA 필요)
cd /app/new_project
python3 datagen.py 54321 1000 0.3

# 또는 스크립트 사용
cd /app
./genData.sh
```

### 모델 훈련
```bash
cd /app/new_project

# 작은 데이터셋으로 빠른 테스트
python3 train.py --dataset small_simulation --epochs 10

# 전체 데이터셋으로 훈련
python3 train.py --dataset full_simulation --epochs 100

# 직접 파일 지정
python3 train.py --files ../train_data/*.txt --epochs 50
```

## 🐛 트러블슈팅

### GPU 관련 문제

#### NVIDIA Docker 설치 확인
```bash
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

#### GPU 접근 권한 문제
```bash
# Docker 그룹에 사용자 추가
sudo usermod -aG docker $USER
newgrp docker
```

### 빌드 문제

#### 캐시 없이 재빌드
```bash
docker compose build --no-cache golexp-gpu
```

#### 이미지 완전 재빌드
```bash
docker system prune -a
docker compose build
```

### 권한 문제

#### 볼륨 권한 설정
```bash
# 컨테이너 내에서 권한 수정
docker compose exec golexp-gpu chown -R $(id -u):$(id -g) /app
```

## 📁 볼륨 마운트

Docker Compose는 다음 디렉토리들을 자동으로 마운트합니다:

- `./` → `/app/` (전체 프로젝트)
- `./train_data/` → `/app/train_data/` (훈련 데이터)
- `./new_project/saved_models/` → `/app/new_project/saved_models/` (저장된 모델)

## 🏷️ 컨테이너 관리

### 컨테이너 상태 확인
```bash
docker compose ps
```

### 로그 확인
```bash
docker compose logs golexp-gpu
```

### 컨테이너 정지
```bash
docker compose down
```

### 이미지 제거
```bash
docker compose down --rmi all
```

## ⚡ 성능 최적화

### 빌드 최적화
- 빌드 컨텍스트 크기 줄이기 (.dockerignore 사용)
- 멀티스테이지 빌드로 최종 이미지 크기 줄이기
- 레이어 캐싱 최대 활용

### 런타임 최적화
- 공유 메모리 크기 증가: `--shm-size=1g`
- CPU 코어 수 제한: `--cpus="4"`
- 메모리 제한: `--memory="8g"`

## 🔗 추가 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [NVIDIA Docker 설정](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose 문서](https://docs.docker.com/compose/)

## 💡 팁

1. **개발 시**: `golexp-dev` 서비스 사용으로 코드 변경사항 즉시 반영
2. **배포 시**: `golexp-gpu` 또는 `golexp-cpu` 서비스 사용
3. **CI/CD**: GitHub Actions에서 Docker 이미지 자동 빌드 가능
4. **클러스터**: Kubernetes로 확장 가능