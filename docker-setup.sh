#!/bin/bash

# 🐳 GOL 실험 프로젝트 Docker 빌드 스크립트

set -e  # 에러 시 중단

echo "🚀 GOL 실험 프로젝트 Docker 환경 구축"

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 함수 정의
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Docker 설치 확인
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되지 않았습니다."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose가 설치되지 않았습니다."
        exit 1
    fi
    
    log_info "Docker 환경 확인 완료"
}

# NVIDIA Docker 확인
check_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        log_info "NVIDIA Docker 지원 확인됨"
        return 0
    else
        log_warn "NVIDIA Docker를 사용할 수 없습니다. CPU 모드로 실행됩니다."
        return 1
    fi
}

# 빌드 타입 선택
select_build_type() {
    echo
    echo "빌드 타입을 선택하세요:"
    echo "1) GPU 지원 (CUDA + PyTorch GPU)"
    echo "2) CPU 전용 (PyTorch CPU)"
    echo "3) 자동 선택 (GPU 지원 확인 후 결정)"
    echo "4) 모든 버전 빌드"
    
    read -p "선택 (1-4): " choice
    
    case $choice in
        1) BUILD_TYPE="gpu" ;;
        2) BUILD_TYPE="cpu" ;;
        3) BUILD_TYPE="auto" ;;
        4) BUILD_TYPE="all" ;;
        *) 
            log_error "잘못된 선택입니다."
            exit 1
            ;;
    esac
}

# Docker 이미지 빌드
build_image() {
    local type=$1
    local dockerfile=$2
    
    log_info "${type} 버전 빌드 시작..."
    
    if docker build -f "$dockerfile" -t "golexp:$type" .; then
        log_info "${type} 빌드 완료!"
    else
        log_error "${type} 빌드 실패!"
        exit 1
    fi
}

# 메인 빌드 로직
build_main() {
    case $BUILD_TYPE in
        "gpu")
            build_image "gpu" "Dockerfile.gpu"
            ;;
        "cpu")
            build_image "cpu" "Dockerfile.cpu"
            ;;
        "auto")
            if check_nvidia_docker; then
                build_image "gpu" "Dockerfile.gpu"
            else
                build_image "cpu" "Dockerfile.cpu"
            fi
            ;;
        "all")
            build_image "gpu" "Dockerfile.gpu"
            build_image "cpu" "Dockerfile.cpu"
            ;;
    esac
}

# 사용법 출력
show_usage() {
    echo
    log_info "빌드된 컨테이너 사용법:"
    echo
    
    if [[ $BUILD_TYPE == "gpu" ]] || [[ $BUILD_TYPE == "all" ]] || ([[ $BUILD_TYPE == "auto" ]] && check_nvidia_docker); then
        echo "🎮 GPU 버전 실행:"
        echo "  docker-compose up -d golexp-gpu"
        echo "  docker-compose exec golexp-gpu bash"
        echo
    fi
    
    if [[ $BUILD_TYPE == "cpu" ]] || [[ $BUILD_TYPE == "all" ]] || ([[ $BUILD_TYPE == "auto" ]] && ! check_nvidia_docker); then
        echo "💻 CPU 버전 실행:"
        echo "  docker-compose up -d golexp-cpu"
        echo "  docker-compose exec golexp-cpu bash"
        echo
    fi
    
    echo "🔧 개발 모드:"
    echo "  docker-compose up -d golexp-dev"
    echo "  docker-compose exec golexp-dev bash"
    echo
    echo "📖 더 자세한 사용법은 DOCKER.md 파일을 참조하세요."
}

# 메인 실행
main() {
    log_info "Docker 환경 구축 시작"
    
    check_docker
    select_build_type
    
    echo
    log_info "선택된 빌드 타입: $BUILD_TYPE"
    
    # 빌드 시작
    build_main
    
    # 사용법 출력
    show_usage
    
    log_info "🎉 Docker 환경 구축 완료!"
}

# 스크립트 실행
main "$@"