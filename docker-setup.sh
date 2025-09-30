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
    
    if ! docker compose version &> /dev/null && ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose가 설치되지 않았습니다. v2를 설치하거나 v1을 사용하세요."
        exit 1
    fi
    
    log_info "Docker 환경 확인 완료"
}

# Docker 권한 확인
check_docker_permissions() {
    if ! docker info &> /dev/null; then
        log_warn "Docker 권한이 없습니다."
        echo
        echo "해결 방법:"
        echo "1) 임시로 sudo 사용: sudo $0"
        echo "2) 사용자를 docker 그룹에 추가:"
        echo "   sudo usermod -aG docker \$USER"
        echo "   newgrp docker  # 또는 로그아웃 후 재로그인"
        echo
        read -p "sudo로 계속 실행하시겠습니까? (y/N): " choice
        case $choice in
            [yY]|[yY][eE][sS])
                log_info "sudo 권한으로 재실행합니다..."
                exec sudo "$0" "$@"
                ;;
            *)
                log_error "Docker 권한 설정 후 다시 실행해주세요."
                exit 1
                ;;
        esac
    fi
    
    log_info "Docker 권한 확인 완료"
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
    echo "🎯 GOL 실험 환경을 선택하세요:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1) 🚀 GPU 지원 (CUDA + PyTorch GPU)"
    echo "   - RTX/GTX 시리즈 그래픽카드 보유자"
    echo "   - 빠른 데이터 생성 및 AI 훈련"
    echo ""
    echo "2) 💻 CPU 전용 (PyTorch CPU)"
    echo "   - 그래픽카드 없음 또는 호환성 문제"
    echo "   - 메모리 최적화된 CPU 데이터 생성"
    echo ""
    echo "3) 🔍 자동 선택 (GPU 지원 확인 후 결정)"
    echo "   - 시스템 환경을 자동으로 감지"
    echo ""
    echo "4) 🛠️  모든 버전 빌드 (개발자용)"
    echo "   - GPU/CPU 버전 모두 빌드"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    read -p "선택 (1-4): " choice
    
    case $choice in
        1) 
            BUILD_TYPE="gpu"
            log_info "🎮 GPU 지원 모드 선택됨"
            ;;
        2) 
            BUILD_TYPE="cpu"
            log_info "💻 CPU 전용 모드 선택됨"
            ;;
        3) 
            BUILD_TYPE="auto"
            log_info "🔍 자동 감지 모드 선택됨"
            ;;
        4) 
            BUILD_TYPE="all"
            log_info "🛠️  전체 빌드 모드 선택됨"
            ;;
        *) 
            log_error "잘못된 선택입니다. (1-4 중 선택)"
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

# 컨테이너 실행 선택
ask_run_container() {
    echo
    echo "🚀 빌드가 완료되었습니다!"
    echo "바로 컨테이너를 실행하시겠습니까?"
    echo
    echo "1) 예 - 바로 실행하고 접속"
    echo "2) 아니오 - 사용법만 보여주기"
    
    read -p "선택 (1-2): " run_choice
    
    case $run_choice in
        1)
            run_container
            ;;
        2)
            show_usage
            ;;
        *)
            log_warn "잘못된 선택, 사용법을 출력합니다."
            show_usage
            ;;
    esac
}

# 컨테이너 실행
run_container() {
    local service=""
    
    # 빌드 타입에 따라 서비스 결정
    case $BUILD_TYPE in
        "gpu")
            service="golexp-gpu"
            ;;
        "cpu")
            service="golexp-cpu"
            ;;
        "auto")
            if check_nvidia_docker; then
                service="golexp-gpu"
            else
                service="golexp-cpu"
            fi
            ;;
        "all")
            echo "여러 버전이 빌드되었습니다. 실행할 버전을 선택하세요:"
            echo "1) GPU 버전"
            echo "2) CPU 버전"
            read -p "선택 (1-2): " version_choice
            case $version_choice in
                1) service="golexp-gpu" ;;
                2) service="golexp-cpu" ;;
                *) service="golexp-cpu" ;;
            esac
            ;;
    esac
    
    log_info "🐳 $service 컨테이너 시작 중..."
    
    # Docker Compose v2를 우선 사용, v1은 fallback
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        log_warn "Docker Compose v1이 감지되었습니다. v2 업그레이드를 권장합니다."
    else
        log_error "Docker Compose가 설치되지 않았습니다."
        exit 1
    fi
    
    # 컨테이너 시작
    if $COMPOSE_CMD up -d $service; then
        log_info "✅ 컨테이너가 성공적으로 시작되었습니다!"
        echo
        echo "🔗 컨테이너에 접속하려면:"
        echo "   $COMPOSE_CMD exec $service bash"
        echo
        echo "📁 프로젝트 디렉토리로 이동:"
        echo "   cd new_project"
        echo
        echo "🎯 데이터 생성 테스트:"
        if [[ $service == *"gpu"* ]]; then
            echo "   python3 datagen.py 12345 100 0.3 --verbose"
        else
            echo "   python3 datagen.py 12345 100 0.3 --cpu --verbose"
        fi
        echo
        echo "🛑 컨테이너 중지:"
        echo "   $COMPOSE_CMD down"
        echo
        read -p "지금 컨테이너에 접속하시겠습니까? (y/N): " connect_choice
        case $connect_choice in
            [yY]|[yY][eE][sS])
                log_info "🚪 컨테이너에 접속합니다..."
                $COMPOSE_CMD exec $service bash
                ;;
            *)
                log_info "나중에 위의 명령어로 접속하실 수 있습니다."
                ;;
        esac
    else
        log_error "컨테이너 시작에 실패했습니다."
        show_usage
    fi
}

# 사용법 출력
show_usage() {
    echo
    log_info "📖 빌드된 컨테이너 사용법:"
    echo
    
    # Docker Compose v2를 우선 사용, v1은 fallback
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        log_warn "Docker Compose v1이 감지되었습니다. v2 업그레이드를 권장합니다."
    else
        log_error "Docker Compose가 설치되지 않았습니다."
        exit 1
    fi
    
    if [[ $BUILD_TYPE == "gpu" ]] || [[ $BUILD_TYPE == "all" ]] || ([[ $BUILD_TYPE == "auto" ]] && check_nvidia_docker); then
        echo "🎮 GPU 버전 실행:"
        echo "  $COMPOSE_CMD up -d golexp-gpu"
        echo "  $COMPOSE_CMD exec golexp-gpu bash"
        echo "  # 데이터 생성: python3 datagen.py 12345 1000 0.3"
        echo
    fi
    
    if [[ $BUILD_TYPE == "cpu" ]] || [[ $BUILD_TYPE == "all" ]] || ([[ $BUILD_TYPE == "auto" ]] && ! check_nvidia_docker); then
        echo "💻 CPU 버전 실행:"
        echo "  $COMPOSE_CMD up -d golexp-cpu"
        echo "  $COMPOSE_CMD exec golexp-cpu bash"
        echo "  # 데이터 생성: python3 datagen.py 12345 1000 0.3 --cpu"
        echo
    fi
    
    echo "🔧 개발 모드:"
    echo "  $COMPOSE_CMD up -d golexp-dev"
    echo "  $COMPOSE_CMD exec golexp-dev bash"
    echo
    echo "🛑 컨테이너 중지:"
    echo "  $COMPOSE_CMD down"
    echo
    echo "📖 더 자세한 사용법은 DOCKER.md 파일을 참조하세요."
}

# 메인 실행
main() {
    log_info "🐳 GOL 실험 프로젝트 Docker 환경 구축 시작"
    
    check_docker
    check_docker_permissions
    select_build_type
    
    echo
    log_info "📦 선택된 빌드 타입: $BUILD_TYPE"
    
    # 빌드 시작
    build_main
    
    # 컨테이너 실행 선택
    ask_run_container
    
    log_info "🎉 Docker 환경 구축 완료!"
    echo "💡 팁: 언제든지 './docker-setup.sh'로 다시 실행할 수 있습니다."
}

# 스크립트 실행
main "$@"