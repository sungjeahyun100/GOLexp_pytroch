#!/bin/bash

# Game of Life GUI 설정 스크립트
# X11 포워딩을 활성화하고 GUI 애플리케이션을 실행할 수 있도록 설정합니다.

echo "🖥️  Game of Life GUI 설정 시작..."

# 운영체제 감지
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "🐧 Linux 환경 감지"
    
    # X11 포워딩 허용
    echo "🔓 X11 포워딩 허용 중..."
    xhost +local:docker
    
    if [ $? -eq 0 ]; then
        echo "✅ X11 포워딩 설정 완료"
        echo "📋 DISPLAY 변수: $DISPLAY"
    else
        echo "⚠️  xhost 명령 실패 - GUI가 작동하지 않을 수 있습니다"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "🍎 macOS 환경 감지"
    echo "📝 XQuartz 설정이 필요합니다:"
    echo "   1. XQuartz 설치: brew install --cask xquartz"
    echo "   2. XQuartz 실행 후 환경설정 > 보안 > '네트워크 클라이언트에서 연결 허용' 체크"
    echo "   3. XQuartz 재시작"
    echo "   4. 이 스크립트 다시 실행"
    
    # XQuartz 실행 여부 확인
    if pgrep -x "XQuartz" > /dev/null; then
        echo "✅ XQuartz가 실행 중입니다"
        export DISPLAY=host.docker.internal:0
        echo "📋 DISPLAY 설정: $DISPLAY"
    else
        echo "❌ XQuartz가 실행되지 않았습니다"
        echo "   XQuartz를 실행한 후 다시 시도하세요"
        exit 1
    fi
    
else
    # Windows (Git Bash/WSL)
    echo "🪟 Windows 환경 감지"
    echo ""
    echo "� Windows 사용자는 PowerShell 스크립트를 사용하세요:"
    echo "   .\setup-gui.ps1"
    echo ""
    echo "�📝 또는 수동으로 VcXsrv 설정:"
    echo "   1. VcXsrv 다운로드: https://sourceforge.net/projects/vcxsrv/"
    echo "   2. XLaunch 실행 시 'Disable access control' 체크"
    echo "   3. Windows IP 확인: ipconfig"
    echo "   4. DISPLAY 환경변수 설정: export DISPLAY=윈도우IP:0.0"
    
    # WSL 환경인지 확인
    if grep -q Microsoft /proc/version 2>/dev/null; then
        echo "🔍 WSL 환경 감지됨 - PowerShell 스크립트를 권장합니다"
        # WSL2의 경우 Windows IP 자동 감지 시도
        WINDOWS_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' 2>/dev/null)
        if [ ! -z "$WINDOWS_IP" ]; then
            export DISPLAY=$WINDOWS_IP:0.0
            echo "📋 임시 설정된 DISPLAY: $DISPLAY"
            echo "   더 나은 설정을 위해 PowerShell 스크립트를 사용하세요"
        fi
    fi
fi

echo ""
echo "🚀 GUI 지원 컨테이너 실행 방법:"
echo "   GPU: docker-compose up -d golexp-gpu"
echo "   CPU: docker-compose up -d golexp-cpu"
echo ""
echo "🎮 GUI 애플리케이션 실행:"
echo "   docker exec -it golexp-gpu python3 new_project/interface.py"
echo "   또는"
echo "   docker exec -it golexp-cpu python3 new_project/interface.py"
echo ""
echo "🔧 문제 해결:"
echo "   GUI가 작동하지 않으면 헤드리스 모드 사용:"
echo "   docker exec -it golexp-gpu python3 new_project/interface.py --headless"

# X11 포워딩 테스트 (Linux만)
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v xeyes >/dev/null 2>&1; then
    echo ""
    read -p "🧪 X11 포워딩 테스트를 하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "👁️  xeyes 테스트 실행 중..."
        timeout 3s xeyes 2>/dev/null || echo "⚠️  xeyes 테스트 실패 - GUI 설정을 확인하세요"
    fi
fi

echo ""
echo "✅ GUI 설정 스크립트 완료!"