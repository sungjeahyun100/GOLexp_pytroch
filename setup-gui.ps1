# Game of Life GUI 설정 스크립트 (Windows PowerShell)
# X11 포워딩을 활성화하고 GUI 애플리케이션을 실행할 수 있도록 설정합니다.

Write-Host "🖥️  Game of Life GUI 설정 시작..." -ForegroundColor Green

# Windows 환경 확인
Write-Host "🪟 Windows 환경 감지됨" -ForegroundColor Cyan

# VcXsrv 프로세스 확인
$vcxsrvRunning = Get-Process -Name "vcxsrv" -ErrorAction SilentlyContinue

if ($vcxsrvRunning) {
    Write-Host "✅ VcXsrv가 실행 중입니다" -ForegroundColor Green
} else {
    Write-Host "❌ VcXsrv가 실행되지 않았습니다" -ForegroundColor Red
    Write-Host ""
    Write-Host "📝 VcXsrv 설치 및 설정이 필요합니다:" -ForegroundColor Yellow
    Write-Host "   1. VcXsrv 다운로드: https://sourceforge.net/projects/vcxsrv/" -ForegroundColor White
    Write-Host "   2. 설치 후 XLaunch 실행" -ForegroundColor White
    Write-Host "   3. 설정 시 다음 옵션들을 선택하세요:" -ForegroundColor White
    Write-Host "      - Multiple windows 선택" -ForegroundColor Gray
    Write-Host "      - Display number: 0" -ForegroundColor Gray
    Write-Host "      - 'Disable access control' 체크 ✅" -ForegroundColor Gray
    Write-Host "      - 'Native opengl' 체크 해제" -ForegroundColor Gray
    Write-Host "   4. VcXsrv 실행 후 이 스크립트를 다시 실행하세요" -ForegroundColor White
    Write-Host ""
    
    $response = Read-Host "VcXsrv를 설치하고 실행했습니까? (y/N)"
    if ($response -match "^[Yy]") {
        Write-Host "🔄 VcXsrv 프로세스 재확인 중..." -ForegroundColor Yellow
        Start-Sleep 2
        $vcxsrvRunning = Get-Process -Name "vcxsrv" -ErrorAction SilentlyContinue
        if (-not $vcxsrvRunning) {
            Write-Host "❌ 여전히 VcXsrv 프로세스를 찾을 수 없습니다" -ForegroundColor Red
            Write-Host "   XLaunch를 실행했는지 확인해주세요" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host "❌ VcXsrv 설정을 완료한 후 다시 실행해주세요" -ForegroundColor Red
        exit 1
    }
}

# 네트워크 IP 확인
Write-Host ""
Write-Host "🌐 네트워크 IP 확인 중..." -ForegroundColor Yellow

# WSL 환경 확인
$isWSL = $false
if ($env:WSL_DISTRO_NAME) {
    Write-Host "🔍 WSL 환경 감지됨" -ForegroundColor Cyan
    $isWSL = $true
    
    # WSL에서 Windows 호스트 IP 가져오기
    try {
        $windowsIP = (Get-Content /etc/resolv.conf | Select-String "nameserver").ToString().Split()[1]
        Write-Host "📋 자동 감지된 Windows IP: $windowsIP" -ForegroundColor Green
        $env:DISPLAY = "$windowsIP:0.0"
    } catch {
        Write-Host "⚠️  자동 IP 감지 실패, 수동 설정이 필요합니다" -ForegroundColor Yellow
        $isWSL = $false
    }
}

if (-not $isWSL) {
    # 일반 Windows 환경
    try {
        $networkConfig = Get-NetIPConfiguration | Where-Object { $_.NetAdapter.Status -eq "Up" -and $_.IPv4Address.IPAddress }
        $localIP = $networkConfig[0].IPv4Address.IPAddress
        
        Write-Host "📋 감지된 로컬 IP 주소들:" -ForegroundColor Cyan
        foreach ($config in $networkConfig) {
            $adapterName = $config.NetAdapter.Name
            $ipAddress = $config.IPv4Address.IPAddress
            Write-Host "   - $adapterName: $ipAddress" -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "💡 일반적으로 다음 중 하나를 사용합니다:" -ForegroundColor Yellow
        Write-Host "   - localhost:0.0 (로컬 테스트용)" -ForegroundColor Gray
        Write-Host "   - $localIP:0.0 (네트워크용)" -ForegroundColor Gray
        
        $userIP = Read-Host "사용할 IP를 입력하세요 (기본값: localhost)"
        if ([string]::IsNullOrEmpty($userIP)) {
            $userIP = "localhost"
        }
        
        $env:DISPLAY = "$userIP:0.0"
    } catch {
        Write-Host "⚠️  자동 IP 감지 실패" -ForegroundColor Yellow
        $env:DISPLAY = "localhost:0.0"
    }
}

Write-Host "📋 설정된 DISPLAY: $($env:DISPLAY)" -ForegroundColor Green

# Docker 실행 가능 여부 확인
Write-Host ""
Write-Host "🐳 Docker 환경 확인 중..." -ForegroundColor Yellow

try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-Host "✅ Docker가 설치되어 있습니다: $dockerVersion" -ForegroundColor Green
        
        # Docker Compose 확인
        $composeVersion = docker-compose --version 2>$null
        if (-not $composeVersion) {
            $composeVersion = docker compose version 2>$null
        }
        
        if ($composeVersion) {
            Write-Host "✅ Docker Compose 사용 가능: $composeVersion" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Docker Compose를 찾을 수 없습니다" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ Docker가 설치되지 않았거나 실행되지 않습니다" -ForegroundColor Red
        Write-Host "   Docker Desktop을 설치하고 실행해주세요" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Docker 상태를 확인할 수 없습니다" -ForegroundColor Red
    exit 1
}

# 사용법 안내
Write-Host ""
Write-Host "🚀 GUI 지원 컨테이너 실행 방법:" -ForegroundColor Green
Write-Host "   GPU: docker-compose up -d golexp-gpu" -ForegroundColor White
Write-Host "   CPU: docker-compose up -d golexp-cpu" -ForegroundColor White
Write-Host ""
Write-Host "🎮 GUI 애플리케이션 실행:" -ForegroundColor Green
Write-Host "   docker exec -it golexp-gpu python3 new_project/interface.py" -ForegroundColor White
Write-Host "   또는" -ForegroundColor Gray
Write-Host "   docker exec -it golexp-cpu python3 new_project/interface.py" -ForegroundColor White
Write-Host ""
Write-Host "🔧 문제 해결:" -ForegroundColor Yellow
Write-Host "   GUI가 작동하지 않으면 헤드리스 모드 사용:" -ForegroundColor Gray
Write-Host "   docker exec -it golexp-gpu python3 new_project/interface.py --headless" -ForegroundColor White
Write-Host ""

# 자동 실행 옵션
$autoRun = Read-Host "지금 바로 컨테이너를 실행하시겠습니까? [GPU/CPU/N]"

switch ($autoRun.ToUpper()) {
    "GPU" {
        Write-Host "🚀 GPU 컨테이너 실행 중..." -ForegroundColor Green
        docker-compose up -d golexp-gpu
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ GPU 컨테이너가 실행되었습니다" -ForegroundColor Green
            Write-Host ""
            Write-Host "🎮 GUI 실행:" -ForegroundColor Cyan
            Write-Host "   docker exec -it golexp-gpu python3 new_project/interface.py" -ForegroundColor White
        } else {
            Write-Host "❌ GPU 컨테이너 실행 실패" -ForegroundColor Red
        }
    }
    "CPU" {
        Write-Host "🚀 CPU 컨테이너 실행 중..." -ForegroundColor Green
        docker-compose up -d golexp-cpu
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ CPU 컨테이너가 실행되었습니다" -ForegroundColor Green
            Write-Host ""
            Write-Host "🎮 GUI 실행:" -ForegroundColor Cyan
            Write-Host "   docker exec -it golexp-cpu python3 new_project/interface.py" -ForegroundColor White
        } else {
            Write-Host "❌ CPU 컨테이너 실행 실패" -ForegroundColor Red
        }
    }
    default {
        Write-Host "📝 수동으로 컨테이너를 실행해주세요" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "✅ Windows GUI 설정 스크립트 완료!" -ForegroundColor Green

# DISPLAY 환경변수를 현재 세션에 유지
Write-Host ""
Write-Host "💡 현재 PowerShell 세션에서 DISPLAY 환경변수가 설정되었습니다" -ForegroundColor Cyan
Write-Host "   영구 설정을 원하면 다음 명령을 실행하세요:" -ForegroundColor Gray
Write-Host "   [Environment]::SetEnvironmentVariable('DISPLAY', '$($env:DISPLAY)', 'User')" -ForegroundColor White