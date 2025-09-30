#!/usr/bin/env pwsh
# PowerShell Docker Setup Script for GOLexp Project
# Windows version of docker-setup.sh

# Colors for output (PowerShell equivalents)
$RED = "`e[31m"
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$MAGENTA = "`e[35m"
$CYAN = "`e[36m"
$NC = "`e[0m"  # No Color

# Function to print colored output
function Write-ColorOutput {
    param(
        [string]$Text,
        [string]$Color = $NC
    )
    Write-Host "$Color$Text$NC"
}

function Show-Header {
    Clear-Host
    Write-ColorOutput "==========================================" $CYAN
    Write-ColorOutput "   GOL Experiment Docker Setup (Windows)" $CYAN
    Write-ColorOutput "==========================================" $CYAN
    Write-Host ""
}

function Test-DockerCompose {
    Write-ColorOutput "Docker Compose 버전 확인 중..." $BLUE
    
    # Try Docker Compose v2 first
    try {
        $v2Output = & docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ Docker Compose v2 감지됨" $GREEN
            Write-ColorOutput "  $v2Output" $GREEN
            return "v2"
        }
    }
    catch {
        # v2 not available, try v1
    }
    
    # Try Docker Compose v1 as fallback
    try {
        $v1Output = & docker-compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ Docker Compose v1 감지됨 (v2 권장)" $YELLOW
            Write-ColorOutput "  $v1Output" $YELLOW
            Write-ColorOutput "  경고: v1은 지원이 중단되었습니다. v2로 업그레이드를 권장합니다." $YELLOW
            return "v1"
        }
    }
    catch {
        Write-ColorOutput "✗ Docker Compose를 찾을 수 없습니다!" $RED
        Write-ColorOutput "Docker Desktop을 설치하고 다시 시도해주세요." $RED
        exit 1
    }
}

function Test-Docker {
    Write-ColorOutput "Docker 설치 확인 중..." $BLUE
    try {
        & docker --version | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ Docker가 설치되어 있습니다." $GREEN
        } else {
            Write-ColorOutput "✗ Docker를 찾을 수 없습니다!" $RED
            Write-ColorOutput "Docker Desktop을 설치하고 다시 시도해주세요." $RED
            exit 1
        }
    }
    catch {
        Write-ColorOutput "✗ Docker를 찾을 수 없습니다!" $RED
        Write-ColorOutput "Docker Desktop을 설치하고 다시 시도해주세요." $RED
        exit 1
    }
}

function Test-NvidiaDocker {
    Write-ColorOutput "NVIDIA Container Toolkit 확인 중..." $BLUE
    try {
        & docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ NVIDIA GPU 지원이 가능합니다." $GREEN
            return $true
        } else {
            Write-ColorOutput "! NVIDIA GPU 지원이 불가능합니다. CPU 모드를 사용하세요." $YELLOW
            return $false
        }
    }
    catch {
        Write-ColorOutput "! NVIDIA GPU 지원이 불가능합니다. CPU 모드를 사용하세요." $YELLOW
        return $false
    }
}

function Show-Menu {
    param([bool]$hasNvidia)
    
    Write-Host ""
    Write-ColorOutput "사용 가능한 서비스:" $MAGENTA
    
    if ($hasNvidia) {
        Write-ColorOutput "1) GPU 버전 (NVIDIA GPU 필요)" $GREEN
        Write-ColorOutput "2) CPU 버전" $BLUE
        Write-ColorOutput "3) 개발 환경 (GPU + X11 포워딩)" $CYAN
        Write-ColorOutput "4) 모든 서비스 빌드" $YELLOW
    } else {
        Write-ColorOutput "1) GPU 버전 (사용 불가 - NVIDIA GPU 없음)" $RED
        Write-ColorOutput "2) CPU 버전" $BLUE
        Write-ColorOutput "3) 개발 환경 (사용 불가 - NVIDIA GPU 없음)" $RED
        Write-ColorOutput "4) 모든 서비스 빌드" $YELLOW
    }
    
    Write-ColorOutput "5) 서비스 상태 확인" $MAGENTA
    Write-ColorOutput "6) 서비스 중지" $RED
    Write-ColorOutput "7) 컨테이너 및 이미지 정리" $RED
    Write-ColorOutput "0) 종료" $YELLOW
    Write-Host ""
}

function Build-Service {
    param(
        [string]$service,
        [string]$composeVersion
    )
    
    $composeCmd = if ($composeVersion -eq "v2") { "docker compose" } else { "docker-compose" }
    
    Write-ColorOutput "서비스 '$service' 빌드 중..." $BLUE
    try {
        if ($service -eq "all") {
            & cmd /c "$composeCmd build"
        } else {
            & cmd /c "$composeCmd build $service"
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ 빌드 완료!" $GREEN
        } else {
            Write-ColorOutput "✗ 빌드 실패!" $RED
            return $false
        }
    }
    catch {
        Write-ColorOutput "✗ 빌드 중 오류 발생!" $RED
        return $false
    }
    return $true
}

function Start-Service {
    param(
        [string]$service,
        [string]$composeVersion
    )
    
    $composeCmd = if ($composeVersion -eq "v2") { "docker compose" } else { "docker-compose" }
    
    Write-ColorOutput "서비스 '$service' 시작 중..." $BLUE
    try {
        & cmd /c "$composeCmd up -d $service"
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✓ 서비스 시작 완료!" $GREEN
            Write-ColorOutput "컨테이너에 접속하려면: $composeCmd exec $service bash" $CYAN
        } else {
            Write-ColorOutput "✗ 서비스 시작 실패!" $RED
        }
    }
    catch {
        Write-ColorOutput "✗ 서비스 시작 중 오류 발생!" $RED
    }
}

function Show-Status {
    param([string]$composeVersion)
    
    $composeCmd = if ($composeVersion -eq "v2") { "docker compose" } else { "docker-compose" }
    
    Write-ColorOutput "현재 서비스 상태:" $BLUE
    & cmd /c "$composeCmd ps"
}

function Stop-Services {
    param([string]$composeVersion)
    
    $composeCmd = if ($composeVersion -eq "v2") { "docker compose" } else { "docker-compose" }
    
    Write-ColorOutput "모든 서비스 중지 중..." $YELLOW
    & cmd /c "$composeCmd down"
    Write-ColorOutput "✓ 서비스 중지 완료!" $GREEN
}

function Clean-Docker {
    param([string]$composeVersion)
    
    $composeCmd = if ($composeVersion -eq "v2") { "docker compose" } else { "docker-compose" }
    
    Write-ColorOutput "Docker 정리를 시작합니다..." $YELLOW
    Write-ColorOutput "경고: 이 작업은 모든 컨테이너와 이미지를 삭제합니다!" $RED
    
    $confirmation = Read-Host "계속하시겠습니까? (y/N)"
    if ($confirmation -eq "y" -or $confirmation -eq "Y") {
        Write-ColorOutput "컨테이너 및 이미지 삭제 중..." $YELLOW
        & cmd /c "$composeCmd down --rmi all --volumes --remove-orphans"
        & docker system prune -f
        Write-ColorOutput "✓ 정리 완료!" $GREEN
    } else {
        Write-ColorOutput "정리 작업이 취소되었습니다." $BLUE
    }
}

# Main execution
function Main {
    Show-Header
    
    # Check prerequisites
    Test-Docker
    $composeVersion = Test-DockerCompose
    $hasNvidia = Test-NvidiaDocker
    
    while ($true) {
        Show-Menu $hasNvidia
        $choice = Read-Host "선택하세요"
        
        switch ($choice) {
            "1" {
                if ($hasNvidia) {
                    if (Build-Service "golexp-gpu" $composeVersion) {
                        Start-Service "golexp-gpu" $composeVersion
                    }
                } else {
                    Write-ColorOutput "NVIDIA GPU가 없어서 GPU 버전을 사용할 수 없습니다." $RED
                }
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "2" {
                if (Build-Service "golexp-cpu" $composeVersion) {
                    Start-Service "golexp-cpu" $composeVersion
                }
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "3" {
                if ($hasNvidia) {
                    Write-ColorOutput "주의: Windows에서는 X11 포워딩이 제한적입니다." $YELLOW
                    Write-ColorOutput "X Server (예: VcXsrv, Xming)가 필요할 수 있습니다." $YELLOW
                    if (Build-Service "golexp-dev" $composeVersion) {
                        Start-Service "golexp-dev" $composeVersion
                    }
                } else {
                    Write-ColorOutput "NVIDIA GPU가 없어서 개발 환경을 사용할 수 없습니다." $RED
                }
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "4" {
                Build-Service "all" $composeVersion
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "5" {
                Show-Status $composeVersion
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "6" {
                Stop-Services $composeVersion
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "7" {
                Clean-Docker $composeVersion
                Read-Host "계속하려면 Enter를 누르세요"
            }
            "0" {
                Write-ColorOutput "설정 스크립트를 종료합니다." $GREEN
                exit 0
            }
            default {
                Write-ColorOutput "잘못된 선택입니다. 다시 선택해주세요." $RED
                Read-Host "계속하려면 Enter를 누르세요"
            }
        }
        
        Show-Header
    }
}

# Run the main function
Main