#!/usr/bin/env pwsh
# PowerShell Game of Life 데이터 자동 생성 스크립트
# 생존 비율 0.01~0.99까지 99개 데이터셋 생성

# Colors for output
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$RED = "`e[31m"
$CYAN = "`e[36m"
$NC = "`e[0m"  # No Color

function Write-ColorOutput {
    param(
        [string]$Text,
        [string]$Color = $NC
    )
    Write-Host "$Color$Text$NC"
}

function Test-NvidiaGPU {
    try {
        # Windows에서 nvidia-smi 확인
        $nvidiaOutput = & nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }
    catch {
        # nvidia-smi가 없거나 실행 실패
    }
    return $false
}

function Start-DataGeneration {
    Write-ColorOutput "🚀 GOL 데이터셋 자동 생성 시작..." $CYAN
    Write-ColorOutput "📊 생존 비율: 0.01 ~ 0.99 (99개 파일)" $CYAN
    Write-ColorOutput "📁 각 파일: 1000개 샘플" $CYAN
    Write-Host ""

    # GPU 감지
    $hasGPU = Test-NvidiaGPU
    if ($hasGPU) {
        Write-ColorOutput "🎮 NVIDIA GPU 감지됨 - GPU 모드로 실행" $GREEN
    } else {
        Write-ColorOutput "💻 GPU 미감지 - CPU 모드로 실행" $YELLOW
    }
    Write-Host ""

    # 진행 상황 추적
    $totalFiles = 99
    $current = 0
    $errors = @()

    # 데이터 생성 (0.01부터 0.99까지)
    for ($i = 1; $i -le 99; $i++) {
        $ratio = ($i * 0.01).ToString("0.00")
        $current++
        $progress = [math]::Round(($current * 100 / $totalFiles), 0)
        
        Write-ColorOutput "[$current/$totalFiles] ($progress%) 생성 중: database-54321_1000_$ratio.txt" $CYAN
        
        # 명령어 구성
        if ($hasGPU) {
            $arguments = @("new_project/datagen.py", "54321", "1000", $ratio, "--one_file")
        } else {
            $arguments = @("new_project/datagen.py", "54321", "1000", $ratio, "--one_file", "--cpu")
            Write-ColorOutput "  ⚠️  GPU 미감지, CPU 모드로 실행..." $YELLOW
        }
        
        # Python 실행
        try {
            $process = Start-Process -FilePath "python3" -ArgumentList $arguments -NoNewWindow -PassThru -Wait
            
            if ($process.ExitCode -ne 0) {
                $errorMsg = "❌ 오류 발생: database-54321_1000_$ratio.txt 생성 실패 (종료 코드: $($process.ExitCode))"
                Write-ColorOutput $errorMsg $RED
                $errors += $errorMsg
                
                Write-ColorOutput "💡 해결 방법:" $YELLOW
                Write-ColorOutput "   1. 빌드 확인: cd build && cmake .. && cmake --build . --config Release" $YELLOW
                Write-ColorOutput "   2. CPU 모드: --cpu 옵션 추가" $YELLOW
                Write-ColorOutput "   3. 권한 확인: 관리자 권한으로 실행" $YELLOW
                
                # 사용자에게 계속할지 물어보기
                $choice = Read-Host "계속 진행하시겠습니까? (y/N)"
                if ($choice -notmatch '^[yY]') {
                    Write-ColorOutput "❌ 사용자에 의해 중단됨" $RED
                    return 1
                }
            } else {
                Write-ColorOutput "  ✅ 성공" $GREEN
            }
        }
        catch {
            $errorMsg = "❌ 실행 오류: $($_.Exception.Message)"
            Write-ColorOutput $errorMsg $RED
            $errors += $errorMsg
            
            # 사용자에게 계속할지 물어보기
            $choice = Read-Host "계속 진행하시겠습니까? (y/N)"
            if ($choice -notmatch '^[yY]') {
                Write-ColorOutput "❌ 사용자에 의해 중단됨" $RED
                return 1
            }
        }
    }

    Write-Host ""
    
    if ($errors.Count -eq 0) {
        Write-ColorOutput "✅ 모든 데이터셋 생성 완료!" $GREEN
        Write-ColorOutput "📊 총 파일: $totalFiles개" $GREEN
        Write-ColorOutput "📁 저장 위치: train_data/" $GREEN
        Write-ColorOutput "💾 총 샘플: $($totalFiles * 1000)개" $GREEN
    } else {
        Write-ColorOutput "⚠️  일부 오류와 함께 완료됨" $YELLOW
        Write-ColorOutput "📊 성공: $($totalFiles - $errors.Count)/$totalFiles개" $YELLOW
        Write-ColorOutput "❌ 실패: $($errors.Count)개" $RED
        Write-Host ""
        Write-ColorOutput "실패한 파일들:" $RED
        foreach ($error in $errors) {
            Write-ColorOutput "  $error" $RED
        }
    }
    
    return 0
}

function Show-Help {
    Write-ColorOutput "GOL 데이터 생성 스크립트 (PowerShell)" $CYAN
    Write-Host ""
    Write-ColorOutput "사용법:" $GREEN
    Write-ColorOutput "  .\genData.ps1              # 자동 데이터 생성" $GREEN
    Write-ColorOutput "  .\genData.ps1 --help       # 도움말 표시" $GREEN
    Write-Host ""
    Write-ColorOutput "기능:" $GREEN
    Write-ColorOutput "  • 생존 비율 0.01~0.99 범위의 99개 데이터셋 생성" $GREEN
    Write-ColorOutput "  • 각 데이터셋마다 1000개 샘플 포함" $GREEN
    Write-ColorOutput "  • GPU/CPU 자동 감지 및 최적 모드 선택" $GREEN
    Write-ColorOutput "  • 실시간 진행률 표시" $GREEN
    Write-ColorOutput "  • 오류 발생 시 복구 옵션 제공" $GREEN
    Write-Host ""
    Write-ColorOutput "출력:" $GREEN
    Write-ColorOutput "  train_data/database-54321_1000_*.txt" $GREEN
}

# 메인 실행 로직
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Show-Help
    exit 0
}

# 현재 디렉토리 확인
if (-not (Test-Path "new_project/datagen.py")) {
    Write-ColorOutput "❌ 오류: new_project/datagen.py를 찾을 수 없습니다." $RED
    Write-ColorOutput "💡 GOL 프로젝트 루트 디렉토리에서 실행하세요." $YELLOW
    Write-ColorOutput "예: cd C:\path\to\GOLexp_pytorch && .\genData.ps1" $YELLOW
    exit 1
}

# Python 설치 확인
try {
    $pythonVersion = & python3 --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        # python3가 없으면 python 시도
        $pythonVersion = & python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            # python으로 교체
            $global:pythonCmd = "python"
        } else {
            throw "Python not found"
        }
    } else {
        $global:pythonCmd = "python3"
    }
    Write-ColorOutput "🐍 Python 감지됨: $pythonVersion" $GREEN
}
catch {
    Write-ColorOutput "❌ 오류: Python이 설치되지 않았습니다." $RED
    Write-ColorOutput "💡 Python 3.8 이상을 설치하세요: https://python.org" $YELLOW
    exit 1
}

# 데이터 생성 시작
$result = Start-DataGeneration
exit $result