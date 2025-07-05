#!/bin/bash
# AutoCI 전역 명령어 설치 스크립트

echo "🚀 AutoCI 전역 명령어 설치 시작..."

# 스크립트 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LAUNCHER_PATH="$SCRIPT_DIR/autoci_launcher.py"

# OS 확인
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows
    echo "Windows 환경 감지..."
    
    # PowerShell 스크립트 생성
    cat > "$SCRIPT_DIR/autoci.ps1" << 'EOF'
$scriptPath = $PSScriptRoot
$launcherPath = Join-Path $scriptPath "autoci_launcher.py"
python $launcherPath $args
EOF
    
    # 배치 파일 생성
    cat > "$SCRIPT_DIR/autoci.bat" << 'EOF'
@echo off
python "%~dp0autoci_launcher.py" %*
EOF
    
    echo "✅ Windows용 스크립트 생성 완료"
    echo ""
    echo "사용법:"
    echo "1. 시스템 환경 변수 PATH에 다음 경로 추가:"
    echo "   $SCRIPT_DIR"
    echo ""
    echo "2. 새 명령 프롬프트에서 사용:"
    echo "   autoci"
    echo "   autoci create --name MyGame --type platformer"
    echo "   autoci learn"
    echo "   autoci learn low"
    
else
    # Linux/Mac
    echo "Linux/Mac 환경 감지..."
    
    # autoci 실행 스크립트 생성
    AUTOCI_BIN="/usr/local/bin/autoci"
    
    # 임시 스크립트 생성
    cat > /tmp/autoci << EOF
#!/bin/bash
python3 "$LAUNCHER_PATH" "\$@"
EOF
    
    # 실행 권한 부여
    chmod +x /tmp/autoci
    chmod +x "$LAUNCHER_PATH"
    
    # 시스템에 설치 (sudo 필요)
    if [ -w "/usr/local/bin" ]; then
        cp /tmp/autoci /usr/local/bin/
        echo "✅ AutoCI가 /usr/local/bin에 설치되었습니다"
    else
        echo "⚠️  sudo 권한이 필요합니다. 다음 명령어를 실행하세요:"
        echo "   sudo cp /tmp/autoci /usr/local/bin/"
        echo "   sudo chmod +x /usr/local/bin/autoci"
    fi
    
    # 사용자 로컬 bin 디렉토리 옵션
    USER_BIN="$HOME/.local/bin"
    if [ ! -d "$USER_BIN" ]; then
        mkdir -p "$USER_BIN"
    fi
    
    cp /tmp/autoci "$USER_BIN/"
    chmod +x "$USER_BIN/autoci"
    
    echo ""
    echo "✅ AutoCI가 $USER_BIN에도 설치되었습니다"
    echo ""
    echo "PATH에 추가하려면 ~/.bashrc 또는 ~/.zshrc에 다음 줄 추가:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "🎉 설치 완료!"
echo ""
echo "사용 가능한 명령어:"
echo "  autoci              - 대화형 모드"
echo "  autoci create       - 게임 자동 생성"
echo "  autoci learn        - AI 학습"
echo "  autoci learn low    - 메모리 최적화 학습"
echo "  autoci fix          - 엔진 개선"
echo "  autoci monitor      - 실시간 모니터링"
echo "  autoci demo         - 빠른 데모"