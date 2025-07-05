#!/bin/bash
# build-godot 명령어 업데이트 스크립트

echo "🔧 build-godot 명령어 업데이트 중..."

# 기존 명령어 백업
if [ -f /usr/local/bin/build-godot ]; then
    sudo cp /usr/local/bin/build-godot /usr/local/bin/build-godot.bak
fi

# 새로운 build-godot 스크립트 생성 (Windows 빌드)
sudo tee /usr/local/bin/build-godot > /dev/null << 'EOF'
#!/bin/bash
# AutoCI AI Godot Windows 빌드 명령어

cd /mnt/d/AutoCI/AutoCI

echo "🤖 AutoCI - AI Godot Windows 빌드 시작"
echo "=========================================="

# Python Windows 빌드 스크립트 실행
if [ -f "build-godot" ]; then
    python3 build-godot
else
    echo "❌ build-godot 파일을 찾을 수 없습니다."
    echo "💡 현재 디렉토리: $(pwd)"
fi
EOF

# 실행 권한 설정
sudo chmod +x /usr/local/bin/build-godot

echo "✅ build-godot 명령어가 Windows 빌드로 업데이트되었습니다!"
echo ""
echo "사용법:"
echo "  build-godot        # Windows exe 빌드"
echo "  build-godot-linux  # Linux 실행 파일 빌드"