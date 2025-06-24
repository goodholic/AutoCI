# 🐧 AutoCI WSL 실행 가이드

## 📋 WSL 환경에서 AutoCI 실행하기

WSL(Windows Subsystem for Linux) 환경에서 AutoCI를 실행하기 위한 가이드입니다.

## 🚀 빠른 시작

### 1단계: WSL 환경 설정
```bash
# WSL 터미널에서 실행
cd /mnt/c/Users/[사용자명]/Desktop/Unity\ Project\(25년도\ 제작\)/26.AutoCI/AutoCI

# 설정 스크립트 실행
chmod +x wsl_setup.sh
./wsl_setup.sh
```

### 2단계: AutoCI 실행
```bash
# WSL 전용 스크립트로 실행
python3 wsl_start_all.py

# 또는 기존 스크립트 사용
python3 start_all.py
```

## 🌐 접속 방법

### WSL 내부에서 접속
- AI 코드 생성: http://localhost:7100/codegen
- 스마트 검색: http://localhost:7100/codesearch
- 프로젝트 Q&A: http://localhost:7100/rag
- 학습 대시보드: http://localhost:8080/dashboard

### Windows에서 접속
1. WSL IP 주소 확인:
   ```bash
   hostname -I
   ```

2. Windows 브라우저에서 접속:
   - AI 코드 생성: http://[WSL_IP]:7100/codegen
   - 스마트 검색: http://[WSL_IP]:7100/codesearch
   - 프로젝트 Q&A: http://[WSL_IP]:7100/rag
   - 학습 대시보드: http://[WSL_IP]:8080/dashboard

## 🔥 Windows 방화벽 설정

Windows에서 WSL 서비스에 접근하려면 PowerShell을 **관리자 권한**으로 실행하고 다음 명령을 입력하세요:

```powershell
# AI 서버 포트
New-NetFirewallRule -DisplayName "WSL Port 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

# 모니터링 API 포트
New-NetFirewallRule -DisplayName "WSL Port 8080" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow

# Backend 포트
New-NetFirewallRule -DisplayName "WSL Port 5049" -Direction Inbound -LocalPort 5049 -Protocol TCP -Action Allow

# Frontend 포트
New-NetFirewallRule -DisplayName "WSL Port 7100" -Direction Inbound -LocalPort 7100 -Protocol TCP -Action Allow
```

## 🛠️ WSL 특화 기능

### `wsl_start_all.py`의 주요 기능:
1. **WSL 환경 자동 감지**: `/proc/version` 파일을 통해 WSL 환경 확인
2. **네트워크 설정 최적화**: 모든 서비스를 `0.0.0.0`으로 바인딩
3. **Windows 프로세스 플래그 제거**: `CREATE_NEW_CONSOLE` 같은 Windows 전용 플래그 제거
4. **WSL IP 자동 표시**: Windows에서 접속할 수 있는 IP 주소 자동 표시
5. **방화벽 설정 가이드**: 필요한 방화벽 명령어 자동 생성

## 📊 시스템 요구사항

### WSL 버전
- WSL 2 권장 (더 나은 성능과 호환성)
- WSL 1에서도 작동하지만 성능이 제한적일 수 있음

### 확인 방법:
```bash
wsl -l -v
```

### WSL 2로 업그레이드:
```powershell
# PowerShell 관리자 권한
wsl --set-version [배포판 이름] 2
```

## 🐛 문제 해결

### 1. "Permission denied" 오류
```bash
# 실행 권한 부여
chmod +x wsl_start_all.py
chmod +x start_expert_learning.py
chmod +x download_model.py
```

### 2. Windows에서 접속이 안 될 때
- Windows Defender 방화벽에서 WSL 포트 허용 확인
- WSL IP 주소가 변경되었는지 확인 (`hostname -I`)
- 서비스가 `0.0.0.0`으로 바인딩되었는지 확인

### 3. 메모리 부족 문제
WSL 2의 메모리 제한 설정:
```bash
# Windows 사용자 홈 디렉토리에 .wslconfig 파일 생성
# C:\Users\[사용자명]\.wslconfig

[wsl2]
memory=16GB
swap=8GB
```

### 4. 디스크 공간 문제
WSL 가상 디스크 확장:
```powershell
# PowerShell 관리자 권한
wsl --shutdown
diskpart
select vdisk file="C:\Users\[사용자명]\AppData\Local\Packages\[WSL패키지]\LocalState\ext4.vhdx"
expand vdisk maximum=100000
```

## 💡 추가 팁

### 1. WSL 파일 시스템 최적화
```bash
# Windows 파일 시스템 대신 WSL 파일 시스템 사용
cp -r /mnt/c/[프로젝트경로] ~/autoci
cd ~/autoci
```

### 2. VS Code에서 WSL 개발
```bash
# WSL 터미널에서
code .
```

### 3. 백그라운드 실행
```bash
# tmux 사용
sudo apt install tmux
tmux new -s autoci
python3 wsl_start_all.py

# 세션 분리: Ctrl+B, D
# 세션 재접속: tmux attach -t autoci
```

## 📝 주의사항

1. **경로 구분자**: WSL에서는 항상 `/` 사용 (Windows의 `\` 아님)
2. **줄 끝 문자**: Git 설정에서 `core.autocrlf=false` 권장
3. **대소문자 구분**: Linux는 파일명 대소문자를 구분함
4. **권한 관리**: chmod로 적절한 파일 권한 설정 필요

## 🤝 도움말

추가 도움이 필요하면:
- WSL 공식 문서: https://docs.microsoft.com/ko-kr/windows/wsl/
- AutoCI Issues: [프로젝트 Issues 페이지]