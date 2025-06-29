# AI Godot 빌드 가이드 🚀

AutoCI가 완전히 제어할 수 있는 AI 수정된 Godot을 빌드하는 방법입니다.

## 📋 빌드 전 확인사항

### 필수 요구사항
- **OS**: Windows 10/11 (WSL2 선택사항)
- **RAM**: 8GB 이상 (16GB 권장)
- **디스크**: 30GB 이상 여유 공간
- **도구**:
  - Python 3.8 이상
  - Git
  - Visual Studio Build Tools 2019/2022 (Windows)
  - 또는 MinGW-w64 (대안)

### 환경 확인
```bash
python3 check_build_status.py
```

## 🔨 빌드 방법

### 방법 1: WSL에서 통합 명령어 사용 (권장) ⭐

```bash
# build-godot 명령어 실행
build-godot

# 메뉴에서 선택:
# 1. AI 수정된 Godot 빌드 (Linux) - 권장
# 2. 빠른 Windows 설정 (5분)
# 3. 전체 Windows 빌드 (1시간)
```

### 방법 2: WSL에서 직접 빌드

```bash
cd /mnt/d/AutoCI/AutoCI
python3 build_ai_godot.py
```

### 방법 3: Windows에서 직접 빌드

1. **Windows 탐색기에서 AutoCI 폴더 열기**
2. **BUILD_AI_GODOT.bat 더블클릭**
3. **빌드 옵션 선택**:
   - 1: 전체 빌드 (AI 기능 모두 포함) - 권장
   - 2: 빠른 빌드 (기본 AI 기능만)
   - 3: 디버그 빌드 (개발용)

## 📊 빌드 진행 과정

1. **환경 준비** (1-2분)
   - 필수 도구 확인
   - 디렉토리 생성

2. **소스코드 다운로드** (5-10분)
   - Godot 4.3 stable 다운로드
   - 약 150MB

3. **AI 패치 적용** (1-2분)
   - 코어 AI 통합
   - 에디터 훅 추가
   - 스크립트 주입 기능
   - 씬 제어 API
   - 네트워크 API
   - 렌더링 접근

4. **빌드 실행** (20분-1시간)
   - SCons로 컴파일
   - Windows 실행 파일 생성

5. **후처리** (1분)
   - 실행 파일 복사
   - 설정 파일 생성

## ✅ 빌드 성공 확인

빌드가 성공하면:
- `godot_ai_build/output/godot.windows.editor.x86_64.exe` 생성
- `godot_build.log`에 빌드 로그 저장
- AutoCI가 자동으로 AI Godot 감지

## 🚀 빌드 후 사용

1. **AutoCI 실행**:
   ```bash
   autoci
   ```

2. **AI Godot이 자동으로 실행됨**

3. **24시간 자동 게임 개발 시작**:
   ```
   > create platformer game
   ```

## ❌ 문제 해결

### "Visual Studio Build Tools가 없습니다"
1. https://visualstudio.microsoft.com/downloads/
2. "Build Tools for Visual Studio" 다운로드
3. C++ 빌드 도구 선택하여 설치

### "Python을 찾을 수 없습니다"
1. https://python.org 에서 Python 3.8+ 다운로드
2. 설치 시 "Add Python to PATH" 체크

### 빌드 실패 시
1. `godot_build.log` 파일 확인
2. 에러 메시지 검색
3. GitHub Issues에 문제 보고

## 🎯 AI 기능 설명

빌드된 AI Godot은 다음 기능을 포함합니다:

### 1. 명령 실행 API
- `ai_execute_command()`: AI 명령 실행
- `ai_batch_operation()`: 일괄 작업

### 2. 스크립트 주입
- `ai_inject_script()`: 실행 중 코드 주입
- `ai_modify_script()`: 메서드 동적 교체

### 3. 씬 조작
- `ai_create_scene()`: AI가 씬 생성
- `ai_add_node()`: 노드 추가
- `ai_manipulate_nodes()`: 노드 조작

### 4. 실시간 제어
- 소켓 통신으로 외부 제어
- 비동기 명령 처리
- 상태 모니터링

## 📞 지원

문제가 있으면:
1. `check_build_status.py` 실행하여 환경 확인
2. `godot_build.log` 확인
3. GitHub Issues에 문제 보고

---

**빌드 시작**: `BUILD_AI_GODOT.bat` 더블클릭! 🚀