# AutoCI 프로젝트 정리 가이드

## 🗑️ 삭제 권장 파일 목록

### 중복 빌드 관련 파일들
다음 파일들은 `build_ai_godot.py`와 기능이 중복되므로 삭제를 권장합니다:

```bash
rm -f simple_godot_build.py
rm -f quick_build_check.py
rm -f start_build_now.py
rm -f start_ai_godot_build.py
rm -f run_build.py
rm -f test_wsl_cmd.py
```

### 임시/테스트 파일들
개발 중 생성된 임시 파일들:

```bash
rm -f setup_custom_godot.py  # build_ai_godot.py로 대체됨
rm -f simple_build.log       # 임시 로그 파일
```

### 중복 폴더
```bash
# godot_ai_build/와 중복되는 폴더
rm -rf godot_modified/
```

## ✅ 유지해야 할 핵심 파일들

### 메인 시스템
- `autoci` - 전역 실행 파일
- `autoci.py` - 메인 진입점
- `autoci_terminal.py` - 터미널 인터페이스
- `autoci_production.py` - 프로덕션 모드

### AI Godot 빌드
- `build_ai_godot.py` - AI Godot 빌드 시스템
- `build-godot` - 빌드 명령어
- `check-godot` - 상태 확인 명령어
- `BUILD_AI_GODOT.bat` - Windows 빌드
- `RUN_SIMPLE_BUILD.bat` - 빠른 시작

### 설정 및 설치
- `install_global_autoci.sh` - AutoCI 설치
- `install_godot_commands.sh` - Godot 명령어 설치
- `setup_ai_godot.py` - AI Godot 설정 도우미
- `wsl_run_build.py` - WSL 빌드 실행

### 문서
- `README.md` - 메인 문서
- `QUICK_START.md` - 빠른 시작 가이드
- `AI_GODOT_BUILD_GUIDE.md` - 빌드 가이드

### 모듈 (modules/ 폴더)
모든 모듈 파일들은 유지:
- `csharp_24h_*.py` - 24시간 학습 시스템
- `godot_*.py` - Godot 통합 시스템
- `monitoring_*.py` - 모니터링 시스템
- `error_handler*.py` - 오류 처리

### 데이터 폴더
- `user_learning_data/` - 학습 데이터
- `game_projects/` - 생성된 프로젝트
- `godot_ai/` - AI 플러그인
- `godot_ai_patches/` - 패치 파일
- `logs/` - 로그 파일

## 🔧 정리 명령어

### 안전한 정리 (권장)
```bash
# 백업 생성
mkdir -p backup
cp -r *.py backup/

# 중복 파일만 삭제
rm -f simple_godot_build.py quick_build_check.py start_build_now.py
rm -f start_ai_godot_build.py run_build.py test_wsl_cmd.py
```

### 로그 정리
```bash
# 7일 이상된 로그 파일 삭제
find logs/ -name "*.log" -mtime +7 -delete
```

### Git 정리
```bash
# 삭제된 파일들을 Git에서도 제거
git add -A
git commit -m "cleanup: 중복 파일 제거 및 프로젝트 구조 정리"
```

## 📝 정리 후 확인사항

1. **필수 명령어 작동 확인**
   ```bash
   autoci --version
   build-godot
   check-godot
   ```

2. **모듈 임포트 테스트**
   ```bash
   python3 -c "from modules import csharp_24h_learning"
   ```

3. **설정 파일 확인**
   ```bash
   ls -la autoci_config.json .godot_config.json
   ```

## ⚠️ 주의사항

- 학습 데이터(`user_learning_data/`)는 절대 삭제하지 마세요
- `.godot_config.json` 파일은 유지하세요
- `godot_ai_build/` 폴더는 빌드 결과물이므로 유지하세요

---

정리 작업 전 반드시 백업을 생성하시기 바랍니다!