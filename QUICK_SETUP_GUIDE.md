# 🚀 AutoCI v3.0 빠른 설정 가이드

## 📋 현재 상황
Ubuntu 24.04에서 **PEP 668 "externally-managed-environment" 에러**가 발생했습니다.
이 문제를 해결하고 AutoCI를 성공적으로 실행하기 위한 단계별 가이드입니다.

## 🔧 1단계: PEP 668 문제 해결

```bash
# 현재 위치에서 실행 (AutoCI 디렉토리)
./fix_pep668_install.sh
```

이 스크립트는 다음을 수행합니다:
- ✅ 기존 가상환경 정리
- ✅ 새로운 Python 가상환경 생성
- ✅ 필요한 패키지 설치
- ✅ 가상환경 활성화 확인

## 🎯 2단계: 가상환경 활성화 (매번 필수!)

```bash
# AutoCI 사용할 때마다 실행해야 함
source autoci_env/bin/activate

# 가상환경이 활성화되었는지 확인
which python
# 출력: /path/to/AutoCI/autoci_env/bin/python
```

**중요**: 터미널을 새로 열 때마다 이 명령어를 실행해야 합니다!

## 🚀 3단계: AutoCI 모델 다운로드 (선택사항)

### 기본 모델만 사용하는 경우:
```bash
# 가상환경 활성화 후
source autoci_env/bin/activate

# AutoCI 바로 시작
python start_autoci_agent.py
```

### 고급 모델 사용하는 경우:
```bash
# 가상환경 활성화 후
source autoci_env/bin/activate

# 고급 모델 다운로드 (150GB+, 1-3시간 소요)
./download_advanced_models.sh

# 모델 성능 테스트 (선택사항)
python benchmark_models.py

# AutoCI 고급 모드로 시작
python start_autoci_agent.py --advanced-models
```

## 📊 4단계: 시스템 상태 확인

```bash
# 가상환경 활성화 확인
echo $VIRTUAL_ENV

# Python 버전 확인
python --version

# 필요한 패키지 확인
python -c "import torch, transformers; print('✅ 모든 패키지 설치됨')"

# GPU 사용 가능 여부 확인
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available())"
```

## 🎮 5단계: AutoCI 첫 실행

```bash
# 1. 가상환경 활성화 (필수!)
source autoci_env/bin/activate

# 2. AutoCI 시작
python start_autoci_agent.py

# 3. 대화 시작 예시
"간단한 2D 플랫포머 게임을 만들어줘"
```

## 🛠️ 문제 해결

### 가상환경 활성화가 안 되는 경우:
```bash
# 가상환경 재생성
rm -rf autoci_env
python3 -m venv autoci_env
source autoci_env/bin/activate
pip install -r requirements.txt
```

### "command not found" 에러:
```bash
# 스크립트 실행 권한 확인
chmod +x *.sh
ls -la *.sh
```

### 메모리 부족 에러:
```bash
# 시스템 메모리 확인
free -h

# 32GB 미만인 경우 기본 모델만 사용
python start_autoci_agent.py  # 고급 모델 플래그 제거
```

## 🏃‍♂️ 빠른 시작 요약

```bash
# 1. PEP 668 문제 해결 (한 번만)
./fix_pep668_install.sh

# 2. 매번 사용 시 (가상환경 활성화)
source autoci_env/bin/activate

# 3. AutoCI 실행
python start_autoci_agent.py
```

## 📱 대시보드 접속

AutoCI 실행 후 다음 주소에서 모니터링 가능:
- **대시보드**: http://localhost:8888
- **API 문서**: http://localhost:8000/docs (서비스 실행 시)

## 🔄 자동 시작 설정 (선택사항)

매번 가상환경을 활성화하는 것이 번거롭다면:

```bash
# ~/.bashrc에 별칭 추가
echo 'alias autoci="cd $(pwd) && source autoci_env/bin/activate && python start_autoci_agent.py"' >> ~/.bashrc
source ~/.bashrc

# 이제 어디서든 'autoci' 명령어로 시작 가능
autoci
```

## 🎯 성공 확인

다음이 모두 작동하면 성공:
- ✅ `source autoci_env/bin/activate` 실행 시 프롬프트 변경
- ✅ `which python`이 가상환경 경로 출력
- ✅ `python start_autoci_agent.py` 에러 없이 실행
- ✅ AutoCI와 한국어 대화 가능

## 🆘 추가 도움이 필요한 경우

1. **에러 로그 확인**: `tail -f logs/*.log`
2. **시스템 정보**: `python benchmark_models.py --help`
3. **문서 확인**: `README.md` 파일의 문제 해결 섹션

---

**🎉 축하합니다! AutoCI v3.0이 준비되었습니다!**

이제 자연어로 게임을 만들어보세요:
- "간단한 퍼즐 게임 만들어줘"
- "2D 플랫포머에 더블점프 추가해줘" 
- "적 AI를 더 똑똑하게 만들어줘" 