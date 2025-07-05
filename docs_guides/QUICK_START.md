# 🚀 AutoCI 빠른 시작 가이드

## 5분 만에 시작하기!

### 1단계: 빠른 설정

#### 옵션 A: WSL에서 AI Godot 빌드 (권장) ⭐
```bash
# Godot 명령어 설치 (처음 한 번만)
chmod +x install_godot_commands.sh
./install_godot_commands.sh

# AI 수정된 Godot 빌드 실행
build-godot
# → 메뉴에서 1번 선택: AI 수정된 Godot 빌드 (Linux)
```

#### 옵션 B: Windows에서 직접 실행
1. **AutoCI 폴더 열기**
2. **`RUN_SIMPLE_BUILD.bat` 더블클릭**

#### 옵션 C: WSL에서 Python으로 실행
```bash
python3 wsl_run_build.py
```

### 2단계: AutoCI 실행

WSL 터미널에서:
```bash
autoci
```

### 3단계: 24시간 자동 게임 개발 시작

```
> create platformer game
```

## 🎮 이제 뭘 할 수 있나요?

- **24시간 자동 게임 개발**: AI가 알아서 게임을 만듭니다
- **실시간 모니터링**: Godot 창에서 개발 과정을 실시간으로 확인
- **24시간 C# 학습**: `autoci learn` 명령어로 학습 시작

## ❓ 문제가 있나요?

1. **Python이 없다고 나올 때**: https://python.org 에서 다운로드
2. **WSL이 없을 때**: Windows 스토어에서 Ubuntu 설치
3. **그 외 문제**: GitHub Issues에 문의

---

**더 자세한 정보는 README.md를 참고하세요!**