# AutoCI Panda3D 통합 가이드 🎮

AI가 자동으로 Panda3D 게임을 개발하는 시스템의 완전한 가이드입니다.

## 📋 목차

1. [개요](#개요)
2. [시스템 구조](#시스템-구조)
3. [설치 및 설정](#설치-및-설정)
4. [사용법](#사용법)
5. [AI 모델 구조](#ai-모델-구조)
6. [게임 개발 프로세스](#게임-개발-프로세스)
7. [API 레퍼런스](#api-레퍼런스)
8. [예제](#예제)

## 개요

AutoCI Panda3D 통합 시스템은 AI가 24시간 동안 자동으로 2.5D/3D 게임을 개발하는 혁신적인 시스템입니다.

### 주요 특징

- **완전 자동화**: AI가 게임 기획부터 구현, 최적화까지 모든 과정을 수행
- **다양한 게임 타입**: 플랫포머, 레이싱, RPG, 퍼즐, 슈터 등 지원
- **실시간 모니터링**: Socket.IO를 통한 개발 과정 실시간 확인
- **강화학습 기반**: PyTorch를 활용한 지속적인 개선
- **딥러닝 코드 생성**: DeepSeek-Coder, Llama 등 최신 AI 모델 활용

## 시스템 구조

```
AutoCI Panda3D System
├── AI Agent (강화학습 + 딥러닝)
│   ├── Policy Network (액션 결정)
│   ├── Code Generator (AI 모델)
│   └── Quality Evaluator
├── Panda3D Controller
│   ├── Project Manager
│   ├── Code Applier
│   └── Build System
├── Socket.IO Server
│   ├── Real-time Updates
│   ├── Multiplayer Support
│   └── Monitoring Dashboard
└── PyTorch Learning System
    ├── Experience Replay
    ├── Model Training
    └── Knowledge Storage
```

## 설치 및 설정

### 1. 필수 요구사항

```bash
# Python 3.8+ 필요
python --version

# CUDA (GPU 사용 시)
nvidia-smi
```

### 2. 의존성 설치

```bash
# Panda3D 설치
pip install panda3d

# PyTorch 설치 (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Socket.IO 설치
pip install python-socketio[asyncio] aiohttp

# 기타 의존성
pip install numpy psutil click
```

### 3. AI 모델 설정

```bash
# 모델 디렉토리 생성
mkdir -p models/pytorch_models

# 환경 변수 설정 (선택사항)
export AUTOCI_MODEL_PATH="./models"
export AUTOCI_DEVICE="cuda"  # 또는 "cpu"
```

## 사용법

### 기본 명령어

```bash
# 1. 새 게임 생성 (24시간 개발)
python run_autoci_panda3d.py create --name MyGame --type platformer

# 2. 빠른 데모 (5분)
python run_autoci_panda3d.py demo

# 3. 게임 분석
python run_autoci_panda3d.py analyze --path game_projects/MyGame

# 4. 실시간 모니터링
python run_autoci_panda3d.py monitor --port 5001
```

### Python API 사용

```python
import asyncio
from modules.autoci_panda3d_integration import AutoCIPanda3DSystem

async def create_my_game():
    # 시스템 초기화
    system = AutoCIPanda3DSystem()
    
    # 플랫폼 게임 생성 (24시간)
    result = await system.create_game(
        project_name="SuperPlatformer",
        game_type="platformer",
        development_hours=24.0
    )
    
    if result["success"]:
        print(f"게임 생성 완료! 품질: {result['quality_score']}/100")
        
        # 게임 수정
        mod_result = await system.modify_game(
            project_name="SuperPlatformer",
            modification_request="Add double jump and wall sliding"
        )

# 실행
asyncio.run(create_my_game())
```

## AI 모델 구조

### 1. 강화학습 정책 네트워크

```python
class GameDevelopmentPolicyNetwork(nn.Module):
    def __init__(self):
        # 3층 신경망
        self.fc1 = nn.Linear(128, 256)  # 상태 -> 은닉층
        self.fc2 = nn.Linear(256, 256)  # 은닉층
        self.fc3 = nn.Linear(256, 256)  # 은닉층
        
        # Actor-Critic 구조
        self.actor = nn.Linear(256, len(ActionSpace))  # 액션 확률
        self.critic = nn.Linear(256, 1)  # 상태 가치
```

### 2. 액션 공간

AI가 선택할 수 있는 액션들:

- **CREATE_PLAYER**: 플레이어 캐릭터 생성
- **ADD_MOVEMENT**: 이동 메커니즘 추가
- **ADD_JUMPING**: 점프 기능 추가
- **CREATE_ENEMY**: 적 AI 생성
- **GENERATE_LEVEL**: 레벨 자동 생성
- **ADD_COLLISION**: 충돌 감지 추가
- **ADD_SCORE_SYSTEM**: 점수 시스템 구현
- **ADD_PARTICLE_EFFECT**: 파티클 효과 추가
- **OPTIMIZE_PERFORMANCE**: 성능 최적화

### 3. 보상 시스템

```python
# 기본 보상
base_reward = 10.0 if success else -5.0

# 중요 기능 보너스
if action in [CREATE_PLAYER, ADD_MOVEMENT, ADD_COLLISION]:
    base_reward *= 1.5

# 단계별 보너스
phase_bonus = {
    "initialization": 2.0,
    "core_mechanics": 3.0,
    "level_design": 2.5,
    "gameplay": 2.0,
    "polish": 1.5
}
```

## 게임 개발 프로세스

### 1. 개발 단계

1. **초기화 (Initialization)**
   - 프로젝트 구조 생성
   - 기본 Panda3D 앱 설정
   - 플레이어 캐릭터 생성

2. **핵심 메커니즘 (Core Mechanics)**
   - 이동 컨트롤 구현
   - 점프/중력 시스템
   - 충돌 감지

3. **레벨 디자인 (Level Design)**
   - 지형 생성
   - 적/장애물 배치
   - 수집 아이템 추가

4. **게임플레이 (Gameplay)**
   - 점수 시스템
   - UI/HUD 구현
   - 난이도 조절

5. **폴리싱 (Polish)**
   - 파티클 효과
   - 사운드 효과
   - 그래픽 개선

6. **최적화 (Optimization)**
   - 성능 개선
   - 버그 수정
   - 코드 리팩토링

### 2. 품질 평가 기준

```python
quality_criteria = {
    "has_player": 10,        # 플레이어 존재
    "has_movement": 10,      # 이동 가능
    "has_enemies": 10,       # 적 AI
    "has_collision": 10,     # 충돌 감지
    "has_level": 10,         # 레벨 디자인
    "has_ui": 5,            # UI 요소
    "has_score": 5,         # 점수 시스템
    "has_sound": 5,         # 사운드
    "has_particles": 5,     # 시각 효과
    "is_playable": 15,      # 플레이 가능
    "is_fun": 10            # 재미 요소
}
```

## API 레퍼런스

### AutoCIPanda3DSystem

```python
class AutoCIPanda3DSystem:
    async def create_game(project_name: str, game_type: str, development_hours: float) -> Dict
    async def modify_game(project_name: str, modification_request: str) -> Dict
    async def analyze_game(project_path: str) -> Dict
    async def start_monitoring(port: int = 5001) -> None
```

### Panda3DAIAgent

```python
class Panda3DAIAgent:
    async def start_development(target_hours: float = 24.0) -> None
    async def _execute_action(action: ActionSpace) -> bool
    def _evaluate_action(action: ActionSpace, success: bool) -> float
    def _update_quality_score() -> None
```

## 예제

### 1. 간단한 플랫폼 게임

```python
# examples/simple_platformer.py
import asyncio
from modules.autoci_panda3d_integration import AutoCIPanda3DSystem

async def main():
    system = AutoCIPanda3DSystem()
    
    # 30분 동안 플랫폼 게임 개발
    result = await system.create_game(
        project_name="QuickPlatformer",
        game_type="platformer", 
        development_hours=0.5
    )
    
    print(f"완성도: {result['completeness']}%")
    print(f"구현된 기능: {result['features']}")

asyncio.run(main())
```

### 2. 멀티플레이어 레이싱 게임

```python
# examples/multiplayer_racing.py
async def create_multiplayer_racing():
    system = AutoCIPanda3DSystem()
    
    # 기본 레이싱 게임 생성
    result = await system.create_game(
        project_name="RacingMultiplayer",
        game_type="racing",
        development_hours=2.0
    )
    
    # 멀티플레이어 기능 추가
    if result["success"]:
        mod_result = await system.modify_game(
            project_name="RacingMultiplayer",
            modification_request="Add Socket.IO multiplayer with 4 players support"
        )
```

### 3. 배치 게임 생성

```python
# examples/batch_creation.py
async def create_game_studio():
    system = AutoCIPanda3DSystem()
    
    games = [
        ("CasualPuzzle", "puzzle", 0.5),
        ("ActionShooter", "shooter", 1.0),
        ("RPGAdventure", "rpg", 2.0)
    ]
    
    # 동시에 여러 게임 개발
    tasks = [
        system.create_game(name, type, hours)
        for name, type, hours in games
    ]
    
    results = await asyncio.gather(*tasks)
    
    for game, result in zip(games, results):
        print(f"{game[0]}: 품질 {result['quality_score']}/100")
```

## 문제 해결

### GPU 메모리 부족
```bash
# CPU 모드로 실행
export AUTOCI_DEVICE="cpu"
```

### Panda3D 창이 열리지 않음
```bash
# Headless 모드 설정
export PANDA3D_WINDOW_HIDDEN=1
```

### Socket.IO 연결 실패
```bash
# 방화벽 포트 열기
sudo ufw allow 5001
```

## 추가 자료

- [Panda3D 공식 문서](https://docs.panda3d.org/)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Socket.IO Python 가이드](https://python-socketio.readthedocs.io/)
- [AutoCI 메인 문서](../README.md)