# Mirror Networking AI 통합 가이드

## 개요

AutoCI가 이제 Unity의 Mirror Networking과 Godot을 동시에 제어하여 AI 기반 멀티플레이어 게임을 자동으로 개발할 수 있습니다.

## 주요 기능

### 1. AI 네트워크 매니저
- 자동 서버/클라이언트 관리
- 플레이어 연결 최적화
- 동적 네트워크 설정

### 2. 지능형 동기화 시스템
- 네트워크 상태 기반 동기화 전략
- 자동 보간 및 예측
- 레이턴시 기반 최적화

### 3. Unity-Godot 브릿지
- 실시간 양방향 통신
- UDP 소켓 기반 메시징
- 자동 데이터 변환

### 4. AI 네트워크 최적화
- 실시간 성능 모니터링
- 자동 대역폭 조정
- 지능형 패킷 우선순위

## 설치 및 설정

### 1단계: Mirror Networking 설치
```bash
autoci mirror install
```

### 2단계: AI 모듈 설정
```bash
autoci mirror setup
```

### 3단계: 상태 확인
```bash
autoci mirror status
```

## 사용 방법

### 멀티플레이어 게임 생성

#### FPS 게임
```bash
autoci create multiplayer fps
```
- 자동 히트박스 동기화
- 레이캐스팅 네트워크 최적화
- 무기 시스템 동기화

#### MOBA 게임
```bash
autoci create multiplayer moba
```
- AI 기반 스킬 시스템
- 미니맵 동기화
- 팀 매칭 시스템

#### Racing 게임
```bash
autoci create multiplayer racing
```
- 물리 예측 시스템
- 위치 보간
- 랩 타임 동기화

### AI 기능 활성화

#### 네트워크 매니저
```bash
autoci mirror ai-network
```

#### 지능형 동기화
```bash
autoci mirror ai-sync
```

#### Godot 브릿지
```bash
autoci mirror godot-bridge
```

## 아키텍처

### 시스템 구조
```
┌─────────────┐     UDP Socket      ┌─────────────┐
│   Unity     │◄─────7777 포트─────►│   Godot     │
│  (Mirror)   │                     │  (Network)  │
└──────┬──────┘                     └──────┬──────┘
       │                                   │
       └──────────►┌──────────┐◄───────────┘
                  │  AutoCI   │
                  │    AI     │
                  └──────────┘
```

### 통신 프로토콜
1. **Unity → Godot**: Mirror 이벤트를 Godot 시그널로 변환
2. **Godot → Unity**: Godot 상태를 Mirror RPC로 전송
3. **AI 제어**: 양쪽 엔진에 최적화된 명령 전송

## 코드 예시

### Unity (C#) - AI Network Manager
```csharp
public class AINetworkManager : NetworkManager
{
    private AIController aiController;
    
    public override void OnServerConnect(NetworkConnectionToClient conn)
    {
        base.OnServerConnect(conn);
        aiController.OptimizePlayerConnection(conn);
    }
    
    public override void OnServerDisconnect(NetworkConnectionToClient conn)
    {
        base.OnServerDisconnect(conn);
        aiController.HandlePlayerDisconnection(conn);
    }
}
```

### Godot (GDScript) - Mirror Bridge
```gdscript
extends Node

var mirror_bridge: UDPServer
var ai_controller: Node

func _ready():
    mirror_bridge = UDPServer.new()
    mirror_bridge.listen(7778)
    ai_controller = preload("res://ai/GodotAIController.gd").new()

func _process(_delta):
    if mirror_bridge.is_connection_available():
        var peer = mirror_bridge.take_connection()
        handle_mirror_message(peer)
```

## 성능 최적화

### AI 자동 최적화
- 네트워크 트래픽 분석
- 동적 업데이트 빈도 조정
- 스마트 LOD 시스템

### 모니터링
```bash
autoci mirror monitor
```
- 실시간 네트워크 상태
- 레이턴시 그래프
- 패킷 손실률

### 최적화 실행
```bash
autoci mirror optimize
```
- AI가 자동으로 네트워크 설정 조정
- 최적의 동기화 전략 선택

## 문제 해결

### Mirror 설치 실패
```bash
# Git이 설치되어 있는지 확인
git --version

# 수동 설치
cd mirror_networking
git clone https://github.com/MirrorNetworking/Mirror.git
```

### 포트 충돌
- 기본 포트: 7777 (Mirror), 7778 (Godot Bridge)
- 환경 변수로 변경 가능:
  ```bash
  export MIRROR_PORT=8888
  export GODOT_BRIDGE_PORT=8889
  ```

### 동기화 문제
- `autoci mirror ai-sync` 재실행
- 네트워크 상태 확인
- AI 로그 분석

## 고급 기능

### 커스텀 동기화 전략
AI가 게임 타입에 따라 자동으로 선택:
- **FPS**: 높은 빈도, 낮은 지연
- **MOBA**: 중간 빈도, 상태 우선
- **Racing**: 물리 예측, 보간 중심

### 확장 가능한 AI 모듈
- 새로운 게임 타입 추가 가능
- 커스텀 네트워크 최적화 규칙
- 플러그인 시스템 지원

## 로드맵

### 계획된 기능
1. **고급 매치메이킹**: AI 기반 플레이어 매칭
2. **자동 밸런싱**: 게임플레이 데이터 분석
3. **클라우드 통합**: 자동 서버 배포
4. **모바일 지원**: 크로스 플랫폼 최적화

### 기여하기
- GitHub Issues로 버그 리포트
- Pull Request 환영
- 커뮤니티 피드백 수렴

---

**버전**: 1.0  
**최종 업데이트**: 2025년 6월 30일  
**라이선스**: MIT