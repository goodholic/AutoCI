#!/usr/bin/env python3
"""
AutoCI 통합 연속 학습 시스템
기존 24시간 학습 시스템과 LLM 기반 Q&A 학습을 통합
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# LLM 관련 imports (조건부)
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        pipeline
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM 라이브러리가 설치되지 않았습니다. 기본 학습 모드만 사용 가능합니다.")

# 기존 학습 시스템 import
from .csharp_24h_user_learning import CSharp24HUserLearning, UserLearningSession
from .csharp_24h_learning_config import LearningConfig

@dataclass
class LearningTopic:
    """통합 학습 주제"""
    id: str
    category: str
    topic: str
    difficulty: int  # 1-5
    korean_keywords: List[str]
    csharp_concepts: List[str]
    godot_integration: Optional[str] = None
    traditional_content: Optional[str] = None  # 기존 학습 내용
    
@dataclass
class QASession:
    """Q&A 세션"""
    session_id: str
    question: Dict[str, Any]
    answer: Dict[str, Any]
    analysis: Dict[str, Any]
    timestamp: datetime
    model_used: Optional[str] = None

class CSharpContinuousLearning(CSharp24HUserLearning):
    """통합 연속 학습 시스템"""
    
    def __init__(self, use_llm: bool = True):
        super().__init__()
        self.use_llm = use_llm and LLM_AVAILABLE
        
        # LLM 관련 설정
        self.models_dir = Path("./models")
        self.continuous_learning_dir = self.user_data_dir / "continuous_learning"
        self.continuous_learning_dir.mkdir(exist_ok=True)
        
        # Q&A 데이터 디렉토리
        self.qa_dir = self.continuous_learning_dir / "qa_sessions"
        self.qa_dir.mkdir(exist_ok=True)
        
        # 진행 상태 디렉토리
        self.progress_dir = self.continuous_learning_dir / "progress"
        self.progress_dir.mkdir(exist_ok=True)
        
        # LLM 모델
        self.llm_models = {}
        if self.use_llm:
            self.load_llm_models()
            
        # 통합 학습 주제
        self.integrated_topics = self._create_integrated_topics()
        
        # Q&A 세션 기록
        self.qa_sessions: List[QASession] = []
        
        # 지식 베이스
        self.knowledge_base = self._load_knowledge_base()
        
        # 통합 진행 상태
        self.integrated_progress = self._load_integrated_progress()
        
    def _create_integrated_topics(self) -> List[LearningTopic]:
        """기존 커리큘럼과 LLM 학습 주제 통합"""
        topics = []
        
        # 기존 커리큘럼을 통합 주제로 변환
        for category, info in self.learning_curriculum.items():
            for i, topic in enumerate(info["topics"]):
                # 한글 키워드와 C# 개념 매핑
                korean_keywords, csharp_concepts = self._extract_keywords(topic)
                
                topics.append(LearningTopic(
                    id=f"{category}_{i}",
                    category=category,
                    topic=topic,
                    difficulty=self._get_difficulty(info["level"]),
                    korean_keywords=korean_keywords,
                    csharp_concepts=csharp_concepts,
                    godot_integration=self._get_godot_integration(category, topic),
                    traditional_content=f"Traditional learning content for {topic}"
                ))
                
        # Godot 네트워킹 및 엔진 개발 관련 주제 추가
        godot_topics = self._create_mirror_topics()  # 함수명은 호환성을 위해 유지
        topics.extend(godot_topics)
        
        return topics
        
    def _create_mirror_topics(self) -> List[LearningTopic]:
        """Godot 네트워킹 및 엔진 개발 관련 학습 주제 생성"""
        godot_topics = [
            # Godot 내장 네트워킹 기초
            LearningTopic(
                id="godot_net_basics_0",
                category="godot_networking",
                topic="Godot MultiplayerAPI 기초",
                difficulty=2,
                korean_keywords=["고닷", "네트워킹", "멀티플레이어", "API", "기초"],
                csharp_concepts=["MultiplayerAPI", "MultiplayerPeer", "ENetMultiplayerPeer", "WebSocketMultiplayerPeer"],
                godot_integration="Godot 내장 네트워킹 시스템 이해",
                traditional_content="Godot의 내장 MultiplayerAPI 기본 개념과 사용법"
            ),
            LearningTopic(
                id="godot_net_basics_1", 
                category="godot_networking",
                topic="Godot에서 서버와 클라이언트 생성",
                difficulty=2,
                korean_keywords=["서버", "클라이언트", "호스트", "연결", "포트"],
                csharp_concepts=["create_server", "create_client", "multiplayer_peer", "peer_connected", "peer_disconnected"],
                godot_integration="ENet과 WebSocket을 이용한 서버/클라이언트 구성",
                traditional_content="Godot에서 멀티플레이어 서버와 클라이언트 생성 방법"
            ),
            # Godot RPC와 동기화
            LearningTopic(
                id="godot_rpc_0",
                category="godot_rpc",
                topic="Godot RPC 시스템",
                difficulty=3,
                korean_keywords=["원격호출", "RPC", "동기화", "네트워크", "통신"],
                csharp_concepts=["@rpc", "call_local", "any_peer", "authority", "reliable"],
                godot_integration="Godot RPC 어노테이션과 메서드 호출",
                traditional_content="Godot의 RPC 시스템을 이용한 네트워크 통신"
            ),
            LearningTopic(
                id="godot_sync_0",
                category="godot_sync",
                topic="MultiplayerSynchronizer와 상태 동기화",
                difficulty=3,
                korean_keywords=["동기화", "상태", "복제", "보간", "예측"],
                csharp_concepts=["MultiplayerSynchronizer", "MultiplayerSpawner", "replication", "interpolation"],
                godot_integration="Godot 4의 새로운 동기화 노드 활용",
                traditional_content="MultiplayerSynchronizer를 이용한 자동 상태 동기화"
            ),
            # Godot 고급 네트워킹
            LearningTopic(
                id="godot_net_advanced_0",
                category="godot_advanced",
                topic="Godot에서 지연 보상과 예측",
                difficulty=4,
                korean_keywords=["지연보상", "예측", "보간", "롤백", "동기화"],
                csharp_concepts=["client_prediction", "server_reconciliation", "interpolation", "extrapolation"],
                godot_integration="클라이언트 사이드 예측과 서버 검증",
                traditional_content="네트워크 지연을 보상하는 고급 기법"
            ),
            LearningTopic(
                id="godot_net_advanced_1",
                category="godot_advanced",
                topic="Godot 네트워크 최적화",
                difficulty=4,
                korean_keywords=["최적화", "압축", "대역폭", "성능", "효율"],
                csharp_concepts=["compression", "delta_compression", "interest_management", "bandwidth"],
                godot_integration="네트워크 트래픽 최적화 기법",
                traditional_content="Godot에서 네트워크 성능 최적화 방법"
            ),
            # Godot 엔진 개발
            LearningTopic(
                id="godot_engine_dev_0",
                category="godot_engine",
                topic="Godot 엔진 아키텍처 이해",
                difficulty=4,
                korean_keywords=["엔진", "아키텍처", "구조", "설계", "시스템"],
                csharp_concepts=["engine_architecture", "scene_system", "node_system", "rendering_pipeline"],
                godot_integration="Godot 엔진 내부 구조와 확장 방법",
                traditional_content="Godot 엔진의 핵심 아키텍처와 설계 철학"
            ),
            LearningTopic(
                id="godot_engine_dev_1",
                category="godot_engine",
                topic="Godot 렌더링 파이프라인 개선",
                difficulty=5,
                korean_keywords=["렌더링", "그래픽", "셰이더", "최적화", "파이프라인"],
                csharp_concepts=["rendering_pipeline", "vulkan", "shader", "optimization", "gpu"],
                godot_integration="Godot 렌더링 시스템 이해와 최적화",
                traditional_content="Godot 렌더링 파이프라인 구조와 개선 방향"
            ),
            # Nakama 서버 개발 (NEW - 5번째 핵심 주제)
            LearningTopic(
                id="nakama_basics_0",
                category="nakama_server",
                topic="Nakama 서버 기초와 설치",
                difficulty=2,
                korean_keywords=["나카마", "서버", "설치", "설정", "구성"],
                csharp_concepts=["nakama_server", "docker", "configuration", "database", "setup"],
                godot_integration="Godot에서 Nakama 서버 연결",
                traditional_content="Nakama 오픈소스 게임 서버 설치 및 기본 설정"
            ),
            LearningTopic(
                id="nakama_basics_1",
                category="nakama_server", 
                topic="Nakama 인증과 세션 관리",
                difficulty=3,
                korean_keywords=["인증", "세션", "로그인", "사용자", "관리"],
                csharp_concepts=["authentication", "session", "user_management", "tokens", "devices"],
                godot_integration="Godot 클라이언트 인증 시스템",
                traditional_content="Nakama에서 사용자 인증과 세션 관리 방법"
            ),
            LearningTopic(
                id="nakama_matchmaking_0",
                category="nakama_matchmaking",
                topic="Nakama AI 기반 매치메이킹",
                difficulty=4,
                korean_keywords=["매치메이킹", "매칭", "로비", "스킬", "AI"],
                csharp_concepts=["matchmaking", "lobby", "skill_based", "ai_matching", "tournaments"],
                godot_integration="Godot에서 지능형 매치메이킹 시스템",
                traditional_content="AI 알고리즘을 활용한 스킬 기반 매치메이킹 구현"
            ),
            LearningTopic(
                id="nakama_storage_0",
                category="nakama_storage",
                topic="Nakama 스토리지와 데이터 관리",
                difficulty=3,
                korean_keywords=["스토리지", "데이터", "저장", "리더보드", "프로필"],
                csharp_concepts=["storage", "leaderboards", "user_data", "persistence", "records"],
                godot_integration="Godot 게임 데이터 자동 동기화",
                traditional_content="Nakama 스토리지 시스템과 리더보드 구현"
            ),
            LearningTopic(
                id="nakama_social_0",
                category="nakama_social",
                topic="Nakama 소셜 기능과 AI 모더레이션",
                difficulty=4,
                korean_keywords=["소셜", "친구", "채팅", "그룹", "AI모더레이션"],
                csharp_concepts=["social", "friends", "chat", "groups", "ai_moderation"],
                godot_integration="Godot에서 소셜 게임 기능 구현",
                traditional_content="AI 기반 자동 채팅 모더레이션과 소셜 시스템"
            ),
            LearningTopic(
                id="nakama_advanced_0",
                category="nakama_advanced",
                topic="Nakama 서버 확장과 AI 최적화",
                difficulty=5,
                korean_keywords=["확장", "최적화", "스케일링", "AI", "성능"],
                csharp_concepts=["scaling", "optimization", "ai_optimization", "server_runtime", "plugins"],
                godot_integration="대규모 멀티플레이어 게임 최적화",
                traditional_content="Nakama 서버 확장과 AI 기반 자동 최적화 시스템"
            ),
            # Godot AI 네트워킹
            LearningTopic(
                id="godot_ai_net_0",
                category="godot_ai",
                topic="AI 제어 Godot 네트워크 매니저",
                difficulty=4,
                korean_keywords=["인공지능", "자동화", "네트워크제어", "최적화", "분석"],
                csharp_concepts=["AINetworkManager", "MultiplayerAPI", "optimization", "automation"],
                godot_integration="AI가 제어하는 Godot 네트워킹",
                traditional_content="AI를 이용한 Godot 네트워크 자동 최적화"
            ),
            LearningTopic(
                id="godot_ai_net_1",
                category="godot_ai",
                topic="지능형 동기화와 예측 시스템",
                difficulty=5,
                korean_keywords=["지능형동기화", "예측", "머신러닝", "최적화", "적응형"],
                csharp_concepts=["IntelligentSync", "prediction", "ML", "adaptive", "optimization"],
                godot_integration="AI 기반 동적 네트워크 최적화",
                traditional_content="머신러닝을 활용한 네트워크 예측과 최적화"
            ),
            # Godot 엔진 발전 방향
            LearningTopic(
                id="godot_future_0",
                category="godot_future",
                topic="Godot 엔진의 미래 발전 방향",
                difficulty=4,
                korean_keywords=["미래", "발전", "로드맵", "개선", "방향성"],
                csharp_concepts=["roadmap", "features", "improvements", "community", "opensource"],
                godot_integration="차세대 Godot 기능과 개선사항",
                traditional_content="Godot 엔진이 나아가야 할 방향과 개선점"
            ),
            LearningTopic(
                id="godot_future_1",
                category="godot_future",
                topic="Godot 에디터 AI 통합",
                difficulty=5,
                korean_keywords=["에디터", "AI통합", "자동화", "어시스턴트", "생산성"],
                csharp_concepts=["editor_ai", "automation", "assistant", "productivity", "tooling"],
                godot_integration="AI 기반 개발 도구와 에디터 개선",
                traditional_content="Godot 에디터에 AI 기능 통합 방안"
            ),
            # Godot 성능과 최적화
            LearningTopic(
                id="godot_perf_0",
                category="godot_performance",
                topic="Godot 엔진 성능 최적화",
                difficulty=4,
                korean_keywords=["성능", "최적화", "프로파일링", "병목", "개선"],
                csharp_concepts=["performance", "optimization", "profiling", "bottleneck", "improvement"],
                godot_integration="Godot 성능 분석과 최적화 기법",
                traditional_content="Godot 엔진의 성능 최적화 방법론"
            ),
            LearningTopic(
                id="godot_platform_0",
                category="godot_platform",
                topic="Godot 크로스 플랫폼 개발",
                difficulty=4,
                korean_keywords=["크로스플랫폼", "모바일", "콘솔", "웹", "이식성"],
                csharp_concepts=["cross_platform", "mobile", "console", "web", "portability"],
                godot_integration="다양한 플랫폼을 위한 Godot 최적화",
                traditional_content="Godot의 멀티 플랫폼 지원과 최적화"
            ),
            # Nakama 서버 기초
            LearningTopic(
                id="nakama_basics_0",
                category="nakama_server",
                topic="Nakama 서버 기초와 설정",
                difficulty=3,
                korean_keywords=["나카마", "게임서버", "백엔드", "설정", "기초"],
                csharp_concepts=["Nakama", "IClient", "ISession", "ISocket", "authentication"],
                godot_integration="Godot에서 Nakama 클라이언트 연동",
                traditional_content="Nakama 오픈소스 게임 서버의 기본 개념과 설정"
            ),
            LearningTopic(
                id="nakama_basics_1",
                category="nakama_server",
                topic="Nakama 매치메이킹 시스템",
                difficulty=3,
                korean_keywords=["매치메이킹", "매칭", "로비", "방", "플레이어"],
                csharp_concepts=["IMatchmakerTicket", "IMatch", "matchmaking", "lobby", "room"],
                godot_integration="Godot에서 Nakama 매치메이킹 구현",
                traditional_content="Nakama의 실시간 매치메이킹과 로비 시스템"
            ),
            # Nakama 데이터 관리
            LearningTopic(
                id="nakama_data_0",
                category="nakama_data",
                topic="Nakama 스토리지와 데이터베이스",
                difficulty=4,
                korean_keywords=["스토리지", "데이터베이스", "저장", "조회", "관리"],
                csharp_concepts=["IApiStorageObject", "collection", "key-value", "query", "storage"],
                godot_integration="Godot에서 플레이어 데이터 저장/조회",
                traditional_content="Nakama의 스토리지 시스템과 데이터 관리"
            ),
            LearningTopic(
                id="nakama_data_1",
                category="nakama_data",
                topic="Nakama 리더보드와 토너먼트",
                difficulty=4,
                korean_keywords=["리더보드", "순위표", "토너먼트", "대회", "점수"],
                csharp_concepts=["IApiLeaderboard", "IApiTournament", "ranking", "score", "competition"],
                godot_integration="Godot 게임에 순위 시스템 통합",
                traditional_content="Nakama의 리더보드와 토너먼트 시스템"
            ),
            # Nakama 소셜 기능
            LearningTopic(
                id="nakama_social_0",
                category="nakama_social",
                topic="Nakama 친구와 그룹 시스템",
                difficulty=3,
                korean_keywords=["친구", "그룹", "길드", "소셜", "커뮤니티"],
                csharp_concepts=["IApiFriend", "IApiGroup", "social", "guild", "community"],
                godot_integration="Godot에서 소셜 기능 구현",
                traditional_content="Nakama의 소셜 네트워킹 기능"
            ),
            LearningTopic(
                id="nakama_social_1",
                category="nakama_social",
                topic="Nakama 실시간 채팅과 알림",
                difficulty=3,
                korean_keywords=["채팅", "메시지", "알림", "실시간", "통신"],
                csharp_concepts=["IChannelMessage", "INotification", "chat", "realtime", "messaging"],
                godot_integration="Godot에서 실시간 채팅 구현",
                traditional_content="Nakama의 채팅과 알림 시스템"
            ),
            # Nakama AI 통합
            LearningTopic(
                id="nakama_ai_0",
                category="nakama_ai",
                topic="AI 제어 Nakama 서버 관리",
                difficulty=5,
                korean_keywords=["AI서버관리", "자동화", "최적화", "모니터링", "분석"],
                csharp_concepts=["AIServerManager", "automation", "monitoring", "analytics", "optimization"],
                godot_integration="AI가 관리하는 Nakama 백엔드",
                traditional_content="AI를 활용한 Nakama 서버 자동 관리"
            ),
            LearningTopic(
                id="nakama_ai_1",
                category="nakama_ai",
                topic="Nakama 지능형 매치메이킹 AI",
                difficulty=5,
                korean_keywords=["AI매치메이킹", "스킬분석", "플레이스타일", "밸런싱", "예측"],
                csharp_concepts=["AIMatchmaker", "skill_analysis", "playstyle", "balancing", "prediction"],
                godot_integration="AI 기반 공정한 매치메이킹",
                traditional_content="머신러닝을 활용한 지능형 매치메이킹"
            ),
            # Nakama 고급 기능
            LearningTopic(
                id="nakama_advanced_0",
                category="nakama_advanced",
                topic="Nakama 서버 확장과 커스터마이징",
                difficulty=5,
                korean_keywords=["서버확장", "커스터마이징", "모듈", "플러그인", "스크립트"],
                csharp_concepts=["server_runtime", "modules", "hooks", "custom_logic", "TypeScript"],
                godot_integration="커스텀 게임 로직 구현",
                traditional_content="Nakama 서버의 확장과 커스터마이징"
            ),
            LearningTopic(
                id="nakama_advanced_1",
                category="nakama_advanced",
                topic="Nakama 대규모 멀티플레이어 최적화",
                difficulty=5,
                korean_keywords=["대규모", "MMO", "최적화", "확장성", "성능"],
                csharp_concepts=["scalability", "load_balancing", "sharding", "performance", "MMO"],
                godot_integration="대규모 멀티플레이어 게임 구현",
                traditional_content="Nakama로 대규모 멀티플레이어 게임 구축"
            )
        ]
        
        return godot_topics
    
    def _extract_keywords(self, topic: str) -> Tuple[List[str], List[str]]:
        """주제에서 한글 키워드와 C# 개념 추출"""
        keyword_map = {
            "변수와 타입": (["변수", "타입", "자료형", "선언"], ["int", "string", "bool", "var"]),
            "연산자": (["연산자", "계산", "논리"], ["operators", "+", "-", "&&", "||"]),
            "조건문": (["조건문", "if", "분기"], ["if", "else", "switch", "?:"]),
            "반복문": (["반복", "루프", "순환"], ["for", "while", "foreach", "do-while"]),
            "클래스": (["클래스", "객체", "인스턴스"], ["class", "object", "new", "constructor"]),
            "상속": (["상속", "부모", "자식"], ["inheritance", "base", "override", "virtual"]),
            "인터페이스": (["인터페이스", "구현", "계약"], ["interface", "implement", "contract"]),
            "제네릭": (["제네릭", "타입매개변수"], ["generic", "T", "where", "constraint"]),
            "LINQ": (["링크", "쿼리", "질의"], ["LINQ", "from", "select", "where"]),
            "async/await": (["비동기", "대기", "동시성"], ["async", "await", "Task", "concurrent"]),
            # Godot 관련 키워드 추가
            "Godot": (["고닷", "엔진", "게임엔진"], ["Godot", "Engine", "GameEngine"]),
            "MultiplayerAPI": (["멀티플레이어", "네트워킹", "API"], ["MultiplayerAPI", "networking", "multiplayer"]),
            "RPC": (["원격호출", "RPC", "통신"], ["@rpc", "remote", "call"]),
            "동기화": (["동기화", "상태", "복제"], ["sync", "replication", "state"]),
            "엔진개발": (["엔진개발", "확장", "개선"], ["engine_dev", "extension", "improvement"]),
            "네트워킹": (["네트워킹", "통신", "연결"], ["networking", "communication", "connection"]),
            "Nakama": (["나카마", "게임서버", "백엔드"], ["Nakama", "gameserver", "backend"]),
            "매치메이킹": (["매치메이킹", "매칭", "로비"], ["matchmaking", "lobby", "match"]),
            "리더보드": (["리더보드", "순위", "토너먼트"], ["leaderboard", "ranking", "tournament"]),
        }
        
        # 기본값
        default_korean = ["학습", "프로그래밍", "개발"]
        default_csharp = ["C#", "code", "programming"]
        
        for key, (korean, csharp) in keyword_map.items():
            if key in topic:
                return korean + default_korean, csharp + default_csharp
                
        return default_korean, default_csharp
        
    def _get_difficulty(self, level: str) -> int:
        """레벨을 난이도(1-5)로 변환"""
        level_map = {
            "beginner": 1,
            "intermediate": 3,
            "advanced": 4,
            "expert": 5
        }
        return level_map.get(level, 2)
        
    def _get_godot_integration(self, category: str, topic: str) -> Optional[str]:
        """Godot 통합 정보 반환"""
        if "godot" in category.lower():
            return f"Godot integration for {topic}"
        elif "게임" in topic:
            return "Game development pattern"
        return None
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """지식 베이스 로드"""
        kb_file = self.continuous_learning_dir / "knowledge_base.json"
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "csharp_patterns": {},
            "korean_translations": {},
            "godot_integrations": {},
            "godot_networking": {},
            "godot_engine_development": {},
            "network_optimizations": {},
            "common_errors": {},
            "best_practices": {},
            "user_progress": {}
        }
        
    def _save_knowledge_base(self):
        """지식 베이스 저장"""
        kb_file = self.continuous_learning_dir / "knowledge_base.json"
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            
    def _load_integrated_progress(self) -> Dict[str, Any]:
        """통합 진행 상태 로드"""
        progress_file = self.progress_dir / "integrated_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                self.logger.info(f"📚 기존 통합 학습 진행 상태를 로드했습니다.")
                self.logger.info(f"  - 총 학습 시간: {progress.get('total_hours', 0):.1f}시간")
                self.logger.info(f"  - 완료된 주제: {progress.get('topics_completed', 0)}개")
                return progress
        return {
            "total_hours": 0,
            "total_questions": 0,
            "total_successful": 0,
            "topics_completed": 0,
            "topics_progress": {},
            "sessions_history": [],
            "last_session_time": None,
            "knowledge_gained": {
                "csharp_basics": 0,
                "csharp_oop": 0,
                "csharp_advanced": 0,
                "korean_translation": 0,
                "korean_concepts": 0,
                "godot_architecture": 0,
                "godot_future": 0,
                "godot_networking": 0,
                "godot_ai_network": 0,
                "nakama_basics": 0,
                "nakama_ai": 0
            }
        }
        
    def _save_integrated_progress(self):
        """통합 진행 상태 저장"""
        # 현재 세션 정보 추가
        if hasattr(self, 'current_session_start'):
            session_duration = time.time() - self.current_session_start
            self.integrated_progress["total_hours"] += session_duration / 3600
            
        # 주제별 진행도 업데이트
        for topic in self.integrated_topics:
            if self._is_topic_completed(topic):
                topic_key = topic.category
                if topic_key not in self.integrated_progress["topics_progress"]:
                    self.integrated_progress["topics_progress"][topic_key] = {
                        "completed": 0,
                        "total": 0,
                        "last_studied": None
                    }
                self.integrated_progress["topics_progress"][topic_key]["completed"] += 1
                self.integrated_progress["topics_progress"][topic_key]["last_studied"] = datetime.now().isoformat()
        
        # 완료된 총 주제 수 계산
        self.integrated_progress["topics_completed"] = sum(
            info["completed"] for info in self.integrated_progress["topics_progress"].values()
        )
        
        self.integrated_progress["last_session_time"] = datetime.now().isoformat()
        
        # 파일로 저장
        progress_file = self.progress_dir / "integrated_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.integrated_progress, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"💾 통합 학습 진행 상태를 저장했습니다.")
        self.logger.info(f"  - 누적 학습 시간: {self.integrated_progress['total_hours']:.1f}시간")
        self.logger.info(f"  - 완료된 주제: {self.integrated_progress['topics_completed']}개")
            
    def load_llm_models(self):
        """LLM 모델 로드"""
        models_info_file = self.models_dir / "installed_models.json"
        if not models_info_file.exists():
            self.logger.warning("LLM 모델이 설치되지 않았습니다. 기본 학습 모드를 사용합니다.")
            self.use_llm = False
            return
            
        try:
            with open(models_info_file, 'r', encoding='utf-8') as f:
                installed_models = json.load(f)
                
            for model_name, info in installed_models.items():
                try:
                    self.logger.info(f"{model_name} 모델을 로드합니다...")
                    model_path = Path(info['path'])
                    
                    # 모델 파일 경로 확인
                    model_files_path = model_path / "model_files"
                    tokenizer_path = model_path / "tokenizer" if (model_path / "tokenizer").exists() else model_files_path
                    model_load_path = model_files_path if model_files_path.exists() else model_path
                    
                    # 토크나이저 로드
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(tokenizer_path),
                        trust_remote_code=True
                    )
                    
                    # 양자화 설정
                    quantization_config = self._get_quantization_config(info['quantization'])
                    
                    # 파이프라인 생성 (CPU 오프로드 포함)
                    pipe = pipeline(
                        "text-generation",
                        model=str(model_load_path),
                        tokenizer=tokenizer,
                        device_map="auto",
                        model_kwargs={
                            "quantization_config": quantization_config,
                            "torch_dtype": torch.bfloat16,
                            "llm_int8_enable_fp32_cpu_offload": True,
                            "low_cpu_mem_usage": True
                        },
                        max_new_tokens=500,
                        temperature=0.7,
                        do_sample=True
                    )
                    
                    self.llm_models[model_name] = {
                        "pipeline": pipe,
                        "features": info['features'],
                        "info": info
                    }
                    
                    self.logger.info(f"✓ {model_name} 로드 완료")
                    
                except Exception as e:
                    self.logger.error(f"✗ {model_name} 로드 실패: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"모델 정보 로드 실패: {str(e)}")
            self.use_llm = False
            
    def _get_quantization_config(self, quantization: str):
        """양자화 설정 반환"""
        if not LLM_AVAILABLE:
            return None
            
        # RTX 2080 8GB에 맞춰 4bit 양자화 강제 적용
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
    async def generate_llm_question(self, topic: LearningTopic) -> Optional[Dict[str, Any]]:
        """LLM을 위한 질문 생성"""
        if not self.use_llm or not self.llm_models:
            return None
            
        question_types = [
            "explain",      # 개념 설명
            "example",      # 예제 코드
            "translate",    # 한글-영어 번역
            "error",        # 오류 수정
            "optimize",     # 최적화
            "integrate"     # Godot 통합
        ]
        
        question_type = random.choice(question_types)
        
        # 질문 템플릿
        templates = {
            "explain": {
                "korean": f"{topic.topic}에 대해 한글로 자세히 설명해주세요. 특히 {random.choice(topic.korean_keywords)}에 초점을 맞춰주세요.",
                "english": f"Explain {topic.topic} in C# with focus on {random.choice(topic.csharp_concepts)}."
            },
            "example": {
                "korean": f"{topic.topic}을 사용하는 C# 코드 예제를 작성하고 한글로 설명해주세요.",
                "english": f"Write a C# code example demonstrating {topic.topic} with comments."
            },
            "translate": {
                "korean": f"다음 C# 개념을 한글로 번역하고 설명하세요: {random.choice(topic.csharp_concepts)}",
                "english": f"Translate and explain this Korean term in C# context: {random.choice(topic.korean_keywords)}"
            },
            "error": {
                "korean": f"{topic.topic} 관련 일반적인 오류와 해결방법을 한글로 설명해주세요.",
                "english": f"What are common errors with {topic.topic} in C# and how to fix them?"
            },
            "optimize": {
                "korean": f"{topic.topic}을 사용할 때 성능 최적화 방법을 한글로 설명해주세요.",
                "english": f"How to optimize performance when using {topic.topic} in C#?"
            },
            "integrate": {
                "korean": f"Godot에서 {topic.topic}을 어떻게 활용하는지 C# 코드와 함께 설명해주세요.",
                "english": f"How to use {topic.topic} in Godot with C#? Provide examples."
            }
        }
        
        # Nakama 관련 질문 추가
        if "nakama" in topic.category.lower() or "gameserver" in topic.category.lower():
            templates.update({
                "nakama_implement": {
                    "korean": f"Nakama 서버를 사용하여 {topic.topic}을 구현하는 방법을 C# 코드와 함께 설명해주세요.",
                    "english": f"How to implement {topic.topic} using Nakama server? Provide C# code examples."
                },
                "server_architecture": {
                    "korean": f"게임 서버에서 {topic.topic}을 구현할 때 고려해야 할 아키텍처 패턴을 설명해주세요.",
                    "english": f"Explain architecture patterns to consider when implementing {topic.topic} in game servers."
                },
                "godot_nakama_bridge": {
                    "korean": f"Godot와 Nakama를 연동할 때 {topic.topic}를 어떻게 처리하는지 설명해주세요.",
                    "english": f"How to handle {topic.topic} when integrating Godot with Nakama? Explain with examples."
                }
            })
            # Nakama 카테고리면 추가 질문 타입 포함
            question_types.extend(["nakama_implement", "server_architecture", "godot_nakama_bridge"])
        
        # 언어 선택 (한글 학습 강조)
        language = "korean" if random.random() < 0.7 else "english"
        question_text = templates[question_type][language]
        
        return {
            "id": f"{topic.id}_{question_type}_{int(time.time())}",
            "topic": topic.topic,
            "type": question_type,
            "language": language,
            "difficulty": topic.difficulty,
            "question": question_text,
            "keywords": topic.korean_keywords if language == "korean" else topic.csharp_concepts
        }
        
    async def ask_llm_model(self, model_name: str, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """LLM 모델에 질문"""
        if not self.use_llm or model_name not in self.llm_models:
            return None
            
        try:
            model_pipeline = self.llm_models[model_name]["pipeline"]
            
            # 프롬프트 구성
            system_prompt = f"""You are an expert in C# programming, Godot game development, Mirror Networking, game server architecture, and Korean language.
Your task is to answer questions about {question['topic']} accurately and educationally.
When answering in Korean, use proper technical terminology and clear explanations.
Always provide practical examples when possible.
If the question involves Mirror Networking or game servers, include specific implementation details and best practices."""
            
            full_prompt = f"{system_prompt}\n\nQuestion: {question['question']}\n\nAnswer:"
            
            # 모델 호출
            start_time = time.time()
            
            # AI 응답 생성 시작 알림
            self.logger.info(f"🤖 AI 응답 생성 시작: {model_name}")
            print(f"🤖 AI 응답 생성 중... (모델: {model_name})")
            
            response = model_pipeline(
                full_prompt,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=model_pipeline.tokenizer.eos_token_id
            )
            
            response_time = time.time() - start_time
            
            # 응답 완료 알림
            print(f"✅ AI 응답 생성 완료! (소요 시간: {response_time:.1f}초)")
            self.logger.info(f"AI 응답 완료: {response_time:.1f}초")
            
            answer_text = response[0]['generated_text'].split("Answer:")[-1].strip()
            
            return {
                "model": model_name,
                "question_id": question["id"],
                "answer": answer_text,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"LLM 모델 오류 {model_name}: {str(e)}")
            return None
            
    def analyze_llm_answer(self, question: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 답변 분석"""
        if not answer:
            return {"success": False, "error": "No answer"}
            
        analysis = {
            "success": True,
            "quality_score": 0,
            "extracted_knowledge": {},
            "new_patterns": [],
            "improvements": []
        }
        
        answer_text = answer.get("answer", "")
        
        # 답변 품질 평가
        quality_factors = {
            "length": len(answer_text) > 100,
            "has_code": "```" in answer_text or "class" in answer_text or "public" in answer_text,
            "has_korean": any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in answer_text),
            "has_explanation": any(word in answer_text.lower() for word in ["because", "therefore", "이유", "때문", "따라서"]),
            "has_example": any(word in answer_text.lower() for word in ["example", "예제", "예시", "다음"])
        }
        
        analysis["quality_score"] = sum(1 for factor in quality_factors.values() if factor) / len(quality_factors)
        
        # 지식 추출 및 저장
        if analysis["quality_score"] > 0.6:
            self._extract_knowledge(question, answer_text, analysis)
            
        return analysis
        
    def _extract_knowledge(self, question: Dict[str, Any], answer_text: str, analysis: Dict[str, Any]):
        """답변에서 지식 추출"""
        topic = question["topic"]
        
        if question["type"] == "translate" and "korean" in question["language"]:
            # 한글 번역 저장
            for keyword in question["keywords"]:
                if keyword in answer_text:
                    self.knowledge_base["korean_translations"][keyword] = answer_text[:200]
                    
        elif question["type"] == "example":
            # 코드 패턴 저장
            self.knowledge_base["csharp_patterns"][topic] = {
                "code": answer_text,
                "language": question["language"],
                "timestamp": datetime.now().isoformat()
            }
            
        elif question["type"] == "error":
            # 오류 패턴 저장
            self.knowledge_base["common_errors"][topic] = answer_text[:300]
            
        elif question["type"] == "integrate" and question.get("godot_integration"):
            # Godot 통합 정보 저장
            self.knowledge_base["godot_integrations"][topic] = answer_text[:400]
            
        elif question["type"] == "mirror_implement":
            # Mirror 구현 패턴 저장
            self.knowledge_base["mirror_networking"][topic] = {
                "implementation": answer_text,
                "timestamp": datetime.now().isoformat()
            }
            
        elif question["type"] == "server_architecture":
            # 게임 서버 아키텍처 패턴 저장
            self.knowledge_base["gameserver_patterns"][topic] = answer_text[:500]
            
        elif question["type"] == "optimize":
            # 네트워크 최적화 방법 저장
            if "mirror" in topic.lower() or "network" in topic.lower():
                self.knowledge_base["network_optimizations"][topic] = answer_text[:400]
            
    async def continuous_learning_session(self, topic: LearningTopic, use_traditional: bool = True, use_llm: bool = True):
        """통합 학습 세션"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🎯 학습 주제: {topic.topic} (난이도: {topic.difficulty}/5)")
        self.logger.info(f"📚 카테고리: {topic.category}")
        
        session_start = time.time()
        
        # 1. 전통적 학습 (기존 방식)
        if use_traditional:
            await self._traditional_learning(topic)
            
        # 2. LLM 기반 Q&A 학습
        if use_llm and self.use_llm and self.llm_models:
            await self._llm_qa_learning(topic)
            
        # 세션 완료
        session_duration = time.time() - session_start
        self.logger.info(f"\n✅ 세션 완료! 소요 시간: {LearningConfig.format_duration(session_duration)}")
        
        # 진행상황 저장
        self._save_session_progress(topic, session_duration)
        
    async def _traditional_learning(self, topic: LearningTopic):
        """전통적 학습 방식"""
        self.logger.info(f"\n📖 전통적 학습 시작...")
        
        # 기존 학습 내용 표시
        if topic.traditional_content:
            self.logger.info(f"내용: {topic.traditional_content}")
            
        # 키워드 학습
        self.logger.info(f"\n🔤 핵심 키워드:")
        self.logger.info(f"  한글: {', '.join(topic.korean_keywords)}")
        self.logger.info(f"  영어: {', '.join(topic.csharp_concepts)}")
        
        # 학습 시뮬레이션
        duration = random.randint(
            LearningConfig.SESSION_DURATION_MIN,
            LearningConfig.SESSION_DURATION_MAX
        )
        actual_duration = LearningConfig.get_actual_duration(duration)
        
        self.logger.info(f"\n⏱️  학습 시간: {LearningConfig.format_duration(actual_duration)}")
        await asyncio.sleep(actual_duration)
        
    async def _llm_qa_learning(self, topic: LearningTopic):
        """LLM 기반 Q&A 학습"""
        self.logger.info(f"\n🤖 AI Q&A 학습 시작...")
        
        # 사용 가능한 모델 확인
        available_models = list(self.llm_models.keys())
        if not available_models:
            self.logger.warning("사용 가능한 LLM 모델이 없습니다.")
            return
            
        # 2-3개의 질문 생성
        num_questions = random.randint(2, 3)
        
        for i in range(num_questions):
            # 질문 생성
            question = await self.generate_llm_question(topic)
            if not question:
                continue
                
            self.logger.info(f"\n❓ 질문 {i+1}/{num_questions}: {question['question'][:100]}...")
            
            # 모델 선택
            model_name = self._select_model_for_question(question, available_models)
            self.logger.info(f"📊 선택된 모델: {model_name}")
            
            # 답변 받기
            answer = await self.ask_llm_model(model_name, question)
            if answer:
                self.logger.info(f"💡 답변 받음 (응답 시간: {answer['response_time']:.1f}초)")
                
                # 답변 분석
                analysis = self.analyze_llm_answer(question, answer)
                self.logger.info(f"📈 답변 품질: {analysis['quality_score']*100:.0f}%")
                
                # Q&A 세션 저장
                qa_session = QASession(
                    session_id=f"{topic.id}_{int(time.time())}",
                    question=question,
                    answer=answer,
                    analysis=analysis,
                    timestamp=datetime.now(),
                    model_used=model_name
                )
                self.qa_sessions.append(qa_session)
                self._save_qa_session(qa_session)
                
                # AI 답변 학습 시간 확보
                if analysis['quality_score'] > 0.5 and answer.get('answer'):
                    answer_length = len(answer['answer'])
                    # 답변 길이에 따른 학습 시간 (100자당 2초, 최소 5초, 최대 30초)
                    learning_time = max(5.0, min(30.0, answer_length / 100 * 2))
                    
                    self.logger.info(f"\n📖 답변 학습 중... ({learning_time:.1f}초)")
                    print(f"📖 답변 학습 중... ({learning_time:.1f}초)")
                    
                    # 답변 내용 일부 표시
                    answer_preview = answer['answer'][:200]
                    if len(answer['answer']) > 200:
                        print(f"💭 학습 내용: {answer_preview}...")
                    else:
                        print(f"💭 학습 내용: {answer_preview}")
                    
                    await asyncio.sleep(learning_time)
                    print(f"✅ 학습 완료!")
                    self.logger.info(f"답변 학습 완료")
                
            # 다음 질문까지 짧은 대기
            await asyncio.sleep(random.uniform(3, 8))
            
    def _select_model_for_question(self, question: Dict[str, Any], available_models: List[str]) -> str:
        """질문에 적합한 모델 선택"""
        # 한글 질문
        if "korean" in question.get("language", ""):
            if "qwen2.5-coder-32b" in available_models:
                return "qwen2.5-coder-32b"
            elif "llama-3.1-8b" in available_models:
                return "llama-3.1-8b"
                
        # 코드 관련 질문
        if question["type"] in ["example", "error", "optimize"]:
            if "codellama-13b" in available_models:
                return "codellama-13b"
                
        # 랜덤 선택
        return random.choice(available_models)
        
    def _save_qa_session(self, qa_session: QASession):
        """Q&A 세션 저장"""
        # 날짜별 디렉토리
        today = datetime.now().strftime("%Y%m%d")
        daily_dir = self.qa_dir / today
        daily_dir.mkdir(exist_ok=True)
        
        # 파일 저장
        filename = f"{qa_session.session_id}.json"
        with open(daily_dir / filename, 'w', encoding='utf-8') as f:
            session_data = {
                "session_id": qa_session.session_id,
                "question": qa_session.question,
                "answer": qa_session.answer,
                "analysis": qa_session.analysis,
                "timestamp": qa_session.timestamp.isoformat(),
                "model_used": qa_session.model_used
            }
            json.dump(session_data, f, indent=2, ensure_ascii=False)
            
    def _save_session_progress(self, topic: LearningTopic, duration: float):
        """세션 진행상황 저장"""
        # 사용자 학습 세션 추가
        session = UserLearningSession(
            topic=topic.topic,
            level=topic.category,
            duration_minutes=int(duration / 60),
            start_time=datetime.now() - timedelta(seconds=duration),
            completion_rate=1.0,
            mastery_score=0.8,
            notes=f"Integrated learning session with {'LLM' if self.use_llm else 'traditional'} mode"
        )
        self.learning_sessions.append(session)
        
        # 진행상황 업데이트
        progress = self.load_progress()
        if topic.category not in progress["completed_topics"]:
            progress["completed_topics"][topic.category] = []
        if topic.topic not in progress["completed_topics"][topic.category]:
            progress["completed_topics"][topic.category].append(topic.topic)
            
        progress["total_time"] += duration
        progress["last_session"] = datetime.now().isoformat()
        
        # 지식 베이스에도 진행상황 저장
        self.knowledge_base["user_progress"][topic.id] = {
            "completed": True,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "qa_sessions": len([s for s in self.qa_sessions if topic.id in s.session_id])
        }
        
        self.save_progress(progress)
        self._save_knowledge_base()
        
    async def start_continuous_learning(self, hours: int = 24, use_traditional: bool = True, use_llm: bool = True):
        """연속 학습 시작"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 AutoCI 통합 연속 학습 시스템 시작!")
        self.logger.info(f"⏰ 학습 시간: {hours}시간")
        self.logger.info(f"📚 전통적 학습: {'활성화' if use_traditional else '비활성화'}")
        self.logger.info(f"🤖 AI Q&A 학습: {'활성화' if use_llm and self.use_llm else '비활성화'}")
        if self.use_llm:
            self.logger.info(f"🔧 사용 가능한 모델: {list(self.llm_models.keys())}")
        
        # 기존 진행 상태 표시
        if self.integrated_progress["total_hours"] > 0:
            self.logger.info(f"\n📊 기존 학습 진행 상태:")
            self.logger.info(f"  - 누적 학습 시간: {self.integrated_progress['total_hours']:.1f}시간")
            self.logger.info(f"  - 완료된 주제: {self.integrated_progress['topics_completed']}개")
            self.logger.info(f"  - 총 질문 수: {self.integrated_progress['total_questions']}")
            if self.integrated_progress['total_questions'] > 0:
                success_rate = (self.integrated_progress['total_successful'] / 
                              self.integrated_progress['total_questions'] * 100)
                self.logger.info(f"  - 전체 성공률: {success_rate:.1f}%")
            
            # 5가지 핵심 주제별 진행도
            self.logger.info(f"\n📚 5가지 핵심 주제 진행도:")
            core_categories = {
                "C# 프로그래밍": ["csharp_basics", "csharp_oop", "csharp_advanced"],
                "한글 용어": ["korean_translation", "korean_concepts"],
                "Godot 엔진": ["godot_architecture", "godot_future"],
                "Godot 네트워킹": ["godot_networking", "godot_ai_network"],
                "Nakama 서버": ["nakama_basics", "nakama_ai"]
            }
            
            for core_name, sub_categories in core_categories.items():
                total_progress = sum(
                    self.integrated_progress["knowledge_gained"].get(cat, 0) 
                    for cat in sub_categories
                )
                self.logger.info(f"  - {core_name}: {total_progress}개 학습")
                
        self.logger.info(f"{'='*60}\n")
        
        self.is_learning = True
        self.current_session_start = time.time()
        start_time = time.time()
        end_time = start_time + (hours * 3600)
        
        # 학습할 주제 선택
        remaining_topics = [t for t in self.integrated_topics if not self._is_topic_completed(t)]
        if not remaining_topics:
            remaining_topics = self.integrated_topics  # 모두 완료했으면 처음부터
            
        topic_index = 0
        save_counter = 0
        
        try:
            while time.time() < end_time and self.is_learning:
                # 현재 주제
                current_topic = remaining_topics[topic_index % len(remaining_topics)]
                
                # 학습 세션 실행
                await self.continuous_learning_session(
                    current_topic,
                    use_traditional=use_traditional,
                    use_llm=use_llm
                )
                
                # 다음 주제로
                topic_index += 1
                save_counter += 1
                
                # Q&A 세션 통계 업데이트
                if self.qa_sessions:
                    recent_qa = len([s for s in self.qa_sessions if s.timestamp > datetime.now() - timedelta(hours=1)])
                    self.integrated_progress["total_questions"] += recent_qa
                    # 성공한 Q&A 계산 (품질 점수 0.6 이상)
                    successful_qa = len([s for s in self.qa_sessions 
                                       if s.timestamp > datetime.now() - timedelta(hours=1) 
                                       and s.analysis.get("quality_score", 0) >= 0.6])
                    self.integrated_progress["total_successful"] += successful_qa
                
                # 주제별 지식 증가 추적
                if current_topic.category in ["csharp_basics", "csharp_oop", "csharp_advanced",
                                            "korean_translation", "korean_concepts",
                                            "godot_architecture", "godot_future",
                                            "godot_networking", "godot_ai_network",
                                            "nakama_basics", "nakama_ai"]:
                    self.integrated_progress["knowledge_gained"][current_topic.category] += 1
                
                # 10개 주제마다 진행 상태 저장
                if save_counter % 10 == 0:
                    self._save_integrated_progress()
                    self.logger.info(f"\n💾 진행 상태 자동 저장 완료")
                
                # 휴식 시간
                if topic_index % 3 == 0:  # 3개 주제마다 긴 휴식
                    break_time = LearningConfig.get_actual_duration(LearningConfig.BREAK_BETWEEN_BLOCKS)
                    self.logger.info(f"\n☕ 휴식 시간: {LearningConfig.format_duration(break_time)}")
                    await asyncio.sleep(break_time)
                else:
                    await asyncio.sleep(random.uniform(10, 30))
                    
                # 진행상황 업데이트
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                progress = (elapsed / (hours * 3600)) * 100
                
                self.logger.info(f"\n📊 전체 진행률: {progress:.1f}%")
                self.logger.info(f"⏱️  남은 시간: {LearningConfig.format_duration(remaining)}")
                
        except KeyboardInterrupt:
            self.logger.info("\n\n⚠️  학습이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"\n\n❌ 학습 중 오류 발생: {str(e)}")
        finally:
            self.is_learning = False
            total_time = time.time() - start_time
            
            # 최종 진행 상태 저장
            self._save_integrated_progress()
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎉 학습 세션 종료!")
            self.logger.info(f"⏱️  총 학습 시간: {LearningConfig.format_duration(total_time)}")
            self.logger.info(f"📚 완료한 주제: {topic_index}개")
            if self.qa_sessions:
                self.logger.info(f"🤖 Q&A 세션: {len(self.qa_sessions)}개")
                
            # 누적 통계 표시
            self.logger.info(f"\n📊 누적 학습 통계:")
            self.logger.info(f"  - 총 누적 시간: {self.integrated_progress['total_hours']:.1f}시간")
            self.logger.info(f"  - 전체 완료 주제: {self.integrated_progress['topics_completed']}개")
            self.logger.info(f"  - 전체 질문 수: {self.integrated_progress['total_questions']}")
            if self.integrated_progress['total_questions'] > 0:
                overall_success = (self.integrated_progress['total_successful'] / 
                                 self.integrated_progress['total_questions'] * 100)
                self.logger.info(f"  - 전체 성공률: {overall_success:.1f}%")
            
            self.logger.info(f"{'='*60}\n")
            
            # 최종 보고서 생성
            self.generate_final_report()
            
    def _is_topic_completed(self, topic: LearningTopic) -> bool:
        """주제 완료 여부 확인"""
        return topic.id in self.knowledge_base.get("user_progress", {})
        
    def generate_final_report(self):
        """최종 학습 보고서 생성"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_sessions": len(self.learning_sessions),
            "total_learning_time": sum(s.duration_minutes * 60 for s in self.learning_sessions),
            "topics_completed": len([t for t in self.integrated_topics if self._is_topic_completed(t)]),
            "qa_sessions": len(self.qa_sessions),
            "knowledge_base": {
                "korean_translations": len(self.knowledge_base.get("korean_translations", {})),
                "csharp_patterns": len(self.knowledge_base.get("csharp_patterns", {})),
                "common_errors": len(self.knowledge_base.get("common_errors", {})),
                "godot_integrations": len(self.knowledge_base.get("godot_integrations", {})),
                "mirror_networking": len(self.knowledge_base.get("mirror_networking", {})),
                "gameserver_patterns": len(self.knowledge_base.get("gameserver_patterns", {})),
                "network_optimizations": len(self.knowledge_base.get("network_optimizations", {}))
            }
        }
        
        # 모델별 사용 통계
        if self.qa_sessions:
            model_usage = {}
            for session in self.qa_sessions:
                model = session.model_used
                if model:
                    model_usage[model] = model_usage.get(model, 0) + 1
            report["model_usage"] = model_usage
            
        # 보고서 저장
        report_file = self.continuous_learning_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"\n📊 최종 보고서가 생성되었습니다: {report_file}")
        
        # 요약 출력
        self.logger.info(f"\n📈 학습 요약:")
        self.logger.info(f"  - 총 학습 시간: {LearningConfig.format_duration(report['total_learning_time'])}")
        self.logger.info(f"  - 완료한 주제: {report['topics_completed']}개")
        if report.get("qa_sessions"):
            self.logger.info(f"  - Q&A 세션: {report['qa_sessions']}개")
        self.logger.info(f"  - 지식 베이스:")
        for key, count in report["knowledge_base"].items():
            if count > 0:
                self.logger.info(f"    • {key}: {count}개")
                
# 편의 함수들
async def start_continuous_learning(hours: int = 24, use_traditional: bool = True, use_llm: bool = True):
    """통합 연속 학습 시작"""
    system = CSharpContinuousLearning(use_llm=use_llm)
    await system.start_continuous_learning(hours, use_traditional, use_llm)
    
async def quick_llm_session(topic_name: str = None):
    """빠른 LLM Q&A 세션"""
    system = CSharpContinuousLearning(use_llm=True)
    
    if not system.llm_models:
        logging.error("LLM 모델이 없습니다. install_llm_models.py를 먼저 실행하세요.")
        return
        
    # 주제 선택
    if topic_name:
        topics = [t for t in system.integrated_topics if topic_name.lower() in t.topic.lower()]
        if not topics:
            logging.error(f"'{topic_name}' 관련 주제를 찾을 수 없습니다.")
            return
        topic = topics[0]
    else:
        topic = random.choice(system.integrated_topics)
        
    # 단일 세션 실행
    await system.continuous_learning_session(topic, use_traditional=False, use_llm=True)