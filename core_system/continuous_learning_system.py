#!/usr/bin/env python3
"""
AutoCI 24시간 연속 학습 시스템 - 저사양 최적화 버전
C#과 한글에 대해 지속적으로 학습하는 시스템
Llama-3.1-8B, CodeLlama-13B, Qwen2.5-Coder-32B 모델을 활용

RTX 2080 GPU 8GB, 32GB 메모리 환경에 최적화됨
autoci learn low 명령어에서 사용됩니다.
Cross-platform support for Windows and WSL
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
import gc
import psutil
import platform
from datetime import datetime, timedelta
from pathlib import Path

# Platform-specific path setup
def get_project_root():
    """Get project root path based on platform"""
    if platform.system() == "Windows":
        # Windows: use script's parent directory
        return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    else:
        # WSL/Linux: use configured path
        return Path("/mnt/d/AutoCI/AutoCI")

# Add project paths
PROJECT_ROOT = get_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'modules'))
sys.path.insert(0, str(PROJECT_ROOT / 'modules_active'))
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
try:
    from transformers.utils.quantization_config import BitsAndBytesConfig
except ImportError:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None
        
try:
    from transformers.pipelines import pipeline
except ImportError:
    try:
        from transformers import pipeline
    except ImportError:
        pipeline = None

# 정보 수집기 임포트
try:
    from modules.intelligent_information_gatherer import get_information_gatherer
    INFORMATION_GATHERER_AVAILABLE = True
    print("🌐 지능형 정보 수집기 활성화!")
except ImportError:
    INFORMATION_GATHERER_AVAILABLE = False
    print("⚠️ 정보 수집기를 로드할 수 없습니다")

# 🎮 AI 모델 완전 제어 시스템 임포트
try:
    from modules.ai_model_controller import AIModelController
    MODEL_CONTROLLER_AVAILABLE = True
    print("🎮 AI 모델 완전 제어 시스템 활성화!")
except ImportError:
    MODEL_CONTROLLER_AVAILABLE = False
    print("⚠️ AI 모델 컨트롤러를 로드할 수 없습니다")

# 로깅 설정
log_file = PROJECT_ROOT / 'continuous_learning.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# PyTorch 딥러닝 모듈 임포트
try:
    from modules.pytorch_deep_learning_module import AutoCIPyTorchLearningSystem
    PYTORCH_AVAILABLE = True
    print("🧠 PyTorch 딥러닝 모듈 활성화!")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch 딥러닝 모듈을 로드할 수 없습니다")

# 지시-응답 데이터셋 빌더 임포트
try:
    from modules.instruction_response_dataset_builder import get_dataset_builder
    DATASET_BUILDER_AVAILABLE = True
    print("📚 지시-응답 데이터셋 빌더 활성화!")
except ImportError:
    DATASET_BUILDER_AVAILABLE = False
    print("⚠️ 데이터셋 빌더를 로드할 수 없습니다")

# 학습 품질 모니터 임포트
try:
    from modules.learning_quality_monitor import get_quality_monitor
    QUALITY_MONITOR_AVAILABLE = True
    print("📊 학습 품질 모니터 활성화!")
except ImportError:
    QUALITY_MONITOR_AVAILABLE = False
    print("⚠️ 품질 모니터를 로드할 수 없습니다")

# Hugging Face 토큰
HF_TOKEN = "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"

# PyTorch 품질 평가 모델
class QualityAssessmentModel(nn.Module):
    """답변 품질을 평가하는 신경망 모델"""
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU()
        )
        self.quality_head = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        quality_score = torch.sigmoid(self.quality_head(features))
        return quality_score

# Experience Replay Buffer
class ExperienceReplayBuffer:
    """학습 경험을 저장하고 재사용하는 버퍼"""
    def __init__(self, capacity=10000, prioritized=True):
        self.buffer = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.prioritized = prioritized
        self.alpha = 0.6  # prioritization 강도
        self.beta = 0.4   # importance sampling
        self.epsilon = 1e-6
        self.pos = 0
        
    def add(self, experience, td_error=None):
        """경험 추가"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < len(self.priorities):
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        # Priority 설정
        priority = max_priority if td_error is None else (abs(td_error) + self.epsilon) ** self.alpha
        self.priorities[self.pos] = priority
        
        self.pos = (self.pos + 1) % len(self.priorities)
        
    def sample(self, batch_size):
        """배치 샘플링"""
        if len(self.buffer) == 0:
            return None, None, None
            
        if self.prioritized:
            # Priority-based sampling
            probs = self.priorities[:len(self.buffer)]
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            
            return samples, weights, indices
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
            samples = [self.buffer[idx] for idx in indices]
            return samples, np.ones(batch_size), indices
            
    def update_priorities(self, indices, td_errors):
        """Priority 업데이트"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

# 멀티태스크 학습 네트워크
class MultiTaskLearningNetwork(nn.Module):
    """5가지 핵심 주제에 특화된 멀티태스크 학습 네트워크"""
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        # 공유 인코더
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 태스크별 전문화 헤드
        self.task_heads = nn.ModuleDict({
            'csharp': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ),
            'korean': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ),
            'godot': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ),
            'socketio': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            ),
            'ai_optimization': nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        })
        
        # 출력 레이어
        self.output_layer = nn.Linear(128, 1)
        
    def forward(self, x, task):
        # 공유 특징 추출
        shared_features = self.shared_encoder(x)
        
        # 태스크별 특징 추출
        if task in self.task_heads:
            task_features = self.task_heads[task](shared_features)
        else:
            # 기본 태스크 (알 수 없는 태스크)
            task_features = self.task_heads['godot'](shared_features)
            
        # 최종 출력
        output = self.output_layer(task_features)
        return torch.sigmoid(output)

# 실시간 피드백 시스템
class RealtimeFeedbackSystem:
    """답변 생성 중 실시간으로 품질을 모니터링하고 개선"""
    def __init__(self, quality_threshold=0.7):
        self.quality_threshold = quality_threshold
        self.feedback_history = deque(maxlen=100)
        self.improvement_strategies = {
            'low_quality': self._improve_low_quality,
            'incomplete': self._improve_incomplete,
            'off_topic': self._improve_off_topic,
            'language_mix': self._improve_language_mix
        }
        
    def analyze_partial_response(self, partial_response, expected_language='korean'):
        """부분 응답 분석"""
        issues = []
        
        # 품질 체크
        if len(partial_response) < 50:
            issues.append('incomplete')
            
        # 언어 혼용 체크
        has_korean = any('\u3131' <= char <= '\u3163' or '\uac00' <= char <= '\ud7a3' for char in partial_response)
        has_english = any('a' <= char.lower() <= 'z' for char in partial_response)
        
        if expected_language == 'korean' and has_english and not has_korean:
            issues.append('language_mix')
        elif expected_language == 'english' and has_korean:
            issues.append('language_mix')
            
        # 코드 완성도 체크
        if '```' in partial_response and partial_response.count('```') % 2 != 0:
            issues.append('incomplete')
            
        return issues
        
    def _improve_low_quality(self, response, context):
        """낮은 품질 개선"""
        prompt_addon = "\n\n더 구체적이고 실용적인 예제를 포함해주세요."
        return prompt_addon
        
    def _improve_incomplete(self, response, context):
        """불완전한 답변 개선"""
        prompt_addon = "\n\n답변을 완성해주세요. 코드 예제가 있다면 완전한 형태로 제공해주세요."
        return prompt_addon
        
    def _improve_off_topic(self, response, context):
        """주제 벗어남 개선"""
        prompt_addon = f"\n\n{context.get('topic', '주제')}에 맞는 답변을 제공해주세요."
        return prompt_addon
        
    def _improve_language_mix(self, response, context):
        """언어 혼용 개선"""
        expected_lang = context.get('language', 'korean')
        if expected_lang == 'korean':
            prompt_addon = "\n\n한국어로 일관되게 답변해주세요."
        else:
            prompt_addon = "\n\nPlease answer consistently in English."
        return prompt_addon
        
    def get_improvement_prompt(self, issues, response, context):
        """개선 프롬프트 생성"""
        prompts = []
        for issue in issues:
            if issue in self.improvement_strategies:
                prompt = self.improvement_strategies[issue](response, context)
                prompts.append(prompt)
                
        return " ".join(prompts)
        
    def record_feedback(self, response, quality_score, issues):
        """피드백 기록"""
        self.feedback_history.append({
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'issues': issues,
            'response_length': len(response)
        })

@dataclass
class LearningTopic:
    """학습 주제"""
    id: str
    category: str
    topic: str
    difficulty: int  # 1-5
    korean_keywords: List[str]
    python_concepts: List[str]
    godot_integration: Optional[str] = None
    
@dataclass
class LearningSession:
    """학습 세션"""
    session_id: str
    start_time: datetime
    topics_covered: List[str]
    questions_asked: int
    successful_answers: int
    models_used: Dict[str, int]
    knowledge_gained: List[Dict[str, Any]]
    
class ContinuousLearningSystem:
    def __init__(self, models_dir: str = None, learning_dir: str = None, max_memory_gb: float = 32.0):
        # Platform-specific default paths
        if models_dir is None:
            models_dir = str(PROJECT_ROOT / "models")
        if learning_dir is None:
            learning_dir = str(PROJECT_ROOT / "continuous_learning")
        
        self.models_dir = Path(models_dir)
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(exist_ok=True, parents=True)
        
        # 🎮 AI 모델 완전 제어 시스템 초기화
        if MODEL_CONTROLLER_AVAILABLE:
            self.model_controller = AIModelController()
            print("🎯 AI 모델 조종권 확보 완료!")
            logger.info("🎮 AI 모델 완전 제어 시스템이 활성화되었습니다.")
        else:
            self.model_controller = None
            logger.warning("⚠️ AI 모델 컨트롤러 없이 실행됩니다.")
        
        # 메모리 관리 설정 (저사양 최적화)
        self.max_memory_gb = max_memory_gb
        self.memory_threshold = 0.70  # 70% 사용 시 모델 언로드 (더 보수적)
        self.currently_loaded_model = None
        self.model_cache = {}  # 언로드된 모델 정보 저장
        
        # 실패한 모델 블랙리스트 (현재 세션 동안만 유효)
        self.failed_models_blacklist = set()  # 로딩 실패한 모델 추적
        
        # 🆕 PyTorch 기반 개선 시스템 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 PyTorch 디바이스: {self.device}")
        
        # 품질 평가 모델
        self.quality_assessor = QualityAssessmentModel().to(self.device)
        self.quality_optimizer = torch.optim.Adam(self.quality_assessor.parameters(), lr=0.001)
        
        # Experience Replay Buffer
        self.experience_buffer = ExperienceReplayBuffer(capacity=10000, prioritized=True)
        logger.info("📦 Experience Replay Buffer 초기화 완료 (용량: 10,000)")
        
        # 멀티태스크 학습 네트워크
        self.multitask_network = MultiTaskLearningNetwork().to(self.device)
        self.multitask_optimizer = torch.optim.Adam(self.multitask_network.parameters(), lr=0.0003)
        
        # 실시간 피드백 시스템
        self.feedback_system = RealtimeFeedbackSystem(quality_threshold=0.7)
        logger.info("🔄 실시간 피드백 시스템 활성화")
        
        # 학습률 스케줄러
        self.quality_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.quality_optimizer, mode='max', patience=10, factor=0.5
        )
        self.multitask_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.multitask_optimizer, mode='max', patience=10, factor=0.5
        )
        
        # 학습 데이터 디렉토리
        self.questions_dir = self.learning_dir / "questions"
        self.answers_dir = self.learning_dir / "answers"
        self.knowledge_base_dir = self.learning_dir / "knowledge_base"
        self.progress_dir = self.learning_dir / "progress"
        
        for dir in [self.questions_dir, self.answers_dir, self.knowledge_base_dir, self.progress_dir]:
            dir.mkdir(exist_ok=True)
            
        # 모델 정보 (실제 로딩은 필요할 때만)
        self.available_models = {}
        self.load_model_info()
        
        # 학습 주제 정의
        self.learning_topics = self._initialize_learning_topics()
        
        # 현재 세션
        self.current_session = None
        
        # 지식 베이스
        self.knowledge_base = self._load_knowledge_base()
        
        # 메모리 사용량 모니터링
        self.memory_usage_history = []
        
        # 학습 진행 상태 로드
        self.learning_progress = self._load_learning_progress()
        
        # 진행형 학습 관리자
        try:
            from modules.progressive_learning_manager import ProgressiveLearningManager
            self.progressive_manager = ProgressiveLearningManager(self.learning_dir)
            logger.info("📈 진행형 학습 관리자 활성화")
        except:
            self.progressive_manager = None
            logger.warning("⚠️ 진행형 학습 관리자를 사용할 수 없습니다")
        
        # 질문 히스토리 추적 시스템
        self.question_history = self._load_question_history()
        
        # 지시-응답 데이터셋 빌더
        self.dataset_builder = None
        if DATASET_BUILDER_AVAILABLE:
            self.dataset_builder = get_dataset_builder()
            logger.info("📚 지시-응답 데이터셋 빌더 연결됨")
        
        # 학습 품질 모니터
        self.quality_monitor = None
        if QUALITY_MONITOR_AVAILABLE:
            self.quality_monitor = get_quality_monitor()
            logger.info("📊 학습 품질 모니터 연결됨")
        self.question_similarity_threshold = 0.85  # 유사도 임계값
        self.max_history_size = 10000  # 최대 히스토리 크기
        
        # PyTorch 딥러닝 시스템 초기화
        if PYTORCH_AVAILABLE:
            try:
                self.pytorch_system = AutoCIPyTorchLearningSystem(base_path=str(Path(__file__).parent.parent))
                logger.info("🧠 PyTorch 딥러닝 시스템 초기화 완료!")
                
                # 기존 모델이 있으면 로드
                pytorch_models_dir = Path(self.models_dir).parent / "pytorch_models"
                if pytorch_models_dir.exists():
                    latest_dl_model = sorted(pytorch_models_dir.glob("deep_learning_model_*.pth"))
                    latest_rl_model = sorted(pytorch_models_dir.glob("rl_agent_*.pth"))
                    
                    if latest_dl_model:
                        self.pytorch_system.load_models(dl_path=str(latest_dl_model[-1]))
                    if latest_rl_model:
                        self.pytorch_system.load_models(rl_path=str(latest_rl_model[-1]))
                        
            except Exception as e:
                self.pytorch_system = None
                logger.error(f"PyTorch 시스템 초기화 실패: {str(e)}")
        else:
            self.pytorch_system = None
            
        # 저장된 PyTorch 모델 로드
        self.load_pytorch_models()
        
    def _initialize_learning_topics(self) -> List[LearningTopic]:
        """5가지 핵심 학습 주제 초기화 (DeepSeek-coder 최적화)"""
        topics = [
            # 1️⃣ C# 프로그래밍 (변형된 Godot용)
            LearningTopic("csharp_basics", "C# 프로그래밍", "C# 기초 문법", 1,
                         ["변수", "데이터타입", "메서드", "클래스"],
                         ["variable", "datatype", "method", "class"],
                         "Godot C# 기초"),
            LearningTopic("csharp_advanced", "C# 프로그래밍", "C# 고급 기능", 3,
                         ["델리게이트", "이벤트", "LINQ", "async/await"],
                         ["delegate", "event", "LINQ", "async"],
                         "Godot C# 고급 기능"),
            # 2️⃣ 한글 프로그래밍 용어
            LearningTopic("korean_terms_basic", "한글 용어", "프로그래밍 기본 용어", 1,
                         ["변수", "함수", "클래스", "객체", "상속"],
                         ["variable", "function", "class", "object", "inheritance"],
                         "한-영 용어 매핑"),
            LearningTopic("korean_terms_advanced", "한글 용어", "고급 프로그래밍 용어", 3,
                         ["다형성", "캡슐화", "추상화", "인터페이스"],
                         ["polymorphism", "encapsulation", "abstraction", "interface"],
                         "전문 용어 이해"),
            # 3️⃣ 변형된 Godot 엔진
            LearningTopic("godot_basics", "변형된 Godot", "Godot 기초", 2,
                         ["노드", "씬", "시그널", "스크립트"],
                         ["Node", "Scene", "Signal", "Script"],
                         "Godot 기본 구조"),
            LearningTopic("godot_advanced", "변형된 Godot", "Godot 고급", 4,
                         ["커스텀노드", "셰이더", "물리엔진", "최적화"],
                         ["CustomNode", "Shader", "Physics2D/3D", "Optimization"],
                         "Godot 확장 개발"),
            # 4️⃣ Socket.IO 네트워킹
            LearningTopic("socketio_basic", "Socket.IO", "실시간 통신 기초", 3,
                         ["소켓", "이벤트", "룸", "네임스페이스"],
                         ["Socket", "Event", "Room", "Namespace"],
                         "Socket.IO 기본 통신"),
            LearningTopic("socketio_advanced", "Socket.IO", "고급 실시간 통신", 5,
                         ["브로드캐스트", "미들웨어", "클러스터링", "Redis"],
                         ["Broadcast", "Middleware", "Clustering", "Redis"],
                         "Socket.IO 고급 기능"),
            # 5️⃣ AI 최적화
            LearningTopic("ai_optimization_basic", "AI 최적화", "AI 코드 생성 기초", 3,
                         ["프롬프트", "컨텍스트", "토큰", "응답"],
                         ["Prompt", "Context", "Token", "Response"],
                         "AI 기반 코드 생성"),
            LearningTopic("ai_optimization_advanced", "AI 최적화", "AI 고급 최적화", 5,
                         ["파인튜닝", "프롬프트엔지니어링", "컨텍스트관리", "체이닝"],
                         ["FineTuning", "PromptEngineering", "ContextManagement", "Chaining"],
                         "AI 성능 최적화"),
            # 6️⃣ 변형된 Godot 전문가 학습
            LearningTopic("godot_expert_architecture", "Godot 전문가", "변형된 Godot 아키텍처", 5,
                         ["커스텀엔진", "렌더파이프라인", "씬시스템", "리소스관리"],
                         ["CustomEngine", "RenderPipeline", "SceneSystem", "ResourceManager"],
                         "Godot 핵심 구조"),
            LearningTopic("godot_expert_csharp", "Godot 전문가", "C# 고급 통합", 5,
                         ["GDExtension", "NativeCall", "메모리관리", "성능최적화"],
                         ["GDExtension", "NativeCall", "MemoryManagement", "Performance"],
                         "C# 고급 게임 개발"),
            # 7️⃣ Godot 엔진 조작 (가상 입력)
            LearningTopic("godot_manipulation_basic", "Godot 조작", "기본 에디터 조작", 2,
                         ["노드생성", "씬구성", "속성설정", "스크립트연결"],
                         ["NodeCreation", "SceneSetup", "PropertyConfig", "ScriptAttach"],
                         "에디터 기본 조작"),
            LearningTopic("godot_manipulation_advanced", "Godot 조작", "고급 자동화 조작", 4,
                         ["복잡한씬구성", "애니메이션설정", "물리설정", "최적화작업"],
                         ["ComplexScene", "AnimationSetup", "PhysicsConfig", "Optimization"],
                         "자동화 워크플로우"),
        ]
        
        # 모든 주제에 DeepSeek-coder 우선 태그 추가
        for topic in topics:
            topic.godot_integration = f"[DeepSeek 우선] {topic.godot_integration}"
            
        return topics
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """지식 베이스 로드"""
        kb_file = self.knowledge_base_dir / "knowledge_base.json"
        if kb_file.exists():
            with open(kb_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "csharp_patterns": {},
            "korean_translations": {},
            "godot_integrations": {},
            "common_errors": {},
            "best_practices": {}
        }
        
    def _save_knowledge_base(self):
        """지식 베이스 저장"""
        kb_file = self.knowledge_base_dir / "knowledge_base.json"
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
    
    def _load_question_history(self) -> List[Dict[str, Any]]:
        """질문 히스토리 로드"""
        history_file = self.learning_dir / "question_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_question_history(self):
        """질문 히스토리 저장"""
        history_file = self.learning_dir / "question_history.json"
        # 히스토리 크기 제한
        if len(self.question_history) > self.max_history_size:
            self.question_history = self.question_history[-self.max_history_size:]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.question_history, f, indent=2, ensure_ascii=False)
    
    def _calculate_question_similarity(self, q1: str, q2: str) -> float:
        """두 질문 간의 유사도 계산 (간단한 자카드 유사도)"""
        # 한글과 영어 모두 처리
        import re
        
        # 단어 추출 (한글, 영어, 숫자)
        pattern = r'[\w가-힣]+'
        words1 = set(re.findall(pattern, q1.lower()))
        words2 = set(re.findall(pattern, q2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # 자카드 유사도
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _is_question_too_similar(self, new_question: str) -> bool:
        """새 질문이 기존 질문들과 너무 유사한지 확인"""
        # 최근 100개 질문만 비교 (성능 최적화)
        recent_questions = self.question_history[-100:] if len(self.question_history) > 100 else self.question_history
        
        for hist_q in recent_questions:
            similarity = self._calculate_question_similarity(new_question, hist_q.get('question', ''))
            if similarity > self.question_similarity_threshold:
                return True
        return False
    
    def _add_to_question_history(self, question: Dict[str, Any]):
        """질문을 히스토리에 추가"""
        self.question_history.append({
            'question': question['question'],
            'type': question['type'],
            'topic': question['topic'],
            'timestamp': datetime.now().isoformat(),
            'keywords': question.get('keywords', [])
        })
        self._save_question_history()
            
    def _load_learning_progress(self) -> Dict[str, Any]:
        """학습 진행 상태 로드"""
        progress_file = self.progress_dir / "learning_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                logger.info(f"기존 학습 진행 상태를 로드했습니다. 총 학습 시간: {progress.get('total_hours', 0)}시간")
                return progress
        return {
            "total_hours": 0,
            "total_questions": 0,
            "total_successful": 0,
            "topics_progress": {},
            "last_session_id": None,
            "last_save_time": None,
            "sessions_completed": []
        }
        
    def _save_learning_progress(self):
        """학습 진행 상태 저장"""
        if self.current_session:
            # 현재 세션 정보 업데이트
            session_duration = (datetime.now() - self.current_session.start_time).total_seconds() / 3600
            
            self.learning_progress["total_hours"] += session_duration
            self.learning_progress["total_questions"] += self.current_session.questions_asked
            self.learning_progress["total_successful"] += self.current_session.successful_answers
            self.learning_progress["last_session_id"] = self.current_session.session_id
            self.learning_progress["last_save_time"] = datetime.now().isoformat()
            
            # 주제별 진행 상태 업데이트
            if self.current_session.topics_covered:
                for topic in self.current_session.topics_covered:
                    if topic not in self.learning_progress["topics_progress"]:
                        self.learning_progress["topics_progress"][topic] = {
                            "questions_asked": 0,
                            "successful_answers": 0,
                            "last_studied": None
                        }
                    self.learning_progress["topics_progress"][topic]["questions_asked"] += 1
                    self.learning_progress["topics_progress"][topic]["last_studied"] = datetime.now().isoformat()
        
        progress_file = self.progress_dir / "learning_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_progress, f, indent=2, ensure_ascii=False)
        logger.info(f"학습 진행 상태를 저장했습니다. 총 학습 시간: {self.learning_progress['total_hours']:.1f}시간")
            
    def load_model_info(self):
        """모델 정보만 로드 (실제 모델은 필요할 때만 로드)"""
        models_info_file = self.models_dir / "installed_models.json"
        if not models_info_file.exists():
            logger.error("모델이 설치되지 않았습니다. install_llm_models.py를 먼저 실행하세요.")
            return
            
        with open(models_info_file, 'r', encoding='utf-8') as f:
            installed_models = json.load(f)
            
        self.available_models = installed_models
        logger.info(f"사용 가능한 모델: {list(self.available_models.keys())}")
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 반환 (GB)"""
        return psutil.virtual_memory().used / (1024**3)
        
    def get_memory_usage_percent(self) -> float:
        """현재 메모리 사용률 반환 (%)"""
        return psutil.virtual_memory().percent
        
    def check_memory_safety(self) -> bool:
        """메모리 사용량이 안전한지 확인"""
        current_usage = self.get_memory_usage()
        
        # 현재 사용량이 최대 허용량의 85%를 넘으면 위험
        return current_usage < (self.max_memory_gb * self.memory_threshold)
        
    def unload_current_model(self):
        """현재 로드된 모델 언로드"""
        if self.currently_loaded_model:
            logger.info(f"메모리 절약을 위해 {self.currently_loaded_model} 모델을 언로드합니다...")
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 파이썬 가비지 컬렉션
            gc.collect()
            
            self.currently_loaded_model = None
            logger.info("모델 언로드 완료")
            
    def load_model(self, model_name: str) -> bool:
        """필요 시 모델 로드"""
        if model_name == self.currently_loaded_model:
            return True  # 이미 로드됨
            
        # 블랙리스트 체크
        if model_name in self.failed_models_blacklist:
            logger.warning(f"🚫 {model_name}은(는) 블랙리스트에 있습니다. 로드하지 않습니다.")
            return False
            
        if model_name not in self.available_models:
            logger.error(f"사용할 수 없는 모델: {model_name}")
            return False
            
        # 메모리 확인 후 필요시 기존 모델 언로드
        if not self.check_memory_safety() and self.currently_loaded_model:
            self.unload_current_model()
            
        try:
            print(f"🔄 {model_name} 모델 로드 시작...")
            logger.info(f"{model_name} 모델을 로드합니다...")
            info = self.available_models[model_name]
            model_id = info['model_id']
            print(f"📍 모델 경로: {model_id}")
            
            # RTX 2080 8GB 최적화 모델만 허용
            rtx_2080_optimized = {
                "bitnet-b1.58-2b": {"max_vram": 1, "device": "cpu"},
                "gemma-4b": {"max_vram": 4, "device": "cuda:0"},
                "phi3-mini": {"max_vram": 6, "device": "cuda:0"},
                "deepseek-coder-7b": {"max_vram": 6, "device": "cuda:0"},
                "mistral-7b": {"max_vram": 7, "device": "cuda:0"}
            }
            
            if model_name not in rtx_2080_optimized:
                logger.warning(f"❌ {model_name}은 RTX 2080 8GB에 최적화되지 않았습니다.")
                logger.info(f"✅ 사용 가능 모델: {', '.join(rtx_2080_optimized.keys())}")
                logger.info("💡 install_llm_models_rtx2080.py를 실행하여 최적화된 모델을 설치하세요.")
                return False
            
            model_config = rtx_2080_optimized[model_name]
            logger.info(f"🎯 RTX 2080 최적화: {model_name} (VRAM: {model_config['max_vram']}GB)")
            
            # AutoTokenizer와 AutoModelForCausalLM을 직접 사용
            from transformers import AutoTokenizer, AutoModelForCausalLM
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                from transformers.utils.quantization_config import BitsAndBytesConfig
            
            hf_token = os.getenv('HF_TOKEN', None)
            
            # 모델별 최적 설정
            device_map = model_config['device']
            torch_dtype = torch.float32 if device_map == "cpu" else torch.float16
            quantization_config = None
            
            # 4bit 양자화 설정 (GPU 모델용)
            if device_map != "cpu" and info.get('quantization') == '4bit':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            # 로컬 설치된 모델용 토크나이저 경로 처리
            tokenizer_path = model_id
            if model_name in ["deepseek-coder-7b", "llama-3.1-8b"]:
                # 로컬 설치된 모델은 토크나이저 폴더 사용
                tokenizer_path = info.get('tokenizer_path', model_id)
                if not tokenizer_path.startswith('./'):
                    tokenizer_path = f"./{tokenizer_path}"
                print(f"📁 토크나이저 경로: {tokenizer_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                token=hf_token,
                trust_remote_code=True,
                local_files_only=model_id.startswith('./models/')  # 로컬 모델은 오프라인 모드
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            if device_map == "cpu":
                model_kwargs["device_map"] = "cpu"
            else:
                model_kwargs["device_map"] = {"": 0}  # GPU 0 사용
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
            
            if hf_token and not model_id.startswith('./models/'):
                model_kwargs["token"] = hf_token
            
            # 로컬 모델은 오프라인 모드로 로드
            if model_id.startswith('./models/'):
                model_kwargs["local_files_only"] = True
                print(f"🔄 로컬 모델 로드 중: {model_id}")
            else:
                print(f"🔄 Hugging Face에서 모델 로드 중: {model_id}")
            
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            
            # 간단한 텍스트 생성 래퍼
            def optimized_generate(prompt):
                try:
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=1024,
                        padding=True
                    )
                    
                    if device_map != "cpu":
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs.get('input_ids'),
                            attention_mask=inputs.get('attention_mask'),
                            max_new_tokens=300,  # 더 상세한 답변을 위해 증가 (기존 150)
                            temperature=0.6,  # 더 일관된 답변을 위해 감소 (기존 0.7)
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(
                        outputs[0][inputs.get('input_ids').shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    return [{"generated_text": response}]
                    
                except Exception as e:
                    logger.error(f"생성 오류: {str(e)}")
                    return [{"generated_text": "죄송합니다. 답변 생성에 실패했습니다."}]
            
            pipe = optimized_generate
            
            self.model_cache[model_name] = {
                "pipeline": pipe,
                "features": info['features'],
                "info": info
            }
            
            self.currently_loaded_model = model_name
            
            # 메모리 사용량 로깅
            memory_usage = self.get_memory_usage()
            self.memory_usage_history.append({
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "memory_gb": memory_usage,
                "memory_percent": self.get_memory_usage_percent()
            })
            
            logger.info(f"✓ {model_name} 로드 완료 (메모리: {memory_usage:.1f}GB)")
            return True
            
        except Exception as e:
            logger.error(f"✗ {model_name} 로드 실패: {str(e)}")
            # 실패한 모델을 블랙리스트에 추가
            self.failed_models_blacklist.add(model_name)
            logger.warning(f"⚠️ {model_name}을(를) 이번 세션 블랙리스트에 추가했습니다.")
            return False
                
    def _get_quantization_config(self, quantization: str):
        """양자화 설정 반환 (RTX 2080 8GB 최적화)"""
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,  # RTX 2080 8GB 필수
                llm_int8_has_fp16_weight=False  # 메모리 절약
            )
        return None
        
    def generate_question(self, topic: LearningTopic) -> Dict[str, Any]:
        """개선된 질문 생성 - 지시-응답 데이터셋 구축에 최적화"""
        
        # 지시-응답 데이터셋 빌더의 고품질 템플릿 우선 사용
        if self.dataset_builder and hasattr(topic, 'category'):
            # Godot 카테고리 매핑
            category_map = {
                "Godot 기초": "concept_explanation",
                "Godot 고급": "code_generation",
                "Godot 전문가 아키텍처": "architecture",
                "Godot 전문가 C#": "optimization",
                "Godot 조작 고급": "debugging"
            }
            
            builder_category = category_map.get(topic.category)
            if builder_category:
                template_question = self.dataset_builder.generate_instruction_from_template(builder_category)
                if template_question:
                    return {
                        'question': template_question,
                        'type': 'template_based',
                        'topic': topic.topic,
                        'difficulty': topic.difficulty_level,
                        'language': 'korean' if any(k in template_question for k in ['해', '줘', '나', '을']) else 'english',
                        'category': topic.category,
                        'quality_priority': True  # 템플릿 기반 질문은 우선순위 높음
                    }
        
        # 기존 질문 타입들 (폴백)
        question_types = [
            "explain",      # 개념 설명
            "example",      # 예제 코드
            "translate",    # 한글-영어 번역
            "error",        # 오류 수정
            "optimize",     # 최적화
            "integrate",    # Godot 통합
            "compare",      # 비교 분석
            "implement",    # 구현 과제
            "debug",        # 디버깅
            "refactor",     # 리팩토링
            "design",       # 설계 패턴
            "test"          # 테스트 작성
        ]
        
        # "Godot 전문가" 주제인 경우, 문서에서 질문 생성
        if topic.category == "Godot 전문가":
            try:
                with open("collected_godot_docs.json", "r", encoding="utf-8") as f:
                    docs_data = json.load(f)
                
                if docs_data:
                    doc = random.choice(docs_data)
                    # 다양한 문서 기반 질문 템플릿
                    doc_templates = [
                        f"{doc['title']}의 핵심 개념을 실제 게임 개발 예제와 함께 설명해주세요.",
                        f"{doc['title']}를 사용한 최적화된 코드를 작성하고, 성능 개선 포인트를 분석해주세요.",
                        f"다음 문서를 바탕으로 {doc['title']}의 고급 활용법을 제시해주세요: \n\n{doc['content'][:300]}",
                        f"{doc['title']}와 관련된 일반적인 문제점과 해결책을 제시해주세요.",
                        f"{doc['title']}를 활용한 실무 프로젝트 구조를 설계해주세요."
                    ]
                    question_text = random.choice(doc_templates)
                    return {
                        "id": f"{topic.id}_from_doc_{int(time.time())}",
                        "topic": topic.topic,
                        "type": "doc_based_qna",
                        "language": "korean",
                        "difficulty": topic.difficulty,
                        "question": question_text,
                        "keywords": topic.korean_keywords
                    }
            except FileNotFoundError:
                pass

        # 가중치 기반 질문 타입 선택 (더 다양한 분포)
        type_weights = {
            "example": 0.20,    # 20%
            "explain": 0.15,    # 15%
            "implement": 0.15,  # 15%
            "integrate": 0.10,  # 10%
            "optimize": 0.10,   # 10%
            "error": 0.08,      # 8%
            "debug": 0.08,      # 8%
            "compare": 0.05,    # 5%
            "design": 0.05,     # 5%
            "translate": 0.02,  # 2%
            "refactor": 0.01,   # 1%
            "test": 0.01        # 1%
        }
        
        question_type = random.choices(
            list(type_weights.keys()),
            weights=list(type_weights.values())
        )[0]
        
        # 키워드 조합 생성 (1-3개)
        num_keywords = random.randint(1, min(3, len(topic.korean_keywords)))
        selected_keywords = random.sample(topic.korean_keywords, num_keywords)
        keyword_str = ", ".join(selected_keywords)
        
        num_concepts = random.randint(1, min(3, len(topic.python_concepts)))
        selected_concepts = random.sample(topic.python_concepts, num_concepts)
        concept_str = ", ".join(selected_concepts)
        
        # 난이도별 컨텍스트
        difficulty_contexts = {
            "basic": ["기초적인", "간단한", "입문자를 위한", "기본적인"],
            "intermediate": ["실무에서 사용하는", "중급 수준의", "실제 프로젝트에 적용할", "프로덕션 레벨의"],
            "advanced": ["고급", "최적화된", "대규모 시스템의", "전문가 수준의"],
            "expert": ["아키텍처 레벨의", "엔터프라이즈급", "최첨단", "혁신적인"]
        }
        
        difficulty_context = random.choice(difficulty_contexts.get(topic.difficulty, ["적절한"]))
        
        # 다양한 질문 템플릿 (각 타입별로 3-5개씩)
        templates = {
            "explain": {
                "korean": [
                    f"{topic.topic}의 {keyword_str} 개념을 {difficulty_context} 예제와 함께 상세히 설명해주세요.",
                    f"{difficulty_context} {topic.topic} 사용법을 {keyword_str}를 중심으로 단계별로 설명해주세요.",
                    f"{topic.topic}에서 {keyword_str}가 어떻게 작동하는지 내부 메커니즘을 포함해 설명해주세요.",
                    f"초보자도 이해할 수 있도록 {topic.topic}의 {keyword_str}를 비유를 들어 설명해주세요.",
                    f"{topic.topic}와 {keyword_str}의 관계를 실제 사례를 들어 설명해주세요."
                ],
                "english": [
                    f"Explain {difficulty_context} {topic.topic} focusing on {concept_str} with practical examples.",
                    f"Deep dive into {topic.topic} architecture, especially {concept_str} components.",
                    f"How does {concept_str} work in {topic.topic}? Include implementation details.",
                    f"Explain the relationship between {topic.topic} and {concept_str} with real-world scenarios."
                ]
            },
            "example": {
                "korean": [
                    f"{topic.topic}을 활용한 {difficulty_context} {keyword_str} 구현 예제를 작성해주세요.",
                    f"{keyword_str}를 사용하여 {topic.topic} 기반의 실용적인 애플리케이션을 만들어주세요.",
                    f"{difficulty_context} 수준의 {topic.topic} 프로젝트에서 {keyword_str}를 어떻게 활용하는지 보여주세요.",
                    f"{topic.topic}과 {keyword_str}를 결합한 창의적인 코드 예제를 제시해주세요.",
                    f"실제 게임/앱에서 사용할 수 있는 {topic.topic}의 {keyword_str} 활용 예제를 만들어주세요."
                ],
                "english": [
                    f"Create a {difficulty_context} example using {topic.topic} with {concept_str}.",
                    f"Build a practical application demonstrating {concept_str} in {topic.topic}.",
                    f"Show creative usage of {topic.topic} combining {concept_str} features.",
                    f"Implement a real-world scenario using {topic.topic} and {concept_str}."
                ]
            },
            "implement": {
                "korean": [
                    f"{topic.topic}을 사용하여 {keyword_str} 기능을 가진 시스템을 구현해주세요.",
                    f"{difficulty_context} {keyword_str} 시스템을 {topic.topic} 기반으로 설계하고 구현해주세요.",
                    f"{topic.topic}과 {keyword_str}를 활용한 미니 프로젝트를 완성해주세요.",
                    f"주어진 요구사항에 맞춰 {topic.topic}로 {keyword_str} 모듈을 개발해주세요."
                ],
                "english": [
                    f"Implement a {difficulty_context} system using {topic.topic} with {concept_str} features.",
                    f"Design and build a {concept_str} module using {topic.topic} best practices.",
                    f"Create a complete mini-project showcasing {topic.topic} and {concept_str}."
                ]
            },
            "optimize": {
                "korean": [
                    f"{topic.topic}에서 {keyword_str} 관련 성능을 최적화하는 방법을 제시해주세요.",
                    f"{difficulty_context} 환경에서 {topic.topic}의 {keyword_str} 병목현상을 해결하는 방법을 보여주세요.",
                    f"{topic.topic} 코드의 메모리 사용량과 실행 속도를 {keyword_str} 관점에서 개선해주세요.",
                    f"프로파일링을 통해 {topic.topic}의 {keyword_str} 성능 문제를 찾고 해결해주세요."
                ],
                "english": [
                    f"Optimize {topic.topic} performance focusing on {concept_str} bottlenecks.",
                    f"Profile and improve {concept_str} usage in {topic.topic} applications.",
                    f"Reduce memory footprint and execution time for {topic.topic} with {concept_str}."
                ]
            },
            "debug": {
                "korean": [
                    f"{topic.topic}에서 발생하는 {keyword_str} 관련 버그를 찾고 수정해주세요.",
                    f"다음 {topic.topic} 코드의 {keyword_str} 부분에서 발생하는 문제를 디버깅해주세요.",
                    f"{difficulty_context} {topic.topic} 프로젝트의 {keyword_str} 오류를 체계적으로 추적하고 해결해주세요."
                ],
                "english": [
                    f"Debug {concept_str} related issues in {topic.topic} code.",
                    f"Find and fix bugs in {topic.topic} implementation focusing on {concept_str}.",
                    f"Systematically trace and resolve {concept_str} errors in {topic.topic}."
                ]
            },
            "compare": {
                "korean": [
                    f"{topic.topic}과 다른 기술의 {keyword_str} 구현 방식을 비교 분석해주세요.",
                    f"{topic.topic}에서 {keyword_str}를 처리하는 여러 방법의 장단점을 비교해주세요.",
                    f"성능, 가독성, 유지보수 관점에서 {topic.topic}의 {keyword_str} 접근법들을 평가해주세요."
                ],
                "english": [
                    f"Compare different approaches to {concept_str} in {topic.topic}.",
                    f"Analyze pros and cons of various {concept_str} implementations in {topic.topic}.",
                    f"Evaluate {topic.topic} {concept_str} patterns from performance and maintainability perspectives."
                ]
            },
            "design": {
                "korean": [
                    f"{topic.topic}을 활용한 {keyword_str} 시스템의 아키텍처를 설계해주세요.",
                    f"{difficulty_context} 프로젝트를 위한 {topic.topic} 기반 {keyword_str} 설계 패턴을 제안해주세요.",
                    f"확장 가능한 {topic.topic} {keyword_str} 구조를 설계하고 다이어그램으로 표현해주세요."
                ],
                "english": [
                    f"Design a scalable architecture for {concept_str} using {topic.topic}.",
                    f"Propose design patterns for {topic.topic} based {concept_str} systems.",
                    f"Create architectural diagrams for {concept_str} implementation in {topic.topic}."
                ]
            },
            "test": {
                "korean": [
                    f"{topic.topic}의 {keyword_str} 기능을 검증하는 테스트 코드를 작성해주세요.",
                    f"{difficulty_context} {topic.topic} 프로젝트의 {keyword_str} 부분에 대한 단위 테스트를 구현해주세요.",
                    f"TDD 방식으로 {topic.topic}의 {keyword_str} 모듈을 개발해주세요."
                ],
                "english": [
                    f"Write comprehensive tests for {concept_str} functionality in {topic.topic}.",
                    f"Implement unit tests for {topic.topic} {concept_str} module.",
                    f"Develop {concept_str} features in {topic.topic} using TDD approach."
                ]
            },
            "refactor": {
                "korean": [
                    f"레거시 {topic.topic} 코드를 {keyword_str} 패턴을 사용해 리팩토링해주세요.",
                    f"{topic.topic}의 {keyword_str} 부분을 더 깔끔하고 유지보수하기 쉽게 개선해주세요.",
                    f"SOLID 원칙에 따라 {topic.topic}의 {keyword_str} 구현을 리팩토링해주세요."
                ],
                "english": [
                    f"Refactor legacy {topic.topic} code using {concept_str} patterns.",
                    f"Improve {concept_str} implementation in {topic.topic} following SOLID principles.",
                    f"Clean up and modernize {topic.topic} {concept_str} codebase."
                ]
            },
            "error": {
                "korean": [
                    f"{topic.topic}에서 {keyword_str} 사용 시 발생하는 일반적인 오류와 해결책을 정리해주세요.",
                    f"{difficulty_context} 개발자가 {topic.topic}의 {keyword_str}에서 자주 실수하는 부분과 예방법을 설명해주세요.",
                    f"{topic.topic} {keyword_str} 관련 에러 메시지를 해석하고 디버깅하는 방법을 가이드해주세요."
                ],
                "english": [
                    f"Common {concept_str} errors in {topic.topic} and how to fix them.",
                    f"Troubleshooting guide for {topic.topic} {concept_str} issues.",
                    f"Error handling best practices for {concept_str} in {topic.topic}."
                ]
            },
            "integrate": {
                "korean": [
                    f"Godot 엔진에서 {topic.topic}의 {keyword_str} 개념을 C#으로 구현해주세요.",
                    f"{topic.topic}과 Godot의 {keyword_str}를 연동하는 실용적인 예제를 만들어주세요.",
                    f"Unity에서 Godot로 마이그레이션할 때 {topic.topic}의 {keyword_str}를 어떻게 변환하는지 보여주세요.",
                    f"Godot 4.x에서 {topic.topic} 기반 {keyword_str} 시스템을 구축하는 방법을 설명해주세요."
                ],
                "english": [
                    f"Integrate {concept_str} from {topic.topic} into Godot with C#.",
                    f"Build a bridge between {topic.topic} {concept_str} and Godot systems.",
                    f"Migrate {concept_str} patterns from Unity to Godot using {topic.topic} principles."
                ]
            },
            "translate": {
                "korean": [
                    f"{concept_str} 용어를 한국어로 번역하고 {topic.topic} 맥락에서의 의미를 설명해주세요.",
                    f"다음 영문 {topic.topic} 문서를 한국어로 번역하고 {keyword_str} 용어를 해설해주세요.",
                    f"{topic.topic}의 {concept_str}에 해당하는 한국어 전문용어와 실무 사용 예를 제시해주세요."
                ],
                "english": [
                    f"Translate {keyword_str} to English and explain in {topic.topic} context.",
                    f"Create a glossary of {topic.topic} terms translating {keyword_str}.",
                    f"Professional translation of {keyword_str} with {topic.topic} usage examples."
                ]
            }
        }
        
        # 언어 선택 (난이도에 따라 가중치 조정)
        language_weight = {
            "basic": 0.8,      # 기초는 한글 80%
            "intermediate": 0.7, # 중급은 한글 70%
            "advanced": 0.6,    # 고급은 한글 60%
            "expert": 0.5       # 전문가는 한글 50%
        }
        
        language = "korean" if random.random() < language_weight.get(topic.difficulty, 0.7) else "english"
        
        # 템플릿 선택
        question_templates = templates.get(question_type, templates["example"])[language]
        question_text = random.choice(question_templates)
        
        # 시나리오 기반 컨텍스트 추가 (30% 확률)
        if random.random() < 0.3:
            scenarios = [
                "실시간 멀티플레이어 게임",
                "모바일 앱",
                "웹 서비스 백엔드",
                "AI 챗봇",
                "데이터 분석 도구",
                "자동화 스크립트",
                "게임 엔진 플러그인",
                "IoT 디바이스",
                "블록체인 dApp",
                "클라우드 서비스"
            ]
            scenario = random.choice(scenarios)
            question_text = f"[{scenario} 개발 상황] " + question_text
        
        # 중복 확인 및 재생성 (최대 5회 시도)
        max_attempts = 5
        for attempt in range(max_attempts):
            if not self._is_question_too_similar(question_text):
                break
            
            # 재생성 시 다른 템플릿, 키워드 조합, 시나리오 사용
            logger.info(f"🔄 유사한 질문 감지, 재생성 중... (시도 {attempt + 1}/{max_attempts})")
            
            # 다른 질문 타입 선택
            if attempt == 1:
                question_type = random.choices(
                    list(type_weights.keys()),
                    weights=list(type_weights.values())
                )[0]
            
            # 다른 키워드 조합
            num_keywords = random.randint(1, min(3, len(topic.korean_keywords)))
            selected_keywords = random.sample(topic.korean_keywords, num_keywords)
            keyword_str = ", ".join(selected_keywords)
            
            num_concepts = random.randint(1, min(3, len(topic.python_concepts)))
            selected_concepts = random.sample(topic.python_concepts, num_concepts)
            concept_str = ", ".join(selected_concepts)
            
            # 다른 난이도 컨텍스트
            difficulty_context = random.choice(difficulty_contexts.get(topic.difficulty, ["적절한"]))
            
            # 다른 템플릿 선택
            question_templates = templates.get(question_type, templates["example"])[language]
            question_text = random.choice(question_templates)
            
            # 다른 시나리오 (확률 높임)
            if random.random() < 0.5:  # 50%로 증가
                scenario = random.choice(scenarios)
                question_text = f"[{scenario} 개발 상황] " + question_text
        
        question_data = {
            "id": f"{topic.id}_{question_type}_{int(time.time())}",
            "topic": topic.topic,
            "type": question_type,
            "language": language,
            "difficulty": topic.difficulty,
            "question": question_text,
            "keywords": topic.korean_keywords if language == "korean" else topic.python_concepts,
            "context": {
                "scenario": scenario if 'scenario' in locals() else None,
                "difficulty_context": difficulty_context,
                "keyword_combination": selected_keywords if language == "korean" else selected_concepts
            }
        }
        
        # 히스토리에 추가
        self._add_to_question_history(question_data)
        
        return question_data
        
    def select_model_for_question(self, question: Dict[str, Any]) -> str:
        """질문에 적합한 모델 선택 (5대 핵심 주제 DeepSeek-coder 최적화)"""
        # RTX 2080 최적화 모델만 고려
        rtx_2080_models = {
            "deepseek-coder-7b": {"priority": 10, "specialties": ["code", "csharp", "godot", "korean", "socketio"], "vram": 6},
            "phi3-mini": {"priority": 8, "specialties": ["reasoning", "math", "python"], "vram": 6},
            "llama-3.1-8b": {"priority": 7, "specialties": ["general", "korean", "python"], "vram": 7},
            "gemma-4b": {"priority": 6, "specialties": ["general", "korean"], "vram": 4},
            "mistral-7b": {"priority": 4, "specialties": ["general", "fast"], "vram": 7}
        }
        
        # RTX 2080 최적화 모델만 필터링 (블랙리스트 제외)
        available_optimized = []
        for model_name in self.available_models:
            if (model_name in rtx_2080_models and 
                self.available_models[model_name].get('rtx_2080_optimized', False) and
                model_name not in self.failed_models_blacklist):  # 블랙리스트 체크
                available_optimized.append(model_name)
        
        if not available_optimized:
            logger.warning("❌ RTX 2080 최적화 모델이 없습니다.")
            if self.failed_models_blacklist:
                logger.info(f"🚫 블랙리스트 모델: {', '.join(self.failed_models_blacklist)}")
            logger.info("💡 python download_deepseek_coder.py로 DeepSeek-coder 6.7B를 설치하세요.")
            # Return a default model if available
            if self.available_models:
                return list(self.available_models.keys())[0]
            return "deepseek-coder-7b"  # fallback model name
            
        # 5가지 핵심 주제 카테고리 확인
        core_categories = ["C# 프로그래밍", "한글 용어", "변형된 Godot", "Socket.IO", "AI 최적화"]
        topic_category = question.get("category", "")
        is_core_topic = topic_category in core_categories
        
        # 현재 로드된 모델이 RTX 2080 최적화이고 적합하면 계속 사용
        if (self.currently_loaded_model and 
            self.currently_loaded_model in available_optimized and
            self._is_model_suitable(self.currently_loaded_model, question)):
            return self.currently_loaded_model
            
        # 질문 특성 분석 (5가지 핵심 주제 포함)
        question_features = set()
        question_text = question.get("question", "").lower()
        topic_text = question.get("topic", "").lower()
        
        # 1️⃣ Python 프로그래밍 특성
        if any(word in question_text + topic_text for word in ['code', 'programming', 'script', 'function', 'class', 'method', 'python', 'py']):
            question_features.add('python')
            question_features.add('code')
        
        # 2️⃣ 한글 프로그래밍 용어 특성
        if any(word in question_text + topic_text for word in ['korean', '한글', '한국어', '번역', '용어', '개념', '언어']):
            question_features.add('korean')
        
        # 3️⃣ Godot 엔진 특성
        if any(word in question_text + topic_text for word in ['godot', 'node', 'scene', '노드', '씬']):
            question_features.add('godot')
        
        # 4️⃣ Socket.IO 네트워킹 특성
        if any(word in question_text + topic_text for word in ['socket', 'socketio', 'realtime', '소켓', '실시간']):
            question_features.add('socketio')
        
        # 5️⃣ AI 최적화 특성
        if any(word in question_text + topic_text for word in ['ai', 'optimize', 'prompt', '최적화', '프롬프트']):
            question_features.add('ai_optimization')
        
        # 모델 점수 계산
        model_scores = []
        for model_name in available_optimized:
            model_info = rtx_2080_models[model_name]
            score = model_info['priority']
            
            # 특성 매칭 점수
            for feature in question_features:
                if feature in model_info['specialties']:
                    score += 5
            
            # 모델별 특별 보너스
            if model_name == "llama-3.1-8b" and 'korean' in question_features:
                score += 20  # 한국어 특화 보너스
            elif model_name == "gemma-4b" and 'korean' in question_features:
                score += 12  # 한국어 특화 보너스
            elif model_name == "deepseek-coder-7b":
                if 'code' in question_features or 'python' in question_features:
                    score += 10  # 코드 특화 보너스
                if is_core_topic:
                    score += 15  # DeepSeek-coder에 핵심 주제 보너스
            
            model_scores.append((model_name, score))
        
        # 점수가 높은 모델 선택
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        if model_scores:
            selected_model = model_scores[0][0]
            logger.info(f"선택된 모델: {selected_model} (점수: {model_scores[0][1]})")
            return selected_model
        else:
            logger.error("적합한 모델을 찾을 수 없습니다")
            return list(self.available_models.keys())[0] if self.available_models else "deepseek-coder-7b"
    
    def _is_model_suitable(self, model_name: str, question: Dict[str, Any]) -> bool:
        """모델이 질문에 적합한지 확인"""
        # 간단한 적합성 검사
        question_text = question.get("question", "").lower()
        
        if model_name == "deepseek-coder-7b":
            # 코드 관련 질문에 매우 적합
            return any(word in question_text for word in ['code', 'function', 'class', 'godot', 'csharp'])
        elif model_name == "llama-3.1-8b":
            # 한국어 질문에 적합
            return any(word in question_text for word in ['한글', '한국어', '번역'])
        
        return True  # 기본적으로 적합하다고 가정
    
    async def ask_and_learn(self, question: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """질문하고 답변 학습"""
        try:
            # 공유 지식 베이스에서 캐시된 정보 확인
            from modules.shared_knowledge_base import get_shared_knowledge_base
            shared_kb = get_shared_knowledge_base()
            
            # 질문 키워드로 캐시 검색
            question_keywords = question.get('question', '').split()[:3]  # 첫 3단어로 검색
            search_keyword = ' '.join(question_keywords)
            
            cached_info = await shared_kb.get_cached_search(search_keyword)
            if cached_info:
                logger.info(f"📚 공유 지식 베이스에서 관련 정보 발견: {search_keyword}")
                # 캐시된 정보를 컨텍스트로 활용하여 더 나은 답변 생성 가능
            
            # 모델 로드 (재시도 포함)
            load_success = False
            for attempt in range(3):
                try:
                    if self.load_model(model_name):
                        load_success = True
                        break
                except Exception as load_error:
                    logger.warning(f"모델 로드 시도 {attempt + 1}/3 실패: {str(load_error)}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                        gc.collect()
            
            if not load_success:
                return {
                    "success": False,
                    "error": f"모델 {model_name} 로드 실패 (모든 시도 실패)"
                }
            
            # 파이프라인 가져오기
            pipe = self.model_cache[model_name]["pipeline"]
            
            # 답변 생성
            logger.info(f"질문: {question['question'][:100]}...")
            
            start_time = time.time()
            answer = None
            generation_error = None
            
            # 답변 생성 (타임아웃 및 재시도 포함)
            for gen_attempt in range(2):
                try:
                    # 타임아웃 설정 (60초)
                    response = await asyncio.wait_for(
                        asyncio.to_thread(lambda: pipe(question['question'])),
                        timeout=60.0
                    )
                    answer = response[0]['generated_text'] if response else "답변 생성 실패"
                    if answer and answer != "답변 생성 실패":
                        break
                except asyncio.TimeoutError:
                    logger.warning(f"답변 생성 타임아웃 (60초) - 시도 {gen_attempt + 1}/2")
                    generation_error = "timeout"
                except Exception as gen_error:
                    logger.warning(f"답변 생성 오류 - 시도 {gen_attempt + 1}/2: {str(gen_error)}")
                    generation_error = str(gen_error)
                
                if gen_attempt == 0:
                    await asyncio.sleep(5)
            
            generation_time = time.time() - start_time
            
            if not answer or answer == "답변 생성 실패":
                return {
                    "success": False,
                    "error": f"답변 생성 실패: {generation_error}"
                }
            logger.info(f"답변 생성 시간: {generation_time:.2f}초")
            
            # 답변 품질 평가 (개선된 PyTorch 모델 사용)
            try:
                quality_score = self._evaluate_answer_quality(question, answer, model_name)
            except Exception as eval_error:
                logger.warning(f"품질 평가 중 오류: {str(eval_error)}")
                quality_score = 0.5  # 기본값
            
            # 실시간 피드백으로 답변 개선 시도
            if quality_score < 0.7:
                logger.info(f"💡 품질이 낮아 개선 시도 중... (현재: {quality_score:.2f})")
                improved_answer = self._improve_answer_with_feedback(question, answer, model_name)
                if improved_answer != answer:
                    answer = improved_answer
                    quality_score = self._evaluate_answer_quality(question, answer, model_name)
                    logger.info(f"✨ 개선 후 품질: {quality_score:.2f}")
            
            # 멀티태스크 학습 네트워크 업데이트
            try:
                task_category = self._get_task_category(question['topic'])
                if task_category:
                    self._update_multitask_network(question, answer, quality_score, task_category)
            except Exception as mt_error:
                logger.warning(f"멀티태스크 네트워크 업데이트 중 오류: {str(mt_error)}")
            
            # 답변 저장
            answer_data = {
                "question_id": question['id'],
                "question": question['question'],
                "answer": answer,
                "model": model_name,
                "quality_score": quality_score,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "topic": question['topic'],
                "language": question['language']
            }
            
            # 답변 디렉토리에 저장
            today = datetime.now().strftime("%Y%m%d")
            answer_dir = self.answers_dir / today
            answer_dir.mkdir(exist_ok=True)
            
            answer_file = answer_dir / f"{question['id']}.json"
            with open(answer_file, 'w', encoding='utf-8') as f:
                json.dump(answer_data, f, indent=2, ensure_ascii=False)
            
            # 지식 베이스 업데이트
            self._update_knowledge_base(question, answer, quality_score)
            
            # 고품질 답변은 공유 지식 베이스에 베스트 프랙티스로 저장
            if quality_score > 0.8:
                await shared_kb.save_best_practice(
                    topic=question.get('topic', 'general'),
                    practice={
                        "question": question['question'],
                        "answer": answer,
                        "model": model_name,
                        "quality_score": quality_score,
                        "language": question.get('language', 'ko')
                    }
                )
                logger.info(f"📚 베스트 프랙티스 저장: {question['topic']}")
            
            # PyTorch 학습 데이터로 추가
            if self.pytorch_system and quality_score > 0.6:
                experience = {
                    'question': question['question'],
                    'answer': answer,
                    'quality_score': quality_score,
                    'topic': question['topic']
                }
                
                # 강화학습 업데이트
                state = {'quality_score': quality_score, 'topic': question['topic']}
                action = 'generate_answer'
                reward = quality_score
                next_state = {'quality_score': quality_score + 0.1, 'topic': question['topic']}
                
                self.pytorch_system.reinforcement_learning_step(state, action, reward, next_state)
            
            # 세션 업데이트
            if self.current_session:
                self.current_session.questions_asked += 1
                if quality_score > 0.7:
                    self.current_session.successful_answers += 1
                
                if model_name not in self.current_session.models_used:
                    self.current_session.models_used[model_name] = 0
                self.current_session.models_used[model_name] += 1
                
                self.current_session.knowledge_gained.append({
                    "topic": question['topic'],
                    "quality": quality_score,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "success": True,
                "answer": answer,
                "quality_score": quality_score,
                "model": model_name,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"학습 중 오류 발생: {str(e)}")
            logger.exception("상세 오류 정보:")
            
            # 현재 모델 언로드하여 메모리 확보
            try:
                self.unload_current_model()
                gc.collect()
            except:
                pass
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_knowledge_base(self, question: Dict[str, Any], answer: str, quality_score: float):
        """지식 베이스 업데이트"""
        if quality_score < 0.6:
            return  # 품질이 낮은 답변은 저장하지 않음
        
        topic = question['topic']
        category = question.get('category', 'general')
        
        # 카테고리별 저장
        if category not in self.knowledge_base:
            self.knowledge_base[category] = {}
        
        if topic not in self.knowledge_base[category]:
            self.knowledge_base[category][topic] = []
        
        # 지식 항목 추가
        knowledge_item = {
            "question": question['question'],
            "answer": answer,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
            "keywords": question.get('keywords', [])
        }
        
        self.knowledge_base[category][topic].append(knowledge_item)
        
        # 주기적으로 저장
        if len(self.knowledge_base[category][topic]) % 10 == 0:
            self._save_knowledge_base()
    
    def calculate_question_diversity_metrics(self) -> Dict[str, Any]:
        """질문 다양성 메트릭 계산"""
        if not self.question_history:
            return {
                "total_questions": 0,
                "unique_questions": 0,
                "diversity_score": 0.0,
                "type_distribution": {},
                "topic_distribution": {},
                "keyword_distribution": {},
                "avg_similarity": 0.0
            }
        
        # 기본 통계
        total_questions = len(self.question_history)
        unique_questions = len(set(q['question'] for q in self.question_history))
        
        # 타입별 분포
        type_counts = {}
        topic_counts = {}
        keyword_counts = {}
        
        for q in self.question_history:
            # 타입 분포
            q_type = q.get('type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            
            # 주제 분포
            q_topic = q.get('topic', 'unknown')
            topic_counts[q_topic] = topic_counts.get(q_topic, 0) + 1
            
            # 키워드 분포
            for keyword in q.get('keywords', []):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # 다양성 점수 계산 (엔트로피 기반)
        import math
        
        def calculate_entropy(counts):
            total = sum(counts.values())
            if total == 0:
                return 0.0
            entropy = 0.0
            for count in counts.values():
                if count > 0:
                    prob = count / total
                    entropy -= prob * math.log2(prob)
            return entropy
        
        type_entropy = calculate_entropy(type_counts)
        topic_entropy = calculate_entropy(topic_counts)
        keyword_entropy = calculate_entropy(keyword_counts)
        
        # 정규화된 다양성 점수 (0-1)
        max_type_entropy = math.log2(len(type_counts)) if len(type_counts) > 1 else 1
        max_topic_entropy = math.log2(len(topic_counts)) if len(topic_counts) > 1 else 1
        max_keyword_entropy = math.log2(len(keyword_counts)) if len(keyword_counts) > 1 else 1
        
        type_diversity = type_entropy / max_type_entropy if max_type_entropy > 0 else 0
        topic_diversity = topic_entropy / max_topic_entropy if max_topic_entropy > 0 else 0
        keyword_diversity = keyword_entropy / max_keyword_entropy if max_keyword_entropy > 0 else 0
        
        # 전체 다양성 점수 (가중 평균)
        diversity_score = (0.3 * type_diversity + 0.3 * topic_diversity + 0.4 * keyword_diversity)
        
        # 평균 유사도 계산 (최근 50개 질문)
        recent_questions = self.question_history[-50:] if len(self.question_history) > 50 else self.question_history
        similarities = []
        
        for i in range(len(recent_questions)):
            for j in range(i + 1, min(i + 10, len(recent_questions))):  # 각 질문과 다음 10개만 비교
                sim = self._calculate_question_similarity(
                    recent_questions[i]['question'],
                    recent_questions[j]['question']
                )
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return {
            "total_questions": total_questions,
            "unique_questions": unique_questions,
            "uniqueness_ratio": unique_questions / total_questions if total_questions > 0 else 0,
            "diversity_score": round(diversity_score, 3),
            "type_distribution": type_counts,
            "topic_distribution": topic_counts,
            "keyword_distribution": dict(sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]),  # Top 20
            "avg_similarity": round(avg_similarity, 3),
            "type_entropy": round(type_entropy, 3),
            "topic_entropy": round(topic_entropy, 3),
            "keyword_entropy": round(keyword_entropy, 3)
        }
    
    def log_diversity_metrics(self):
        """다양성 메트릭을 로그에 출력"""
        metrics = self.calculate_question_diversity_metrics()
        
        logger.info("📊 질문 다양성 메트릭:")
        logger.info(f"  - 총 질문 수: {metrics['total_questions']}")
        logger.info(f"  - 고유 질문 수: {metrics['unique_questions']} ({metrics['uniqueness_ratio']:.1%})")
        logger.info(f"  - 다양성 점수: {metrics['diversity_score']:.3f} (0-1)")
        logger.info(f"  - 평균 유사도: {metrics['avg_similarity']:.3f}")
        logger.info(f"  - 타입 엔트로피: {metrics['type_entropy']:.3f}")
        logger.info(f"  - 주제 엔트로피: {metrics['topic_entropy']:.3f}")
        logger.info(f"  - 키워드 엔트로피: {metrics['keyword_entropy']:.3f}")
        
        # 타입별 분포 상위 5개
        if metrics['type_distribution']:
            logger.info("  - 질문 타입 분포 (상위 5개):")
            for q_type, count in sorted(metrics['type_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"    • {q_type}: {count}회")
    
    async def continuous_learning_loop(self, duration_hours: float = 24):
        """연속 학습 루프"""
        logger.info(f"🚀 {duration_hours}시간 연속 학습 시작!")
        
        # 세션 시작
        self.current_session = LearningSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now(),
            topics_covered=[],
            questions_asked=0,
            successful_answers=0,
            models_used={},
            knowledge_gained=[]
        )
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        # PyTorch 학습 데이터 수집
        pytorch_experiences = []
        
        # 학습 모니터링 시작
        if self.quality_monitor:
            self.quality_monitor.start_monitoring(
                model_name="AutoCI Learning System",
                dataset_size=len(self.knowledge_base)
            )
        
        # 학습 전 기존 지식 베이스를 지시-응답 데이터셋으로 변환
        if self.dataset_builder and len(self.knowledge_base) > 0:
            logger.info("📚 기존 지식을 지시-응답 데이터셋으로 변환 중...")
            kb_path = self.knowledge_base_dir / "knowledge_base.json"
            if kb_path.exists():
                imported = self.dataset_builder.import_from_existing_knowledge(kb_path)
                logger.info(f"✅ {imported}개의 고품질 데이터 임포트 완료")
        
        try:
            while datetime.now() < end_time:
                # 학습 조기 종료 체크
                if self.quality_monitor and self.quality_monitor.should_stop:
                    logger.warning("🛑 과적합 방지를 위한 조기 종료!")
                    break
                # 메모리 체크 및 관리
                try:
                    memory_usage = self.get_memory_usage_percent()
                    if memory_usage > 85:
                        logger.warning(f"메모리 사용량 높음: {memory_usage:.1f}%")
                        self.unload_current_model()
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        await asyncio.sleep(10)
                    elif memory_usage > 95:
                        logger.critical(f"메모리 위험 수준: {memory_usage:.1f}%")
                        # 모든 모델 언로드
                        self.model_cache.clear()
                        self.currently_loaded_model = None
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        await asyncio.sleep(30)
                except Exception as mem_error:
                    logger.error(f"메모리 체크 중 오류: {str(mem_error)}")
                    # 안전을 위해 메모리 정리
                    gc.collect()
                
                # 학습 주제 선택
                topic = random.choice(self.learning_topics)
                
                # 질문 생성
                question = self.generate_question(topic)
                question['category'] = topic.category
                
                # 모델 선택
                model_name = self.select_model_for_question(question)
                if not model_name:
                    logger.error("적합한 모델을 찾을 수 없습니다")
                    await asyncio.sleep(10)
                    continue
                
                # 학습 실행 (재시도 로직 포함)
                result = None
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        result = await self.ask_and_learn(question, model_name)
                        break  # 성공하면 재시도 루프 종료
                    except Exception as e:
                        logger.warning(f"학습 시도 {retry + 1}/{max_retries} 실패: {str(e)}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(10)  # 재시도 전 대기
                            # 다른 모델로 시도
                            alternative_models = [m for m in self.available_models if m != model_name and m not in self.failed_models_blacklist]
                            if alternative_models:
                                model_name = random.choice(alternative_models)
                                logger.info(f"대체 모델로 재시도: {model_name}")
                        else:
                            result = {"success": False, "error": str(e)}
                
                if result and result['success']:
                    logger.info(f"✓ 학습 성공! 품질: {result['quality_score']:.2f}")
                    
                    # 고품질 데이터를 지시-응답 데이터셋에 추가
                    if self.dataset_builder and result['quality_score'] > 0.7:
                        self.dataset_builder.add_instruction_response_pair(
                            instruction=question['question'],
                            output=result['answer'],
                            category=topic.category,
                            difficulty=topic.difficulty_level,
                            source="continuous_learning",
                            verified=result['quality_score'] > 0.8
                        )
                        logger.info("📚 고품질 지시-응답 쌍 데이터셋에 추가됨")
                    
                    # PyTorch 학습용 데이터 수집
                    if result['quality_score'] > 0.6:
                        pytorch_experiences.append({
                            'question': question['question'],
                            'answer': result['answer'],
                            'quality_score': result['quality_score'],
                            'topic': topic.topic
                        })
                    
                    # 주제 커버리지 업데이트
                    if topic.id not in self.current_session.topics_covered:
                        self.current_session.topics_covered.append(topic.id)
                    
                    # 품질 모니터링 업데이트
                    if self.quality_monitor:
                        self.quality_monitor.update_metrics(
                            epoch=self.current_session.questions_asked // 100,
                            iteration=self.current_session.questions_asked,
                            train_loss=1.0 - result['quality_score'],  # 품질 점수를 loss로 변환
                            validation_loss=None  # 실제 validation 데이터가 있으면 사용
                        )
                else:
                    logger.error(f"✗ 학습 실패: {result.get('error', '알 수 없는 오류')}")
                
                # PyTorch 배치 학습 (100개 경험마다)
                if len(pytorch_experiences) >= 100 and self.pytorch_system:
                    try:
                        logger.info("🧠 PyTorch 배치 학습 시작...")
                        self.pytorch_system.train_on_experience(pytorch_experiences, epochs=5)
                        pytorch_experiences = []  # 리셋
                    except Exception as pt_error:
                        logger.error(f"PyTorch 학습 중 오류: {str(pt_error)}")
                        # 일부 경험만 삭제하여 메모리 확보
                        pytorch_experiences = pytorch_experiences[-50:]
                
                # 품질 평가 모델 배치 학습 (32개 경험마다)
                if len(self.experience_buffer) >= 32 and self.current_session.questions_asked % 5 == 0:
                    try:
                        self.train_quality_assessor_batch()
                    except Exception as qa_error:
                        logger.error(f"품질 평가 모델 학습 중 오류: {str(qa_error)}")
                
                # 진행 상황 저장 (10분마다)
                if self.current_session.questions_asked % 10 == 0:
                    try:
                        self._save_learning_progress()
                        # 다양성 메트릭 출력 (10개 질문마다)
                        self.log_diversity_metrics()
                        # 블랙리스트 상태 출력
                        if self.failed_models_blacklist:
                            logger.info(f"🚫 현재 블랙리스트: {', '.join(self.failed_models_blacklist)}")
                        # PyTorch 모델 저장 (50개 질문마다)
                        if self.current_session.questions_asked % 50 == 0:
                            self.save_pytorch_models()
                    except Exception as save_error:
                        logger.error(f"진행 상황 저장 중 오류: {str(save_error)}")
                
                # 대기 시간 (너무 빠른 반복 방지)
                await asyncio.sleep(random.uniform(5, 15))
                
                # 주기적 헬스체크 (1시간마다)
                if self.current_session.questions_asked > 0 and self.current_session.questions_asked % 60 == 0:
                    elapsed_hours = (datetime.now() - self.current_session.start_time).total_seconds() / 3600
                    remaining_hours = (end_time - datetime.now()).total_seconds() / 3600
                    success_rate = self.current_session.successful_answers / max(1, self.current_session.questions_asked)
                    
                    logger.info("\n" + "="*60)
                    logger.info("🏥 시스템 헬스체크")
                    logger.info(f"⏱️  경과 시간: {elapsed_hours:.1f}시간 / 남은 시간: {remaining_hours:.1f}시간")
                    logger.info(f"📊 진행 상황: {self.current_session.questions_asked}개 질문 / {self.current_session.successful_answers}개 성공")
                    logger.info(f"✅ 성공률: {success_rate:.1%}")
                    logger.info(f"💾 메모리 사용: {self.get_memory_usage():.1f}GB / {self.max_memory_gb}GB")
                    logger.info(f"🤖 현재 모델: {self.currently_loaded_model}")
                    logger.info(f"📚 커버된 주제: {len(self.current_session.topics_covered)}개")
                    logger.info("="*60 + "\n")
                
                # 주기적 메모리 정리 (2시간마다)
                if self.current_session.questions_asked > 0 and self.current_session.questions_asked % 120 == 0:
                    logger.info("🧹 주기적 메모리 정리 시작...")
                    try:
                        # 현재 모델 언로드 및 재로드
                        current_model = self.currently_loaded_model
                        if current_model:
                            self.unload_current_model()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            await asyncio.sleep(10)
                            
                            # 모델 재로드
                            logger.info(f"🔄 {current_model} 모델 재로드...")
                            self.load_model(current_model)
                        
                        # Experience buffer 크기 제한
                        if len(self.experience_buffer) > 5000:
                            logger.info("📦 Experience buffer 크기 조정...")
                            # 가장 오래된 경험 제거
                            while len(self.experience_buffer) > 3000:
                                self.experience_buffer.buffer.popleft()
                        
                        # 메모리 사용량 히스토리 정리
                        if len(self.memory_usage_history) > 1000:
                            self.memory_usage_history = self.memory_usage_history[-500:]
                        
                        logger.info("✅ 메모리 정리 완료")
                    except Exception as cleanup_error:
                        logger.error(f"메모리 정리 중 오류: {str(cleanup_error)}")
                
        except KeyboardInterrupt:
            logger.info("학습이 사용자에 의해 중단되었습니다")
        except Exception as e:
            logger.error(f"학습 루프 오류: {str(e)}")
            logger.exception("상세 오류 정보:")
            # 오류 발생 시에도 진행 상황 저장
            try:
                self._save_learning_progress()
                self._save_knowledge_base()
                logger.info("오류 발생 후 진행 상황 저장 완료")
            except Exception as save_error:
                logger.error(f"진행 상황 저장 중 오류: {str(save_error)}")
        finally:
            # 마지막 PyTorch 학습
            if pytorch_experiences and self.pytorch_system:
                logger.info("🧠 최종 PyTorch 배치 학습...")
                self.pytorch_system.train_on_experience(pytorch_experiences, epochs=5)
            
            # 세션 종료
            session_duration = (datetime.now() - self.current_session.start_time).total_seconds() / 3600
            logger.info(f"📊 학습 세션 완료!")
            logger.info(f"- 총 시간: {session_duration:.1f}시간")
            logger.info(f"- 질문 수: {self.current_session.questions_asked}")
            logger.info(f"- 성공적인 답변: {self.current_session.successful_answers}")
            logger.info(f"- 커버된 주제: {len(self.current_session.topics_covered)}")
            
            # 최종 저장
            self._save_learning_progress()
            self._save_knowledge_base()
            
            # 최종 다양성 메트릭 출력
            logger.info("\n🎯 최종 질문 다양성 분석:")
            self.log_diversity_metrics()
            
            # 세션 보고서 생성
            self._generate_session_report()
            
            # 데이터셋 큐레이션 및 내보내기
            if self.dataset_builder:
                logger.info("📚 데이터셋 큐레이션 시작...")
                curated_count = self.dataset_builder.curate_dataset(min_quality=0.7)
                if curated_count > 0:
                    # 학습용 포맷으로 내보내기
                    training_path = self.dataset_builder.export_for_training(format="alpaca")
                    logger.info(f"✅ 학습용 데이터셋 생성 완료: {training_path}")
                    
                    # 데이터셋 검증
                    validation_results = self.dataset_builder.validate_dataset()
                    logger.info(f"📊 데이터셋 검증 결과:")
                    logger.info(f"  - 총 큐레이션된 데이터: {validation_results['total_curated']}")
                    logger.info(f"  - 카테고리 분포: {validation_results['category_distribution']}")
                    logger.info(f"  - 검증 통과: {'✅' if validation_results['validation_passed'] else '❌'}")
                
                self.dataset_builder.save_stats()
            
            # 학습 품질 보고서 생성
            if self.quality_monitor:
                logger.info("📊 학습 품질 보고서 생성 중...")
                report_path = self.quality_monitor.export_report()
                logger.info(f"✅ 품질 보고서 생성 완료: {report_path}")
                
                # 학습 요약 출력
                summary = self.quality_monitor.get_learning_summary()
                logger.info(f"📈 학습 요약:")
                logger.info(f"  - 총 반복: {summary['total_iterations']}")
                logger.info(f"  - 평균 품질 점수: {summary['average_quality_score']:.3f}")
                logger.info(f"  - 과적합 감지: {'⚠️ 예' if summary['overfitting_detected'] else '✅ 아니오'}")
            
            # 블랙리스트 초기화 (세션 종료 시)
            if self.failed_models_blacklist:
                logger.info(f"🔄 블랙리스트 초기화: {', '.join(self.failed_models_blacklist)}")
                self.failed_models_blacklist.clear()
    
    def _evaluate_answer_quality(self, question: Dict[str, Any], answer: str, model_name: str) -> float:
        """개선된 답변 품질 평가"""
        try:
            # 텍스트를 벡터로 변환 (간단한 예시 - 실제로는 더 정교한 임베딩 사용)
            # 여기서는 길이, 키워드 포함 여부 등을 특징으로 사용
            features = self._extract_features(question, answer)
            
            # PyTorch 텐서로 변환
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 품질 평가 모델 실행
            self.quality_assessor.eval()
            with torch.no_grad():
                quality_score = self.quality_assessor(feature_tensor).item()
            
            # 추가 휴리스틱 평가
            heuristic_score = self._heuristic_evaluation(question, answer)
            
            # 최종 점수 (가중 평균)
            final_score = 0.7 * quality_score + 0.3 * heuristic_score
            
            # Experience Buffer에 저장
            experience = {
                'question': question,
                'answer': answer,
                'model': model_name,
                'quality_score': final_score,
                'features': features
            }
            self.experience_buffer.add(experience)
            
            return final_score
            
        except Exception as e:
            logger.error(f"품질 평가 중 오류: {str(e)}")
            return 0.5  # 기본값
    
    def _extract_features(self, question: Dict[str, Any], answer: str) -> List[float]:
        """질문과 답변에서 특징 추출"""
        features = []
        
        # 1. 답변 길이 (정규화)
        features.append(min(len(answer) / 1000.0, 1.0))
        
        # 2. 키워드 포함 비율
        keywords = question.get('keywords', [])
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
        features.append(keyword_count / max(len(keywords), 1))
        
        # 3. 코드 블록 존재 여부
        features.append(1.0 if '```' in answer else 0.0)
        
        # 4. 완성도 (코드 블록이 올바르게 닫혔는지)
        code_blocks = answer.count('```')
        features.append(1.0 if code_blocks % 2 == 0 else 0.0)
        
        # 5. 언어 일관성
        question_lang = question.get('language', 'korean')
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in answer)
        has_english = any('a' <= char.lower() <= 'z' for char in answer)
        
        if question_lang == 'korean':
            features.append(1.0 if has_korean and not has_english else 0.5)
        else:
            features.append(1.0 if has_english and not has_korean else 0.5)
        
        # 6. 구조화 수준 (번호, 불릿 포인트 등)
        structured = any(marker in answer for marker in ['1.', '2.', '•', '-', '*'])
        features.append(1.0 if structured else 0.0)
        
        # 7. 주제 관련성 (간단한 키워드 매칭)
        topic_keywords = question.get('topic', '').split()
        topic_match = sum(1 for keyword in topic_keywords if keyword.lower() in answer.lower())
        features.append(min(topic_match / max(len(topic_keywords), 1), 1.0))
        
        # 768차원으로 패딩 (실제 임베딩 크기에 맞춤)
        while len(features) < 768:
            features.append(0.0)
            
        return features[:768]
    
    def _heuristic_evaluation(self, question: Dict[str, Any], answer: str) -> float:
        """휴리스틱 기반 평가"""
        score = 0.5  # 기본 점수
        
        # 답변 길이
        if len(answer) > 100:
            score += 0.1
        if len(answer) > 300:
            score += 0.1
            
        # 코드 예제 포함
        if '```' in answer and answer.count('```') >= 2:
            score += 0.15
            
        # 설명 구조화
        if any(marker in answer for marker in ['1.', '2.', '단계', 'Step']):
            score += 0.1
            
        # 주제별 특별 평가
        topic = question.get('topic', '')
        if 'Godot' in topic and any(godot_term in answer for godot_term in ['Node', 'Scene', '노드', '씬']):
            score += 0.05
        elif 'C#' in topic and any(csharp_term in answer for csharp_term in ['class', 'method', 'async']):
            score += 0.05
            
        return min(score, 1.0)
    
    def _improve_answer_with_feedback(self, question: Dict[str, Any], answer: str, model_name: str) -> str:
        """실시간 피드백을 통한 답변 개선"""
        try:
            # 답변 분석
            issues = self.feedback_system.analyze_partial_response(
                answer, 
                expected_language=question.get('language', 'korean')
            )
            
            if not issues:
                return answer
                
            # 개선 프롬프트 생성
            context = {
                'topic': question.get('topic', ''),
                'language': question.get('language', 'korean')
            }
            improvement_prompt = self.feedback_system.get_improvement_prompt(issues, answer, context)
            
            # 모델이 로드되어 있는지 확인
            if not self.currently_loaded_model or model_name != self.currently_loaded_model:
                if not self.load_model(model_name):
                    return answer
                    
            # 개선된 답변 생성
            pipe = self.model_cache[model_name]["pipeline"]
            improved_prompt = f"{question['question']}\n\n기존 답변:\n{answer}\n\n{improvement_prompt}"
            
            response = pipe(improved_prompt)
            improved_answer = response[0]['generated_text'] if response else answer
            
            # 피드백 기록
            self.feedback_system.record_feedback(
                improved_answer,
                self._heuristic_evaluation(question, improved_answer),
                issues
            )
            
            return improved_answer
            
        except Exception as e:
            logger.error(f"답변 개선 중 오류: {str(e)}")
            return answer
    
    def _get_task_category(self, topic: str) -> Optional[str]:
        """주제를 태스크 카테고리로 매핑"""
        topic_lower = topic.lower()
        
        if 'c#' in topic_lower or 'csharp' in topic_lower:
            return 'csharp'
        elif '한글' in topic_lower or '용어' in topic_lower or 'korean' in topic_lower:
            return 'korean'
        elif 'godot' in topic_lower:
            return 'godot'
        elif 'socket' in topic_lower:
            return 'socketio'
        elif 'ai' in topic_lower or '최적화' in topic_lower:
            return 'ai_optimization'
        else:
            return None
    
    def _update_multitask_network(self, question: Dict[str, Any], answer: str, quality_score: float, task_category: str):
        """멀티태스크 학습 네트워크 업데이트"""
        try:
            # 특징 추출
            features = self._extract_features(question, answer)
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 타겟 품질 점수
            target = torch.tensor([quality_score], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Forward pass
            self.multitask_network.train()
            predicted_quality = self.multitask_network(feature_tensor, task_category)
            
            # Loss 계산
            loss = F.mse_loss(predicted_quality, target)
            
            # Backward pass
            self.multitask_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.multitask_network.parameters(), 1.0)
            self.multitask_optimizer.step()
            
            # 학습률 스케줄러 업데이트
            self.multitask_scheduler.step(quality_score)
            
            logger.debug(f"멀티태스크 네트워크 업데이트 - 태스크: {task_category}, Loss: {loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"멀티태스크 네트워크 업데이트 중 오류: {str(e)}")
    
    def train_quality_assessor_batch(self):
        """Experience Buffer에서 배치로 품질 평가 모델 학습"""
        if len(self.experience_buffer) < 32:
            return
            
        try:
            # 배치 샘플링
            experiences, weights, indices = self.experience_buffer.sample(32)
            if experiences is None:
                return
                
            # 배치 데이터 준비
            features_batch = []
            targets_batch = []
            
            for exp in experiences:
                features_batch.append(exp['features'])
                targets_batch.append(exp['quality_score'])
                
            # 텐서로 변환
            features_tensor = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
            targets_tensor = torch.tensor(targets_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)
            
            # Forward pass
            self.quality_assessor.train()
            predictions = self.quality_assessor(features_tensor)
            
            # Weighted loss
            loss = F.mse_loss(predictions, targets_tensor, reduction='none')
            weighted_loss = (loss * weights_tensor).mean()
            
            # Backward pass
            self.quality_optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.quality_assessor.parameters(), 1.0)
            self.quality_optimizer.step()
            
            # TD errors for priority update
            td_errors = (predictions - targets_tensor).detach().cpu().numpy().flatten()
            self.experience_buffer.update_priorities(indices, td_errors)
            
            # 학습률 스케줄러 업데이트
            avg_quality = targets_tensor.mean().item()
            self.quality_scheduler.step(avg_quality)
            
            logger.debug(f"품질 평가 모델 학습 - Loss: {weighted_loss.item():.4f}, Avg Quality: {avg_quality:.2f}")
            
        except Exception as e:
            logger.error(f"품질 평가 모델 학습 중 오류: {str(e)}")
    
    def save_pytorch_models(self):
        """PyTorch 모델들 저장"""
        try:
            models_dir = self.learning_dir / "pytorch_models"
            models_dir.mkdir(exist_ok=True)
            
            # 품질 평가 모델 저장
            torch.save({
                'model_state_dict': self.quality_assessor.state_dict(),
                'optimizer_state_dict': self.quality_optimizer.state_dict(),
                'scheduler_state_dict': self.quality_scheduler.state_dict()
            }, models_dir / f"quality_assessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            
            # 멀티태스크 네트워크 저장
            torch.save({
                'model_state_dict': self.multitask_network.state_dict(),
                'optimizer_state_dict': self.multitask_optimizer.state_dict(),
                'scheduler_state_dict': self.multitask_scheduler.state_dict()
            }, models_dir / f"multitask_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            
            logger.info("📁 PyTorch 모델 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류: {str(e)}")
    
    def load_pytorch_models(self):
        """저장된 PyTorch 모델 로드"""
        try:
            models_dir = self.learning_dir / "pytorch_models"
            if not models_dir.exists():
                return
                
            # 최신 품질 평가 모델 로드
            quality_models = sorted(models_dir.glob("quality_assessor_*.pth"))
            if quality_models:
                checkpoint = torch.load(quality_models[-1], map_location=self.device)
                self.quality_assessor.load_state_dict(checkpoint['model_state_dict'])
                self.quality_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.quality_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"✅ 품질 평가 모델 로드: {quality_models[-1].name}")
                
            # 최신 멀티태스크 네트워크 로드
            multitask_models = sorted(models_dir.glob("multitask_network_*.pth"))
            if multitask_models:
                checkpoint = torch.load(multitask_models[-1], map_location=self.device)
                self.multitask_network.load_state_dict(checkpoint['model_state_dict'])
                self.multitask_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.multitask_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"✅ 멀티태스크 네트워크 로드: {multitask_models[-1].name}")
                
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")

    def _generate_session_report(self):
        """세션 보고서 생성"""
        if not self.current_session:
            return
        
        report = {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_hours": (datetime.now() - self.current_session.start_time).total_seconds() / 3600,
            "questions_asked": self.current_session.questions_asked,
            "successful_answers": self.current_session.successful_answers,
            "success_rate": self.current_session.successful_answers / max(1, self.current_session.questions_asked),
            "topics_covered": self.current_session.topics_covered,
            "models_used": self.current_session.models_used,
            "knowledge_gained": len(self.current_session.knowledge_gained),
            "memory_usage_history": self.memory_usage_history[-100:],  # 최근 100개만
            "experience_buffer_size": len(self.experience_buffer),
            "feedback_history": list(self.feedback_system.feedback_history)
        }
        
        # PyTorch 통계 추가
        if self.pytorch_system:
            report['pytorch_stats'] = self.pytorch_system.training_stats
        
        # 질문 다양성 메트릭 추가
        report['diversity_metrics'] = self.calculate_question_diversity_metrics()
        
        # 보고서 저장
        report_file = self.learning_dir / f"session_{self.current_session.session_id}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"세션 보고서 저장: {report_file}")

# 메인 실행 부분
async def main():
    """메인 실행 함수 (자동 재시작 기능 포함)"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoCI 연속 학습 시스템")
    parser.add_argument("duration", type=float, nargs='?', default=24, 
                        help="학습 시간 (시간 단위, 기본값: 24)")
    parser.add_argument("memory", type=float, nargs='?', default=32.0,
                        help="최대 메모리 사용량 (GB, 기본값: 32)")
    
    args = parser.parse_args()
    
    # 총 실행 시간 추적
    total_start_time = datetime.now()
    target_end_time = total_start_time + timedelta(hours=args.duration)
    restart_count = 0
    max_restarts = 5
    
    logger.info(f"🚀 AutoCI 학습 시작 - 목표 시간: {args.duration}시간")
    logger.info(f"💾 메모리 제한: {args.memory}GB")
    
    while datetime.now() < target_end_time and restart_count < max_restarts:
        try:
            # 시스템 초기화
            learning_system = ContinuousLearningSystem(max_memory_gb=args.memory)
            
            # 남은 시간 계산
            remaining_hours = (target_end_time - datetime.now()).total_seconds() / 3600
            if remaining_hours <= 0:
                break
            
            logger.info(f"📚 학습 세션 시작 - 남은 시간: {remaining_hours:.1f}시간")
            
            # 학습 실행
            await learning_system.continuous_learning_loop(duration_hours=remaining_hours)
            
            # 정상 완료
            logger.info("✅ 학습 세션 정상 완료")
            break
            
        except KeyboardInterrupt:
            logger.info("🛑 사용자에 의해 학습이 중단되었습니다")
            break
            
        except Exception as e:
            restart_count += 1
            logger.error(f"❌ 학습 중 오류 발생 (재시작 {restart_count}/{max_restarts}): {str(e)}")
            logger.exception("상세 오류 정보:")
            
            if restart_count < max_restarts:
                wait_time = min(60 * restart_count, 300)  # 최대 5분
                logger.info(f"⏳ {wait_time}초 후 자동 재시작...")
                await asyncio.sleep(wait_time)
                
                # 메모리 정리
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.critical("❌ 최대 재시작 횟수 초과 - 학습 중단")
                break
    
    # 최종 요약
    total_duration = (datetime.now() - total_start_time).total_seconds() / 3600
    logger.info(f"\n📊 최종 학습 요약:")
    logger.info(f"- 총 실행 시간: {total_duration:.1f}시간")
    logger.info(f"- 재시작 횟수: {restart_count}회")
    logger.info(f"- 상태: {'완료' if restart_count == 0 else '부분 완료'}")

if __name__ == "__main__":
    asyncio.run(main())