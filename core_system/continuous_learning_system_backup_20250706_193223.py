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

# Hugging Face 토큰
HF_TOKEN = "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"

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
        """PyTorch 스타일의 다양한 학습 질문 생성"""
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
        
        # RTX 2080 최적화 모델만 필터링
        available_optimized = []
        for model_name in self.available_models:
            if (model_name in rtx_2080_models and 
                self.available_models[model_name].get('rtx_2080_optimized', False)):
                available_optimized.append(model_name)
        
        if not available_optimized:
            logger.warning("❌ RTX 2080 최적화 모델이 없습니다.")
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
            
            # 모델 로드
            if not self.load_model(model_name):
                return {
                    "success": False,
                    "error": f"모델 {model_name} 로드 실패"
                }
            
            # 파이프라인 가져오기
            pipe = self.model_cache[model_name]["pipeline"]
            
            # 답변 생성
            logger.info(f"질문: {question['question'][:100]}...")
            
            start_time = time.time()
            response = pipe(question['question'])
            answer = response[0]['generated_text'] if response else "답변 생성 실패"
            
            generation_time = time.time() - start_time
            logger.info(f"답변 생성 시간: {generation_time:.2f}초")
            
            # 답변 품질 평가 (PyTorch 시스템 활용)
            quality_score = 0.7  # 기본값
            if self.pytorch_system and False:  # 임시 비활성화 - 학습되지 않은 모델이 0.505 반환하는 문제 해결
                quality_score = self.pytorch_system.assess_quality(answer)
                
                # 주제 분류
                classified_topic = self.pytorch_system.classify_topic(answer)
                logger.info(f"분류된 주제: {classified_topic}, 품질 점수: {quality_score:.2f}")
            
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
        
        try:
            while datetime.now() < end_time:
                # 메모리 체크
                memory_usage = self.get_memory_usage_percent()
                if memory_usage > 85:
                    logger.warning(f"메모리 사용량 높음: {memory_usage:.1f}%")
                    self.unload_current_model()
                    gc.collect()
                    await asyncio.sleep(5)
                
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
                
                # 학습 실행
                result = await self.ask_and_learn(question, model_name)
                
                if result['success']:
                    logger.info(f"✓ 학습 성공! 품질: {result['quality_score']:.2f}")
                    
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
                else:
                    logger.error(f"✗ 학습 실패: {result.get('error', '알 수 없는 오류')}")
                
                # PyTorch 배치 학습 (100개 경험마다)
                if len(pytorch_experiences) >= 100 and self.pytorch_system:
                    logger.info("🧠 PyTorch 배치 학습 시작...")
                    self.pytorch_system.train_on_experience(pytorch_experiences, epochs=5)
                    pytorch_experiences = []  # 리셋
                
                # 진행 상황 저장 (10분마다)
                if self.current_session.questions_asked % 10 == 0:
                    self._save_learning_progress()
                    # 다양성 메트릭 출력 (10개 질문마다)
                    self.log_diversity_metrics()
                
                # 대기 시간 (너무 빠른 반복 방지)
                await asyncio.sleep(random.uniform(5, 15))
                
        except KeyboardInterrupt:
            logger.info("학습이 사용자에 의해 중단되었습니다")
        except Exception as e:
            logger.error(f"학습 루프 오류: {str(e)}")
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
            "memory_usage_history": self.memory_usage_history[-100:]  # 최근 100개만
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
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AutoCI 연속 학습 시스템")
    parser.add_argument("duration", type=float, nargs='?', default=24, 
                        help="학습 시간 (시간 단위, 기본값: 24)")
    parser.add_argument("memory", type=float, nargs='?', default=32.0,
                        help="최대 메모리 사용량 (GB, 기본값: 32)")
    
    args = parser.parse_args()
    
    # 시스템 초기화
    learning_system = ContinuousLearningSystem(max_memory_gb=args.memory)
    
    # 학습 실행
    await learning_system.continuous_learning_loop(duration_hours=args.duration)

if __name__ == "__main__":
    asyncio.run(main())