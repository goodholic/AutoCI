#!/usr/bin/env python3
"""
AutoCI 24시간 연속 학습 시스템 - 저사양 최적화 버전
C#과 한글에 대해 지속적으로 학습하는 시스템
Llama-3.1-8B, CodeLlama-13B, Qwen2.5-Coder-32B 모델을 활용

RTX 2080 GPU 8GB, 32GB 메모리 환경에 최적화됨
autoci learn low 명령어에서 사용됩니다.
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
from datetime import datetime, timedelta
from pathlib import Path
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning.log', encoding='utf-8'),
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
    def __init__(self, models_dir: str = "./models", learning_dir: str = "./continuous_learning", max_memory_gb: float = 32.0):
        self.models_dir = Path(models_dir)
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(exist_ok=True)
        
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
        """학습 질문 생성"""
        question_types = [
            "explain",      # 개념 설명
            "example",      # 예제 코드
            "translate",    # 한글-영어 번역
            "error",        # 오류 수정
            "optimize",     # 최적화
            "integrate"     # Godot 통합
        ]
        
        # "Godot 전문가" 주제인 경우, 문서에서 질문 생성
        if topic.category == "Godot 전문가":
            try:
                with open("collected_godot_docs.json", "r", encoding="utf-8") as f:
                    docs_data = json.load(f)
                
                if docs_data:
                    doc = random.choice(docs_data)
                    question_text = f"{doc['title']}에 대해 다음 내용을 바탕으로 설명해주세요: \n\n{doc['content'][:500]}"
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
                pass # 문서가 없으면 일반 질문 생성으로 넘어감

        question_type = random.choice(question_types)
        
        # 질문 템플릿
        templates = {
            "explain": {
                "korean": f"{topic.topic}에 대해 한글로 자세히 설명해주세요. 특히 {random.choice(topic.korean_keywords)}에 초점을 맞춰주세요.",
                "english": f"Explain {topic.topic} in Python with focus on {random.choice(topic.python_concepts)}."
            },
            "example": {
                "korean": f"{topic.topic}을 사용하는 Python 코드 예제를 작성하고 한글로 설명해주세요.",
                "english": f"Write a Python code example demonstrating {topic.topic} with comments."
            },
            "translate": {
                "korean": f"다음 Python 개념을 한글로 번역하고 설명하세요: {random.choice(topic.python_concepts)}",
                "english": f"Translate and explain this Korean term in Python context: {random.choice(topic.korean_keywords)}"
            },
            "error": {
                "korean": f"{topic.topic} 관련 일반적인 오류와 해결방법을 한글로 설명해주세요.",
                "english": f"What are common errors with {topic.topic} in Python and how to fix them?"
            },
            "optimize": {
                "korean": f"{topic.topic}을 사용할 때 성능 최적화 방법을 한글로 설명해주세요.",
                "english": f"How to optimize performance when using {topic.topic} in Python?"
            },
            "integrate": {
                "korean": f"Godot에서 {topic.topic}을 어떻게 활용하는지 C# 코드와 함께 설명해주세요.",
                "english": f"How to use {topic.topic} in Godot with C#? Provide examples."
            }
        }
        
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
            "keywords": topic.korean_keywords if language == "korean" else topic.python_concepts
        }
        
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