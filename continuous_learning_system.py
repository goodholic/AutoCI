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
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

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
    csharp_concepts: List[str]
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
        
    def _initialize_learning_topics(self) -> List[LearningTopic]:
        """5가지 핵심 학습 주제 초기화 (DeepSeek-coder 최적화)"""
        topics = [
            # ... (기존 주제들)
            # 6️⃣ Godot 전문가 학습 (문서 기반)
            LearningTopic("godot_expert_nodes", "Godot 전문가", "노드와 씬 심층 분석", 5,
                         ["노드", "씬", "트리", "상속", "인스턴스"],
                         ["Node", "Scene", "SceneTree", "inheritance", "instance"],
                         "Godot 핵심 아키텍처"),
            LearningTopic("godot_expert_scripting", "Godot 전문가", "고급 스크립팅 기술", 5,
                         ["GDScript", "C#", "시그널", "코루틴", "툴 스크립트"],
                         ["GDScript", "C#", "Signal", "Coroutine", "Tool Script"],
                         "효율적인 게임 로직 구현"),
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
        usage_percent = self.get_memory_usage_percent()
        
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
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
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
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=150,  # RTX 2080 최적화
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1]:], 
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
        
    def select_model_for_question(self, question: Dict[str, Any]) -> str:
        """질문에 적합한 모델 선택 (5대 핵심 주제 DeepSeek-coder 최적화)"""
        # RTX 2080 최적화 모델만 고려
        rtx_2080_models = {
            "deepseek-coder-7b": {"priority": 10, "specialties": ["code", "csharp", "godot", "korean", "nakama"], "vram": 6},
            "phi3-mini": {"priority": 8, "specialties": ["reasoning", "math", "csharp"], "vram": 6},
            "llama-3.1-8b": {"priority": 7, "specialties": ["general", "korean", "csharp"], "vram": 7},
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
            return None
            
        # 5가지 핵심 주제 카테고리 확인
        core_categories = ["C# 프로그래밍", "한글 용어", "Godot 엔진", "Godot 네트워킹", "Nakama 서버"]
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
        
        # 1️⃣ C# 프로그래밍 특성
        if any(word in question_text + topic_text for word in ['code', 'programming', 'script', 'function', 'class', 'method', 'c#', 'csharp']):
            question_features.add('csharp')
            question_features.add('code')
        
        # 2️⃣ 한글 용어 특성
        if any(word in question_text + topic_text for word in ['korean', '한글', '한국어', '번역', '용어', '개념']):
            question_features.add('korean')
        
        # 3️⃣ Godot 엔진 특성
        if any(word in question_text + topic_text for word in ['godot', 'engine', '엔진', '노드', '씬', '아키텍처']):
            question_features.add('godot')
        
        # 4️⃣ Godot 네트워킹 특성
        if any(word in question_text + topic_text for word in ['multiplayer', 'network', '네트워킹', 'rpc', '동기화', 'AI제어']):
            question_features.add('networking')
        
        # 5️⃣ Nakama 서버 특성
        if any(word in question_text + topic_text for word in ['nakama', 'server', '서버', '매치메이킹', 'backend']):
            question_features.add('nakama')
        
        # 기타 특성
        if any(word in question_text for word in ['reasoning', 'math', '수학', '추론', '논리']):
            question_features.add('reasoning')
        
        # 모델별 점수 계산
        model_scores = {}
        for model_name in available_optimized:
            if model_name not in rtx_2080_models:
                continue
                
            model_config = rtx_2080_models[model_name]
            score = model_config['priority']
            
            # 🔥 핵심 주제에서 DeepSeek-coder 강력 우선순위
            if model_name == "deepseek-coder-7b":
                # 5가지 핵심 주제 중 하나라면 무조건 DeepSeek-coder 선택
                if is_core_topic:
                    score += 50  # 핵심 주제 특대 보너스
                    logger.info(f"🔥 핵심 주제 '{topic_category}' 감지! DeepSeek-coder 최우선 선택")
                
                # 세부 특성별 추가 보너스
                if 'code' in question_features or 'csharp' in question_features:
                    score += 25  # C# 코딩 특화 보너스
                if 'korean' in question_features:
                    score += 20  # 한글 번역 특화 보너스
                if 'godot' in question_features:
                    score += 20  # Godot 특화 보너스
                if 'networking' in question_features:
                    score += 15  # 네트워킹 특화 보너스
                if 'nakama' in question_features:
                    score += 15  # 서버 특화 보너스
                
                # 일반 질문에도 기본 보너스
                score += 10
            
            # 다른 모델들의 특기 분야
            elif model_name == "phi3-mini" and 'reasoning' in question_features:
                score += 15
            elif model_name == "gemma-4b" and 'korean' in question_features and not is_core_topic:
                score += 12  # 핵심 주제가 아닐 때만
            elif model_name == "llama-3.1-8b" and 'korean' in question_features and not is_core_topic:
                score += 10  # 핵심 주제가 아닐 때만
            
            # 특기 분야 매칭 보너스
            for specialty in model_config['specialties']:
                if specialty in question_features:
                    if model_name == "deepseek-coder-7b":
                        score += 12  # DeepSeek-coder에 더 높은 보너스
                    else:
                        score += 6
            
            # VRAM 효율성 보너스 (낮은 VRAM 사용량 선호)
            score += (8 - model_config['vram'])
            
            model_scores[model_name] = score
        
        # 최고 점수 모델 선택
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])[0]
            selected_score = model_scores[best_model]
            
            # 로그 출력 (실시간 UI용)
            if best_model == "deepseek-coder-7b" and is_core_topic:
                logger.info(f"🎯 핵심 주제 최적화: {best_model} 선택 (점수: {selected_score}) - {topic_category}")
                print(f"Selected model: {best_model} (🔥 DeepSeek-coder 핵심 주제)")
            else:
                logger.info(f"🎯 RTX 2080 최적화: {best_model} 선택 (점수: {selected_score})")
                print(f"Selected model: {best_model}")
            
            return best_model
        
        # 기본 우선순위: DeepSeek > Phi3 > Llama > Gemma > Mistral
        for fallback in ["deepseek-coder-7b", "phi3-mini", "llama-3.1-8b", "gemma-4b", "mistral-7b"]:
            if fallback in available_optimized:
                logger.info(f"🎯 RTX 2080 기본 모델: {fallback}")
                return fallback
        
        return None
        
    def _is_model_suitable(self, model_name: str, question: Dict[str, Any]) -> bool:
        """모델이 질문에 적합한지 확인"""
        if model_name not in self.available_models:
            return False
            
        model_features = self.available_models[model_name]['features']
        
        # 한글 질문인 경우
        if "korean" in question["language"]:
            return "korean" in model_features
            
        # 코드 관련 질문인 경우
        if question["type"] in ["example", "error", "optimize"]:
            return "code" in model_features or "csharp" in model_features
            
        return True  # 기본적으로 적합하다고 가정
        
    async def ask_model(self, model_name: str, question: Dict[str, Any]) -> Dict[str, Any]:
        """🎮 모델에 질문하고 답변 받기 - 완전 제어 모드"""
        # 모델 로드 확인 및 필요시 로드
        if not self.load_model(model_name):
            return {"error": f"Model {model_name} failed to load"}
            
        if model_name not in self.model_cache:
            return {"error": f"Model {model_name} not in cache"}
        
        # 🎯 AI 모델 완전 제어: 여러 번 시도하여 우리 기준에 맞는 답변 확보
        max_attempts = 3 if self.model_controller else 1
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                model_pipeline = self.model_cache[model_name]["pipeline"]
                
                # 메모리 상태 확인
                memory_before = self.get_memory_usage()
                if not self.check_memory_safety():
                    logger.warning(f"메모리 사용량 높음: {memory_before:.1f}GB")
                
                # 🎮 커스텀 프롬프트 (모델 컨트롤러가 있으면 더 세밀하게)
                if self.model_controller:
                    custom_prompt = self.model_controller.get_custom_prompt(model_name, question.get('type', 'general'))
                    system_prompt = custom_prompt
                else:
                    system_prompt = f"""C# programming and Godot expert. Answer about {question['topic']} in Korean with examples."""
                
                full_prompt = f"{system_prompt}\n\nQ: {question['question']}\nA:"
                
                # 모델 호출 (에러 추적 강화)
                start_time = time.time()
                
                # 입력 타입 확인
                if not isinstance(full_prompt, str):
                    raise ValueError(f"프롬프트가 문자열이 아님: {type(full_prompt)}")
                
                logger.debug(f"시도 {attempt}: 프롬프트 길이 {len(full_prompt)} 문자")
                
                # AI 응답 생성 시작 알림
                print(f"🤖 AI 응답 생성 중... (모델: {model_name})")
                logger.info(f"AI 응답 생성 시작: {model_name}")
                
                response = model_pipeline(full_prompt)  # optimized_generate는 prompt만 받음
                
                # 응답 완료 확인
                print(f"✅ AI 응답 생성 완료! (소요 시간: {time.time() - start_time:.1f}초)")
                
                if not response or len(response) == 0:
                    raise ValueError("모델이 빈 응답을 반환함")
                
                if not isinstance(response[0], dict) or 'generated_text' not in response[0]:
                    raise ValueError(f"잘못된 응답 형식: {type(response[0])}")
                
                answer_text = response[0]['generated_text'].strip()
                
                if not answer_text or len(answer_text) < 10:
                    raise ValueError("답변이 너무 짧거나 비어있음")
                
                response_time = time.time() - start_time
                
                # 🎯 품질 평가 (모델 컨트롤러가 있으면)
                if self.model_controller:
                    quality = self.model_controller.evaluate_response_quality(question, answer_text, model_name)
                    self.model_controller.log_response_quality(question, answer_text, quality, model_name)
                    
                    if quality.is_acceptable:
                        print(f"🎯 품질 통과 (시도 {attempt}): {quality.score:.2f}")
                        logger.info(f"✅ 품질 기준 통과: {model_name} (점수: {quality.score:.2f})")
                    else:
                        print(f"❌ 품질 실패 (시도 {attempt}): {quality.score:.2f} - {', '.join(quality.issues)}")
                        
                        # 재시도 결정
                        if self.model_controller.should_retry(quality, model_name, attempt):
                            logger.warning(f"품질 기준 미달, 재시도 중... (시도 {attempt}/{max_attempts})")
                            continue
                        else:
                            logger.warning(f"최대 시도 횟수 도달, 현재 답변 사용")
                
                # 메모리 사용량 추적
                memory_after = self.get_memory_usage()
                memory_delta = memory_after - memory_before
                
                final_answer = {
                    "model": model_name,
                    "question_id": question["id"],
                    "answer": answer_text,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt,
                    "memory_usage": {
                        "before_gb": memory_before,
                        "after_gb": memory_after,
                        "delta_gb": memory_delta
                    }
                }
                
                # 품질 정보 추가
                if self.model_controller:
                    final_answer["quality"] = {
                        "score": quality.score,
                        "is_acceptable": quality.is_acceptable,
                        "issues": quality.issues
                    }
                
                return final_answer
                
            except Exception as model_error:
                logger.error(f"생성 오류: {str(model_error)}")
                if attempt >= max_attempts:
                    logger.error(f"모든 시도 실패: {str(model_error)}")
                    # 🔍 에러는 error 필드에만 넣고 answer 필드는 넣지 않기
                    return {"error": str(model_error), "model": model_name, "attempts": attempt}
                else:
                    # 잠시 대기 후 재시도
                    await asyncio.sleep(1.0)
                    continue
        
        # 🔍 최대 시도 횟수 초과 시에도 error 필드에만 넣기
        return {"error": "최대 시도 횟수 초과", "model": model_name, "attempts": max_attempts}
            
    def analyze_answer(self, question: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
        """답변 분석 및 지식 추출 - 엄격한 성공/실패 판단"""
        # 🔍 1차: error 필드가 있으면 무조건 실패
        if "error" in answer:
            error_msg = answer["error"]
            logger.error(f"❌ 답변 실패 (error 필드): {error_msg}")
            return {"success": False, "error": error_msg, "model": answer.get("model", "unknown")}
        
        # 🔍 2차: 답변이 없거나 비어있는 경우
        answer_text = answer.get("answer", "")
        if not answer_text or len(answer_text.strip()) < 5:
            logger.error(f"❌ 답변 실패 (비어있음): '{answer_text}'")
            return {"success": False, "error": "답변이 비어있거나 너무 짧음"}
        
        # 🔍 3차: 실패 패턴 체크 (강화된 패턴)
        failure_patterns = [
            "죄송합니다. 답변 생성에 실패했습니다",
            "답변을 생성할 수 없습니다",
            "오류가 발생했습니다", 
            "생성에 실패",
            "죄송합니다",
            "잘 모르겠습니다",
            "확실하지 않습니다",
            "답변 생성에 실패"
        ]
        
        answer_lower = answer_text.lower()
        for pattern in failure_patterns:
            if pattern.lower() in answer_lower:
                logger.error(f"❌ 답변 실패 (실패 패턴): '{pattern}' 감지")
                return {"success": False, "error": f"실패 패턴 감지: {pattern}"}
        
        # 🔍 4차: 최소 품질 체크
        if len(answer_text.strip()) < 20:
            logger.error(f"❌ 답변 실패 (너무 짧음): 길이 {len(answer_text)}")
            return {"success": False, "error": f"답변이 너무 짧음 (길이: {len(answer_text)})"}
        
        # ✅ 여기까지 통과하면 성공
        logger.info(f"✅ 답변 성공 (길이: {len(answer_text)})")
            
        analysis = {
            "success": True,
            "quality_score": 0,
            "extracted_knowledge": {},
            "new_patterns": [],
            "improvements": []
        }
        
        # 답변 품질 평가 (간단한 휴리스틱)
        quality_factors = {
            "length": len(answer_text) > 100,
            "has_code": "```" in answer_text or "class" in answer_text or "public" in answer_text,
            "has_korean": any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in answer_text),
            "has_explanation": any(word in answer_text.lower() for word in ["because", "therefore", "이유", "때문", "따라서"]),
            "has_example": any(word in answer_text.lower() for word in ["example", "예제", "예시", "다음"])
        }
        
        analysis["quality_score"] = sum(1 for factor in quality_factors.values() if factor) / len(quality_factors)
        
        # 지식 추출
        if question["type"] == "translate" and quality_factors["has_korean"]:
            # 한글 번역 저장
            for keyword in question["keywords"]:
                if keyword in answer_text:
                    self.knowledge_base["korean_translations"][keyword] = answer_text[:200]
                    
        elif question["type"] == "example" and quality_factors["has_code"]:
            # 코드 패턴 저장
            code_pattern = {
                "topic": question["topic"],
                "code": answer_text,
                "language": question["language"]
            }
            self.knowledge_base["csharp_patterns"][question["topic"]] = code_pattern
            
        elif question["type"] == "error":
            # 일반적인 오류 패턴 저장
            self.knowledge_base["common_errors"][question["topic"]] = answer_text[:300]
            
        # 새로운 패턴 발견
        if analysis["quality_score"] > 0.7:
            analysis["new_patterns"].append({
                "topic": question["topic"],
                "pattern": "High quality answer",
                "model": answer["model"]
            })
            
        return analysis
        
    def save_qa_pair(self, question: Dict[str, Any], answer: Dict[str, Any], analysis: Dict[str, Any]):
        """질문-답변 쌍 저장"""
        qa_data = {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "session_id": self.current_session.session_id if self.current_session else None
        }
        
        # 날짜별 디렉토리
        today = datetime.now().strftime("%Y%m%d")
        daily_dir = self.answers_dir / today
        daily_dir.mkdir(exist_ok=True)
        
        # 파일 저장
        filename = f"{question['id']}.json"
        with open(daily_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
    async def learning_cycle(self, duration_hours: int = 24):
        """학습 사이클 실행 (메모리 최적화)"""
        logger.info(f"Starting {duration_hours} hour learning cycle...")
        logger.info(f"Max memory limit: {self.max_memory_gb:.1f}GB")
        
        # 기존 학습 진행 상태 표시
        if self.learning_progress["total_hours"] > 0:
            logger.info(f"📚 기존 학습 진행 상태:")
            logger.info(f"  - 총 학습 시간: {self.learning_progress['total_hours']:.1f}시간")
            logger.info(f"  - 총 질문 수: {self.learning_progress['total_questions']}")
            logger.info(f"  - 성공한 답변: {self.learning_progress['total_successful']}")
            if self.learning_progress['total_questions'] > 0:
                overall_success_rate = (self.learning_progress['total_successful'] / 
                                      self.learning_progress['total_questions'] * 100)
                logger.info(f"  - 전체 성공률: {overall_success_rate:.1f}%")
            logger.info(f"  - 학습한 주제 수: {len(self.learning_progress['topics_progress'])}")
            print(f"\n📊 누적 학습 통계:")
            print(f"  총 {self.learning_progress['total_hours']:.1f}시간 학습")
            print(f"  {self.learning_progress['total_questions']}개 질문 중 {self.learning_progress['total_successful']}개 성공")
            print(f"  {len(self.learning_progress['topics_progress'])}개 주제 학습\n")
        
        # 세션 시작
        self.current_session = LearningSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now(),
            topics_covered=[],
            questions_asked=0,
            successful_answers=0,
            models_used={model: 0 for model in self.available_models.keys()},
            knowledge_gained=[]
        )
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        cycle_count = 0
        model_rotation_count = 0
        
        while datetime.now() < end_time:
            cycle_count += 1
            logger.info(f"\n--- Learning Cycle {cycle_count} ---")
            
            # 메모리 상태 체크
            current_memory = self.get_memory_usage()
            memory_percent = self.get_memory_usage_percent()
            logger.info(f"Memory: {current_memory:.1f}GB ({memory_percent:.1f}%)")
            
            # 메모리 임계치 초과시 모델 교체 (저사양 최적화: 10사이클마다)
            if not self.check_memory_safety() or (cycle_count % 10 == 0):
                self.unload_current_model()
                model_rotation_count += 1
                logger.info(f"Model rotation #{model_rotation_count}")
            
            # 진행형 주제 선택 (난이도 기반)
            if self.progressive_manager:
                # 현재 난이도 확인
                current_difficulty = self.progressive_manager.get_current_difficulty()
                
                # 진행 상태 요약 출력 (5사이클마다)
                if cycle_count % 5 == 1:
                    summary = self.progressive_manager.get_progress_summary()
                    print(f"\n📊 학습 진행 상태:")
                    print(f"  현재 난이도: {summary['difficulty_name']} (레벨 {current_difficulty})")
                    print(f"  전체 진행률: {summary['overall']['total_questions']}문제, 성공률 {summary['overall']['success_rate']:.1%}")
                    for diff, info in summary['difficulties'].items():
                        status = "✅ 마스터" if info['mastered'] else "📖 학습중"
                        print(f"  난이도 {diff} ({info['name']}): {info['total']}문제, 성공률 {info['rate']:.1%} {status}")
                    print()
                
                # 난이도에 맞는 주제 선택
                topic = self.progressive_manager.select_topic_by_difficulty(self.learning_topics, current_difficulty)
                if not topic:
                    # 폴백: 랜덤 선택
                    topic = random.choice(self.learning_topics)
                    logger.warning(f"진행형 선택 실패, 랜덤 선택: {topic.topic}")
            else:
                # 진행형 관리자가 없으면 기존 랜덤 방식
                topic = random.choice(self.learning_topics)
            
            # 5가지 핵심 주제 감지
            core_categories = ["C# 프로그래밍", "한글 용어", "Godot 엔진", "Godot 네트워킹", "Nakama 서버"]
            is_core_topic = topic.category in core_categories
            
            # 질문 생성
            question = self.generate_question(topic)
            
            # 실시간 UI용 출력
            print(f"Topic: {topic.topic} | Category: {topic.category} | Type: {question['type']}")
            if is_core_topic:
                print(f"🔥 핵심 주제 감지: {topic.category} - DeepSeek-coder 최우선 사용!")
                logger.info(f"🔥 핵심 주제: {topic.category} - {topic.topic}")
            else:
                logger.info(f"📚 일반 주제: {topic.category} - {topic.topic}")
            
            logger.info(f"Topic: {topic.topic} | Type: {question['type']} | Language: {question['language']}")
            logger.info(f"Question: {question['question'][:100]}...")
            
            # 모델 선택 (메모리 고려)
            model_name = self.select_model_for_question(question)
            if not model_name:
                logger.error("No models available")
                break
                
            logger.info(f"Selected model: {model_name}")
            
            # 질문하고 답변 받기
            answer = await self.ask_model(model_name, question)
            
            # 답변 분석 (에러 체크 포함)
            analysis = self.analyze_answer(question, answer)
            
            # 🎮 실시간 AI 제어 상태 출력
            if not analysis.get("success", False):
                error_msg = analysis.get('error', '알 수 없는 오류')
                attempts = answer.get("attempts", 1)
                print(f"❌ 제어 실패: {error_msg} (시도: {attempts})")
                logger.error(f"답변 실패: {error_msg}")
            else:
                answer_preview = answer.get('answer', '')[:100]
                attempts = answer.get("attempts", 1)
                quality_info = answer.get("quality", {})
                
                if quality_info:
                    score = quality_info.get("score", 0.0)
                    is_acceptable = quality_info.get("is_acceptable", False)
                    quality_status = "🎯 품질 통과" if is_acceptable else "⚠️ 품질 미달"
                    print(f"✅ {quality_status}: {answer_preview}... (점수: {score:.2f}, 시도: {attempts})")
                else:
                    print(f"✅ 답변 성공: {answer_preview}... (시도: {attempts})")
                
                logger.info(f"답변 성공 (길이: {len(answer.get('answer', ''))}, 품질: {quality_info.get('score', 'N/A')})")
            
            # 세션 업데이트 (모든 시도 기록)
            self.current_session.questions_asked += 1
            self.current_session.models_used[model_name] += 1
            if topic.topic not in self.current_session.topics_covered:
                self.current_session.topics_covered.append(topic.topic)
                
            # 성공한 경우에만 성공 카운트 증가
            success = analysis.get("success", False)
            if success:
                self.current_session.successful_answers += 1
                logger.info(f"✅ 성공한 답변 수: {self.current_session.successful_answers}")
            else:
                logger.warning(f"❌ 실패한 답변. 이유: {analysis.get('error', '알 수 없음')}")
            
            # 진행형 학습 관리자에 결과 업데이트
            if self.progressive_manager:
                self.progressive_manager.update_topic_progress(topic.id, topic.difficulty, success)
                    
            # 결과 저장
            self.save_qa_pair(question, answer, analysis)
            
            # AI 답변이 있는 경우 학습 시간 확보
            if analysis.get("success", False) and answer.get('answer'):
                answer_length = len(answer.get('answer', ''))
                # 답변 길이에 따른 학습 시간 계산 (100자당 2초)
                learning_time = max(5.0, min(30.0, answer_length / 100 * 2))
                
                print(f"\n📖 답변 학습 중... ({learning_time:.1f}초)")
                logger.info(f"답변 학습 시간: {learning_time:.1f}초 (답변 길이: {answer_length}자)")
                
                # 답변 내용 일부 표시 (학습 중임을 보여주기 위해)
                answer_text = answer.get('answer', '')
                if len(answer_text) > 200:
                    print(f"💭 학습 내용: {answer_text[:200]}...")
                else:
                    print(f"💭 학습 내용: {answer_text}")
                
                await asyncio.sleep(learning_time)
                print(f"✅ 학습 완료!\n")
            
            # 지식 베이스 업데이트
            if cycle_count % 10 == 0:
                self._save_knowledge_base()
                self._save_learning_progress()  # 진행 상태도 함께 저장

            # 50 사이클마다 웹에서 새로운 정보 수집
            if INFORMATION_GATHERER_AVAILABLE and cycle_count % 50 == 0:
                print("🌐 웹에서 새로운 코드 정보를 수집합니다...")
                gatherer = get_information_gatherer()
                asyncio.create_task(gatherer.gather_and_process_csharp_code())
                
            # 메모리 사용량 히스토리 저장 및 가비지 컬렉션 (저사양 최적화)
            if cycle_count % 5 == 0:
                self.save_memory_usage_log()
                gc.collect()  # 더 빈번한 가비지 컬렉션
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # GPU 메모리 정리
                
            # 진행 상황 출력 (실시간 UI용 - 정확한 성공률)
            success_rate = (self.current_session.successful_answers / 
                          self.current_session.questions_asked * 100)
            failed_count = self.current_session.questions_asked - self.current_session.successful_answers
            print(f"Progress: {self.current_session.questions_asked} questions, {success_rate:.1f}% success ({self.current_session.successful_answers}✅/{failed_count}❌), {model_rotation_count} rotations")
            logger.info(f"Progress: {self.current_session.questions_asked} questions, "
                       f"{success_rate:.1f}% success rate ({self.current_session.successful_answers} success, {failed_count} failed), {model_rotation_count} rotations")
            
            # 다음 질문까지 짧은 휴식
            wait_time = random.uniform(3, 8)
            print(f"⏳ 다음 질문까지 {wait_time:.1f}초 대기...")
            await asyncio.sleep(wait_time)
            
        # 최종 메모리 정리
        self.unload_current_model()
        
        # 세션 종료
        self.save_session_summary()
        
    def save_session_summary(self):
        """세션 요약 저장"""
        if not self.current_session:
            return
            
        summary = {
            "session": asdict(self.current_session),
            "duration": str(datetime.now() - self.current_session.start_time),
            "knowledge_base_size": {
                "patterns": len(self.knowledge_base["csharp_patterns"]),
                "translations": len(self.knowledge_base["korean_translations"]),
                "errors": len(self.knowledge_base["common_errors"])
            }
        }
        
        summary_file = self.learning_dir / f"session_{self.current_session.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
        # 진행 상태 저장
        self._save_learning_progress()
        
        # 세션을 완료 목록에 추가
        if self.current_session.session_id not in self.learning_progress["sessions_completed"]:
            self.learning_progress["sessions_completed"].append(self.current_session.session_id)
            self._save_learning_progress()
            
        logger.info(f"\nSession Summary:")
        logger.info(f"Duration: {summary['duration']}")
        logger.info(f"Questions: {self.current_session.questions_asked}")
        logger.info(f"Success Rate: {self.current_session.successful_answers / max(1, self.current_session.questions_asked) * 100:.1f}%")
        logger.info(f"Topics Covered: {len(self.current_session.topics_covered)}")
        logger.info(f"Models Used: {self.current_session.models_used}")
        
        # 누적 통계 표시
        logger.info(f"\n📊 누적 학습 통계:")
        logger.info(f"총 학습 시간: {self.learning_progress['total_hours']:.1f}시간")
        logger.info(f"총 질문 수: {self.learning_progress['total_questions']}")
        logger.info(f"총 성공 답변: {self.learning_progress['total_successful']}")
        logger.info(f"완료된 세션 수: {len(self.learning_progress['sessions_completed'])}")
        
    def generate_learning_report(self):
        """학습 보고서 생성"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_sessions": 0,
            "total_questions": 0,
            "total_successful": 0,
            "topics_mastered": [],
            "korean_vocabulary": len(self.knowledge_base["korean_translations"]),
            "code_patterns": len(self.knowledge_base["csharp_patterns"]),
            "model_performance": {}
        }
        
        # 모든 세션 분석
        for session_file in self.learning_dir.glob("session_*.json"):
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                report["total_sessions"] += 1
                report["total_questions"] += session_data["session"]["questions_asked"]
                report["total_successful"] += session_data["session"]["successful_answers"]
                
        # 보고서 저장
        report_file = self.learning_dir / f"learning_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\nLearning Report Generated: {report_file}")
        return report
        
    def save_memory_usage_log(self):
        """메모리 사용량 로그 저장"""
        if not self.memory_usage_history:
            return
            
        log_file = self.learning_dir / "memory_usage.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory_usage_history, f, indent=2, ensure_ascii=False)

async def main():
    """메인 함수"""
    # 메모리 제한 설정 (기본 32GB, 명령행에서 변경 가능)
    max_memory = 32.0
    if len(sys.argv) > 2:
        try:
            max_memory = float(sys.argv[2])
            logger.info(f"메모리 제한을 {max_memory}GB로 설정했습니다.")
        except ValueError:
            logger.warning("잘못된 메모리 제한값. 기본값 32GB를 사용합니다.")
    
    system = ContinuousLearningSystem(max_memory_gb=max_memory)
    
    if not system.available_models:
        logger.error("No models available. Please run install_llm_models.py first.")
        return
        
    # 학습 시간 설정 (기본 24시간)
    duration = 24
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            logger.error("Invalid duration. Using default 24 hours.")
            
    logger.info(f"Starting continuous learning for {duration} hours...")
    
    # DeepSeek-coder 우선 확인 및 안내
    deepseek_available = (
        "deepseek-coder-7b" in system.available_models and 
        system.available_models["deepseek-coder-7b"].get('status') == 'installed'
    )
    
    # 실시간 UI용 DeepSeek-coder 상태 출력
    if deepseek_available:
        print("🔥 DeepSeek-coder-v2 6.7B 모델을 5가지 핵심 주제에 최우선 사용합니다:")
        print("   1️⃣ C# 프로그래밍 → DeepSeek-coder 특화")
        print("   2️⃣ 한글 용어 → DeepSeek-coder 번역")
        print("   3️⃣ Godot 엔진 → DeepSeek-coder 엔진")
        print("   4️⃣ Godot 네트워킹 → DeepSeek-coder 네트워킹")
        print("   5️⃣ Nakama 서버 → DeepSeek-coder 서버")
        logger.info("🔥 DeepSeek-coder-v2 6.7B 모델을 5가지 핵심 주제에 최우선 사용합니다")
    else:
        print("⚠️  DeepSeek-coder가 설치되지 않았습니다.")
        print("💡 python download_deepseek_coder.py로 설치하면 더 나은 학습 가능")
        logger.warning("⚠️  DeepSeek-coder가 설치되지 않았습니다.")
        logger.info("💡 python download_deepseek_coder.py로 설치하면 더 나은 학습 가능")
    
    logger.info(f"Available models: {list(system.available_models.keys())}")
    logger.info(f"Memory limit: {max_memory}GB")
    
    # 초기 메모리 상태 확인
    initial_memory = system.get_memory_usage()
    print(f"📊 초기 메모리 사용량: {initial_memory:.1f}GB / {max_memory}GB")
    print(f"📚 사용 가능한 모델: {list(system.available_models.keys())}")
    print(f"⏰ 학습 시간: {duration}시간")
    print("🚀 학습을 시작합니다...")
    print("=" * 60)
    
    logger.info(f"Initial memory usage: {initial_memory:.1f}GB")
    
    try:
        # 학습 사이클 실행
        await system.learning_cycle(duration)
        
        # 최종 보고서 생성
        system.generate_learning_report()
        
    except KeyboardInterrupt:
        logger.info("\nLearning interrupted by user.")
        system.save_session_summary()
        system.generate_learning_report()
        
    except Exception as e:
        logger.error(f"Error during learning: {str(e)}")
        system.save_session_summary()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 AutoCI 저사양 최적화 AI 학습 시스템")
    print("="*60)
    print("💻 RTX 2080 GPU 8GB, 32GB 메모리 환경 최적화")
    print("🚀 autoci learn low 명령어에서 실행됨")
    print("🔥 DeepSeek-coder-v2 6.7B 최우선 사용")
    print("📚 5가지 핵심 주제: C#, 한글, Godot 엔진, Godot 네트워킹, Nakama 서버")
    print("="*60 + "\n")
    
    # 직접 실행
    asyncio.run(main())