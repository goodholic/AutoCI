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
        
        # 메모리 관리 설정 (저사양 최적화)
        self.max_memory_gb = max_memory_gb
        self.memory_threshold = 0.70  # 70% 사용 시 모델 언로드 (더 보수적)
        self.currently_loaded_model = None
        self.model_cache = {}  # 언로드된 모델 정보 저장
        
        # 학습 데이터 디렉토리
        self.questions_dir = self.learning_dir / "questions"
        self.answers_dir = self.learning_dir / "answers"
        self.knowledge_base_dir = self.learning_dir / "knowledge_base"
        
        for dir in [self.questions_dir, self.answers_dir, self.knowledge_base_dir]:
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
        
    def _initialize_learning_topics(self) -> List[LearningTopic]:
        """5가지 핵심 학습 주제 초기화 (DeepSeek-coder 최적화)"""
        topics = [
            # 1️⃣ C# 프로그래밍 언어 전문 학습 (DeepSeek-coder 특화)
            LearningTopic("core_csharp_basics", "C# 프로그래밍", "C# 기초 문법", 2,
                         ["변수", "타입", "연산자", "조건문", "반복문", "배열"],
                         ["int", "string", "bool", "var", "if", "for", "foreach", "array"],
                         "Godot Node 기본 프로퍼티"),
            LearningTopic("core_csharp_oop", "C# 프로그래밍", "객체지향 프로그래밍", 3,
                         ["클래스", "객체", "상속", "다형성", "캡슐화", "인터페이스"],
                         ["class", "object", "inheritance", "polymorphism", "interface", "abstract"],
                         "Godot 노드 상속 구조"),
            LearningTopic("core_csharp_advanced", "C# 프로그래밍", "고급 C# 기능", 4,
                         ["제네릭", "비동기", "LINQ", "델리게이트", "람다", "속성"],
                         ["generics", "async", "await", "Task", "LINQ", "delegate", "lambda"],
                         "Godot 고급 스크립팅"),
            
            # 2️⃣ 한글 프로그래밍 용어 학습 (DeepSeek-coder 번역 특화)
            LearningTopic("core_korean_translation", "한글 용어", "프로그래밍 용어 번역", 2,
                         ["변수", "함수", "클래스", "객체", "상속", "인터페이스", "알고리즘"],
                         ["variable", "function", "class", "object", "inheritance", "interface", "algorithm"],
                         "Godot 용어 한글화"),
            LearningTopic("core_korean_concepts", "한글 용어", "한국어 코딩 개념", 3,
                         ["자료구조", "디자인패턴", "아키텍처", "프레임워크", "라이브러리"],
                         ["data structure", "design pattern", "architecture", "framework", "library"],
                         "Godot 아키텍처 이해"),
            
            # 3️⃣ Godot 엔진 개발 방향성 분석 (DeepSeek-coder Godot 특화)
            LearningTopic("core_godot_architecture", "Godot 엔진", "Godot 4.x 아키텍처", 4,
                         ["노드시스템", "씬트리", "리소스", "시그널", "렌더링"],
                         ["Node", "SceneTree", "Resource", "Signal", "RenderingServer"],
                         "현대적 게임 엔진 설계"),
            LearningTopic("core_godot_future", "Godot 엔진", "Godot 미래 방향성", 5,
                         ["웹어셈블리", "모바일최적화", "VR지원", "AI통합", "클라우드"],
                         ["WebAssembly", "mobile", "VR", "AI", "cloud", "C# bindings"],
                         "차세대 게임 개발"),
            
            # 4️⃣ Godot 내장 네트워킹 (AI 제어) (DeepSeek-coder 네트워킹 특화)
            LearningTopic("core_godot_networking", "Godot 네트워킹", "MultiplayerAPI 시스템", 4,
                         ["멀티플레이어", "서버", "클라이언트", "동기화", "RPC", "피어"],
                         ["MultiplayerAPI", "server", "client", "sync", "RPC", "peer"],
                         "실시간 멀티플레이어"),
            LearningTopic("core_godot_ai_network", "Godot 네트워킹", "AI 네트워크 제어", 5,
                         ["AI제어", "자동동기화", "지능형매칭", "예측보상", "최적화"],
                         ["AI control", "auto sync", "intelligent matching", "prediction", "optimization"],
                         "AI 기반 네트워킹"),
            
            # 5️⃣ Nakama 서버 개발 (AI 최적화) (DeepSeek-coder 서버 특화)
            LearningTopic("core_nakama_basics", "Nakama 서버", "Nakama 기본 구조", 3,
                         ["게임서버", "인증", "세션", "매치메이킹", "리더보드"],
                         ["game server", "authentication", "session", "matchmaking", "leaderboard"],
                         "백엔드 서비스 통합"),
            LearningTopic("core_nakama_ai", "Nakama 서버", "AI 통합 Nakama", 5,
                         ["AI매칭", "지능형스토리지", "자동스케일링", "예측분석"],
                         ["AI matching", "intelligent storage", "auto scaling", "predictive analytics"],
                         "차세대 게임 백엔드"),
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
            logger.info(f"{model_name} 모델을 로드합니다...")
            info = self.available_models[model_name]
            model_id = info['model_id']
            
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
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                token=hf_token,
                trust_remote_code=True
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
            
            if hf_token:
                model_kwargs["token"] = hf_token
            
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
        """모델에 질문하고 답변 받기"""
        # 모델 로드 확인 및 필요시 로드
        if not self.load_model(model_name):
            return {"error": f"Model {model_name} failed to load"}
            
        if model_name not in self.model_cache:
            return {"error": f"Model {model_name} not in cache"}
            
        try:
            model_pipeline = self.model_cache[model_name]["pipeline"]
            
            # 메모리 상태 확인
            memory_before = self.get_memory_usage()
            if not self.check_memory_safety():
                logger.warning(f"메모리 사용량 높음: {memory_before:.1f}GB")
            
            # 프롬프트 구성 (간단화)
            system_prompt = f"""C# programming and Godot expert. Answer about {question['topic']} in Korean with examples."""
            full_prompt = f"{system_prompt}\n\nQ: {question['question']}\nA:"
            
            # 모델 호출 (새로운 간단한 방식)
            start_time = time.time()
            response = model_pipeline(full_prompt)
            
            answer_text = response[0]['generated_text'].strip()
            response_time = time.time() - start_time
            
            # 메모리 사용량 추적
            memory_after = self.get_memory_usage()
            memory_delta = memory_after - memory_before
            
            return {
                "model": model_name,
                "question_id": question["id"],
                "answer": answer_text,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
                "memory_usage": {
                    "before_gb": memory_before,
                    "after_gb": memory_after,
                    "delta_gb": memory_delta
                }
            }
            
        except Exception as e:
            logger.error(f"Error with model {model_name}: {str(e)}")
            return {"error": str(e), "model": model_name}
            
    def analyze_answer(self, question: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
        """답변 분석 및 지식 추출"""
        if "error" in answer:
            return {"success": False, "error": answer["error"]}
            
        analysis = {
            "success": True,
            "quality_score": 0,
            "extracted_knowledge": {},
            "new_patterns": [],
            "improvements": []
        }
        
        answer_text = answer["answer"]
        
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
            
            # 랜덤 주제 선택 (5가지 핵심 주제 강화)
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
            
            # 답변 분석
            analysis = self.analyze_answer(question, answer)
            
            # 세션 업데이트
            self.current_session.questions_asked += 1
            if analysis.get("success", False):
                self.current_session.successful_answers += 1
                self.current_session.models_used[model_name] += 1
                if topic.topic not in self.current_session.topics_covered:
                    self.current_session.topics_covered.append(topic.topic)
                    
            # 결과 저장
            self.save_qa_pair(question, answer, analysis)
            
            # 지식 베이스 업데이트
            if cycle_count % 10 == 0:
                self._save_knowledge_base()
                
            # 메모리 사용량 히스토리 저장 및 가비지 컬렉션 (저사양 최적화)
            if cycle_count % 5 == 0:
                self.save_memory_usage_log()
                gc.collect()  # 더 빈번한 가비지 컬렉션
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # GPU 메모리 정리
                
            # 진행 상황 출력 (실시간 UI용)
            success_rate = (self.current_session.successful_answers / 
                          self.current_session.questions_asked * 100)
            print(f"Progress: {self.current_session.questions_asked} questions, {success_rate:.1f}% success, {model_rotation_count} rotations")
            logger.info(f"Progress: {self.current_session.questions_asked} questions, "
                       f"{success_rate:.1f}% success rate, {model_rotation_count} rotations")
            
            # 메모리 상황에 따른 대기 시간 조정
            if not self.check_memory_safety():
                wait_time = random.uniform(15, 30)  # 메모리 부족시 더 오래 대기
                logger.info(f"Memory high, waiting {wait_time:.1f}s...")
            else:
                wait_time = random.uniform(5, 15)  # 정상시 짧은 대기
                
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
            
        logger.info(f"\nSession Summary:")
        logger.info(f"Duration: {summary['duration']}")
        logger.info(f"Questions: {self.current_session.questions_asked}")
        logger.info(f"Success Rate: {self.current_session.successful_answers / max(1, self.current_session.questions_asked) * 100:.1f}%")
        logger.info(f"Topics Covered: {len(self.current_session.topics_covered)}")
        logger.info(f"Models Used: {self.current_session.models_used}")
        
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