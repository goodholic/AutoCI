#!/usr/bin/env python3
"""
기업급 AI 모델 통합 및 최적화 시스템

여러 AI 모델을 통합 관리하고 최적화하여 상용 수준의 성능을 제공하는 시스템
- 다중 모델 앙상블
- 동적 모델 선택
- 실시간 성능 모니터링
- 자동 스케일링
- 메모리 최적화
- 에러 복구
"""

import os
import sys
import json
import time
import asyncio
import threading
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np
import psutil
import gc
import logging

# AI 프레임워크
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        AutoModelForQuestionAnswering, AutoConfig,
        BitsAndBytesConfig, TrainingArguments,
        Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import accelerate
    from accelerate import Accelerator
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch and Transformers not available. Some features will be disabled.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False


@dataclass
class ModelConfig:
    """모델 설정"""
    name: str
    model_type: str  # llm, code_gen, qa, embedding, classifier
    model_path: str
    tokenizer_path: Optional[str] = None
    device: str = "auto"
    precision: str = "fp16"  # fp32, fp16, int8, int4
    max_length: int = 2048
    batch_size: int = 1
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    use_cache: bool = True
    offload_folder: Optional[str] = None
    low_cpu_mem_usage: bool = True
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    priority: int = 1
    specialized_tasks: List[str] = field(default_factory=list)
    memory_limit_gb: float = 8.0
    max_concurrent_requests: int = 10


@dataclass
class ModelPerformance:
    """모델 성능 메트릭"""
    model_name: str
    task_type: str
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    gpu_utilization: float
    accuracy_score: float
    error_rate: float
    request_count: int
    last_updated: str


@dataclass
class InferenceRequest:
    """추론 요청"""
    request_id: str
    task_type: str
    input_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    max_tokens: int = 512
    temperature: float = 0.1
    timeout: float = 30.0
    callback: Optional[Callable] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AIModelInterface(ABC):
    """AI 모델 인터페이스"""
    
    @abstractmethod
    async def load(self) -> bool:
        """모델 로드"""
        pass
    
    @abstractmethod
    async def unload(self) -> bool:
        """모델 언로드"""
        pass
    
    @abstractmethod
    async def predict(self, request: InferenceRequest) -> Dict[str, Any]:
        """예측 수행"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """메모리 사용량 반환 (MB)"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """로드 상태 확인"""
        pass


class TransformersModel(AIModelInterface):
    """Transformers 기반 모델"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.loaded = False
        self.generation_config = None
        
    async def load(self) -> bool:
        """모델 로드"""
        try:
            # 디바이스 설정
            self.device = self._determine_device()
            
            # 토크나이저 로드
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 모델 설정
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
                "device_map": "auto" if self.config.device == "auto" else None,
            }
            
            # 정밀도 설정
            if self.config.precision == "fp16":
                model_kwargs["torch_dtype"] = torch.float16
            elif self.config.precision == "int8":
                model_kwargs["load_in_8bit"] = True
            elif self.config.precision == "int4":
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            
            # 오프로드 설정
            if self.config.offload_folder:
                model_kwargs["offload_folder"] = self.config.offload_folder
            
            # 모델 로드
            if self.config.model_type == "qa":
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.config.model_path, **model_kwargs
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path, **model_kwargs
                )
            
            # 평가 모드
            self.model.eval()
            
            # 생성 설정
            self.generation_config = {
                "max_new_tokens": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": self.config.use_cache
            }
            
            self.loaded = True
            return True
            
        except Exception as e:
            logging.error(f"모델 로드 실패 {self.config.name}: {e}")
            return False
    
    async def unload(self) -> bool:
        """모델 언로드"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            gc.collect()
            
            self.loaded = False
            return True
            
        except Exception as e:
            logging.error(f"모델 언로드 실패 {self.config.name}: {e}")
            return False
    
    async def predict(self, request: InferenceRequest) -> Dict[str, Any]:
        """예측 수행"""
        if not self.loaded:
            raise RuntimeError(f"모델 {self.config.name}이 로드되지 않음")
        
        try:
            start_time = time.time()
            
            # 입력 토크나이징
            inputs = self.tokenizer(
                request.input_text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            )
            
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성 설정 조정
            generation_config = self.generation_config.copy()
            generation_config.update({
                "max_new_tokens": min(request.max_tokens, self.config.max_length),
                "temperature": request.temperature or self.config.temperature
            })
            
            # 예측
            with torch.no_grad():
                if self.config.model_type == "qa":
                    outputs = self.model(**inputs)
                    # QA 모델 처리
                    answer_start = torch.argmax(outputs.start_logits)
                    answer_end = torch.argmax(outputs.end_logits) + 1
                    answer = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][0][answer_start:answer_end]
                        )
                    )
                    output_text = answer
                else:
                    # 생성 모델
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                    
                    # 디코딩
                    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                    output_text = self.tokenizer.decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )
            
            end_time = time.time()
            
            return {
                "success": True,
                "output": output_text,
                "model_name": self.config.name,
                "latency_ms": (end_time - start_time) * 1000,
                "input_tokens": len(inputs["input_ids"][0]),
                "output_tokens": len(generated_ids) if 'generated_ids' in locals() else 0,
                "request_id": request.request_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_name": self.config.name,
                "request_id": request.request_id
            }
    
    def get_memory_usage(self) -> float:
        """메모리 사용량 반환"""
        if not self.loaded or not self.model:
            return 0.0
        
        # GPU 메모리
        if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # CPU 메모리 (추정)
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
    
    def is_loaded(self) -> bool:
        """로드 상태 확인"""
        return self.loaded and self.model is not None
    
    def _determine_device(self) -> str:
        """디바이스 결정"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device


class EnterpriseAIModelSystem:
    """기업급 AI 모델 시스템"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("/mnt/d/AutoCI/AutoCI/ai_models")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 모델 관리
        self.models: Dict[str, AIModelInterface] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # 요청 큐
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.processing_requests = {}
        self.completed_requests = deque(maxlen=10000)
        
        # 로드 밸런싱
        self.model_load_balance = defaultdict(int)
        self.model_availability = defaultdict(bool)
        
        # 캐시
        self.response_cache = {}
        self.cache_ttl = 3600  # 1시간
        self.max_cache_size = 10000
        
        # 모니터링
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_latency": 0.0,
            "models_loaded": 0,
            "total_memory_usage": 0.0
        }
        
        # 설정
        self.config = {
            "max_concurrent_requests": 50,
            "model_timeout": 30.0,
            "auto_unload_threshold": 0.85,  # 메모리 사용률 85%
            "performance_check_interval": 60,  # 1분
            "auto_scale": True,
            "enable_caching": True,
            "log_level": "INFO"
        }
        
        # 초기화
        self._setup_logging()
        self._initialize_default_models()
        
        # 백그라운드 작업
        self.is_running = True
        self.worker_tasks = []
        self.monitor_task = None
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_default_models(self):
        """기본 모델 설정"""
        default_models = [
            ModelConfig(
                name="deepseek-coder-6.7b",
                model_type="code_gen",
                model_path="deepseek-ai/deepseek-coder-6.7b-instruct",
                device="auto",
                precision="fp16",
                max_length=4096,
                specialized_tasks=["code_generation", "code_review", "debugging"],
                memory_limit_gb=8.0,
                priority=1
            ),
            ModelConfig(
                name="llama-3.1-8b",
                model_type="llm",
                model_path="meta-llama/Llama-3.1-8B-Instruct",
                device="auto",
                precision="int8",
                max_length=2048,
                specialized_tasks=["general_chat", "explanation", "planning"],
                memory_limit_gb=6.0,
                priority=2
            ),
            ModelConfig(
                name="qwen2.5-coder-7b",
                model_type="code_gen",
                model_path="Qwen/Qwen2.5-Coder-7B-Instruct",
                device="auto",
                precision="fp16",
                max_length=8192,
                specialized_tasks=["code_generation", "godot_scripting"],
                memory_limit_gb=7.0,
                priority=1
            )
        ]
        
        for config in default_models:
            self.model_configs[config.name] = config
    
    async def start(self):
        """시스템 시작"""
        self.logger.info("AI 모델 시스템 시작 중...")
        
        # 워커 태스크 시작
        for i in range(min(4, self.config["max_concurrent_requests"])):
            task = asyncio.create_task(self._worker())
            self.worker_tasks.append(task)
        
        # 모니터링 태스크 시작
        self.monitor_task = asyncio.create_task(self._monitor())
        
        # 기본 모델 로드
        await self._load_essential_models()
        
        self.logger.info("AI 모델 시스템 시작 완료")
    
    async def stop(self):
        """시스템 중지"""
        self.logger.info("AI 모델 시스템 중지 중...")
        
        self.is_running = False
        
        # 모든 요청 완료 대기
        await self.request_queue.join()
        
        # 워커 태스크 정리
        for task in self.worker_tasks:
            task.cancel()
        
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # 모든 모델 언로드
        await self._unload_all_models()
        
        self.logger.info("AI 모델 시스템 중지 완료")
    
    async def _load_essential_models(self):
        """필수 모델 로드"""
        # GPU 메모리에 따라 로드할 모델 결정
        available_memory = self._get_available_gpu_memory()
        
        # 우선순위가 높은 모델부터 로드
        sorted_configs = sorted(
            self.model_configs.values(),
            key=lambda x: x.priority
        )
        
        total_memory_used = 0
        for config in sorted_configs:
            if total_memory_used + config.memory_limit_gb <= available_memory * 0.8:
                await self.load_model(config.name)
                total_memory_used += config.memory_limit_gb
            else:
                break
    
    async def load_model(self, model_name: str) -> bool:
        """모델 로드"""
        if model_name not in self.model_configs:
            self.logger.error(f"모델 설정 없음: {model_name}")
            return False
        
        if model_name in self.models and self.models[model_name].is_loaded():
            self.logger.info(f"모델 이미 로드됨: {model_name}")
            return True
        
        config = self.model_configs[model_name]
        
        try:
            # 메모리 확인
            if not self._check_memory_availability(config):
                # 메모리 부족 시 다른 모델 언로드
                await self._free_memory_for_model(config)
            
            # 모델 생성 및 로드
            if TORCH_AVAILABLE:
                model = TransformersModel(config)
            else:
                raise RuntimeError("PyTorch를 사용할 수 없음")
            
            self.logger.info(f"모델 로드 시작: {model_name}")
            success = await model.load()
            
            if success:
                self.models[model_name] = model
                self.model_availability[model_name] = True
                self.metrics["models_loaded"] += 1
                
                # 성능 메트릭 초기화
                self.model_performance[model_name] = ModelPerformance(
                    model_name=model_name,
                    task_type=config.model_type,
                    latency_ms=0.0,
                    throughput_tokens_per_sec=0.0,
                    memory_usage_mb=model.get_memory_usage(),
                    gpu_utilization=0.0,
                    accuracy_score=0.0,
                    error_rate=0.0,
                    request_count=0,
                    last_updated=datetime.now().isoformat()
                )
                
                self.logger.info(f"모델 로드 완료: {model_name}")
                return True
            else:
                self.logger.error(f"모델 로드 실패: {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"모델 로드 오류 {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """모델 언로드"""
        if model_name not in self.models:
            return True
        
        try:
            model = self.models[model_name]
            success = await model.unload()
            
            if success:
                del self.models[model_name]
                self.model_availability[model_name] = False
                self.metrics["models_loaded"] -= 1
                self.logger.info(f"모델 언로드 완료: {model_name}")
                return True
            else:
                self.logger.error(f"모델 언로드 실패: {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"모델 언로드 오류 {model_name}: {e}")
            return False
    
    async def predict(
        self,
        task_type: str,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """예측 요청"""
        request_id = hashlib.md5(
            f"{input_text}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # 캐시 확인
        if self.config["enable_caching"]:
            cache_key = hashlib.md5(f"{task_type}_{input_text}".encode()).hexdigest()
            if cache_key in self.response_cache:
                cache_entry = self.response_cache[cache_key]
                if datetime.now() - cache_entry["timestamp"] < timedelta(seconds=self.cache_ttl):
                    self.metrics["cache_hits"] += 1
                    return cache_entry["response"]
        
        # 요청 생성
        request = InferenceRequest(
            request_id=request_id,
            task_type=task_type,
            input_text=input_text,
            context=context or {},
            priority=priority,
            timeout=timeout
        )
        
        # 큐에 추가
        await self.request_queue.put(request)
        self.metrics["total_requests"] += 1
        
        # 결과 대기
        future = asyncio.Future()
        self.processing_requests[request_id] = future
        
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            
            # 캐시에 저장
            if self.config["enable_caching"] and response.get("success"):
                if len(self.response_cache) >= self.max_cache_size:
                    # 오래된 캐시 항목 제거
                    oldest_key = min(
                        self.response_cache.keys(),
                        key=lambda k: self.response_cache[k]["timestamp"]
                    )
                    del self.response_cache[oldest_key]
                
                self.response_cache[cache_key] = {
                    "response": response,
                    "timestamp": datetime.now()
                }
            
            return response
            
        except asyncio.TimeoutError:
            self.logger.error(f"요청 타임아웃: {request_id}")
            return {
                "success": False,
                "error": "Request timeout",
                "request_id": request_id
            }
        finally:
            if request_id in self.processing_requests:
                del self.processing_requests[request_id]
    
    async def _worker(self):
        """워커 스레드"""
        while self.is_running:
            try:
                # 요청 가져오기
                request = await self.request_queue.get()
                
                # 모델 선택
                model_name = await self._select_best_model(request.task_type)
                
                if not model_name:
                    # 사용 가능한 모델 없음
                    response = {
                        "success": False,
                        "error": "No available model for task",
                        "request_id": request.request_id
                    }
                else:
                    # 예측 수행
                    model = self.models[model_name]
                    response = await model.predict(request)
                    
                    # 성능 메트릭 업데이트
                    self._update_performance_metrics(model_name, response)
                
                # 결과 반환
                if request.request_id in self.processing_requests:
                    future = self.processing_requests[request.request_id]
                    if not future.done():
                        future.set_result(response)
                
                # 요청 완료 표시
                self.request_queue.task_done()
                self.completed_requests.append({
                    "request_id": request.request_id,
                    "completed_at": datetime.now().isoformat(),
                    "success": response.get("success", False)
                })
                
                # 통계 업데이트
                if response.get("success"):
                    self.metrics["successful_requests"] += 1
                else:
                    self.metrics["failed_requests"] += 1
                
            except Exception as e:
                self.logger.error(f"워커 오류: {e}")
                await asyncio.sleep(1)
    
    async def _select_best_model(self, task_type: str) -> Optional[str]:
        """최적 모델 선택"""
        available_models = []
        
        # 태스크에 특화된 모델 찾기
        for name, config in self.model_configs.items():
            if (name in self.models and 
                self.model_availability.get(name, False) and
                (task_type in config.specialized_tasks or 
                 config.model_type == task_type or
                 not config.specialized_tasks)):  # 범용 모델
                
                available_models.append(name)
        
        if not available_models:
            # 사용 가능한 모델이 없으면 로드 시도
            await self._load_model_for_task(task_type)
            return await self._select_best_model(task_type)
        
        # 로드 밸런싱과 성능을 고려하여 선택
        best_model = min(available_models, key=lambda name: (
            self.model_load_balance[name],  # 부하가 적은 모델
            -self.model_configs[name].priority,  # 우선순위 높은 모델
            self.model_performance.get(name, ModelPerformance(
                model_name=name, task_type="", latency_ms=float('inf'),
                throughput_tokens_per_sec=0, memory_usage_mb=0,
                gpu_utilization=0, accuracy_score=0, error_rate=1,
                request_count=0, last_updated=""
            )).latency_ms  # 지연시간 낮은 모델
        ))
        
        self.model_load_balance[best_model] += 1
        return best_model
    
    async def _load_model_for_task(self, task_type: str):
        """태스크에 맞는 모델 로드"""
        # 태스크에 특화된 모델 찾기
        candidates = [
            name for name, config in self.model_configs.items()
            if (task_type in config.specialized_tasks or
                config.model_type == task_type)
            and name not in self.models
        ]
        
        if candidates:
            # 우선순위가 높은 모델부터 로드 시도
            candidate = min(candidates, key=lambda name: self.model_configs[name].priority)
            await self.load_model(candidate)
    
    def _update_performance_metrics(self, model_name: str, response: Dict[str, Any]):
        """성능 메트릭 업데이트"""
        if model_name not in self.model_performance:
            return
        
        metrics = self.model_performance[model_name]
        metrics.request_count += 1
        
        if response.get("success"):
            # 지연시간 업데이트 (이동 평균)
            new_latency = response.get("latency_ms", 0)
            metrics.latency_ms = (metrics.latency_ms * 0.9 + new_latency * 0.1)
            
            # 처리량 계산
            input_tokens = response.get("input_tokens", 0)
            output_tokens = response.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            
            if new_latency > 0:
                throughput = (total_tokens * 1000) / new_latency  # tokens/sec
                metrics.throughput_tokens_per_sec = (
                    metrics.throughput_tokens_per_sec * 0.9 + throughput * 0.1
                )
        else:
            # 에러율 업데이트
            error_count = metrics.error_rate * metrics.request_count
            error_count += 1
            metrics.error_rate = error_count / metrics.request_count
        
        # 메모리 사용량 업데이트
        if model_name in self.models:
            metrics.memory_usage_mb = self.models[model_name].get_memory_usage()
        
        metrics.last_updated = datetime.now().isoformat()
    
    async def _monitor(self):
        """시스템 모니터링"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config["performance_check_interval"])
                
                # 메모리 사용량 확인
                total_memory = self._get_total_memory_usage()
                self.metrics["total_memory_usage"] = total_memory
                
                # 자동 메모리 관리
                if self.config["auto_scale"]:
                    memory_usage_ratio = total_memory / self._get_total_available_memory()
                    
                    if memory_usage_ratio > self.config["auto_unload_threshold"]:
                        await self._auto_unload_models()
                
                # 평균 지연시간 계산
                if self.model_performance:
                    avg_latency = np.mean([
                        m.latency_ms for m in self.model_performance.values()
                        if m.latency_ms > 0
                    ])
                    self.metrics["average_latency"] = avg_latency
                
                # 로드 밸런싱 리셋
                for model_name in self.model_load_balance:
                    self.model_load_balance[model_name] *= 0.9  # 점진적 감소
                
                self.logger.debug(f"시스템 메트릭: {self.metrics}")
                
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
    
    async def _auto_unload_models(self):
        """자동 모델 언로드"""
        # 사용량이 적고 우선순위가 낮은 모델 언로드
        loaded_models = [
            (name, model) for name, model in self.models.items()
            if model.is_loaded()
        ]
        
        # 정렬: 사용량 적음, 우선순위 낮음, 메모리 사용량 많음
        loaded_models.sort(key=lambda x: (
            -self.model_load_balance[x[0]],  # 사용량 적음
            self.model_configs[x[0]].priority,  # 우선순위 낮음
            -x[1].get_memory_usage()  # 메모리 많이 사용
        ))
        
        # 하위 25% 모델 언로드 고려
        unload_count = max(1, len(loaded_models) // 4)
        
        for name, model in loaded_models[:unload_count]:
            if self.model_load_balance[name] < 1:  # 최근 사용량이 매우 적음
                await self.unload_model(name)
                self.logger.info(f"메모리 부족으로 모델 언로드: {name}")
                break
    
    def _check_memory_availability(self, config: ModelConfig) -> bool:
        """메모리 가용성 확인"""
        current_usage = self._get_total_memory_usage()
        available = self._get_total_available_memory()
        
        return (current_usage + config.memory_limit_gb * 1024) < (available * 0.85)
    
    async def _free_memory_for_model(self, config: ModelConfig):
        """모델을 위한 메모리 확보"""
        required_memory = config.memory_limit_gb * 1024  # MB
        freed_memory = 0
        
        # 우선순위가 낮은 모델부터 언로드
        loaded_models = [
            name for name, model in self.models.items()
            if model.is_loaded()
        ]
        
        loaded_models.sort(key=lambda name: (
            -self.model_configs[name].priority,
            self.model_load_balance[name]
        ))
        
        for model_name in loaded_models:
            if freed_memory >= required_memory:
                break
            
            model = self.models[model_name]
            memory_usage = model.get_memory_usage()
            
            await self.unload_model(model_name)
            freed_memory += memory_usage
            
            self.logger.info(f"메모리 확보를 위해 모델 언로드: {model_name}")
    
    def _get_available_gpu_memory(self) -> float:
        """사용 가능한 GPU 메모리 (GB)"""
        if not torch.cuda.is_available():
            return 0.0
        
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    def _get_total_memory_usage(self) -> float:
        """총 메모리 사용량 (MB)"""
        total = 0
        for model in self.models.values():
            if model.is_loaded():
                total += model.get_memory_usage()
        return total
    
    def _get_total_available_memory(self) -> float:
        """총 가용 메모리 (MB)"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**2
        else:
            return psutil.virtual_memory().total / 1024**2
    
    async def _unload_all_models(self):
        """모든 모델 언로드"""
        for model_name in list(self.models.keys()):
            await self.unload_model(model_name)
    
    def get_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "metrics": self.metrics,
            "loaded_models": list(self.models.keys()),
            "model_performance": {
                name: asdict(perf) for name, perf in self.model_performance.items()
            },
            "queue_size": self.request_queue.qsize(),
            "memory_usage_mb": self._get_total_memory_usage(),
            "available_memory_mb": self._get_total_available_memory(),
            "cache_size": len(self.response_cache),
            "is_running": self.is_running
        }
    
    def add_model_config(self, config: ModelConfig):
        """모델 설정 추가"""
        self.model_configs[config.name] = config
        self.logger.info(f"모델 설정 추가: {config.name}")
    
    async def warm_up_model(self, model_name: str, sample_inputs: List[str]):
        """모델 워밍업"""
        if model_name not in self.models:
            await self.load_model(model_name)
        
        self.logger.info(f"모델 워밍업 시작: {model_name}")
        
        for input_text in sample_inputs:
            request = InferenceRequest(
                request_id=f"warmup_{hashlib.md5(input_text.encode()).hexdigest()[:8]}",
                task_type="warmup",
                input_text=input_text,
                max_tokens=50
            )
            
            model = self.models[model_name]
            await model.predict(request)
        
        self.logger.info(f"모델 워밍업 완료: {model_name}")


async def demo():
    """데모 실행"""
    print("기업급 AI 모델 시스템 데모")
    print("-" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch가 설치되지 않았습니다. 데모를 실행할 수 없습니다.")
        return
    
    # 시스템 생성
    ai_system = EnterpriseAIModelSystem()
    
    try:
        # 시스템 시작
        print("\n1. AI 모델 시스템 시작...")
        await ai_system.start()
        
        # 모델 로드 (작은 모델로 테스트)
        print("\n2. 테스트 모델 설정...")
        test_config = ModelConfig(
            name="test-model",
            model_type="llm",
            model_path="microsoft/DialoGPT-small",  # 작은 테스트 모델
            device="cpu",  # CPU로 테스트
            precision="fp32",
            max_length=512,
            memory_limit_gb=1.0
        )
        
        ai_system.add_model_config(test_config)
        
        print("\n3. 모델 로드...")
        success = await ai_system.load_model("test-model")
        print(f"   모델 로드 {'성공' if success else '실패'}")
        
        if success:
            # 예측 테스트
            print("\n4. 예측 테스트...")
            response = await ai_system.predict(
                task_type="llm",
                input_text="Hello, how are you?",
                timeout=10.0
            )
            
            print(f"   응답: {response.get('output', 'N/A')}")
            print(f"   지연시간: {response.get('latency_ms', 0):.1f}ms")
        
        # 상태 확인
        print("\n5. 시스템 상태:")
        status = ai_system.get_status()
        print(f"   로드된 모델: {status['loaded_models']}")
        print(f"   총 요청: {status['metrics']['total_requests']}")
        print(f"   메모리 사용량: {status['memory_usage_mb']:.1f}MB")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 시스템 정리
        print("\n6. 시스템 종료...")
        await ai_system.stop()
        print("   종료 완료")


if __name__ == "__main__":
    asyncio.run(demo())