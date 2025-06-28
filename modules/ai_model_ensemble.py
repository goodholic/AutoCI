#!/usr/bin/env python3
"""
AI Model Ensemble for AutoCI
여러 AI 모델을 통합하여 최적의 결과를 생성하는 앙상블 시스템
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

# AutoCI 모듈 임포트
try:
    from .ollama_integration import get_llama_interface
    from .unified_autoci import AutoCILlamaEngine, GeminiCLIAdapter
except ImportError:
    # 개발 중 직접 실행시
    from ollama_integration import get_llama_interface
    from unified_autoci import AutoCILlamaEngine, GeminiCLIAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelResponse:
    """모델 응답 데이터"""
    model_name: str
    response: str
    confidence: float
    metadata: Dict[str, Any]

class BaseModel(ABC):
    """AI 모델 베이스 클래스"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """모델 능력 반환"""
        pass

class OllamaModel(BaseModel):
    """Ollama 기반 모델"""
    
    def __init__(self, model_name: str = "codellama:7b-instruct"):
        self.model_name = model_name
        self.llama = get_llama_interface()
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        try:
            response = await self.llama.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2048)
            )
            
            return ModelResponse(
                model_name=self.model_name,
                response=response,
                confidence=0.85,  # Ollama 모델 기본 신뢰도
                metadata={'source': 'ollama', 'local': True}
            )
        except Exception as e:
            logger.error(f"Ollama 생성 오류: {e}")
            return ModelResponse(
                model_name=self.model_name,
                response="",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def get_capabilities(self) -> List[str]:
        return ['code_generation', 'code_analysis', 'general_text']

class LocalLlamaModel(BaseModel):
    """기존 로컬 Llama 모델"""
    
    def __init__(self):
        self.engine = AutoCILlamaEngine()
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        try:
            # 기존 AutoCI Llama 엔진 사용
            response = await asyncio.to_thread(
                self.engine.process_command,
                prompt,
                kwargs.get('context', {})
            )
            
            return ModelResponse(
                model_name="local_llama",
                response=response,
                confidence=0.8,
                metadata={'source': 'local', 'cached': True}
            )
        except Exception as e:
            logger.error(f"Local Llama 오류: {e}")
            return ModelResponse(
                model_name="local_llama",
                response="",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def get_capabilities(self) -> List[str]:
        return ['code_generation', 'godot_specific', 'csharp_expert']

class GeminiModel(BaseModel):
    """Gemini CLI 어댑터 모델"""
    
    def __init__(self):
        self.adapter = GeminiCLIAdapter()
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        try:
            # Gemini CLI 어댑터 사용
            response = await asyncio.to_thread(
                self.adapter.parse_natural_command,
                prompt
            )
            
            # 구조화된 명령어로 변환
            if isinstance(response, dict):
                response = json.dumps(response, ensure_ascii=False, indent=2)
            
            return ModelResponse(
                model_name="gemini_cli",
                response=response,
                confidence=0.9,  # 명령어 파싱은 높은 신뢰도
                metadata={'source': 'gemini', 'structured': True}
            )
        except Exception as e:
            logger.error(f"Gemini CLI 오류: {e}")
            return ModelResponse(
                model_name="gemini_cli",
                response="",
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def get_capabilities(self) -> List[str]:
        return ['command_parsing', 'intent_detection', 'workflow_planning']

class ModelEnsemble:
    """여러 모델을 조합하는 앙상블 시스템"""
    
    def __init__(self):
        self.models = {
            'ollama_code': OllamaModel('codellama:7b-instruct'),
            'ollama_general': OllamaModel('llama2:7b'),
            'local_llama': LocalLlamaModel(),
            'gemini_cli': GeminiModel()
        }
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def select_models(self, task_type: str) -> List[str]:
        """작업 유형에 따라 적절한 모델 선택"""
        model_selection = {
            'code_generation': ['ollama_code', 'local_llama'],
            'code_analysis': ['ollama_code', 'local_llama'],
            'game_design': ['ollama_general', 'local_llama'],
            'command_parsing': ['gemini_cli', 'ollama_general'],
            'godot_specific': ['local_llama', 'ollama_code'],
            'general': ['ollama_general', 'gemini_cli']
        }
        
        return model_selection.get(task_type, ['ollama_general'])
    
    async def generate_ensemble(self, 
                               prompt: str,
                               task_type: str = 'general',
                               strategy: str = 'weighted_average',
                               **kwargs) -> str:
        """앙상블 생성"""
        selected_models = self.select_models(task_type)
        
        # 병렬로 모든 모델 실행
        tasks = []
        for model_name in selected_models:
            if model_name in self.models:
                model = self.models[model_name]
                task = model.generate(prompt, **kwargs)
                tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 오류 필터링
        valid_responses = [
            r for r in responses 
            if isinstance(r, ModelResponse) and r.confidence > 0
        ]
        
        if not valid_responses:
            logger.error("모든 모델이 실패했습니다.")
            return ""
        
        # 전략에 따라 결과 조합
        if strategy == 'weighted_average':
            return self._weighted_average(valid_responses)
        elif strategy == 'best_confidence':
            return self._best_confidence(valid_responses)
        elif strategy == 'majority_vote':
            return self._majority_vote(valid_responses)
        else:
            return self._concatenate(valid_responses)
    
    def _weighted_average(self, responses: List[ModelResponse]) -> str:
        """가중 평균 방식"""
        if len(responses) == 1:
            return responses[0].response
        
        # 신뢰도 기반 가중치 계산
        total_confidence = sum(r.confidence for r in responses)
        
        # 가장 높은 신뢰도의 응답을 기본으로 사용
        best_response = max(responses, key=lambda r: r.confidence)
        
        # 다른 응답들의 좋은 부분을 통합
        combined = f"{best_response.response}\n\n"
        combined += "=== 추가 제안사항 ===\n"
        
        for resp in responses:
            if resp != best_response and resp.confidence > 0.7:
                weight = resp.confidence / total_confidence
                combined += f"\n[{resp.model_name} (신뢰도: {resp.confidence:.2f})]:\n"
                combined += f"{resp.response[:500]}...\n"  # 일부만 포함
        
        return combined
    
    def _best_confidence(self, responses: List[ModelResponse]) -> str:
        """가장 높은 신뢰도의 응답 선택"""
        best = max(responses, key=lambda r: r.confidence)
        return best.response
    
    def _majority_vote(self, responses: List[ModelResponse]) -> str:
        """다수결 방식 (코드 생성에 유용)"""
        # 간단한 구현: 가장 긴 공통 부분 찾기
        if len(responses) == 1:
            return responses[0].response
        
        # 모든 응답에서 공통된 라인 추출
        lines_sets = [set(r.response.split('\n')) for r in responses]
        common_lines = set.intersection(*lines_sets)
        
        if common_lines:
            return '\n'.join(sorted(common_lines))
        else:
            return self._best_confidence(responses)
    
    def _concatenate(self, responses: List[ModelResponse]) -> str:
        """모든 응답 연결"""
        result = ""
        for resp in responses:
            result += f"\n=== {resp.model_name} ===\n"
            result += resp.response
            result += f"\n(신뢰도: {resp.confidence:.2f})\n"
        return result

class AutoCIEnhancedEngine:
    """강화된 AutoCI 엔진"""
    
    def __init__(self):
        self.ensemble = ModelEnsemble()
        self.context_memory = {}
    
    async def process_request(self, 
                            request: str,
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        """요청 처리"""
        # 작업 유형 감지
        task_type = self._detect_task_type(request)
        
        # 컨텍스트 준비
        full_context = {**self.context_memory, **(context or {})}
        
        # 앙상블 생성
        response = await self.ensemble.generate_ensemble(
            prompt=request,
            task_type=task_type,
            strategy='weighted_average',
            context=full_context
        )
        
        # 결과 구조화
        result = {
            'response': response,
            'task_type': task_type,
            'models_used': self.ensemble.select_models(task_type),
            'context': full_context
        }
        
        # 컨텍스트 업데이트
        self._update_context(request, response)
        
        return result
    
    def _detect_task_type(self, request: str) -> str:
        """요청에서 작업 유형 감지"""
        request_lower = request.lower()
        
        if any(keyword in request_lower for keyword in ['코드', 'code', '함수', 'function', '클래스', 'class']):
            return 'code_generation'
        elif any(keyword in request_lower for keyword in ['분석', 'analyze', '검토', 'review']):
            return 'code_analysis'
        elif any(keyword in request_lower for keyword in ['godot', 'gdscript', '씬', 'scene']):
            return 'godot_specific'
        elif any(keyword in request_lower for keyword in ['게임', 'game', '디자인', 'design', '메커니즘']):
            return 'game_design'
        elif any(keyword in request_lower for keyword in ['명령', 'command', '실행', 'run']):
            return 'command_parsing'
        else:
            return 'general'
    
    def _update_context(self, request: str, response: str):
        """컨텍스트 메모리 업데이트"""
        # 최근 10개 상호작용만 유지
        if len(self.context_memory) > 10:
            # 가장 오래된 항목 제거
            oldest_key = min(self.context_memory.keys())
            del self.context_memory[oldest_key]
        
        # 새 컨텍스트 추가
        import time
        self.context_memory[time.time()] = {
            'request': request[:200],  # 요약
            'response': response[:200],  # 요약
            'timestamp': time.time()
        }

# 전역 엔진 인스턴스
_enhanced_engine = None

def get_enhanced_engine() -> AutoCIEnhancedEngine:
    """싱글톤 엔진 반환"""
    global _enhanced_engine
    if _enhanced_engine is None:
        _enhanced_engine = AutoCIEnhancedEngine()
    return _enhanced_engine

# 사용 예제
async def main():
    engine = get_enhanced_engine()
    
    # 코드 생성 예제
    result = await engine.process_request(
        "Python으로 빠른 정렬 알고리즘을 구현해주세요. 재귀적 방법으로."
    )
    print("코드 생성 결과:")
    print(result['response'])
    
    # Godot 특화 요청
    result = await engine.process_request(
        "Godot에서 2D 플랫포머 캐릭터 움직임 스크립트를 만들어주세요"
    )
    print("\nGodot 스크립트:")
    print(result['response'])

if __name__ == "__main__":
    asyncio.run(main())