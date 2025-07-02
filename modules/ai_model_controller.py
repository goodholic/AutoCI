"""
AI 모델 완전 제어 시스템 - 응답 품질 관리자
우리가 AI 모델의 조종권을 완전히 갖기 위한 핵심 시스템

주요 제어 기능:
1. 응답 품질 실시간 검증 및 자동 재시도
2. 모델별 커스텀 프롬프트 및 파라미터 완전 제어
3. 금지된 응답 패턴 자동 차단
4. 우리 기준에 맞는 답변만 허용
5. 상세한 품질 로깅 및 통계 분석
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResponseQuality:
    """응답 품질 평가"""
    score: float  # 0.0 - 1.0
    is_acceptable: bool
    issues: List[str]
    improvements: List[str]
    confidence: float
    
@dataclass
class ModelControl:
    """모델 제어 설정"""
    model_name: str
    max_attempts: int
    quality_threshold: float
    custom_prompts: Dict[str, str]
    parameter_overrides: Dict[str, Any]
    
class AIModelController:
    """AI 모델 완전 제어 시스템"""
    
    def __init__(self):
        self.quality_standards = self._load_quality_standards()
        self.model_controls = self._load_model_controls()
        self.response_history = []
        self.banned_phrases = self._load_banned_phrases()
        self.required_patterns = self._load_required_patterns()
        
    def _load_quality_standards(self) -> Dict[str, Any]:
        """응답 품질 기준 로드"""
        return {
            "min_length": 50,
            "max_length": 2000,
            "required_korean_ratio": 0.3,  # 한글 질문시 한글 응답 비율
            "code_quality_patterns": [
                r"class\s+\w+",
                r"public\s+\w+",
                r"//.*",  # 주석
                r"{\s*.*\s*}"  # 코드 블록
            ],
            "explanation_patterns": [
                r"(왜냐하면|because|따라서|therefore)",
                r"(예를 들어|for example|예시)",
                r"(방법은|method|해결책)"
            ],
            "forbidden_phrases": [
                "죄송합니다",
                "잘 모르겠습니다",
                "확실하지 않습니다",
                "I don't know",
                "I'm not sure"
            ]
        }
    
    def _load_model_controls(self) -> Dict[str, ModelControl]:
        """모델별 제어 설정 - 우리가 완전히 제어합니다!"""
        return {
            "deepseek-coder-7b": ModelControl(
                model_name="deepseek-coder-7b",
                max_attempts=3,
                quality_threshold=0.7,
                custom_prompts={
                    "system": "You are an expert C# and Godot developer. Always provide detailed, practical answers in Korean with code examples.",
                    "code_request": "Write clean, well-commented C# code. Explain each part in Korean.",
                    "korean_terms": "Provide accurate Korean translations for programming terms with examples.",
                    "quality_enforcer": "You MUST provide complete, working examples. Never say 'I don't know' or 'I'm not sure'."
                },
                parameter_overrides={
                    "temperature": 0.6,  # 더 일관된 응답
                    "max_new_tokens": 200,  # 더 상세한 응답
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "do_sample": True,
                    "num_beams": 1
                }
            ),
            "llama-3.1-8b": ModelControl(
                model_name="llama-3.1-8b",
                max_attempts=2,
                quality_threshold=0.6,
                custom_prompts={
                    "system": "You are a helpful programming assistant. Answer in Korean when asked in Korean.",
                    "quality_enforcer": "Provide concrete, actionable answers. No apologies or uncertainty."
                },
                parameter_overrides={
                    "temperature": 0.7,
                    "max_new_tokens": 150,
                    "top_p": 0.95
                }
            ),
            "codellama-13b": ModelControl(
                model_name="codellama-13b",
                max_attempts=3,
                quality_threshold=0.75,
                custom_prompts={
                    "system": "You are a code generation expert. Provide production-ready code with Korean explanations.",
                    "code_request": "Generate optimized, well-structured code with detailed comments in Korean.",
                    "quality_enforcer": "Always provide complete, runnable code examples."
                },
                parameter_overrides={
                    "temperature": 0.5,
                    "max_new_tokens": 250,
                    "top_p": 0.9,
                    "repetition_penalty": 1.15
                }
            ),
            "qwen2.5-coder-32b": ModelControl(
                model_name="qwen2.5-coder-32b",
                max_attempts=2,
                quality_threshold=0.8,
                custom_prompts={
                    "system": "You are an advanced coding assistant specializing in C#, Godot, and game development.",
                    "korean_mode": "한국어로 질문받으면 반드시 한국어로 답변하세요. 코드는 영어로, 설명은 한국어로.",
                    "quality_enforcer": "Provide expert-level, detailed solutions. Never give incomplete answers."
                },
                parameter_overrides={
                    "temperature": 0.55,
                    "max_new_tokens": 300,
                    "top_p": 0.92,
                    "repetition_penalty": 1.1
                }
            )
        }
    
    def _load_banned_phrases(self) -> List[str]:
        """금지된 응답 패턴"""
        return [
            "죄송합니다. 답변 생성에 실패했습니다",
            "I cannot",
            "I'm unable to",
            "죄송하지만",
            "불가능합니다"
        ]
    
    def _load_required_patterns(self) -> Dict[str, List[str]]:
        """요구되는 응답 패턴"""
        return {
            "code_question": [
                r"(class|public|private|protected)",
                r"{\s*.*\s*}",  # 코드 블록
                r"//"  # 주석
            ],
            "korean_question": [
                r"[가-힣]{10,}",  # 최소 10자 이상 한글
            ],
            "explanation": [
                r"(이유|방법|예시|설명)"
            ]
        }
    
    def evaluate_response_quality(self, question: Dict[str, Any], response: str, model_name: str) -> ResponseQuality:
        """응답 품질 종합 평가 - 우리의 기준으로!"""
        issues = []
        improvements = []
        score_factors = []
        
        # 1. 기본 길이 검사
        if len(response) < self.quality_standards["min_length"]:
            issues.append("응답이 너무 짧음")
            score_factors.append(0.2)
        elif len(response) > self.quality_standards["max_length"]:
            issues.append("응답이 너무 김")
            score_factors.append(0.8)
        else:
            score_factors.append(1.0)
        
        # 2. 금지된 구문 검사
        banned_found = False
        for banned in self.quality_standards["forbidden_phrases"]:
            if banned.lower() in response.lower():
                issues.append(f"금지된 구문 발견: {banned}")
                score_factors.append(0.1)
                banned_found = True
                break
        if not banned_found:
            score_factors.append(1.0)
        
        # 3. 한글 질문에 대한 한글 응답 비율
        if question.get("language") == "korean":
            korean_chars = len(re.findall(r'[가-힣]', response))
            total_chars = len(re.sub(r'\s+', '', response))
            korean_ratio = korean_chars / max(total_chars, 1)
            
            if korean_ratio < self.quality_standards["required_korean_ratio"]:
                issues.append(f"한글 응답 비율 부족: {korean_ratio:.2f}")
                score_factors.append(korean_ratio)
            else:
                score_factors.append(1.0)
        
        # 4. 코드 관련 질문에 대한 코드 품질
        if question.get("type") in ["example", "error", "optimize"]:
            code_score = self._evaluate_code_quality(response)
            score_factors.append(code_score)
            if code_score < 0.5:
                issues.append("코드 품질 부족")
        
        # 5. 설명 품질 평가
        explanation_score = self._evaluate_explanation_quality(response, question.get("language", "english"))
        score_factors.append(explanation_score)
        if explanation_score < 0.6:
            issues.append("설명 품질 부족")
        
        # 종합 점수 계산
        final_score = sum(score_factors) / len(score_factors)
        threshold = self.model_controls.get(model_name, ModelControl("default", 3, 0.5, {}, {})).quality_threshold
        
        # 개선 제안 생성
        if final_score < threshold:
            improvements = self._generate_improvements(issues, question, model_name)
        
        return ResponseQuality(
            score=final_score,
            is_acceptable=final_score >= threshold,
            issues=issues,
            improvements=improvements,
            confidence=final_score
        )
    
    def _evaluate_code_quality(self, response: str) -> float:
        """코드 품질 평가"""
        score = 0.0
        
        # 코드 블록 존재
        if "```" in response or re.search(r'{\s*.*\s*}', response):
            score += 0.3
        
        # 주석 존재
        if "//" in response or "/*" in response:
            score += 0.2
        
        # C# 키워드 존재
        csharp_keywords = ["class", "public", "private", "void", "string", "int", "bool"]
        if any(keyword in response for keyword in csharp_keywords):
            score += 0.3
        
        # 의미있는 변수명
        if re.search(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', response):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_explanation_quality(self, response: str, language: str) -> float:
        """설명 품질 평가"""
        score = 0.0
        
        if language == "korean":
            # 한글 설명 패턴
            patterns = [
                r'(이유는|왜냐하면|때문에)',
                r'(예를 들어|예시)',
                r'(방법은|방식)',
                r'(따라서|그러므로|결국)'
            ]
        else:
            # 영어 설명 패턴
            patterns = [
                r'(because|since|due to)',
                r'(for example|such as)',
                r'(method|way|approach)',
                r'(therefore|thus|consequently)'
            ]
        
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.25
        
        return min(score, 1.0)
    
    def _generate_improvements(self, issues: List[str], question: Dict[str, Any], model_name: str) -> List[str]:
        """개선 제안 생성"""
        improvements = []
        
        if "응답이 너무 짧음" in issues:
            improvements.append("더 상세한 설명과 예제를 추가하세요")
        
        if "한글 응답 비율 부족" in issues:
            improvements.append("한글로 더 많이 설명하세요")
        
        if "코드 품질 부족" in issues:
            improvements.append("주석이 포함된 완전한 코드 예제를 제공하세요")
        
        if "설명 품질 부족" in issues:
            improvements.append("'왜냐하면', '예를 들어' 등을 사용해 더 체계적으로 설명하세요")
        
        return improvements
    
    def get_custom_prompt(self, model_name: str, question_type: str) -> str:
        """모델별 커스텀 프롬프트 생성"""
        model_control = self.model_controls.get(model_name)
        if not model_control:
            return "Answer the question accurately and completely."
        
        base_prompt = model_control.custom_prompts.get("system", "")
        
        if question_type in ["example", "error", "optimize"]:
            code_prompt = model_control.custom_prompts.get("code_request", "")
            return f"{base_prompt}\n{code_prompt}"
        elif question_type == "translate":
            korean_prompt = model_control.custom_prompts.get("korean_terms", "")
            return f"{base_prompt}\n{korean_prompt}"
        
        return base_prompt
    
    def get_parameter_overrides(self, model_name: str) -> Dict[str, Any]:
        """모델별 파라미터 오버라이드"""
        model_control = self.model_controls.get(model_name)
        return model_control.parameter_overrides if model_control else {}
    
    def should_retry(self, quality: ResponseQuality, model_name: str, attempt: int) -> bool:
        """재시도 여부 결정"""
        model_control = self.model_controls.get(model_name)
        max_attempts = model_control.max_attempts if model_control else 2
        
        return not quality.is_acceptable and attempt < max_attempts
    
    def log_response_quality(self, question: Dict[str, Any], response: str, quality: ResponseQuality, model_name: str):
        """응답 품질 로깅"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "question_id": question.get("id"),
            "question_type": question.get("type"),
            "language": question.get("language"),
            "response_length": len(response),
            "quality_score": quality.score,
            "is_acceptable": quality.is_acceptable,
            "issues": quality.issues,
            "improvements": quality.improvements
        }
        
        self.response_history.append(log_entry)
        
        # 실시간 로깅
        if quality.is_acceptable:
            logger.info(f"✅ 품질 통과: {model_name} (점수: {quality.score:.2f})")
        else:
            logger.warning(f"❌ 품질 실패: {model_name} (점수: {quality.score:.2f}) - {', '.join(quality.issues)}")
    
    def get_quality_report(self) -> Dict[str, Any]:
        """품질 관리 리포트"""
        if not self.response_history:
            return {"message": "아직 응답 기록이 없습니다"}
        
        total_responses = len(self.response_history)
        acceptable_responses = sum(1 for entry in self.response_history if entry["is_acceptable"])
        avg_score = sum(entry["quality_score"] for entry in self.response_history) / total_responses
        
        model_stats = {}
        for entry in self.response_history:
            model = entry["model"]
            if model not in model_stats:
                model_stats[model] = {"total": 0, "acceptable": 0, "avg_score": 0}
            
            model_stats[model]["total"] += 1
            if entry["is_acceptable"]:
                model_stats[model]["acceptable"] += 1
            model_stats[model]["avg_score"] += entry["quality_score"]
        
        # 평균 점수 계산
        for model in model_stats:
            model_stats[model]["avg_score"] /= model_stats[model]["total"]
            model_stats[model]["success_rate"] = model_stats[model]["acceptable"] / model_stats[model]["total"]
        
        return {
            "total_responses": total_responses,
            "overall_success_rate": acceptable_responses / total_responses,
            "average_quality_score": avg_score,
            "model_performance": model_stats,
            "generated_at": datetime.now().isoformat()
        }
    
    def force_quality_response(self, question: Dict[str, Any], model_name: str, max_attempts: int = 5) -> Tuple[str, ResponseQuality]:
        """품질 기준을 만족할 때까지 강제로 재시도"""
        logger.info(f"🎯 품질 응답 강제 실행: {model_name}")
        
        for attempt in range(1, max_attempts + 1):
            # 시뮬레이션: 실제로는 모델을 호출하는 부분
            response = f"[모델 응답 시뮬레이션 - 시도 {attempt}]"
            quality = self.evaluate_response_quality(question, response, model_name)
            
            if quality.is_acceptable:
                logger.info(f"✅ 시도 {attempt}: 품질 통과 (점수: {quality.score:.2f})")
                return response, quality
            else:
                logger.warning(f"❌ 시도 {attempt}: 품질 실패 (점수: {quality.score:.2f})")
                logger.info(f"   문제점: {', '.join(quality.issues)}")
                
                # 다음 시도를 위한 프롬프트 강화
                if attempt < max_attempts:
                    logger.info("   → 프롬프트 강화 후 재시도...")
        
        logger.error(f"⚠️ {max_attempts}회 시도 후에도 품질 기준 미달")
        return response, quality
    
    def get_model_control_status(self) -> Dict[str, Any]:
        """AI 모델 제어 상태 종합 보고"""
        status = {
            "control_system": "ACTIVE",
            "quality_standards": {
                "min_length": self.quality_standards["min_length"],
                "max_length": self.quality_standards["max_length"],
                "korean_ratio": self.quality_standards["required_korean_ratio"],
                "forbidden_phrases": len(self.quality_standards["forbidden_phrases"])
            },
            "controlled_models": {},
            "recent_performance": {}
        }
        
        # 제어 중인 모델 정보
        for model_name, control in self.model_controls.items():
            status["controlled_models"][model_name] = {
                "quality_threshold": control.quality_threshold,
                "max_attempts": control.max_attempts,
                "custom_prompts": len(control.custom_prompts),
                "parameter_overrides": len(control.parameter_overrides)
            }
        
        # 최근 성능 통계
        if self.response_history:
            recent_history = self.response_history[-100:]  # 최근 100개
            
            for entry in recent_history:
                model = entry["model"]
                if model not in status["recent_performance"]:
                    status["recent_performance"][model] = {
                        "attempts": 0,
                        "successes": 0,
                        "avg_score": 0
                    }
                
                status["recent_performance"][model]["attempts"] += 1
                if entry["is_acceptable"]:
                    status["recent_performance"][model]["successes"] += 1
                status["recent_performance"][model]["avg_score"] += entry["quality_score"]
            
            # 평균 계산
            for model in status["recent_performance"]:
                attempts = status["recent_performance"][model]["attempts"]
                status["recent_performance"][model]["avg_score"] /= attempts
                status["recent_performance"][model]["success_rate"] = (
                    status["recent_performance"][model]["successes"] / attempts
                )
        
        status["total_models_controlled"] = len(self.model_controls)
        status["control_message"] = "🎯 우리가 AI 모델의 완전한 조종권을 갖고 있습니다!"
        
        return status 