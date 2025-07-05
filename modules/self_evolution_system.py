#!/usr/bin/env python3
"""
AutoCI 자가 진화 시스템 (Self-Evolution System)
사용자 질문을 통해 스스로 학습하고 발전하는 집단 지성 시스템
"""

import os
import sys
import json
import time
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('self_evolution.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class UserQuestion:
    """사용자 질문 데이터"""
    question_id: str
    user_id: str  # 익명화된 사용자 ID
    timestamp: datetime
    question: str
    context: Dict[str, Any]  # 프로젝트 타입, 언어, 엔진 버전 등
    category: Optional[str] = None  # 자동 분류된 카테고리
    tags: List[str] = None  # 자동 태깅
    
@dataclass
class AIResponse:
    """AI 응답 데이터"""
    response_id: str
    question_id: str
    timestamp: datetime
    response: str
    confidence_score: float  # AI의 자신감 점수
    model_used: str
    generation_time: float  # 응답 생성 시간
    
@dataclass
class UserFeedback:
    """사용자 피드백 데이터"""
    feedback_id: str
    response_id: str
    timestamp: datetime
    is_helpful: bool  # 도움이 되었는지
    rating: Optional[int] = None  # 1-5 점수
    comment: Optional[str] = None  # 추가 의견
    
@dataclass
class SelfEvaluation:
    """자가 평가 데이터"""
    evaluation_id: str
    response_id: str
    timestamp: datetime
    accuracy_score: float  # 정확도 점수
    completeness_score: float  # 완성도 점수
    relevance_score: float  # 관련성 점수
    technical_score: float  # 기술적 정확성
    improvement_suggestions: List[str]  # 개선 제안
    
@dataclass
class EvolutionInsight:
    """진화 인사이트"""
    insight_id: str
    timestamp: datetime
    pattern_type: str  # 발견된 패턴 타입
    pattern_data: Dict[str, Any]
    confidence: float
    impact_score: float  # 예상 영향도
    implementation_ready: bool

class SelfEvolutionSystem:
    """AutoCI 자가 진화 시스템"""
    
    def __init__(self, data_dir: str = "./evolution_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 데이터 저장 디렉토리
        self.questions_dir = self.data_dir / "questions"
        self.responses_dir = self.data_dir / "responses"
        self.feedback_dir = self.data_dir / "feedback"
        self.evaluations_dir = self.data_dir / "evaluations"
        self.insights_dir = self.data_dir / "insights"
        self.knowledge_dir = self.data_dir / "collective_knowledge"
        
        # 모든 디렉토리 생성
        for dir_path in [self.questions_dir, self.responses_dir, 
                        self.feedback_dir, self.evaluations_dir,
                        self.insights_dir, self.knowledge_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 집단 지성 데이터베이스
        self.collective_knowledge = self._load_collective_knowledge()
        
        # 진화 메트릭스
        self.evolution_metrics = {
            "total_questions": 0,
            "total_responses": 0,
            "average_accuracy": 0.0,
            "learning_rate": 0.0,
            "knowledge_domains": defaultdict(int),
            "common_issues": defaultdict(int),
            "successful_patterns": [],
            "improvement_areas": []
        }
        
        # 자가 평가 모델 (간단한 규칙 기반 + 통계)
        self.evaluation_criteria = {
            "code_quality": ["syntax", "style", "efficiency", "readability"],
            "answer_quality": ["completeness", "accuracy", "clarity", "usefulness"],
            "technical_accuracy": ["correct_api", "best_practices", "security", "performance"]
        }
        
        # 진화 알고리즘 파라미터
        self.evolution_params = {
            "learning_rate": 0.01,
            "min_confidence_threshold": 0.7,
            "pattern_detection_threshold": 5,  # 최소 발생 횟수
            "evolution_cycle": 100  # 진화 주기 (질문 수)
        }
        
        logger.info("자가 진화 시스템이 초기화되었습니다.")
    
    def _load_collective_knowledge(self) -> Dict[str, Any]:
        """집단 지성 지식 베이스 로드"""
        knowledge_file = self.knowledge_dir / "knowledge_base.json"
        if knowledge_file.exists():
            with open(knowledge_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 초기 지식 베이스
        return {
            "patterns": {},  # 발견된 패턴들
            "solutions": {},  # 검증된 솔루션들
            "common_questions": {},  # 자주 묻는 질문들
            "best_practices": {},  # 모범 사례들
            "error_solutions": {},  # 오류 해결법들
            "optimization_tips": {},  # 최적화 팁들
            "version": "1.0",
            "last_updated": datetime.now().isoformat()
        }
    
    async def process_user_question(self, question: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """사용자 질문 처리 및 응답 생성"""
        # 1. 질문 기록
        question_obj = UserQuestion(
            question_id=self._generate_id("Q"),
            user_id=context.get("user_id", "anonymous"),
            timestamp=datetime.now(),
            question=question,
            context=context,
            category=self._categorize_question(question),
            tags=self._extract_tags(question)
        )
        
        # 질문 저장
        await self._save_question(question_obj)
        
        # 2. 집단 지성에서 유사 질문 검색
        similar_questions = await self._find_similar_questions(question)
        
        # 3. AI 응답 생성 (집단 지성 활용)
        response = await self._generate_response(question, context, similar_questions)
        
        # 4. 자가 평가
        evaluation = await self._self_evaluate(question_obj, response)
        
        # 5. 집단 지성 업데이트
        await self._update_collective_knowledge(question_obj, response, evaluation)
        
        return response.response, response.response_id
    
    def _generate_id(self, prefix: str) -> str:
        """고유 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_part}"
    
    def _categorize_question(self, question: str) -> str:
        """질문 자동 분류"""
        question_lower = question.lower()
        
        # 카테고리 키워드 매핑
        categories = {
            "godot_engine": ["godot", "엔진", "씬", "노드", "gdscript"],
            "csharp": ["c#", "csharp", "씨샵", "클래스", "메서드"],
            "networking": ["네트워크", "멀티플레이어", "서버", "클라이언트", "동기화"],
            "nakama": ["nakama", "나카마", "매치메이킹", "리더보드"],
            "ai_integration": ["ai", "인공지능", "llm", "모델"],
            "optimization": ["최적화", "성능", "메모리", "속도"],
            "debugging": ["오류", "에러", "버그", "디버그", "문제"],
            "deployment": ["배포", "빌드", "export", "릴리즈"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _extract_tags(self, question: str) -> List[str]:
        """질문에서 태그 추출"""
        tags = []
        
        # 기술 스택 태그
        tech_tags = {
            "godot4": ["godot 4", "고도 4"],
            "csharp": ["c#", "csharp"],
            "gdscript": ["gdscript", "gd스크립트"],
            "multiplayer": ["멀티플레이어", "multiplayer", "네트워크"],
            "mobile": ["모바일", "안드로이드", "ios"],
            "vr": ["vr", "가상현실"],
            "2d": ["2d", "2디"],
            "3d": ["3d", "3디"]
        }
        
        question_lower = question.lower()
        for tag, keywords in tech_tags.items():
            if any(keyword in question_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    async def _find_similar_questions(self, question: str) -> List[Dict[str, Any]]:
        """유사 질문 검색"""
        similar = []
        
        # 간단한 키워드 기반 유사도 검색
        keywords = set(question.lower().split())
        
        # 기존 질문들과 비교
        for q_file in self.questions_dir.glob("*.json"):
            with open(q_file, 'r', encoding='utf-8') as f:
                q_data = json.load(f)
                
            q_keywords = set(q_data['question'].lower().split())
            similarity = len(keywords & q_keywords) / len(keywords | q_keywords)
            
            if similarity > 0.3:  # 30% 이상 유사도
                similar.append({
                    "question": q_data['question'],
                    "response_id": q_data.get('best_response_id'),
                    "similarity": similarity,
                    "rating": q_data.get('average_rating', 0)
                })
        
        # 유사도 순으로 정렬
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:5]  # 상위 5개
    
    async def _generate_response(self, question: str, context: Dict[str, Any], 
                               similar_questions: List[Dict[str, Any]]) -> AIResponse:
        """AI 응답 생성 (집단 지성 활용)"""
        start_time = time.time()
        
        # 집단 지성에서 관련 정보 수집
        collective_context = self._gather_collective_context(question, similar_questions)
        
        # 프롬프트 구성
        prompt = f"""
질문: {question}

프로젝트 컨텍스트: {json.dumps(context, ensure_ascii=False)}

유사 질문 및 답변:
{self._format_similar_questions(similar_questions)}

집단 지성 정보:
{json.dumps(collective_context, ensure_ascii=False)}

위 정보를 참고하여 정확하고 도움이 되는 답변을 제공하세요.
"""
        
        # 실제 AI 모델 호출 (여기서는 시뮬레이션)
        # TODO: 실제 LLM 통합
        response_text = await self._call_ai_model(prompt)
        
        # 응답 객체 생성
        response = AIResponse(
            response_id=self._generate_id("R"),
            question_id=question,  # 실제로는 question_obj.question_id
            timestamp=datetime.now(),
            response=response_text,
            confidence_score=self._calculate_confidence(response_text, collective_context),
            model_used="autoci-evolution-1.0",
            generation_time=time.time() - start_time
        )
        
        # 응답 저장
        await self._save_response(response)
        
        return response
    
    def _gather_collective_context(self, question: str, similar_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """집단 지성에서 관련 컨텍스트 수집"""
        context = {
            "relevant_patterns": [],
            "proven_solutions": [],
            "common_pitfalls": [],
            "best_practices": []
        }
        
        # 질문 카테고리에 따른 관련 정보 수집
        category = self._categorize_question(question)
        
        if category in self.collective_knowledge.get("patterns", {}):
            context["relevant_patterns"] = self.collective_knowledge["patterns"][category][:3]
        
        if category in self.collective_knowledge.get("solutions", {}):
            context["proven_solutions"] = self.collective_knowledge["solutions"][category][:3]
        
        # 유사 질문들의 성공적인 답변 참고
        for sim_q in similar_questions:
            if sim_q.get("rating", 0) >= 4:  # 높은 평점의 답변
                context["proven_solutions"].append({
                    "question": sim_q["question"],
                    "rating": sim_q["rating"]
                })
        
        return context
    
    def _format_similar_questions(self, similar_questions: List[Dict[str, Any]]) -> str:
        """유사 질문 포맷팅"""
        if not similar_questions:
            return "유사한 질문이 없습니다."
        
        formatted = []
        for i, sq in enumerate(similar_questions[:3], 1):
            formatted.append(f"{i}. 질문: {sq['question']}")
            formatted.append(f"   유사도: {sq['similarity']:.1%}")
            if sq.get('rating'):
                formatted.append(f"   평점: {sq['rating']}/5")
            formatted.append("")
        
        return "\n".join(formatted)
    
    async def _call_ai_model(self, prompt: str) -> str:
        """AI 모델 호출 (시뮬레이션)"""
        # TODO: 실제 LLM 통합
        # 여기서는 간단한 템플릿 응답 생성
        
        await asyncio.sleep(0.5)  # AI 처리 시간 시뮬레이션
        
        # 실제로는 LLM이 생성할 응답
        response = f"""
해당 질문에 대한 답변입니다.

[집단 지성 기반 답변]
AutoCI가 다른 사용자들의 유사한 질문과 검증된 솔루션을 분석한 결과입니다.

1. 주요 해결 방법:
   - 첫 번째 접근법
   - 두 번째 접근법

2. 주의 사항:
   - 자주 발생하는 문제
   - 해결 방법

3. 추가 팁:
   - 최적화 방법
   - 모범 사례

이 답변은 AutoCI의 자가 학습 시스템을 통해 지속적으로 개선됩니다.
"""
        
        return response
    
    def _calculate_confidence(self, response: str, context: Dict[str, Any]) -> float:
        """응답 신뢰도 계산"""
        confidence = 0.5  # 기본 신뢰도
        
        # 집단 지성 정보 활용도에 따라 신뢰도 증가
        if context.get("proven_solutions"):
            confidence += 0.2
        
        if context.get("relevant_patterns"):
            confidence += 0.15
        
        # 응답 길이와 구조에 따른 보정
        if len(response) > 100:
            confidence += 0.1
        
        if any(keyword in response for keyword in ["해결", "방법", "코드", "예제"]):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    async def _self_evaluate(self, question: UserQuestion, response: AIResponse) -> SelfEvaluation:
        """자가 평가 수행"""
        scores = {
            "accuracy": 0.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "technical": 0.0
        }
        
        improvements = []
        
        # 1. 정확도 평가
        scores["accuracy"] = self._evaluate_accuracy(question, response)
        
        # 2. 완성도 평가
        scores["completeness"] = self._evaluate_completeness(response.response)
        
        # 3. 관련성 평가
        scores["relevance"] = self._evaluate_relevance(question.question, response.response)
        
        # 4. 기술적 정확성 평가
        scores["technical"] = self._evaluate_technical_accuracy(response.response, question.category)
        
        # 개선 제안 생성
        if scores["accuracy"] < 0.7:
            improvements.append("정확도 향상: 더 많은 검증된 데이터 필요")
        
        if scores["completeness"] < 0.7:
            improvements.append("완성도 향상: 더 상세한 설명과 예제 추가")
        
        if scores["relevance"] < 0.7:
            improvements.append("관련성 향상: 질문 의도 파악 개선 필요")
        
        # 평가 객체 생성
        evaluation = SelfEvaluation(
            evaluation_id=self._generate_id("E"),
            response_id=response.response_id,
            timestamp=datetime.now(),
            accuracy_score=scores["accuracy"],
            completeness_score=scores["completeness"],
            relevance_score=scores["relevance"],
            technical_score=scores["technical"],
            improvement_suggestions=improvements
        )
        
        # 평가 저장
        await self._save_evaluation(evaluation)
        
        return evaluation
    
    def _evaluate_accuracy(self, question: UserQuestion, response: AIResponse) -> float:
        """정확도 평가"""
        score = 0.5  # 기본 점수
        
        # 카테고리별 키워드 체크
        category_keywords = {
            "godot_engine": ["Node", "Scene", "Signal", "export"],
            "csharp": ["class", "method", "namespace", "using"],
            "networking": ["RPC", "MultiplayerAPI", "peer", "sync"],
        }
        
        if question.category in category_keywords:
            keywords = category_keywords[question.category]
            found = sum(1 for kw in keywords if kw.lower() in response.response.lower())
            score += (found / len(keywords)) * 0.3
        
        # 응답 신뢰도 반영
        score += response.confidence_score * 0.2
        
        return min(score, 1.0)
    
    def _evaluate_completeness(self, response_text: str) -> float:
        """완성도 평가"""
        score = 0.0
        
        # 구조적 요소 체크
        if "해결" in response_text or "방법" in response_text:
            score += 0.3
        
        if "예제" in response_text or "코드" in response_text:
            score += 0.3
        
        if "주의" in response_text or "참고" in response_text:
            score += 0.2
        
        # 길이 체크
        if len(response_text) > 200:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_relevance(self, question: str, response: str) -> float:
        """관련성 평가"""
        # 간단한 키워드 매칭
        q_keywords = set(question.lower().split())
        r_keywords = set(response.lower().split())
        
        if not q_keywords:
            return 0.5
        
        overlap = len(q_keywords & r_keywords) / len(q_keywords)
        return min(overlap * 1.5, 1.0)  # 1.5배 부스트
    
    def _evaluate_technical_accuracy(self, response: str, category: str) -> float:
        """기술적 정확성 평가"""
        score = 0.6  # 기본 점수
        
        # 카테고리별 기술 용어 정확성 체크
        technical_terms = {
            "godot_engine": {
                "correct": ["Node2D", "Node3D", "_ready", "_process"],
                "incorrect": ["GameObject", "Update", "Start"]  # Unity 용어
            },
            "csharp": {
                "correct": ["public", "private", "class", "interface"],
                "incorrect": ["function", "var", "let"]  # JS 용어
            }
        }
        
        if category in technical_terms:
            terms = technical_terms[category]
            
            # 올바른 용어 사용
            for term in terms["correct"]:
                if term in response:
                    score += 0.1
            
            # 잘못된 용어 사용 감점
            for term in terms["incorrect"]:
                if term in response:
                    score -= 0.15
        
        return max(0.0, min(score, 1.0))
    
    async def _update_collective_knowledge(self, question: UserQuestion, 
                                         response: AIResponse, 
                                         evaluation: SelfEvaluation):
        """집단 지성 지식 베이스 업데이트"""
        # 질문 패턴 업데이트
        if question.category not in self.collective_knowledge["patterns"]:
            self.collective_knowledge["patterns"][question.category] = []
        
        # 높은 점수의 응답은 지식 베이스에 추가
        avg_score = (evaluation.accuracy_score + evaluation.completeness_score + 
                    evaluation.relevance_score + evaluation.technical_score) / 4
        
        if avg_score >= 0.8:  # 80% 이상의 점수
            pattern = {
                "question_pattern": question.question,
                "response_template": response.response,
                "score": avg_score,
                "usage_count": 1,
                "last_used": datetime.now().isoformat()
            }
            
            self.collective_knowledge["patterns"][question.category].append(pattern)
            
            # 자주 묻는 질문 업데이트
            q_hash = hashlib.md5(question.question.encode()).hexdigest()
            if q_hash not in self.collective_knowledge["common_questions"]:
                self.collective_knowledge["common_questions"][q_hash] = {
                    "question": question.question,
                    "count": 0,
                    "best_response": None,
                    "average_score": 0.0
                }
            
            self.collective_knowledge["common_questions"][q_hash]["count"] += 1
            
            if avg_score > self.collective_knowledge["common_questions"][q_hash]["average_score"]:
                self.collective_knowledge["common_questions"][q_hash]["best_response"] = response.response
                self.collective_knowledge["common_questions"][q_hash]["average_score"] = avg_score
        
        # 진화 메트릭스 업데이트
        self.evolution_metrics["total_questions"] += 1
        self.evolution_metrics["total_responses"] += 1
        self.evolution_metrics["knowledge_domains"][question.category] += 1
        
        # 주기적으로 진화 사이클 실행
        if self.evolution_metrics["total_questions"] % self.evolution_params["evolution_cycle"] == 0:
            await self._run_evolution_cycle()
        
        # 지식 베이스 저장
        await self._save_collective_knowledge()
    
    async def _run_evolution_cycle(self):
        """진화 사이클 실행"""
        logger.info("진화 사이클 시작...")
        
        # 1. 패턴 분석
        insights = await self._analyze_patterns()
        
        # 2. 개선 영역 식별
        improvements = await self._identify_improvements()
        
        # 3. 새로운 인사이트 생성
        for insight in insights:
            if insight.confidence >= self.evolution_params["min_confidence_threshold"]:
                await self._save_insight(insight)
                
                # 실행 가능한 인사이트는 즉시 적용
                if insight.implementation_ready:
                    await self._implement_insight(insight)
        
        # 4. 학습률 조정
        self._adjust_learning_rate()
        
        logger.info(f"진화 사이클 완료: {len(insights)} 개의 새로운 인사이트 발견")
    
    async def _analyze_patterns(self) -> List[EvolutionInsight]:
        """패턴 분석 및 인사이트 도출"""
        insights = []
        
        # 자주 묻는 질문 분석
        for q_hash, q_data in self.collective_knowledge["common_questions"].items():
            if q_data["count"] >= self.evolution_params["pattern_detection_threshold"]:
                insight = EvolutionInsight(
                    insight_id=self._generate_id("I"),
                    timestamp=datetime.now(),
                    pattern_type="frequent_question",
                    pattern_data={
                        "question": q_data["question"],
                        "frequency": q_data["count"],
                        "best_response": q_data["best_response"]
                    },
                    confidence=min(q_data["count"] / 100, 1.0),
                    impact_score=q_data["count"] / self.evolution_metrics["total_questions"],
                    implementation_ready=True
                )
                insights.append(insight)
        
        # 카테고리별 트렌드 분석
        total_questions = self.evolution_metrics["total_questions"]
        for category, count in self.evolution_metrics["knowledge_domains"].items():
            if count / total_questions > 0.1:  # 10% 이상의 질문
                insight = EvolutionInsight(
                    insight_id=self._generate_id("I"),
                    timestamp=datetime.now(),
                    pattern_type="category_trend",
                    pattern_data={
                        "category": category,
                        "percentage": count / total_questions,
                        "growth_rate": self._calculate_growth_rate(category)
                    },
                    confidence=0.9,
                    impact_score=count / total_questions,
                    implementation_ready=False
                )
                insights.append(insight)
        
        return insights
    
    async def _identify_improvements(self) -> List[Dict[str, Any]]:
        """개선 영역 식별"""
        improvements = []
        
        # 낮은 점수 응답 분석
        low_score_patterns = defaultdict(list)
        
        for eval_file in self.evaluations_dir.glob("*.json"):
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            avg_score = (eval_data["accuracy_score"] + eval_data["completeness_score"] + 
                        eval_data["relevance_score"] + eval_data["technical_score"]) / 4
            
            if avg_score < 0.6:  # 60% 미만
                for suggestion in eval_data["improvement_suggestions"]:
                    low_score_patterns[suggestion].append(eval_data["response_id"])
        
        # 개선 우선순위 결정
        for suggestion, response_ids in low_score_patterns.items():
            if len(response_ids) >= 3:  # 3개 이상 발생
                improvements.append({
                    "area": suggestion,
                    "frequency": len(response_ids),
                    "priority": "high" if len(response_ids) >= 10 else "medium",
                    "affected_responses": response_ids[:5]  # 샘플
                })
        
        self.evolution_metrics["improvement_areas"] = improvements
        return improvements
    
    def _calculate_growth_rate(self, category: str) -> float:
        """카테고리 성장률 계산"""
        # 간단한 구현: 최근 질문 비율 계산
        recent_count = 0
        total_recent = 0
        
        # 최근 100개 질문 중 해당 카테고리 비율
        recent_files = sorted(self.questions_dir.glob("*.json"), 
                            key=lambda x: x.stat().st_mtime, 
                            reverse=True)[:100]
        
        for q_file in recent_files:
            with open(q_file, 'r', encoding='utf-8') as f:
                q_data = json.load(f)
            
            total_recent += 1
            if q_data.get("category") == category:
                recent_count += 1
        
        if total_recent == 0:
            return 0.0
        
        recent_ratio = recent_count / total_recent
        overall_ratio = self.evolution_metrics["knowledge_domains"][category] / self.evolution_metrics["total_questions"]
        
        return (recent_ratio - overall_ratio) / (overall_ratio + 0.001)  # 성장률
    
    def _adjust_learning_rate(self):
        """학습률 동적 조정"""
        # 평균 점수가 높으면 학습률 감소, 낮으면 증가
        recent_scores = []
        
        for eval_file in sorted(self.evaluations_dir.glob("*.json"), 
                               key=lambda x: x.stat().st_mtime, 
                               reverse=True)[:50]:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            avg_score = (eval_data["accuracy_score"] + eval_data["completeness_score"] + 
                        eval_data["relevance_score"] + eval_data["technical_score"]) / 4
            recent_scores.append(avg_score)
        
        if recent_scores:
            avg_recent_score = sum(recent_scores) / len(recent_scores)
            
            if avg_recent_score > 0.8:  # 성능 좋음
                self.evolution_params["learning_rate"] *= 0.95
            elif avg_recent_score < 0.6:  # 성능 나쁨
                self.evolution_params["learning_rate"] *= 1.05
            
            # 학습률 범위 제한
            self.evolution_params["learning_rate"] = max(0.001, 
                                                       min(0.1, self.evolution_params["learning_rate"]))
            
            self.evolution_metrics["learning_rate"] = self.evolution_params["learning_rate"]
            self.evolution_metrics["average_accuracy"] = avg_recent_score
    
    async def _implement_insight(self, insight: EvolutionInsight):
        """인사이트 자동 구현"""
        if insight.pattern_type == "frequent_question":
            # 자주 묻는 질문은 빠른 응답 캐시에 추가
            q_data = insight.pattern_data
            
            if "solutions" not in self.collective_knowledge:
                self.collective_knowledge["solutions"] = {}
            
            # 카테고리 추출
            category = self._categorize_question(q_data["question"])
            
            if category not in self.collective_knowledge["solutions"]:
                self.collective_knowledge["solutions"][category] = []
            
            # 검증된 솔루션으로 추가
            self.collective_knowledge["solutions"][category].append({
                "question": q_data["question"],
                "solution": q_data["best_response"],
                "usage_count": q_data["frequency"],
                "confidence": insight.confidence,
                "added_date": datetime.now().isoformat()
            })
            
            logger.info(f"자주 묻는 질문 솔루션 추가: {q_data['question'][:50]}...")
    
    async def receive_user_feedback(self, response_id: str, is_helpful: bool, 
                                   rating: Optional[int] = None, 
                                   comment: Optional[str] = None):
        """사용자 피드백 수신"""
        feedback = UserFeedback(
            feedback_id=self._generate_id("F"),
            response_id=response_id,
            timestamp=datetime.now(),
            is_helpful=is_helpful,
            rating=rating,
            comment=comment
        )
        
        # 피드백 저장
        await self._save_feedback(feedback)
        
        # 피드백 기반 학습
        await self._learn_from_feedback(feedback)
    
    async def _learn_from_feedback(self, feedback: UserFeedback):
        """피드백으로부터 학습"""
        # 응답 정보 로드
        response_file = self.responses_dir / f"{feedback.response_id}.json"
        if not response_file.exists():
            return
        
        with open(response_file, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
        
        # 긍정적 피드백은 패턴 강화
        if feedback.is_helpful and (feedback.rating is None or feedback.rating >= 4):
            # 해당 응답 패턴 강화
            category = self._categorize_question(response_data.get("question", ""))
            
            if category in self.collective_knowledge["patterns"]:
                for pattern in self.collective_knowledge["patterns"][category]:
                    if pattern["response_template"] == response_data["response"]:
                        pattern["usage_count"] += 1
                        pattern["last_used"] = datetime.now().isoformat()
                        break
        
        # 부정적 피드백은 개선 필요 영역으로 마킹
        elif not feedback.is_helpful or (feedback.rating and feedback.rating <= 2):
            if feedback.comment:
                # 코멘트 분석하여 개선점 추출
                self.evolution_metrics["improvement_areas"].append({
                    "response_id": feedback.response_id,
                    "feedback": feedback.comment,
                    "timestamp": feedback.timestamp.isoformat()
                })
        
        # 지식 베이스 저장
        await self._save_collective_knowledge()
    
    async def get_evolution_status(self) -> Dict[str, Any]:
        """진화 상태 조회"""
        # 최근 성능 지표 계산
        recent_performance = await self._calculate_recent_performance()
        
        return {
            "metrics": self.evolution_metrics,
            "recent_performance": recent_performance,
            "knowledge_domains": dict(self.evolution_metrics["knowledge_domains"]),
            "top_questions": self._get_top_questions(),
            "improvement_areas": self.evolution_metrics["improvement_areas"][:5],
            "evolution_stage": self._determine_evolution_stage(),
            "collective_knowledge_size": self._calculate_knowledge_size()
        }
    
    async def _calculate_recent_performance(self) -> Dict[str, float]:
        """최근 성능 계산"""
        recent_evals = []
        
        for eval_file in sorted(self.evaluations_dir.glob("*.json"), 
                               key=lambda x: x.stat().st_mtime, 
                               reverse=True)[:100]:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            recent_evals.append({
                "accuracy": eval_data["accuracy_score"],
                "completeness": eval_data["completeness_score"],
                "relevance": eval_data["relevance_score"],
                "technical": eval_data["technical_score"]
            })
        
        if not recent_evals:
            return {"accuracy": 0, "completeness": 0, "relevance": 0, "technical": 0}
        
        # 평균 계산
        performance = {
            "accuracy": sum(e["accuracy"] for e in recent_evals) / len(recent_evals),
            "completeness": sum(e["completeness"] for e in recent_evals) / len(recent_evals),
            "relevance": sum(e["relevance"] for e in recent_evals) / len(recent_evals),
            "technical": sum(e["technical"] for e in recent_evals) / len(recent_evals)
        }
        
        return performance
    
    def _get_top_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """가장 많이 묻는 질문들"""
        top_questions = []
        
        for q_hash, q_data in self.collective_knowledge["common_questions"].items():
            top_questions.append({
                "question": q_data["question"],
                "count": q_data["count"],
                "average_score": q_data["average_score"]
            })
        
        # 빈도순 정렬
        top_questions.sort(key=lambda x: x["count"], reverse=True)
        
        return top_questions[:limit]
    
    def _determine_evolution_stage(self) -> str:
        """진화 단계 결정"""
        total_q = self.evolution_metrics["total_questions"]
        avg_acc = self.evolution_metrics["average_accuracy"]
        
        if total_q < 100:
            return "초기 학습 단계"
        elif total_q < 1000:
            if avg_acc > 0.7:
                return "빠른 성장 단계"
            else:
                return "기초 학습 단계"
        elif total_q < 10000:
            if avg_acc > 0.8:
                return "지식 확장 단계"
            else:
                return "지식 개선 단계"
        else:
            if avg_acc > 0.85:
                return "전문가 단계"
            elif avg_acc > 0.75:
                return "숙련 단계"
            else:
                return "지속 개선 단계"
    
    def _calculate_knowledge_size(self) -> Dict[str, int]:
        """지식 베이스 크기 계산"""
        size = {
            "patterns": sum(len(patterns) for patterns in self.collective_knowledge["patterns"].values()),
            "solutions": sum(len(sols) for sols in self.collective_knowledge.get("solutions", {}).values()),
            "common_questions": len(self.collective_knowledge["common_questions"]),
            "best_practices": len(self.collective_knowledge.get("best_practices", {})),
            "total_insights": len(list(self.insights_dir.glob("*.json")))
        }
        
        size["total"] = sum(size.values())
        
        return size
    
    # 데이터 저장 메서드들
    async def _save_question(self, question: UserQuestion):
        """질문 저장"""
        file_path = self.questions_dir / f"{question.question_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(question), f, ensure_ascii=False, indent=2, default=str)
    
    async def _save_response(self, response: AIResponse):
        """응답 저장"""
        file_path = self.responses_dir / f"{response.response_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(response), f, ensure_ascii=False, indent=2, default=str)
    
    async def _save_feedback(self, feedback: UserFeedback):
        """피드백 저장"""
        file_path = self.feedback_dir / f"{feedback.feedback_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(feedback), f, ensure_ascii=False, indent=2, default=str)
    
    async def _save_evaluation(self, evaluation: SelfEvaluation):
        """평가 저장"""
        file_path = self.evaluations_dir / f"{evaluation.evaluation_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(evaluation), f, ensure_ascii=False, indent=2, default=str)
    
    async def _save_insight(self, insight: EvolutionInsight):
        """인사이트 저장"""
        file_path = self.insights_dir / f"{insight.insight_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(insight), f, ensure_ascii=False, indent=2, default=str)
    
    async def _save_collective_knowledge(self):
        """집단 지성 저장"""
        self.collective_knowledge["last_updated"] = datetime.now().isoformat()
        
        file_path = self.knowledge_dir / "knowledge_base.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.collective_knowledge, f, ensure_ascii=False, indent=2)
        
        # 백업 생성
        backup_path = self.knowledge_dir / f"knowledge_base_backup_{datetime.now().strftime('%Y%m%d')}.json"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(self.collective_knowledge, f, ensure_ascii=False, indent=2)


    async def collect_experiences(self) -> List[Dict[str, Any]]:
        """경험 데이터 수집 (학습을 위한 질문-답변 쌍)"""
        experiences = []
        
        # 저장된 질문-응답 쌍에서 학습 데이터 수집
        for q_file in self.questions_dir.glob("*.json"):
            try:
                with open(q_file, 'r', encoding='utf-8') as f:
                    question_data = json.load(f)
                
                # 해당 질문의 응답 찾기
                response_id = question_data.get('best_response_id')
                if response_id:
                    response_file = self.responses_dir / f"{response_id}.json"
                    if response_file.exists():
                        with open(response_file, 'r', encoding='utf-8') as f:
                            response_data = json.load(f)
                        
                        # 평가 정보 가져오기
                        eval_file = self.evaluations_dir / f"eval_{response_id}.json"
                        eval_data = {}
                        if eval_file.exists():
                            with open(eval_file, 'r', encoding='utf-8') as f:
                                eval_data = json.load(f)
                        
                        # 경험 데이터 구성
                        experience = {
                            'question': question_data['question'],
                            'answer': response_data['response'],
                            'category': question_data.get('category', 'general'),
                            'tags': question_data.get('tags', []),
                            'quality_score': eval_data.get('total_score', 0.5),
                            'topic': question_data.get('category', 'general'),
                            'timestamp': question_data.get('timestamp'),
                            'context': question_data.get('context', {})
                        }
                        
                        experiences.append(experience)
                        
            except Exception as e:
                logger.error(f"경험 데이터 수집 중 오류: {e}")
                continue
        
        # 경험이 없으면 샘플 데이터 생성
        if not experiences:
            experiences = self._generate_sample_experiences()
        
        logger.info(f"수집된 경험 데이터: {len(experiences)}개")
        return experiences
    
    def _generate_sample_experiences(self) -> List[Dict[str, Any]]:
        """샘플 경험 데이터 생성"""
        return [
            {
                'question': 'C#에서 Godot 노드를 동적으로 생성하는 방법은?',
                'answer': '''Godot에서 C#을 사용하여 노드를 동적으로 생성하는 방법:

```csharp
// 새 노드 생성
var newNode = new Node2D();
newNode.Name = "DynamicNode";

// 속성 설정
newNode.Position = new Vector2(100, 100);

// 현재 씬에 추가
AddChild(newNode);

// 특정 타입의 노드 생성
var sprite = new Sprite2D();
sprite.Texture = GD.Load<Texture2D>("res://icon.png");
newNode.AddChild(sprite);
```''',
                'category': 'csharp',
                'tags': ['godot4', 'csharp', 'nodes'],
                'quality_score': 0.85,
                'topic': 'C# 프로그래밍',
                'timestamp': datetime.now().isoformat(),
                'context': {'godot_version': '4.2', 'language': 'csharp'}
            },
            {
                'question': '변형된 Godot에서 Socket.IO를 통한 실시간 통신 구현 방법은?',
                'answer': '''Socket.IO를 사용한 실시간 통신 구현:

1. 서버 측 (Node.js + Socket.IO):
```javascript
const io = require('socket.io')(3000);

io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    socket.on('game_action', (data) => {
        // 모든 클라이언트에게 브로드캐스트
        io.emit('game_update', data);
    });
});
```

2. 클라이언트 측 (Godot C#):
```csharp
public partial class NetworkManager : Node
{
    private SocketIOClient.SocketIO socket;
    
    public override void _Ready()
    {
        socket = new SocketIOClient.SocketIO("http://localhost:3000");
        socket.ConnectAsync();
        
        socket.On("game_update", response =>
        {
            var data = response.GetValue<GameData>();
            UpdateGameState(data);
        });
    }
}
```''',
                'category': 'networking',
                'tags': ['socketio', 'multiplayer', 'realtime'],
                'quality_score': 0.90,
                'topic': '네트워킹',
                'timestamp': datetime.now().isoformat(),
                'context': {'project_type': 'multiplayer', 'network_lib': 'socketio'}
            },
            {
                'question': 'AI 모델을 Godot 게임에 통합하는 최적화 방법은?',
                'answer': '''AI 모델 통합 최적화 전략:

1. 모델 경량화:
- ONNX 형식으로 변환하여 추론 속도 향상
- 양자화(Quantization) 적용으로 모델 크기 감소

2. 비동기 처리:
```csharp
public async Task<string> GetAIResponse(string input)
{
    return await Task.Run(() => 
    {
        // AI 모델 추론을 별도 스레드에서 실행
        return aiModel.Predict(input);
    });
}
```

3. 캐싱 전략:
- 자주 사용되는 응답 캐싱
- 유사 입력에 대한 결과 재사용

4. 배치 처리:
- 여러 요청을 모아서 한 번에 처리''',
                'category': 'ai_integration',
                'tags': ['ai', 'optimization', 'performance'],
                'quality_score': 0.88,
                'topic': 'AI 최적화',
                'timestamp': datetime.now().isoformat(),
                'context': {'optimization_target': 'inference_speed'}
            }
        ]


# 전역 인스턴스
_evolution_system = None

def get_evolution_system() -> SelfEvolutionSystem:
    """자가 진화 시스템 싱글톤 인스턴스 반환"""
    global _evolution_system
    if _evolution_system is None:
        _evolution_system = SelfEvolutionSystem()
    return _evolution_system


async def main():
    """테스트 및 데모"""
    system = get_evolution_system()
    
    # 테스트 질문
    test_question = "Godot 4에서 멀티플레이어 게임을 만들 때 RPC 동기화는 어떻게 하나요?"
    test_context = {
        "user_id": "test_user_001",
        "project_type": "multiplayer_fps",
        "godot_version": "4.2",
        "language": "csharp"
    }
    
    print("🧬 AutoCI 자가 진화 시스템 테스트")
    print("=" * 60)
    
    # 질문 처리
    response, response_id = await system.process_user_question(test_question, test_context)
    
    print(f"질문: {test_question}")
    print(f"응답 ID: {response_id}")
    print(f"응답:\n{response}")
    print("=" * 60)
    
    # 피드백 시뮬레이션
    await system.receive_user_feedback(response_id, True, 5, "매우 도움이 되었습니다!")
    
    # 진화 상태 확인
    status = await system.get_evolution_status()
    print(f"진화 상태: {status['evolution_stage']}")
    print(f"총 질문 수: {status['metrics']['total_questions']}")
    print(f"평균 정확도: {status['metrics']['average_accuracy']:.2%}")
    print(f"지식 베이스 크기: {status['collective_knowledge_size']['total']}")


if __name__ == "__main__":
    asyncio.run(main())