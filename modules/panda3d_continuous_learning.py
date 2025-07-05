"""
Panda3D 연속 학습 시스템
README에 정의된 5가지 핵심 주제에 대해 지속적으로 학습
"""

import os
import sys
import json
import time
import random
import logging
import gc
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import asyncio

# AI 모델 통합
from .ai_model_integration import get_ai_integration
from .enterprise_ai_model_system import EnterpriseAIModelSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LearningTopic:
    """학습 주제 정의"""
    name: str
    description: str
    difficulty_levels: List[str]
    sample_questions: List[str]
    keywords: List[str]
    
    
class Panda3DContinuousLearning:
    """Panda3D 기반 연속 학습 시스템"""
    
    def __init__(self, duration_hours: int = 24, memory_limit_gb: float = 16.0):
        self.duration_hours = duration_hours
        self.memory_limit_gb = memory_limit_gb
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        
        # AI 모델 시스템
        self.ai_system = EnterpriseAIModelSystem()
        self.ai_integration = get_ai_integration()
        
        # 학습 통계
        self.stats = {
            "total_questions": 0,
            "quality_answers": 0,
            "topics_covered": {},
            "model_usage": {},
            "learning_progress": {}
        }
        
        # 5가지 핵심 학습 주제 정의 (README 기반)
        self.learning_topics = self._initialize_learning_topics()
        
        # 학습 데이터 경로
        self.base_path = Path("continuous_learning")
        self.base_path.mkdir(exist_ok=True)
        
        # 지식 베이스
        self.knowledge_base = self._load_knowledge_base()
        
    def _initialize_learning_topics(self) -> Dict[str, LearningTopic]:
        """5가지 핵심 학습 주제 초기화"""
        return {
            "python_programming": LearningTopic(
                name="Python 프로그래밍",
                description="기초부터 고급까지 - 비동기, 객체지향, 라이브러리 활용",
                difficulty_levels=["basic", "intermediate", "advanced", "expert"],
                sample_questions=[
                    "Python에서 비동기 프로그래밍은 어떻게 구현하나요?",
                    "데코레이터 패턴을 Python으로 구현하는 방법은?",
                    "Python의 GIL이란 무엇이고 멀티스레딩에 어떤 영향을 미치나요?",
                    "메타클래스를 사용하여 싱글톤 패턴을 구현하는 방법은?"
                ],
                keywords=["async", "await", "class", "decorator", "generator", "lambda", "comprehension"]
            ),
            "korean_programming_terms": LearningTopic(
                name="한글 프로그래밍 용어",
                description="프로그래밍 개념의 한국어 번역 및 설명",
                difficulty_levels=["basic", "intermediate", "advanced"],
                sample_questions=[
                    "객체지향 프로그래밍의 주요 개념을 한글로 설명해주세요",
                    "디자인 패턴의 종류와 각각의 한글 설명은?",
                    "알고리즘 복잡도를 한글로 어떻게 설명하나요?",
                    "함수형 프로그래밍 개념을 한글로 설명해주세요"
                ],
                keywords=["객체", "클래스", "상속", "캡슐화", "다형성", "추상화", "인터페이스"]
            ),
            "panda3d_engine": LearningTopic(
                name="Panda3D 엔진",
                description="아키텍처, 렌더링, 성능 최적화, 2.5D/3D 개발",
                difficulty_levels=["basic", "intermediate", "advanced", "expert"],
                sample_questions=[
                    "Panda3D에서 기본적인 3D 씬을 구성하는 방법은?",
                    "Panda3D의 렌더링 파이프라인은 어떻게 작동하나요?",
                    "Panda3D에서 셰이더를 사용하는 방법은?",
                    "Panda3D로 2.5D 게임을 만들 때 최적화 기법은?",
                    "Panda3D의 물리 엔진 통합 방법은?"
                ],
                keywords=["ShowBase", "NodePath", "render", "camera", "loader", "task", "shader"]
            ),
            "networking_socketio": LearningTopic(
                name="네트워킹 (Socket.IO)",
                description="실시간 통신, 멀티플레이어 게임 구현",
                difficulty_levels=["basic", "intermediate", "advanced"],
                sample_questions=[
                    "Socket.IO를 사용한 기본적인 실시간 통신 구현 방법은?",
                    "Socket.IO에서 room과 namespace의 차이점은?",
                    "멀티플레이어 게임에서 지연 보상 기법은?",
                    "Socket.IO와 WebRTC를 함께 사용하는 방법은?"
                ],
                keywords=["socket", "emit", "broadcast", "room", "namespace", "realtime", "websocket"]
            ),
            "ai_model_optimization": LearningTopic(
                name="AI 모델 최적화",
                description="학습 데이터, 프롬프트 엔지니어링, 모델 경량화",
                difficulty_levels=["intermediate", "advanced", "expert"],
                sample_questions=[
                    "LLM 모델의 양자화(Quantization) 기법은?",
                    "프롬프트 엔지니어링의 best practice는?",
                    "모델 파인튜닝과 few-shot learning의 차이는?",
                    "지식 증류(Knowledge Distillation) 기법은?",
                    "LoRA와 QLoRA의 차이점과 사용 사례는?"
                ],
                keywords=["quantization", "pruning", "distillation", "prompt", "fine-tuning", "LoRA"]
            )
        }
    
    def _load_knowledge_base(self) -> Dict[str, List[Dict[str, Any]]]:
        """저장된 지식 베이스 로드"""
        kb_path = self.base_path / "knowledge_base"
        kb_path.mkdir(exist_ok=True)
        
        knowledge_base = {}
        for topic in self.learning_topics:
            topic_file = kb_path / f"{topic}_kb.json"
            if topic_file.exists():
                with open(topic_file, 'r', encoding='utf-8') as f:
                    knowledge_base[topic] = json.load(f)
            else:
                knowledge_base[topic] = []
        
        return knowledge_base
    
    def _save_knowledge_base(self):
        """지식 베이스 저장"""
        kb_path = self.base_path / "knowledge_base"
        kb_path.mkdir(exist_ok=True)
        
        for topic, data in self.knowledge_base.items():
            topic_file = kb_path / f"{topic}_kb.json"
            with open(topic_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def start_learning(self):
        """연속 학습 시작"""
        logger.info(f"🚀 Panda3D 연속 학습 시작 (목표: {self.duration_hours}시간)")
        logger.info(f"📚 학습 주제: {', '.join(self.learning_topics.keys())}")
        
        # AI 모델 시스템 시작
        await self.ai_system.start()
        
        try:
            while datetime.now() < self.end_time:
                # 학습 주제 선택
                topic = self._select_learning_topic()
                
                # 질문 생성
                question = await self._generate_question(topic)
                
                # 답변 생성
                answer = await self._generate_answer(topic, question)
                
                # 품질 평가
                quality_score = self._evaluate_answer_quality(question, answer)
                
                # 고품질 답변은 지식 베이스에 저장
                if quality_score > 0.7:
                    self._save_to_knowledge_base(topic, question, answer, quality_score)
                    self.stats["quality_answers"] += 1
                
                # 통계 업데이트
                self._update_statistics(topic, quality_score)
                
                # 진행 상황 출력
                if self.stats["total_questions"] % 10 == 0:
                    self._print_progress()
                
                # 메모리 체크 및 정리
                if self.stats["total_questions"] % 50 == 0:
                    await self._memory_cleanup()
                
                # 잠시 대기 (CPU 과부하 방지)
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("학습 중단 요청됨")
        finally:
            await self._finalize_learning()
    
    def _select_learning_topic(self) -> str:
        """학습 주제 선택 (균형있게)"""
        # 가장 적게 학습한 주제 우선
        topic_counts = self.stats["topics_covered"]
        
        if not topic_counts:
            return random.choice(list(self.learning_topics.keys()))
        
        min_count = min(topic_counts.values()) if topic_counts else 0
        candidates = [t for t in self.learning_topics.keys() 
                     if topic_counts.get(t, 0) == min_count]
        
        return random.choice(candidates) if candidates else random.choice(list(self.learning_topics.keys()))
    
    async def _generate_question(self, topic: str) -> str:
        """주제에 맞는 질문 생성"""
        topic_info = self.learning_topics[topic]
        
        # 난이도 선택
        progress = self.stats["learning_progress"].get(topic, {})
        current_level = progress.get("level", 0)
        difficulty = topic_info.difficulty_levels[min(current_level, len(topic_info.difficulty_levels) - 1)]
        
        # 질문 템플릿
        templates = [
            f"{topic_info.name}에서 {difficulty} 수준의 개념을 설명해주세요.",
            f"{random.choice(topic_info.keywords)}에 대해 자세히 설명해주세요.",
            f"{topic_info.name} 관련 실무 예제를 만들어주세요.",
            random.choice(topic_info.sample_questions)
        ]
        
        question = random.choice(templates)
        
        # AI를 통한 더 구체적인 질문 생성
        if self.ai_integration.is_model_loaded():
            prompt = f"""
            주제: {topic_info.name}
            난이도: {difficulty}
            설명: {topic_info.description}
            
            위 주제에 대한 구체적이고 교육적인 질문을 하나 생성해주세요.
            """
            
            generated = await self.ai_integration.generate_code(prompt, {}, max_length=200)
            if generated and "code" in generated and len(generated["code"]) > 20:
                question = generated["code"].strip()
        
        return question
    
    async def _generate_answer(self, topic: str, question: str) -> str:
        """질문에 대한 답변 생성"""
        topic_info = self.learning_topics[topic]
        
        # 컨텍스트 구성
        context = f"""
        주제: {topic_info.name}
        설명: {topic_info.description}
        관련 키워드: {', '.join(topic_info.keywords)}
        
        질문: {question}
        
        위 질문에 대해 정확하고 자세한 답변을 제공해주세요.
        코드 예제가 필요한 경우 Panda3D나 Python 코드를 포함해주세요.
        """
        
        # AI 모델을 통한 답변 생성
        answer = ""
        
        # 여러 모델 중 적절한 것 선택
        if "Panda3D" in question or "엔진" in question:
            model_name = "deepseek-coder"
        elif "한글" in question or "용어" in question:
            model_name = "llama-3.1"
        else:
            model_name = None  # 자동 선택
        
        if self.ai_system.models:
            try:
                response = await self.ai_system.generate(
                    context,
                    model_name=model_name,
                    max_length=1000,
                    temperature=0.7
                )
                answer = response.get("text", "")
            except Exception as e:
                logger.error(f"AI 답변 생성 실패: {e}")
        
        # 폴백: 기본 템플릿 답변
        if not answer:
            answer = self._generate_template_answer(topic, question)
        
        return answer
    
    def _generate_template_answer(self, topic: str, question: str) -> str:
        """템플릿 기반 답변 생성 (폴백)"""
        topic_info = self.learning_topics[topic]
        
        if topic == "panda3d_engine":
            return f"""
{question}에 대한 답변:

Panda3D는 강력한 3D 게임 엔진으로, 다음과 같은 특징이 있습니다:

1. **기본 구조**:
   - ShowBase 클래스를 상속하여 애플리케이션 생성
   - render 노드를 통한 씬 그래프 관리
   - taskMgr를 통한 업데이트 루프 관리

2. **예제 코드**:
```python
from direct.showbase.ShowBase import ShowBase
from panda3d.core import *

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        
        # 모델 로드
        self.model = self.loader.loadModel("models/environment")
        self.model.reparentTo(self.render)
        
        # 카메라 설정
        self.camera.setPos(0, -20, 5)
        self.camera.lookAt(0, 0, 0)

app = MyApp()
app.run()
```

이 기본 구조를 바탕으로 더 복잡한 게임을 개발할 수 있습니다.
"""
        elif topic == "korean_programming_terms":
            return f"""
{question}에 대한 한글 설명:

프로그래밍 개념을 한글로 이해하는 것은 매우 중요합니다:

1. **객체지향 프로그래밍 (OOP)**:
   - 객체(Object): 데이터와 메서드를 포함하는 독립적인 단위
   - 클래스(Class): 객체를 생성하기 위한 템플릿
   - 상속(Inheritance): 부모 클래스의 특성을 자식이 물려받음
   - 캡슐화(Encapsulation): 데이터와 메서드를 하나로 묶고 은닉
   - 다형성(Polymorphism): 같은 인터페이스로 다른 구현 제공

2. **실제 적용 예**:
   - 게임 캐릭터를 클래스로 정의
   - 플레이어와 적을 캐릭터 클래스에서 상속
   - 각각 다른 행동 구현 (다형성)
"""
        else:
            return f"{question}에 대한 기본 답변입니다. 더 자세한 내용은 추가 학습이 필요합니다."
    
    def _evaluate_answer_quality(self, question: str, answer: str) -> float:
        """답변 품질 평가 (0.0 ~ 1.0)"""
        score = 0.0
        
        # 길이 체크
        if len(answer) > 100:
            score += 0.2
        if len(answer) > 500:
            score += 0.1
        
        # 코드 포함 여부
        if "```" in answer or "def " in answer or "class " in answer:
            score += 0.2
        
        # 구조화된 답변 (번호나 불릿 포인트)
        if any(marker in answer for marker in ["1.", "2.", "-", "*", "•"]):
            score += 0.1
        
        # 키워드 포함 여부
        question_keywords = question.lower().split()
        answer_lower = answer.lower()
        keyword_matches = sum(1 for kw in question_keywords if kw in answer_lower)
        score += min(0.2, keyword_matches * 0.05)
        
        # 한글 포함 비율 (한글 주제의 경우)
        if "한글" in question or "korean" in question.lower():
            korean_chars = sum(1 for c in answer if '가' <= c <= '힣')
            korean_ratio = korean_chars / max(len(answer), 1)
            score += min(0.2, korean_ratio)
        
        return min(1.0, score)
    
    def _save_to_knowledge_base(self, topic: str, question: str, answer: str, quality_score: float):
        """고품질 Q&A를 지식 베이스에 저장"""
        entry = {
            "question": question,
            "answer": answer,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
            "model_used": self.stats.get("last_model_used", "unknown")
        }
        
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        
        self.knowledge_base[topic].append(entry)
        
        # 주기적으로 디스크에 저장
        if len(self.knowledge_base[topic]) % 10 == 0:
            self._save_knowledge_base()
    
    def _update_statistics(self, topic: str, quality_score: float):
        """학습 통계 업데이트"""
        self.stats["total_questions"] += 1
        
        # 주제별 카운트
        if topic not in self.stats["topics_covered"]:
            self.stats["topics_covered"][topic] = 0
        self.stats["topics_covered"][topic] += 1
        
        # 학습 진도
        if topic not in self.stats["learning_progress"]:
            self.stats["learning_progress"][topic] = {
                "level": 0,
                "total_score": 0,
                "question_count": 0
            }
        
        progress = self.stats["learning_progress"][topic]
        progress["total_score"] += quality_score
        progress["question_count"] += 1
        
        # 레벨 업 체크 (평균 점수 0.8 이상이면)
        avg_score = progress["total_score"] / progress["question_count"]
        if avg_score > 0.8 and progress["question_count"] >= 20:
            progress["level"] = min(progress["level"] + 1, 3)
            progress["total_score"] = 0
            progress["question_count"] = 0
            logger.info(f"🎉 {topic} 레벨 업! 현재 레벨: {progress['level']}")
    
    def _print_progress(self):
        """진행 상황 출력"""
        elapsed = datetime.now() - self.start_time
        remaining = self.end_time - datetime.now()
        
        logger.info(f"""
📊 학습 진행 상황:
⏱️  경과 시간: {elapsed}
⏳  남은 시간: {remaining}
📝  총 질문 수: {self.stats['total_questions']}
✨  고품질 답변: {self.stats['quality_answers']}
📈  품질 비율: {self.stats['quality_answers'] / max(self.stats['total_questions'], 1) * 100:.1f}%
📚  주제별 진행:
""")
        
        for topic, count in self.stats["topics_covered"].items():
            progress = self.stats["learning_progress"].get(topic, {})
            level = progress.get("level", 0)
            logger.info(f"    - {topic}: {count}개 질문, 레벨 {level}")
    
    async def _memory_cleanup(self):
        """메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 메모리 사용량 체크
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
        if memory_usage > self.memory_limit_gb * 0.9:
            logger.warning(f"⚠️ 메모리 사용량 높음: {memory_usage:.1f}GB")
            # 필요시 모델 언로드
            await self.ai_system.cleanup_memory()
    
    async def _finalize_learning(self):
        """학습 종료 및 보고서 생성"""
        # 지식 베이스 최종 저장
        self._save_knowledge_base()
        
        # AI 시스템 종료
        await self.ai_system.stop()
        
        # 최종 보고서 생성
        report = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": str(datetime.now() - self.start_time),
            "statistics": self.stats,
            "knowledge_base_size": {
                topic: len(entries) for topic, entries in self.knowledge_base.items()
            }
        }
        
        report_path = self.base_path / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"""
🎓 학습 완료!
📊 최종 통계:
   - 총 질문: {self.stats['total_questions']}
   - 고품질 답변: {self.stats['quality_answers']}
   - 지식 베이스 크기: {sum(len(entries) for entries in self.knowledge_base.values())}
   - 보고서 저장: {report_path}
""")


# CLI 실행
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Panda3D 연속 학습 시스템")
    parser.add_argument("--hours", type=int, default=24, help="학습 시간 (기본: 24시간)")
    parser.add_argument("--memory", type=float, default=16.0, help="메모리 제한 GB (기본: 16GB)")
    
    args = parser.parse_args()
    
    # 학습 시스템 생성 및 실행
    learning_system = Panda3DContinuousLearning(
        duration_hours=args.hours,
        memory_limit_gb=args.memory
    )
    
    # 비동기 실행
    asyncio.run(learning_system.start_learning())