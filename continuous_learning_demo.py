#!/usr/bin/env python3
"""
연속 학습 시스템 데모 버전
실제 모델 없이 시스템 동작을 시연
"""

import os
import sys
import json
import time
import random
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning_demo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DemoContinuousLearning:
    def __init__(self):
        self.models_dir = Path("./models")
        self.learning_dir = Path("./continuous_learning_demo")
        self.learning_dir.mkdir(exist_ok=True)
        
        # 데모 모델 정보 로드
        self.load_demo_models()
        
        # 학습 주제
        self.topics = self._init_learning_topics()
        
        # 데모 응답 템플릿
        self.demo_responses = self._init_demo_responses()
        
    def load_demo_models(self):
        """데모 모델 정보 로드"""
        models_file = self.models_dir / "installed_models.json"
        if models_file.exists():
            with open(models_file, 'r', encoding='utf-8') as f:
                self.models = json.load(f)
        else:
            logger.warning("모델 정보가 없습니다. setup_demo_models.py를 먼저 실행하세요.")
            self.models = {}
            
    def _init_learning_topics(self):
        """학습 주제 초기화"""
        return [
            {
                "id": "csharp_variables",
                "category": "C# 기초",
                "topic": "변수와 데이터 타입",
                "korean_terms": ["변수", "자료형", "선언", "초기화"],
                "concepts": ["int", "string", "bool", "var", "const"],
                "difficulty": 1
            },
            {
                "id": "csharp_methods",
                "category": "C# 기초",
                "topic": "메서드와 함수",
                "korean_terms": ["메서드", "함수", "매개변수", "반환값"],
                "concepts": ["void", "return", "parameters", "overloading"],
                "difficulty": 2
            },
            {
                "id": "csharp_classes",
                "category": "C# OOP",
                "topic": "클래스와 객체",
                "korean_terms": ["클래스", "객체", "인스턴스", "생성자"],
                "concepts": ["class", "object", "constructor", "properties"],
                "difficulty": 3
            },
            {
                "id": "csharp_async",
                "category": "C# 고급",
                "topic": "비동기 프로그래밍",
                "korean_terms": ["비동기", "대기", "태스크", "동시성"],
                "concepts": ["async", "await", "Task", "concurrent"],
                "difficulty": 4
            },
            {
                "id": "godot_nodes",
                "category": "Godot",
                "topic": "노드 시스템",
                "korean_terms": ["노드", "씬", "트리", "부모자식"],
                "concepts": ["Node", "Scene", "Tree", "GetNode"],
                "difficulty": 3
            },
            {
                "id": "godot_signals",
                "category": "Godot",
                "topic": "시그널 시스템",
                "korean_terms": ["시그널", "이벤트", "연결", "방출"],
                "concepts": ["signal", "emit", "connect", "delegate"],
                "difficulty": 3
            }
        ]
        
    def _init_demo_responses(self):
        """데모 응답 템플릿"""
        return {
            "explain": [
                "{topic}은(는) C# 프로그래밍의 핵심 개념입니다. {korean_term}로도 불리며, {concept}를 사용하여 구현합니다.",
                "{topic}에 대해 설명하면, 이는 {category}의 중요한 부분으로 {korean_term}이라고 합니다.",
                "C#에서 {topic}은(는) {concept}를 통해 구현되며, 한글로는 {korean_term}입니다."
            ],
            "example": [
                "```csharp\n// {topic} 예제\npublic class Example {{\n    private {concept} value;\n    \n    public void Demo() {{\n        // {korean_term} 사용 예시\n    }}\n}}\n```",
                "```csharp\n// {topic} 구현\nusing Godot;\n\npublic partial class Demo : Node {{\n    // {korean_term} 활용\n    public override void _Ready() {{\n        GD.Print(\"{concept} 예제\");\n    }}\n}}\n```"
            ],
            "translate": [
                "{korean_term}은(는) 영어로 {concept}이며, C#에서 중요한 개념입니다.",
                "C# 용어 {concept}은(는) 한글로 {korean_term}이라고 번역됩니다.",
                "{korean_term}({concept})은(는) {category}에서 사용되는 핵심 개념입니다."
            ],
            "error": [
                "{topic} 사용 시 흔한 오류는 {concept}를 잘못 사용하는 것입니다. {korean_term}을(를) 올바르게 이해해야 합니다.",
                "{topic} 관련 일반적인 실수는 {korean_term} 초기화를 잊는 것입니다."
            ],
            "optimize": [
                "{topic} 성능 최적화를 위해서는 {concept}를 효율적으로 사용해야 합니다.",
                "{korean_term} 사용 시 메모리 관리에 주의하여 {topic}의 성능을 향상시킬 수 있습니다."
            ]
        }
        
    def generate_question(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """학습 질문 생성"""
        question_types = ["explain", "example", "translate", "error", "optimize"]
        q_type = random.choice(question_types)
        
        # 한글/영어 비율 (70% 한글)
        use_korean = random.random() < 0.7
        
        if q_type == "explain":
            question = f"{topic['topic']}에 대해 {'한글로' if use_korean else '영어로'} 설명해주세요."
        elif q_type == "example":
            question = f"{topic['topic']}을(를) 사용하는 C# 코드 예제를 {'한글 주석과 함께' if use_korean else ''} 작성해주세요."
        elif q_type == "translate":
            if use_korean:
                term = random.choice(topic['concepts'])
                question = f"C# 용어 '{term}'을(를) 한글로 설명해주세요."
            else:
                term = random.choice(topic['korean_terms'])
                question = f"'{term}'의 영어 기술 용어와 의미를 설명해주세요."
        elif q_type == "error":
            question = f"{topic['topic']} 사용 시 흔한 오류와 해결 방법을 {'한글로' if use_korean else ''} 설명해주세요."
        else:  # optimize
            question = f"{topic['topic']}의 성능 최적화 방법을 {'한글로' if use_korean else ''} 설명해주세요."
            
        return {
            "id": f"{topic['id']}_{q_type}_{int(time.time())}",
            "topic": topic['topic'],
            "category": topic['category'],
            "type": q_type,
            "question": question,
            "language": "korean" if use_korean else "english",
            "difficulty": topic['difficulty']
        }
        
    def generate_demo_answer(self, question: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """데모 답변 생성"""
        # 해당 주제 찾기
        topic = next((t for t in self.topics if t['topic'] == question['topic']), None)
        if not topic:
            return {"error": "Topic not found"}
            
        # 템플릿 선택 및 채우기
        templates = self.demo_responses.get(question['type'], ["기본 응답"])
        template = random.choice(templates)
        
        answer = template.format(
            topic=topic['topic'],
            category=topic['category'],
            korean_term=random.choice(topic['korean_terms']),
            concept=random.choice(topic['concepts'])
        )
        
        # 모델별 특성 추가
        if model_name == "qwen2.5-coder-32b":
            answer += "\n\n[Qwen2.5: 고급 코드 분석 및 한글 지원 강화]"
        elif model_name == "codellama-13b":
            answer += "\n\n[CodeLlama: 코드 최적화 및 Godot 통합 특화]"
        elif model_name == "llama-3.1-8b":
            answer += "\n\n[Llama-3.1: 일반적인 설명 및 다국어 지원]"
            
        return {
            "model": model_name,
            "question_id": question['id'],
            "answer": answer,
            "response_time": random.uniform(0.5, 2.0),
            "timestamp": datetime.now().isoformat()
        }
        
    async def learning_session(self, duration_hours: float = 0.1):
        """학습 세션 실행 (데모)"""
        logger.info(f"\n{'='*60}")
        logger.info(f"24시간 연속 학습 시스템 데모 시작")
        logger.info(f"실행 시간: {duration_hours}시간")
        logger.info(f"사용 가능 모델: {list(self.models.keys())}")
        logger.info(f"{'='*60}\n")
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.learning_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        stats = {
            "questions_asked": 0,
            "topics_covered": set(),
            "models_used": {m: 0 for m in self.models.keys()},
            "korean_questions": 0,
            "english_questions": 0
        }
        
        cycle = 0
        while datetime.now() < end_time:
            cycle += 1
            
            # 주제 선택
            topic = random.choice(self.topics)
            
            # 질문 생성
            question = self.generate_question(topic)
            
            logger.info(f"\n--- 학습 사이클 {cycle} ---")
            logger.info(f"카테고리: {question['category']}")
            logger.info(f"주제: {question['topic']}")
            logger.info(f"유형: {question['type']} | 언어: {question['language']}")
            logger.info(f"질문: {question['question']}")
            
            # 모델 선택 (질문 유형에 따라)
            if question['language'] == 'korean':
                model_preferences = ['qwen2.5-coder-32b', 'llama-3.1-8b', 'codellama-13b']
            else:
                model_preferences = ['codellama-13b', 'llama-3.1-8b', 'qwen2.5-coder-32b']
                
            model_name = next((m for m in model_preferences if m in self.models), list(self.models.keys())[0])
            
            # 답변 생성
            answer = self.generate_demo_answer(question, model_name)
            
            logger.info(f"선택된 모델: {model_name}")
            logger.info(f"답변: {answer['answer'][:200]}...")
            logger.info(f"응답 시간: {answer['response_time']:.2f}초")
            
            # 통계 업데이트
            stats['questions_asked'] += 1
            stats['topics_covered'].add(question['topic'])
            stats['models_used'][model_name] += 1
            if question['language'] == 'korean':
                stats['korean_questions'] += 1
            else:
                stats['english_questions'] += 1
                
            # QA 쌍 저장
            qa_pair = {
                "cycle": cycle,
                "question": question,
                "answer": answer,
                "stats_snapshot": {
                    "total_questions": stats['questions_asked'],
                    "topics_count": len(stats['topics_covered'])
                }
            }
            
            with open(session_dir / f"qa_{cycle:04d}.json", 'w', encoding='utf-8') as f:
                json.dump(qa_pair, f, indent=2, ensure_ascii=False)
                
            # 진행률 표시
            elapsed = (datetime.now() - start_time).total_seconds()
            total_seconds = duration_hours * 3600
            progress = (elapsed / total_seconds) * 100
            
            logger.info(f"\n진행률: {progress:.1f}% | "
                       f"질문 수: {stats['questions_asked']} | "
                       f"다룬 주제: {len(stats['topics_covered'])}")
            
            # 대기 (실제 처리 시뮬레이션)
            await asyncio.sleep(random.uniform(3, 8))
            
        # 세션 요약
        summary = {
            "session_id": session_id,
            "duration": str(datetime.now() - start_time),
            "total_questions": stats['questions_asked'],
            "topics_covered": list(stats['topics_covered']),
            "models_used": stats['models_used'],
            "language_distribution": {
                "korean": stats['korean_questions'],
                "english": stats['english_questions']
            },
            "knowledge_gained": {
                "csharp_concepts": len([t for t in stats['topics_covered'] if 'C#' in t]),
                "godot_concepts": len([t for t in stats['topics_covered'] if 'Godot' in t]),
                "korean_terms": stats['korean_questions']
            }
        }
        
        with open(session_dir / "session_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # 최종 보고서
        logger.info(f"\n{'='*60}")
        logger.info("학습 세션 완료!")
        logger.info(f"{'='*60}")
        logger.info(f"세션 ID: {session_id}")
        logger.info(f"실행 시간: {summary['duration']}")
        logger.info(f"총 질문 수: {summary['total_questions']}")
        logger.info(f"다룬 주제: {len(summary['topics_covered'])}개")
        logger.info(f"한글 질문: {summary['language_distribution']['korean']}개")
        logger.info(f"영어 질문: {summary['language_distribution']['english']}개")
        logger.info("\n모델별 사용 횟수:")
        for model, count in summary['models_used'].items():
            logger.info(f"  - {model}: {count}회")
        logger.info(f"\n결과 저장 위치: {session_dir}")
        
async def main():
    """메인 함수"""
    demo = DemoContinuousLearning()
    
    # 데모 모델 확인
    if not demo.models:
        logger.error("데모 모델이 설정되지 않았습니다.")
        logger.info("다음 명령을 실행하세요: python setup_demo_models.py")
        return
        
    # 실행 시간 설정
    duration = 0.05  # 3분 데모
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            duration = 0.05
            
    logger.info(f"데모 학습을 {duration}시간 동안 실행합니다...")
    
    try:
        await demo.learning_session(duration)
    except KeyboardInterrupt:
        logger.info("\n사용자가 학습을 중단했습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 데모 모델 설정 확인
    if not Path("./models/installed_models.json").exists():
        logger.info("데모 모델을 먼저 설정합니다...")
        os.system("python setup_demo_models.py")
        
    asyncio.run(main())