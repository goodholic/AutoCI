#!/usr/bin/env python3
"""
간단한 연속 학습 시스템
작은 모델들을 사용하여 C#과 한글 학습
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_learning.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleLearningSystem:
    def __init__(self):
        self.models_dir = Path("./models")
        self.learning_dir = Path("./simple_learning")
        self.learning_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.learning_topics = self._init_topics()
        self.load_available_models()
        
    def _init_topics(self):
        """학습 주제 초기화"""
        return [
            {
                "id": "cs_basic_1",
                "topic": "C# 변수와 타입",
                "korean": ["변수", "자료형", "선언", "int", "string"],
                "questions": [
                    "C#에서 정수형 변수를 선언하는 방법은?",
                    "string 타입의 특징을 한글로 설명하세요.",
                    "var 키워드는 언제 사용하나요?"
                ]
            },
            {
                "id": "cs_basic_2", 
                "topic": "C# 메서드",
                "korean": ["메서드", "함수", "반환값", "매개변수"],
                "questions": [
                    "C#에서 메서드를 정의하는 기본 구조는?",
                    "void 반환 타입의 의미를 한글로 설명하세요.",
                    "메서드 오버로딩이란 무엇인가요?"
                ]
            },
            {
                "id": "cs_basic_3",
                "topic": "C# 클래스",
                "korean": ["클래스", "객체", "인스턴스", "생성자"],
                "questions": [
                    "C# 클래스의 기본 구조를 보여주세요.",
                    "생성자의 역할을 한글로 설명하세요.",
                    "public과 private의 차이는?"
                ]
            },
            {
                "id": "godot_1",
                "topic": "Godot Node",
                "korean": ["노드", "씬", "트리", "부모", "자식"],
                "questions": [
                    "Godot에서 Node란 무엇인가요?",
                    "씬 트리 구조를 한글로 설명하세요.",
                    "C#으로 노드에 접근하는 방법은?"
                ]
            },
            {
                "id": "godot_2",
                "topic": "Godot Signal",
                "korean": ["시그널", "이벤트", "연결", "방출"],
                "questions": [
                    "Godot 시그널 시스템을 설명하세요.",
                    "C#에서 시그널을 연결하는 방법은?",
                    "커스텀 시그널을 만드는 방법은?"
                ]
            }
        ]
        
    def load_available_models(self):
        """사용 가능한 모델 로드"""
        config_file = self.models_dir / "model_config.json"
        
        if not config_file.exists():
            logger.warning("모델 설정 파일이 없습니다. install_llm_models_simple.py를 먼저 실행하세요.")
            return
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        for model_name, info in config["installed_models"].items():
            if info["status"] == "ready":
                try:
                    logger.info(f"{model_name} 모델 로드 중...")
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        f"{info['path']}/tokenizer"
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        f"{info['path']}/model",
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    self.models[model_name] = {
                        "tokenizer": tokenizer,
                        "model": model,
                        "info": info
                    }
                    
                    logger.info(f"✓ {model_name} 로드 완료!")
                    
                except Exception as e:
                    logger.error(f"✗ {model_name} 로드 실패: {str(e)}")
                    
    def generate_learning_session(self, duration_hours=1):
        """학습 세션 실행"""
        if not self.models:
            logger.error("사용 가능한 모델이 없습니다!")
            return
            
        logger.info(f"\n=== {duration_hours}시간 학습 세션 시작 ===")
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.learning_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        qa_pairs = []
        question_count = 0
        
        while time.time() < end_time:
            # 랜덤 주제 선택
            topic = random.choice(self.learning_topics)
            question = random.choice(topic["questions"])
            
            logger.info(f"\n주제: {topic['topic']}")
            logger.info(f"질문: {question}")
            
            # 모델 선택 (현재는 하나뿐이지만)
            model_name = list(self.models.keys())[0]
            model_data = self.models[model_name]
            
            # 프롬프트 생성
            prompt = f"""You are a C# and Godot expert. Answer in Korean when Korean terms are mentioned.

Question: {question}

Answer with a clear explanation and code example if applicable:"""
            
            # 답변 생성
            inputs = model_data["tokenizer"](prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model_data["model"].generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=model_data["tokenizer"].eos_token_id
                )
            
            answer = model_data["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            answer = answer.split("Answer with a clear explanation")[1] if "Answer with a clear explanation" in answer else answer
            
            logger.info(f"답변: {answer[:200]}...")
            
            # QA 쌍 저장
            qa_pair = {
                "topic": topic["topic"],
                "question": question,
                "answer": answer,
                "model": model_name,
                "timestamp": datetime.now().isoformat()
            }
            qa_pairs.append(qa_pair)
            
            # 파일로 저장
            with open(session_dir / f"qa_{question_count}.json", 'w', encoding='utf-8') as f:
                json.dump(qa_pair, f, ensure_ascii=False, indent=2)
                
            question_count += 1
            
            # 진행 상황
            elapsed = time.time() - start_time
            progress = (elapsed / (duration_hours * 3600)) * 100
            logger.info(f"진행률: {progress:.1f}% | 질문 수: {question_count}")
            
            # 짧은 대기
            time.sleep(random.uniform(5, 10))
            
        # 세션 요약
        summary = {
            "session_id": session_id,
            "duration_hours": duration_hours,
            "question_count": question_count,
            "topics_covered": list(set(qa["topic"] for qa in qa_pairs)),
            "models_used": list(self.models.keys())
        }
        
        with open(session_dir / "session_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        logger.info(f"\n=== 세션 완료 ===")
        logger.info(f"총 질문: {question_count}")
        logger.info(f"결과 저장: {session_dir}")
        
    def analyze_learning_progress(self):
        """학습 진행 상황 분석"""
        logger.info("\n=== 학습 진행 상황 분석 ===")
        
        total_questions = 0
        topics_covered = set()
        
        for session_dir in self.learning_dir.iterdir():
            if session_dir.is_dir():
                summary_file = session_dir / "session_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        total_questions += summary["question_count"]
                        topics_covered.update(summary["topics_covered"])
                        
        logger.info(f"총 학습 세션: {len(list(self.learning_dir.iterdir()))}")
        logger.info(f"총 질문 수: {total_questions}")
        logger.info(f"다룬 주제: {', '.join(topics_covered)}")

def main():
    """메인 함수"""
    system = SimpleLearningSystem()
    
    if not system.models:
        logger.error("모델이 없습니다. install_llm_models_simple.py를 먼저 실행하세요.")
        return
        
    # 학습 시간 설정
    duration = 0.1  # 6분 테스트
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            logger.error("잘못된 시간 형식입니다.")
            return
            
    logger.info(f"{duration}시간 동안 학습을 시작합니다...")
    
    try:
        system.generate_learning_session(duration)
        system.analyze_learning_progress()
    except KeyboardInterrupt:
        logger.info("\n사용자가 학습을 중단했습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()