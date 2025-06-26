#!/usr/bin/env python3
"""
한글 명령어 학습 시스템
자연어 처리를 통해 한글 명령을 이해하고 적절한 작업으로 매핑
"""
import json
import os
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import re

# 한글 자연어 처리를 위한 간단한 시스템
class KoreanCommandTrainer:
    def __init__(self):
        self.data_file = "korean_commands_dataset.json"
        self.model_file = "korean_command_model.json"
        self.load_dataset()
        self.load_model()
        
    def load_dataset(self):
        """한글 명령어 데이터셋 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        else:
            # 초기 학습 데이터셋
            self.dataset = {
                "commands": [
                    # 플레이어 관련
                    {"input": "플레이어 만들어줘", "intent": "create_player", "entities": ["플레이어"]},
                    {"input": "플레이어 스크립트 생성해주세요", "intent": "create_player", "entities": ["플레이어", "스크립트"]},
                    {"input": "캐릭터 컨트롤러 만들고 싶어요", "intent": "create_player", "entities": ["캐릭터", "컨트롤러"]},
                    {"input": "주인공 움직임 구현해줘", "intent": "create_player", "entities": ["주인공", "움직임"]},
                    {"input": "플레이어 이동 스크립트 작성", "intent": "create_player", "entities": ["플레이어", "이동"]},
                    
                    # 적 AI 관련
                    {"input": "적 AI 만들어줘", "intent": "create_enemy", "entities": ["적", "AI"]},
                    {"input": "몬스터 인공지능 구현해주세요", "intent": "create_enemy", "entities": ["몬스터", "인공지능"]},
                    {"input": "적 추적 시스템 만들어줘", "intent": "create_enemy", "entities": ["적", "추적"]},
                    {"input": "AI 적 생성하고 싶어요", "intent": "create_enemy", "entities": ["AI", "적"]},
                    {"input": "enemy 스크립트 추가", "intent": "create_enemy", "entities": ["enemy", "스크립트"]},
                    
                    # 게임 매니저 관련
                    {"input": "게임 매니저 만들어줘", "intent": "create_manager", "entities": ["게임", "매니저"]},
                    {"input": "게임매니저 스크립트 생성", "intent": "create_manager", "entities": ["게임매니저", "스크립트"]},
                    {"input": "씬 관리자 구현해주세요", "intent": "create_manager", "entities": ["씬", "관리자"]},
                    {"input": "게임 상태 관리 시스템", "intent": "create_manager", "entities": ["게임", "상태", "관리"]},
                    {"input": "매니저 클래스 작성해줘", "intent": "create_manager", "entities": ["매니저", "클래스"]},
                    
                    # 코드 개선 관련
                    {"input": "코드 개선해줘", "intent": "improve_code", "entities": ["코드", "개선"]},
                    {"input": "전체 스크립트 최적화", "intent": "improve_code", "entities": ["전체", "스크립트", "최적화"]},
                    {"input": "성능 향상시켜주세요", "intent": "improve_code", "entities": ["성능", "향상"]},
                    {"input": "리팩토링 진행해줘", "intent": "improve_code", "entities": ["리팩토링"]},
                    {"input": "코드 품질 개선하고 싶어요", "intent": "improve_code", "entities": ["코드", "품질", "개선"]}
                ],
                "synonyms": {
                    "플레이어": ["player", "캐릭터", "주인공", "유저", "사용자"],
                    "적": ["enemy", "몬스터", "적군", "에너미"],
                    "매니저": ["manager", "관리자", "매니져", "관리"],
                    "개선": ["improve", "향상", "최적화", "optimize", "enhancement"],
                    "만들다": ["생성", "작성", "구현", "추가", "create", "make", "add"],
                    "AI": ["인공지능", "ai", "에이아이", "A.I."]
                },
                "patterns": [
                    {"pattern": r"(.+?)(?:을|를|이|가)?\s*(?:만들|생성|작성|구현|추가)", "type": "create"},
                    {"pattern": r"(.+?)(?:을|를|이|가)?\s*(?:개선|향상|최적화|리팩토링)", "type": "improve"},
                    {"pattern": r"(.+?)\s*(?:해줘|해주세요|하고\s*싶|부탁)", "type": "request"}
                ]
            }
            self.save_dataset()
    
    def save_dataset(self):
        """데이터셋 저장"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
    
    def load_model(self):
        """학습된 모델 로드"""
        if os.path.exists(self.model_file):
            with open(self.model_file, 'r', encoding='utf-8') as f:
                self.model = json.load(f)
        else:
            # 초기 모델 (키워드 가중치)
            self.model = {
                "keyword_weights": {
                    "플레이어": {"create_player": 0.9, "create_enemy": 0.1, "create_manager": 0.1, "improve_code": 0.1},
                    "적": {"create_player": 0.1, "create_enemy": 0.9, "create_manager": 0.1, "improve_code": 0.1},
                    "매니저": {"create_player": 0.1, "create_enemy": 0.1, "create_manager": 0.9, "improve_code": 0.1},
                    "개선": {"create_player": 0.1, "create_enemy": 0.1, "create_manager": 0.1, "improve_code": 0.9},
                    "AI": {"create_player": 0.2, "create_enemy": 0.8, "create_manager": 0.1, "improve_code": 0.1},
                    "코드": {"create_player": 0.2, "create_enemy": 0.2, "create_manager": 0.2, "improve_code": 0.8}
                },
                "intent_mapping": {
                    "create_player": "unity_player",
                    "create_enemy": "unity_enemy", 
                    "create_manager": "game_manager",
                    "improve_code": "improve_all"
                }
            }
            self.save_model()
    
    def save_model(self):
        """모델 저장"""
        with open(self.model_file, 'w', encoding='utf-8') as f:
            json.dump(self.model, f, ensure_ascii=False, indent=2)
    
    def preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 및 토큰화"""
        # 소문자 변환
        text = text.lower().strip()
        
        # 동의어 치환
        for main_word, synonyms in self.dataset["synonyms"].items():
            for synonym in synonyms:
                text = text.replace(synonym.lower(), main_word)
        
        # 기본 토큰화 (공백 기준)
        tokens = text.split()
        
        return tokens
    
    def extract_entities(self, text: str) -> List[str]:
        """텍스트에서 엔티티 추출"""
        entities = []
        tokens = self.preprocess_text(text)
        
        # 키워드 기반 엔티티 추출
        keywords = list(self.model["keyword_weights"].keys())
        for token in tokens:
            for keyword in keywords:
                if keyword in token or token in keyword:
                    entities.append(keyword)
        
        return list(set(entities))
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """입력 텍스트의 의도 예측"""
        tokens = self.preprocess_text(text)
        entities = self.extract_entities(text)
        
        # 의도별 점수 계산
        intent_scores = {
            "create_player": 0.0,
            "create_enemy": 0.0,
            "create_manager": 0.0,
            "improve_code": 0.0
        }
        
        # 키워드 가중치 적용
        for entity in entities:
            if entity in self.model["keyword_weights"]:
                for intent, weight in self.model["keyword_weights"][entity].items():
                    intent_scores[intent] += weight
        
        # 패턴 매칭 보너스
        for pattern_info in self.dataset["patterns"]:
            pattern = pattern_info["pattern"]
            if re.search(pattern, text):
                if pattern_info["type"] == "create":
                    if "플레이어" in entities or "캐릭터" in entities:
                        intent_scores["create_player"] += 0.5
                    elif "적" in entities or "AI" in entities:
                        intent_scores["create_enemy"] += 0.5
                    elif "매니저" in entities:
                        intent_scores["create_manager"] += 0.5
                elif pattern_info["type"] == "improve":
                    intent_scores["improve_code"] += 0.5
        
        # 최고 점수 의도 선택
        if sum(intent_scores.values()) == 0:
            return None, 0.0
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent] / sum(intent_scores.values())
        
        return best_intent, confidence
    
    def add_training_example(self, text: str, intent: str):
        """새로운 학습 예제 추가"""
        entities = self.extract_entities(text)
        
        # 데이터셋에 추가
        self.dataset["commands"].append({
            "input": text,
            "intent": intent,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        })
        
        # 키워드 가중치 업데이트
        for entity in entities:
            if entity not in self.model["keyword_weights"]:
                self.model["keyword_weights"][entity] = {
                    "create_player": 0.25,
                    "create_enemy": 0.25,
                    "create_manager": 0.25,
                    "improve_code": 0.25
                }
            
            # 해당 의도의 가중치 증가
            self.model["keyword_weights"][entity][intent] += 0.1
            
            # 정규화
            total = sum(self.model["keyword_weights"][entity].values())
            for key in self.model["keyword_weights"][entity]:
                self.model["keyword_weights"][entity][key] /= total
        
        self.save_dataset()
        self.save_model()
    
    def get_command_type(self, text: str) -> Dict:
        """한글 명령어를 분석하여 명령 타입 반환"""
        intent, confidence = self.predict_intent(text)
        
        if intent and confidence > 0.3:  # 신뢰도 임계값
            command_type = self.model["intent_mapping"].get(intent)
            entities = self.extract_entities(text)
            
            return {
                "success": True,
                "command_type": command_type,
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "original_text": text
            }
        else:
            return {
                "success": False,
                "message": "명령을 이해할 수 없습니다.",
                "suggestion": "예: '플레이어 만들어줘', '적 AI 구현', '게임 매니저 생성', '코드 개선'"
            }
    
    def generate_training_report(self):
        """학습 현황 리포트 생성"""
        report = {
            "total_examples": len(self.dataset["commands"]),
            "intents": {},
            "keywords": list(self.model["keyword_weights"].keys()),
            "recent_examples": self.dataset["commands"][-5:]
        }
        
        # 의도별 통계
        for cmd in self.dataset["commands"]:
            intent = cmd["intent"]
            if intent not in report["intents"]:
                report["intents"][intent] = 0
            report["intents"][intent] += 1
        
        return report


# 웹 API를 위한 함수
def train_korean_command(text: str, command_type: str) -> Dict:
    """한글 명령어 학습"""
    trainer = KoreanCommandTrainer()
    
    # 의도 매핑
    intent_map = {
        "unity_player": "create_player",
        "unity_enemy": "create_enemy",
        "game_manager": "create_manager",
        "improve_all": "improve_code"
    }
    
    intent = intent_map.get(command_type)
    if intent:
        trainer.add_training_example(text, intent)
        return {"success": True, "message": "학습 완료"}
    else:
        return {"success": False, "message": "알 수 없는 명령 타입"}


def analyze_korean_command(text: str) -> Dict:
    """한글 명령어 분석"""
    trainer = KoreanCommandTrainer()
    return trainer.get_command_type(text)


if __name__ == "__main__":
    # 테스트
    trainer = KoreanCommandTrainer()
    
    test_commands = [
        "유니티에서 플레이어 캐릭터 컨트롤러를 만들어주세요",
        "적 AI 스크립트를 구현하고 싶습니다",
        "게임 전체를 관리하는 매니저를 작성해줘",
        "모든 코드를 최적화하고 개선해주세요",
        "몬스터 추적 시스템 만들어줘"
    ]
    
    print("=== 한글 명령어 분석 테스트 ===\n")
    for cmd in test_commands:
        result = trainer.get_command_type(cmd)
        print(f"입력: {cmd}")
        if result["success"]:
            print(f"  → 명령 타입: {result['command_type']}")
            print(f"  → 신뢰도: {result['confidence']:.2f}")
            print(f"  → 추출된 엔티티: {', '.join(result['entities'])}")
        else:
            print(f"  → 실패: {result['message']}")
        print()
    
    # 학습 리포트
    report = trainer.generate_training_report()
    print("\n=== 학습 현황 ===")
    print(f"총 학습 예제: {report['total_examples']}개")
    print(f"학습된 키워드: {', '.join(report['keywords'])}")
    print(f"의도별 분포: {report['intents']}")