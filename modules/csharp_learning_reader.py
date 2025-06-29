#!/usr/bin/env python3
"""
C# 학습 데이터 읽기 시스템
- 관리자용 학습 데이터를 읽기 전용으로 참조
- 사용자에게 학습 내용 제공
- 파일 변경 불가
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

class CSharpLearningReader:
    """C# 학습 데이터 읽기 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("CSharpLearningReader")
        self.project_root = Path(__file__).parent.parent
        
        # 관리자 데이터 참조 (읽기 전용)
        self.admin_data_dir = self.project_root / "admin" / "csharp_learning_data"
        
        # 사용자 학습 진행 상태 (이것만 변경 가능)
        self.user_progress_file = self.project_root / "user_learning_progress.json"
        
        # 기본 커리큘럼 (간소화 버전)
        self.basic_curriculum = {
            "basics": {
                "level": "beginner",
                "topics": ["변수와 타입", "조건문", "반복문", "메서드"],
                "description": "C# 기초 문법 학습"
            },
            "oop": {
                "level": "intermediate", 
                "topics": ["클래스", "상속", "다형성", "인터페이스"],
                "description": "객체지향 프로그래밍 개념"
            },
            "advanced": {
                "level": "advanced",
                "topics": ["제네릭", "LINQ", "async/await", "델리게이트"],
                "description": "고급 C# 기능"
            },
            "godot": {
                "level": "expert",
                "topics": ["Node 시스템", "Signal", "리소스", "C# 바인딩"],
                "description": "Godot 엔진과 C# 통합"
            }
        }
    
    async def get_available_topics(self) -> List[str]:
        """사용 가능한 학습 주제 목록"""
        topics = []
        for section in self.basic_curriculum.values():
            topics.extend(section["topics"])
        return topics
    
    async def get_learning_content(self, topic: str) -> Dict[str, Any]:
        """특정 주제의 학습 내용 가져오기"""
        # 관리자 데이터에서 참조
        content = await self._read_admin_learning_data(topic)
        
        if not content:
            # 기본 콘텐츠 제공
            content = await self._generate_basic_content(topic)
        
        return content
    
    async def _read_admin_learning_data(self, topic: str) -> Optional[Dict[str, Any]]:
        """관리자 학습 데이터 읽기 (읽기 전용)"""
        try:
            if not self.admin_data_dir.exists():
                return None
            
            # 관리자 세션 데이터 검색
            sessions_dir = self.admin_data_dir / "sessions"
            if not sessions_dir.exists():
                return None
            
            # 주제와 일치하는 세션 찾기
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    session_file = session_dir / "session_data.json"
                    if session_file.exists():
                        try:
                            with open(session_file, 'r', encoding='utf-8') as f:
                                session_data = json.load(f)
                            
                            if topic.lower() in session_data.get('topic', '').lower():
                                return {
                                    "topic": session_data['topic'],
                                    "level": session_data['level'],
                                    "duration_minutes": session_data['duration_minutes'],
                                    "mastery_score": session_data['mastery_score'],
                                    "notes": session_data['notes'],
                                    "source": "admin_data",
                                    "protected": True
                                }
                        except Exception as e:
                            self.logger.warning(f"세션 데이터 읽기 실패: {e}")
                            continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"관리자 데이터 읽기 오류: {e}")
            return None
    
    async def _generate_basic_content(self, topic: str) -> Dict[str, Any]:
        """기본 학습 콘텐츠 생성"""
        # 주제별 기본 내용 매핑
        content_map = {
            "변수와 타입": {
                "description": "C#의 기본 데이터 타입과 변수 선언",
                "code_example": "int number = 42;\nstring text = \"Hello C#\";\nbool flag = true;",
                "key_concepts": ["기본 타입", "변수 선언", "타입 추론"],
                "difficulty": "beginner"
            },
            "조건문": {
                "description": "if문과 switch문을 이용한 조건부 실행",
                "code_example": "if (score >= 90) {\n    grade = \"A\";\n} else if (score >= 80) {\n    grade = \"B\";\n}",
                "key_concepts": ["if문", "else문", "switch문", "논리 연산"],
                "difficulty": "beginner"
            },
            "반복문": {
                "description": "for, while, foreach를 이용한 반복 처리",
                "code_example": "for (int i = 0; i < 10; i++) {\n    Console.WriteLine(i);\n}",
                "key_concepts": ["for문", "while문", "foreach문", "break/continue"],
                "difficulty": "beginner"
            },
            "클래스": {
                "description": "객체지향 프로그래밍의 기본인 클래스",
                "code_example": "public class Player {\n    public string Name { get; set; }\n    public int Health { get; set; }\n}",
                "key_concepts": ["클래스 정의", "속성", "메서드", "생성자"],
                "difficulty": "intermediate"
            },
            "async/await": {
                "description": "비동기 프로그래밍 패턴",
                "code_example": "public async Task<string> GetDataAsync() {\n    var result = await client.GetStringAsync(url);\n    return result;\n}",
                "key_concepts": ["Task", "async 키워드", "await 키워드", "비동기 패턴"],
                "difficulty": "advanced"
            }
        }
        
        base_content = content_map.get(topic, {
            "description": f"{topic} 학습 내용",
            "code_example": f"// {topic} 관련 코드 예제",
            "key_concepts": [topic],
            "difficulty": "intermediate"
        })
        
        return {
            "topic": topic,
            "level": base_content["difficulty"],
            "description": base_content["description"],
            "code_example": base_content["code_example"],
            "key_concepts": base_content["key_concepts"],
            "source": "basic_content",
            "protected": False,
            "estimated_time": "30분"
        }
    
    async def start_quick_learning_session(self, topic: str):
        """빠른 학습 세션 시작"""
        print(f"📚 C# 학습 세션: {topic}")
        print("=" * 60)
        
        # 학습 내용 가져오기
        content = await self.get_learning_content(topic)
        
        # 학습 내용 표시
        print(f"📖 주제: {content['topic']}")
        print(f"📈 난이도: {content['level']}")
        print(f"📝 설명: {content['description']}")
        
        if 'estimated_time' in content:
            print(f"⏰ 예상 시간: {content['estimated_time']}")
        
        # 코드 예제 표시
        if 'code_example' in content:
            print(f"\n💻 코드 예제:")
            print("```csharp")
            print(content['code_example'])
            print("```")
        
        # 핵심 개념
        if 'key_concepts' in content:
            print(f"\n🎯 핵심 개념:")
            for concept in content['key_concepts']:
                print(f"  • {concept}")
        
        # 학습 시뮬레이션
        print(f"\n🔄 학습 진행 중...")
        for i in range(11):
            progress = i * 10
            bar = "█" * (i * 2) + "░" * ((10 - i) * 2)
            print(f"\r⏳ [{bar}] {progress}%", end="", flush=True)
            await asyncio.sleep(0.3)
        
        print(f"\n✅ '{topic}' 학습 완료!")
        
        # 사용자 진행상황 업데이트
        await self._update_user_progress(topic, content['level'])
        
        return content
    
    async def _update_user_progress(self, topic: str, level: str):
        """사용자 학습 진행상황 업데이트"""
        try:
            # 기존 진행상황 로드
            progress = {}
            if self.user_progress_file.exists():
                with open(self.user_progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
            
            # 새 진행상황 추가
            if 'completed_topics' not in progress:
                progress['completed_topics'] = []
            
            if topic not in progress['completed_topics']:
                progress['completed_topics'].append(topic)
            
            progress['last_updated'] = datetime.now().isoformat()
            progress['total_topics_completed'] = len(progress['completed_topics'])
            
            # 저장
            with open(self.user_progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"진행상황 업데이트 실패: {e}")
    
    async def get_user_progress(self) -> Dict[str, Any]:
        """사용자 학습 진행상황 조회"""
        try:
            if not self.user_progress_file.exists():
                return {
                    "completed_topics": [],
                    "total_topics_completed": 0,
                    "last_updated": None
                }
            
            with open(self.user_progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            
            # 추가 통계 계산
            available_topics = await self.get_available_topics()
            progress['total_available_topics'] = len(available_topics)
            progress['completion_percentage'] = (
                progress.get('total_topics_completed', 0) / len(available_topics) * 100
                if available_topics else 0
            )
            
            return progress
            
        except Exception as e:
            self.logger.error(f"진행상황 조회 실패: {e}")
            return {"error": str(e)}
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """학습 통계 조회"""
        progress = await self.get_user_progress()
        
        # 기본 통계
        stats = {
            "기본_통계": {
                "완료된_주제": progress.get('total_topics_completed', 0),
                "전체_주제": progress.get('total_available_topics', 0),
                "완료율": f"{progress.get('completion_percentage', 0):.1f}%",
                "마지막_학습": progress.get('last_updated')
            },
            "사용_가능한_주제": await self.get_available_topics(),
            "완료된_주제": progress.get('completed_topics', [])
        }
        
        # 관리자 데이터 상태 확인
        admin_available = self.admin_data_dir.exists()
        stats["관리자_데이터_상태"] = {
            "사용_가능": admin_available,
            "위치": str(self.admin_data_dir) if admin_available else None,
            "읽기_전용": True
        }
        
        return stats

# 독립 실행용
async def main():
    """테스트 실행"""
    reader = CSharpLearningReader()
    
    print("📚 C# 학습 데이터 읽기 시스템 테스트")
    
    mode = input("모드 선택 (1: 주제 학습, 2: 진행상황, 3: 통계): ")
    
    if mode == "1":
        topics = await reader.get_available_topics()
        print(f"사용 가능한 주제: {', '.join(topics)}")
        topic = input("학습할 주제 입력: ") or topics[0]
        await reader.start_quick_learning_session(topic)
    elif mode == "2":
        progress = await reader.get_user_progress()
        print(f"진행상황: {json.dumps(progress, indent=2, ensure_ascii=False)}")
    else:
        stats = await reader.get_learning_statistics()
        print(f"학습 통계: {json.dumps(stats, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main())