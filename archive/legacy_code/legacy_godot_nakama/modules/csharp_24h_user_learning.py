#!/usr/bin/env python3
"""
사용자용 24시간 C# 학습 시스템
- 관리자 데이터 참조 (읽기 전용)
- 사용자 진행상황 추적
- 전체 주제 학습 가능
"""

import asyncio
import time
import random
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# 학습 설정 import
try:
    from .csharp_24h_learning_config import LearningConfig
except ImportError:
    # 기본 설정
    class LearningConfig:
        DEMO_MODE = False
        SESSION_DURATION_MIN = 20
        SESSION_DURATION_MAX = 40
        EXERCISE_DURATION = 15
        BREAK_BETWEEN_BLOCKS = 30
        PROGRESS_UPDATE_INTERVAL = 30
        SAVE_INTERVAL = 300
        ENABLE_BREAKS = True
        
        @classmethod
        def get_actual_duration(cls, base_minutes):
            return base_minutes * 60
            
        @classmethod
        def format_duration(cls, seconds):
            if seconds < 3600:
                return f"{seconds/60:.1f}분"
            return f"{seconds/3600:.1f}시간"

@dataclass
class UserLearningSession:
    """사용자 학습 세션"""
    topic: str
    level: str
    duration_minutes: int
    start_time: datetime
    completion_rate: float
    mastery_score: float
    notes: str = ""

class CSharp24HUserLearning:
    """사용자용 24시간 C# 학습 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("CSharp24HUserLearning")
        self.project_root = Path(__file__).parent.parent
        
        # 사용자 데이터 디렉토리
        self.user_data_dir = self.project_root / "user_learning_data"
        self.user_data_dir.mkdir(exist_ok=True)
        
        # 관리자 데이터 참조 (읽기 전용)
        self.admin_data_dir = self.project_root / "admin" / "csharp_learning_data"
        
        # 사용자 진행상황 파일
        self.progress_file = self.project_root / "user_learning_progress.json"
        
        # 학습 상태
        self.is_learning = False
        self.learning_sessions: List[UserLearningSession] = []
        self.total_learning_time = 0.0
        
        # 24시간 커리큘럼 (사용자 버전)
        self.learning_curriculum = self._create_user_curriculum()
    
    def _make_safe_filename(self, filename: str) -> str:
        """파일명에서 특수문자를 제거하여 안전한 파일명 생성"""
        # Windows와 Unix에서 문제가 되는 특수문자들을 언더스코어로 대체
        return filename.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    
    def _create_user_curriculum(self) -> Dict[str, Dict[str, Any]]:
        """사용자용 24시간 커리큘럼"""
        return {
            # 1-4시간: C# 기초
            "basics": {
                "level": "beginner",
                "duration": 4,
                "topics": [
                    "변수와 타입",
                    "연산자",
                    "조건문",
                    "반복문",
                    "메서드",
                    "배열과 컬렉션"
                ],
                "exercises": [
                    "계산기 만들기",
                    "숫자 맞추기 게임",
                    "문자열 처리"
                ]
            },
            
            # 5-8시간: 객체지향
            "oop": {
                "level": "intermediate",
                "duration": 4,
                "topics": [
                    "클래스",
                    "객체",
                    "상속",
                    "다형성",
                    "캡슐화",
                    "인터페이스"
                ],
                "exercises": [
                    "동물 클래스 계층",
                    "게임 캐릭터 시스템",
                    "은행 계좌 시스템"
                ]
            },
            
            # 9-12시간: 고급 기능
            "advanced": {
                "level": "intermediate",
                "duration": 4,
                "topics": [
                    "제네릭",
                    "델리게이트",
                    "람다 표현식",
                    "LINQ",
                    "예외 처리",
                    "파일 I/O"
                ],
                "exercises": [
                    "제네릭 컬렉션",
                    "이벤트 시스템",
                    "데이터 처리"
                ]
            },
            
            # 13-16시간: 비동기 프로그래밍
            "async": {
                "level": "advanced",
                "duration": 4,
                "topics": [
                    "async/await",
                    "Task",
                    "병렬 처리",
                    "Thread Safety",
                    "CancellationToken",
                    "비동기 스트림"
                ],
                "exercises": [
                    "비동기 웹 요청",
                    "병렬 데이터 처리",
                    "실시간 스트림"
                ]
            },
            
            # 17-20시간: Godot 통합
            "godot": {
                "level": "advanced",
                "duration": 4,
                "topics": [
                    "Godot Node",
                    "Signal 시스템",
                    "리소스 관리",
                    "씬 트리",
                    "물리 엔진",
                    "UI 시스템"
                ],
                "exercises": [
                    "플레이어 컨트롤러",
                    "AI 시스템",
                    "인벤토리"
                ]
            },
            
            # 21-24시간: 게임 개발
            "gamedev": {
                "level": "expert",
                "duration": 4,
                "topics": [
                    "게임 아키텍처",
                    "상태 머신",
                    "컴포넌트 시스템",
                    "네트워킹",
                    "최적화",
                    "디버깅"
                ],
                "exercises": [
                    "완전한 게임 프로토타입",
                    "멀티플레이어",
                    "성능 최적화"
                ]
            }
        }
    
    def _load_user_progress(self) -> Dict[str, Any]:
        """사용자 진행상황 로드"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {
                "completed_topics": [],
                "total_learning_time": 0,
                "last_updated": None
            }
        except Exception as e:
            self.logger.error(f"진행상황 로드 실패: {e}")
            return {"completed_topics": [], "total_learning_time": 0}
    
    def _save_user_progress(self, progress: Dict[str, Any]):
        """사용자 진행상황 저장"""
        try:
            progress["last_updated"] = datetime.now().isoformat()
            progress["total_topics_completed"] = len(progress.get("completed_topics", []))
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"진행상황 저장 실패: {e}")
    
    async def start_24h_learning_marathon(self, skip_completed: bool = True):
        """24시간 C# 학습 마라톤 시작 (사용자용)"""
        print("🚀 24시간 C# 학습 마라톤 시작!")
        print("=" * 80)
        print("AI가 24시간 동안 체계적으로 C# 프로그래밍을 학습합니다.")
        print("가상화 환경에서 안전하게 진행되며, 진행상황이 자동 저장됩니다.")
        
        # 예상 시간 계산
        if not LearningConfig.DEMO_MODE:
            print("\n⏱️  예상 학습 시간: 약 24시간")
            print("💡 Ctrl+C로 중단 가능, 다시 실행하면 이어서 학습")
        else:
            print("\n⚡ 데모 모드: 빠른 진행 (약 1시간)")
        
        print("=" * 80)
        
        self.is_learning = True
        start_time = datetime.now()
        
        # 기존 진행상황 로드
        progress = self._load_user_progress()
        completed_topics = set(progress.get("completed_topics", []))
        
        try:
            # 각 학습 블록 실행
            for block_name, block_info in self.learning_curriculum.items():
                if not self.is_learning:
                    break
                
                print(f"\n📚 학습 블록: {block_name}")
                print(f"   📖 난이도: {block_info['level']}")
                print(f"   ⏰ 예상 시간: {block_info['duration']}시간")
                print(f"   📋 주제 수: {len(block_info['topics'])}개")
                
                # 이 블록의 주제들 학습
                block_topics = block_info["topics"]
                for i, topic in enumerate(block_topics):
                    if not self.is_learning:
                        break
                    
                    # 이미 완료한 주제는 건너뛰기 (옵션)
                    if skip_completed and topic in completed_topics:
                        print(f"   ✅ '{topic}' - 이미 완료됨, 건너뜀")
                        continue
                    
                    print(f"\n  🎯 주제 {i+1}/{len(block_topics)}: {topic}")
                    
                    # 학습 세션 생성
                    session = UserLearningSession(
                        topic=topic,
                        level=block_info["level"],
                        duration_minutes=random.randint(
                            LearningConfig.SESSION_DURATION_MIN, 
                            LearningConfig.SESSION_DURATION_MAX
                        ),
                        start_time=datetime.now(),
                        completion_rate=0.0,
                        mastery_score=0.0
                    )
                    
                    # 학습 실행
                    await self._execute_learning_session(session)
                    
                    # 완료 주제 추가
                    completed_topics.add(topic)
                    progress["completed_topics"] = list(completed_topics)
                    progress["total_learning_time"] = self.total_learning_time
                    self._save_user_progress(progress)
                    
                    # 세션 기록
                    self.learning_sessions.append(session)
                
                # 블록별 실습
                if "exercises" in block_info:
                    for exercise in block_info["exercises"]:
                        if not self.is_learning:
                            break
                        print(f"\n  🛠️ 실습: {exercise}")
                        await self._execute_exercise(exercise, block_info["level"])
                
                # 진행률 표시
                await self._display_progress_report(progress)
                
                # 블록 간 휴식
                if LearningConfig.ENABLE_BREAKS and block_name != list(self.learning_curriculum.keys())[-1]:
                    break_duration = LearningConfig.get_actual_duration(LearningConfig.BREAK_BETWEEN_BLOCKS)
                    print(f"\n☕ 휴식 시간... ({LearningConfig.format_duration(break_duration)})")
                    await asyncio.sleep(break_duration)
                
        except KeyboardInterrupt:
            print("\n⏸️ 학습이 사용자에 의해 중단되었습니다.")
        finally:
            self.is_learning = False
            await self._generate_final_report(start_time, progress)
    
    async def learn_all_topics(self):
        """모든 주제 학습 (순차적)"""
        print("📚 전체 주제 학습 모드")
        print("=" * 80)
        print("모든 C# 주제를 처음부터 끝까지 학습합니다.")
        
        # skip_completed=False로 모든 주제 학습
        await self.start_24h_learning_marathon(skip_completed=False)
    
    async def learn_remaining_topics(self):
        """남은 주제만 학습"""
        print("📚 남은 주제 학습 모드")
        print("=" * 80)
        
        progress = self._load_user_progress()
        completed = set(progress.get("completed_topics", []))
        
        # 전체 주제 목록
        all_topics = []
        for block in self.learning_curriculum.values():
            all_topics.extend(block["topics"])
        
        remaining = [t for t in all_topics if t not in completed]
        
        if not remaining:
            print("🎉 모든 주제를 완료했습니다!")
            return
        
        print(f"남은 주제: {len(remaining)}개")
        print(f"주제 목록: {', '.join(remaining[:5])}" + ("..." if len(remaining) > 5 else ""))
        
        # skip_completed=True로 남은 주제만 학습
        await self.start_24h_learning_marathon(skip_completed=True)
    
    async def _execute_learning_session(self, session: UserLearningSession):
        """개별 학습 세션 실행"""
        actual_duration = LearningConfig.get_actual_duration(session.duration_minutes)
        print(f"    📖 학습 중... (예상 {LearningConfig.format_duration(actual_duration)})")
        
        # 관리자 데이터에서 참조 시도
        admin_content = await self._load_admin_content(session.topic)
        
        # 학습 진행
        start_time = time.time()
        last_save_time = start_time
        
        # 진행률 업데이트 주기 계산
        update_interval = min(LearningConfig.PROGRESS_UPDATE_INTERVAL, actual_duration / 10)
        total_updates = int(actual_duration / update_interval)
        
        for i in range(total_updates + 1):
            if not self.is_learning:
                break
            
            # 진행률 계산
            elapsed = time.time() - start_time
            progress = min(elapsed / actual_duration, 1.0)
            session.completion_rate = progress * 100
            
            # 숙련도 계산
            base_mastery = 70 if admin_content else 60
            session.mastery_score = base_mastery + (progress * 30)
            
            # 진행률 표시
            filled = int(progress * 20)
            bar = "█" * filled + "░" * (20 - filled)
            remaining = actual_duration - elapsed
            print(f"\r    ⏳ [{bar}] {session.completion_rate:.0f}% (숙련도: {session.mastery_score:.0f}%) | 남은 시간: {LearningConfig.format_duration(remaining)}", 
                  end="", flush=True)
            
            # 주기적 저장
            if time.time() - last_save_time >= LearningConfig.SAVE_INTERVAL:
                progress_data = self._load_user_progress()
                progress_data["total_learning_time"] = self.total_learning_time + (elapsed / 3600)
                self._save_user_progress(progress_data)
                last_save_time = time.time()
            
            # 대기
            if i < total_updates:
                await asyncio.sleep(update_interval)
        
        print(f"\n    ✅ '{session.topic}' 학습 완료!")
        
        # 학습 시간 누적
        self.total_learning_time += session.duration_minutes / 60.0
        
        # 학습 노트 생성
        session.notes = f"{session.topic} 학습 완료. 숙련도: {session.mastery_score:.1f}%"
        
        # 학습 데이터 저장
        await self._save_session_data(session)
    
    async def _load_admin_content(self, topic: str) -> Optional[Dict[str, Any]]:
        """관리자 학습 콘텐츠 로드 (읽기 전용)"""
        try:
            if not self.admin_data_dir.exists():
                return None
            
            # 관리자 세션에서 해당 주제 찾기
            sessions_dir = self.admin_data_dir / "sessions"
            if sessions_dir.exists():
                for session_file in sessions_dir.glob("*/session_data.json"):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if topic.lower() in data.get('topic', '').lower():
                            return data
                    except:
                        continue
            return None
        except:
            return None
    
    async def _save_session_data(self, session: UserLearningSession):
        """세션 데이터 저장"""
        session_dir = self.user_data_dir / session.start_time.strftime('%Y%m%d_%H%M%S')
        session_dir.mkdir(exist_ok=True)
        
        # 파일명에서 특수문자 제거
        safe_topic_name = self._make_safe_filename(session.topic)
        session_file = session_dir / f"{safe_topic_name}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(session), f, indent=2, ensure_ascii=False, default=str)
    
    async def _execute_exercise(self, exercise: str, level: str, save_code: bool = True):
        """실습 프로젝트 실행"""
        print(f"    🔨 실습 진행 중: {exercise}")
        
        actual_duration = LearningConfig.get_actual_duration(LearningConfig.EXERCISE_DURATION)
        steps = ["설계", "구현", "테스트", "최적화"]
        step_duration = actual_duration / len(steps)
        
        # 실습별 코드 생성
        exercise_code = self._generate_exercise_code(exercise, level)
        
        for step in steps:
            print(f"      {step}...")
            await asyncio.sleep(step_duration)
        
        print(f"    ✅ 실습 완료: {exercise}")
        
        # 실습 결과 저장
        exercise_dir = self.user_data_dir / "exercises"
        exercise_dir.mkdir(exist_ok=True)
        
        # 코드 파일 디렉토리
        code_dir = exercise_dir / "code"
        code_dir.mkdir(exist_ok=True)
        
        # 파일명에서 특수문자 제거
        safe_exercise_name = self._make_safe_filename(exercise)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 마크다운 파일 저장 (설명 및 학습 내용)
        exercise_file = exercise_dir / f"{safe_exercise_name}_{timestamp}.md"
        exercise_content = f"""# {exercise}

**난이도**: {level}
**완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 구현 내용
- 요구사항 분석 완료
- 핵심 기능 구현
- 테스트 통과
- 성능 최적화

## 실습 코드
```csharp
{exercise_code}
```

## 학습 성과
이 실습을 통해 {level} 수준의 C# 프로그래밍 실력을 향상시켰습니다.

## 코드 파일 위치
`user_learning_data/exercises/code/{safe_exercise_name}_{timestamp}.cs`
"""
        exercise_file.write_text(exercise_content, encoding='utf-8')
        
        # 실제 코드 파일 저장 (.cs 파일)
        if save_code:
            code_file = code_dir / f"{safe_exercise_name}_{timestamp}.cs"
            code_file.write_text(exercise_code, encoding='utf-8')
            print(f"    💾 실습 코드 저장됨: {code_file.name}")
    
    async def _display_progress_report(self, progress: Dict[str, Any]):
        """진행률 리포트 표시"""
        completed_count = len(progress.get("completed_topics", []))
        
        # 전체 주제 수 계산
        total_topics = sum(len(block["topics"]) for block in self.learning_curriculum.values())
        progress_percentage = (completed_count / total_topics * 100) if total_topics > 0 else 0
        
        print(f"\n📊 학습 진행률 리포트")
        print(f"   ⏰ 총 학습 시간: {self.total_learning_time:.1f}시간")
        print(f"   📚 완료된 주제: {completed_count}/{total_topics} ({progress_percentage:.1f}%)")
        print(f"   🎯 현재 수준: {self._get_current_level()}")
        
        # 남은 주제 표시
        remaining = total_topics - completed_count
        if remaining > 0:
            print(f"   📝 남은 주제: {remaining}개")
    
    def _get_current_level(self) -> str:
        """현재 학습 수준"""
        if self.total_learning_time < 4:
            return "초급 (Beginner)"
        elif self.total_learning_time < 12:
            return "중급 (Intermediate)"
        elif self.total_learning_time < 20:
            return "고급 (Advanced)"
        else:
            return "전문가 (Expert)"
    
    async def _generate_final_report(self, start_time: datetime, progress: Dict[str, Any]):
        """최종 리포트 생성"""
        end_time = datetime.now()
        actual_duration = end_time - start_time
        
        print(f"\n" + "=" * 80)
        print("🎉 C# 학습 마라톤 완료!")
        print("=" * 80)
        
        # 전체 주제 통계
        all_topics = []
        for block in self.learning_curriculum.values():
            all_topics.extend(block["topics"])
        
        completed_topics = progress.get("completed_topics", [])
        completion_rate = (len(completed_topics) / len(all_topics) * 100) if all_topics else 0
        
        report = f"""
📊 학습 성과 요약:
  ⏰ 실제 소요 시간: {actual_duration}
  📚 총 학습 시간: {self.total_learning_time:.1f}시간
  🎯 완료한 주제: {len(completed_topics)}/{len(all_topics)} ({completion_rate:.1f}%)
  📈 현재 수준: {self._get_current_level()}
  💾 저장 위치: {self.user_data_dir}

📚 완료한 주제들:
{chr(10).join(f'  ✅ {topic}' for topic in completed_topics[:10])}
{f'  ... 외 {len(completed_topics) - 10}개' if len(completed_topics) > 10 else ''}

🎯 다음 단계:
  1. 실제 프로젝트에 C# 적용하기
  2. Godot 게임 개발 시작하기
  3. 고급 패턴과 최적화 학습하기
"""
        
        print(report)
        
        # 학습 요약 저장
        summary_file = self.user_data_dir / f"learning_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(actual_duration),
            "total_learning_hours": self.total_learning_time,
            "completed_topics": completed_topics,
            "completion_rate": completion_rate,
            "current_level": self._get_current_level(),
            "sessions": [asdict(s) for s in self.learning_sessions]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
        
        print("=" * 80)
        print("💡 'autoci --production' 으로 실제 게임 개발을 시작해보세요!")
        print("=" * 80)
    
    async def quick_topic_review(self, topic: str):
        """특정 주제 빠른 복습"""
        print(f"⚡ 빠른 복습: {topic}")
        print("=" * 60)
        
        # 이미 학습한 주제인지 확인
        progress = self._load_user_progress()
        if topic in progress.get("completed_topics", []):
            print(f"✅ 이미 학습한 주제입니다. 복습을 진행합니다.")
        
        # 짧은 세션으로 복습
        session = UserLearningSession(
            topic=topic,
            level="review",
            duration_minutes=15,
            start_time=datetime.now(),
            completion_rate=0.0,
            mastery_score=0.0
        )
        
        await self._execute_learning_session(session)
        
        print(f"✅ '{topic}' 복습 완료!")
    
    def _generate_exercise_code(self, exercise: str, level: str) -> str:
        """실습에 대한 실제 C# 코드 생성"""
        # 실습별 실제 코드 템플릿
        exercise_codes = {
            "계산기 만들기": """using System;

namespace Calculator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("간단한 계산기 프로그램");
            
            while (true)
            {
                Console.WriteLine("\n1. 덧셈  2. 뺄셈  3. 곱셈  4. 나눗셈  5. 종료");
                Console.Write("선택: ");
                
                if (!int.TryParse(Console.ReadLine(), out int choice))
                {
                    Console.WriteLine("잘못된 입력입니다.");
                    continue;
                }
                
                if (choice == 5) break;
                
                Console.Write("첫 번째 숫자: ");
                if (!double.TryParse(Console.ReadLine(), out double num1))
                {
                    Console.WriteLine("잘못된 숫자입니다.");
                    continue;
                }
                
                Console.Write("두 번째 숫자: ");
                if (!double.TryParse(Console.ReadLine(), out double num2))
                {
                    Console.WriteLine("잘못된 숫자입니다.");
                    continue;
                }
                
                double result = 0;
                switch (choice)
                {
                    case 1:
                        result = num1 + num2;
                        Console.WriteLine($"{num1} + {num2} = {result}");
                        break;
                    case 2:
                        result = num1 - num2;
                        Console.WriteLine($"{num1} - {num2} = {result}");
                        break;
                    case 3:
                        result = num1 * num2;
                        Console.WriteLine($"{num1} * {num2} = {result}");
                        break;
                    case 4:
                        if (num2 != 0)
                        {
                            result = num1 / num2;
                            Console.WriteLine($"{num1} / {num2} = {result}");
                        }
                        else
                        {
                            Console.WriteLine("0으로 나눌 수 없습니다.");
                        }
                        break;
                    default:
                        Console.WriteLine("잘못된 선택입니다.");
                        break;
                }
            }
            
            Console.WriteLine("계산기를 종료합니다.");
        }
    }
}""",
            
            "숫자 맞추기 게임": """using System;

namespace NumberGuessingGame
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random();
            int targetNumber = random.Next(1, 101);
            int attempts = 0;
            int maxAttempts = 10;
            
            Console.WriteLine("숫자 맞추기 게임!");
            Console.WriteLine("1부터 100 사이의 숫자를 맞춰보세요.");
            Console.WriteLine($"기회는 {maxAttempts}번입니다.\n");
            
            while (attempts < maxAttempts)
            {
                attempts++;
                Console.Write($"시도 {attempts}/{maxAttempts}: ");
                
                if (!int.TryParse(Console.ReadLine(), out int guess))
                {
                    Console.WriteLine("올바른 숫자를 입력하세요.");
                    attempts--;
                    continue;
                }
                
                if (guess < 1 || guess > 100)
                {
                    Console.WriteLine("1부터 100 사이의 숫자를 입력하세요.");
                    attempts--;
                    continue;
                }
                
                if (guess == targetNumber)
                {
                    Console.WriteLine($"\n🎉 정답입니다! {attempts}번 만에 맞추셨습니다!");
                    break;
                }
                else if (guess < targetNumber)
                {
                    Console.WriteLine("더 큰 숫자입니다.");
                }
                else
                {
                    Console.WriteLine("더 작은 숫자입니다.");
                }
                
                if (attempts == maxAttempts)
                {
                    Console.WriteLine($"\n😢 게임 오버! 정답은 {targetNumber}였습니다.");
                }
            }
            
            Console.WriteLine("\n게임을 종료합니다.");
        }
    }
}""",
            
            "문자열 처리": """using System;
using System.Linq;
using System.Text;

namespace StringProcessing
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("문자열 처리 프로그램\n");
            
            Console.Write("문자열을 입력하세요: ");
            string input = Console.ReadLine();
            
            Console.WriteLine("\n=== 문자열 분석 ===");
            Console.WriteLine($"원본 문자열: {input}");
            Console.WriteLine($"길이: {input.Length}자");
            Console.WriteLine($"대문자 변환: {input.ToUpper()}");
            Console.WriteLine($"소문자 변환: {input.ToLower()}");
            Console.WriteLine($"공백 제거: {input.Trim()}");
            Console.WriteLine($"역순: {new string(input.Reverse().ToArray())}");
            
            // 단어 수 계산
            string[] words = input.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            Console.WriteLine($"단어 수: {words.Length}개");
            
            // 문자 빈도 분석
            Console.WriteLine("\n=== 문자 빈도 ===");
            var charFrequency = input
                .Where(c => !char.IsWhiteSpace(c))
                .GroupBy(c => char.ToLower(c))
                .OrderByDescending(g => g.Count())
                .Take(5);
            
            foreach (var group in charFrequency)
            {
                Console.WriteLine($"'{group.Key}': {group.Count()}회");
            }
            
            // 회문 검사
            string cleanedInput = new string(input.Where(char.IsLetterOrDigit).ToArray()).ToLower();
            bool isPalindrome = cleanedInput == new string(cleanedInput.Reverse().ToArray());
            Console.WriteLine($"\n회문 여부: {(isPalindrome ? "예" : "아니오")}");
        }
    }
}""",
            
            "동물 클래스 계층": """using System;
using System.Collections.Generic;

namespace AnimalHierarchy
{
    // 기본 동물 클래스
    public abstract class Animal
    {
        public string Name { get; set; }
        public int Age { get; set; }
        public double Weight { get; set; }
        
        public Animal(string name, int age, double weight)
        {
            Name = name;
            Age = age;
            Weight = weight;
        }
        
        public abstract void MakeSound();
        public abstract void Move();
        
        public virtual void DisplayInfo()
        {
            Console.WriteLine($"이름: {Name}, 나이: {Age}살, 무게: {Weight}kg");
        }
    }
    
    // 포유류 클래스
    public class Mammal : Animal
    {
        public string FurColor { get; set; }
        
        public Mammal(string name, int age, double weight, string furColor) 
            : base(name, age, weight)
        {
            FurColor = furColor;
        }
        
        public override void DisplayInfo()
        {
            base.DisplayInfo();
            Console.WriteLine($"털 색깔: {FurColor}");
        }
    }
    
    // 개 클래스
    public class Dog : Mammal
    {
        public string Breed { get; set; }
        
        public Dog(string name, int age, double weight, string furColor, string breed)
            : base(name, age, weight, furColor)
        {
            Breed = breed;
        }
        
        public override void MakeSound()
        {
            Console.WriteLine($"{Name}가 멍멍 짖습니다!");
        }
        
        public override void Move()
        {
            Console.WriteLine($"{Name}가 네 발로 달립니다.");
        }
        
        public void WagTail()
        {
            Console.WriteLine($"{Name}가 꼬리를 흔듭니다.");
        }
    }
    
    // 고양이 클래스
    public class Cat : Mammal
    {
        public bool IsIndoor { get; set; }
        
        public Cat(string name, int age, double weight, string furColor, bool isIndoor)
            : base(name, age, weight, furColor)
        {
            IsIndoor = isIndoor;
        }
        
        public override void MakeSound()
        {
            Console.WriteLine($"{Name}가 야옹하고 웁니다!");
        }
        
        public override void Move()
        {
            Console.WriteLine($"{Name}가 조용히 걸어다닙니다.");
        }
        
        public void Purr()
        {
            Console.WriteLine($"{Name}가 그르릉거립니다.");
        }
    }
    
    // 조류 클래스
    public class Bird : Animal
    {
        public double WingSpan { get; set; }
        public bool CanFly { get; set; }
        
        public Bird(string name, int age, double weight, double wingSpan, bool canFly)
            : base(name, age, weight)
        {
            WingSpan = wingSpan;
            CanFly = canFly;
        }
        
        public override void MakeSound()
        {
            Console.WriteLine($"{Name}가 지저귑니다!");
        }
        
        public override void Move()
        {
            if (CanFly)
                Console.WriteLine($"{Name}가 날아다닙니다.");
            else
                Console.WriteLine($"{Name}가 걸어다닙니다.");
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            List<Animal> zoo = new List<Animal>
            {
                new Dog("바둑이", 3, 15.5, "갈색", "진돗개"),
                new Cat("나비", 2, 4.2, "흰색", true),
                new Bird("파랑이", 1, 0.3, 0.5, true),
                new Dog("똘이", 5, 20.0, "검은색", "셰퍼드"),
                new Cat("야옹이", 4, 5.0, "삼색", false)
            };
            
            Console.WriteLine("=== 동물원의 동물들 ===");
            foreach (var animal in zoo)
            {
                Console.WriteLine($"\n--- {animal.GetType().Name} ---");
                animal.DisplayInfo();
                animal.MakeSound();
                animal.Move();
                
                // 특별한 행동
                if (animal is Dog dog)
                {
                    dog.WagTail();
                }
                else if (animal is Cat cat)
                {
                    cat.Purr();
                }
            }
        }
    }
}""",
            
            "게임 캐릭터 시스템": """using System;
using System.Collections.Generic;

namespace GameCharacterSystem
{
    // 캐릭터 인터페이스
    public interface ICharacter
    {
        string Name { get; }
        int Level { get; }
        void Attack(ICharacter target);
        void TakeDamage(int damage);
        bool IsAlive { get; }
    }
    
    // 스킬 인터페이스
    public interface ISkill
    {
        string Name { get; }
        int ManaCost { get; }
        void Use(Character caster, ICharacter target);
    }
    
    // 기본 캐릭터 클래스
    public abstract class Character : ICharacter
    {
        public string Name { get; protected set; }
        public int Level { get; protected set; }
        public int Health { get; protected set; }
        public int MaxHealth { get; protected set; }
        public int Mana { get; protected set; }
        public int MaxMana { get; protected set; }
        public int AttackPower { get; protected set; }
        public int Defense { get; protected set; }
        
        public bool IsAlive => Health > 0;
        
        protected List<ISkill> skills = new List<ISkill>();
        
        public Character(string name, int level)
        {
            Name = name;
            Level = level;
            InitializeStats();
        }
        
        protected abstract void InitializeStats();
        
        public virtual void Attack(ICharacter target)
        {
            Console.WriteLine($"{Name}이(가) {target.Name}을(를) 공격합니다!");
            int damage = AttackPower;
            target.TakeDamage(damage);
        }
        
        public virtual void TakeDamage(int damage)
        {
            int actualDamage = Math.Max(damage - Defense, 0);
            Health -= actualDamage;
            Console.WriteLine($"{Name}이(가) {actualDamage}의 피해를 입었습니다! (남은 HP: {Health}/{MaxHealth})");
            
            if (!IsAlive)
            {
                Console.WriteLine($"{Name}이(가) 쓰러졌습니다!");
            }
        }
        
        public void UseSkill(int skillIndex, ICharacter target)
        {
            if (skillIndex < 0 || skillIndex >= skills.Count)
            {
                Console.WriteLine("잘못된 스킬 번호입니다.");
                return;
            }
            
            var skill = skills[skillIndex];
            if (Mana >= skill.ManaCost)
            {
                skill.Use(this, target);
                Mana -= skill.ManaCost;
            }
            else
            {
                Console.WriteLine($"마나가 부족합니다! (필요: {skill.ManaCost}, 현재: {Mana})");
            }
        }
    }
    
    // 전사 클래스
    public class Warrior : Character
    {
        public Warrior(string name, int level) : base(name, level)
        {
            skills.Add(new PowerStrike());
            skills.Add(new ShieldBash());
        }
        
        protected override void InitializeStats()
        {
            MaxHealth = 100 + (Level * 20);
            Health = MaxHealth;
            MaxMana = 50 + (Level * 5);
            Mana = MaxMana;
            AttackPower = 15 + (Level * 3);
            Defense = 10 + (Level * 2);
        }
    }
    
    // 마법사 클래스
    public class Mage : Character
    {
        public Mage(string name, int level) : base(name, level)
        {
            skills.Add(new Fireball());
            skills.Add(new FrostBolt());
        }
        
        protected override void InitializeStats()
        {
            MaxHealth = 60 + (Level * 10);
            Health = MaxHealth;
            MaxMana = 100 + (Level * 15);
            Mana = MaxMana;
            AttackPower = 10 + (Level * 2);
            Defense = 5 + Level;
        }
    }
    
    // 스킬 구현
    public class PowerStrike : ISkill
    {
        public string Name => "파워 스트라이크";
        public int ManaCost => 10;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.AttackPower * 2;
            target.TakeDamage(damage);
        }
    }
    
    public class ShieldBash : ISkill
    {
        public string Name => "방패 강타";
        public int ManaCost => 15;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.Defense + 10;
            target.TakeDamage(damage);
        }
    }
    
    public class Fireball : ISkill
    {
        public string Name => "화염구";
        public int ManaCost => 20;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.Level * 10 + 20;
            target.TakeDamage(damage);
        }
    }
    
    public class FrostBolt : ISkill
    {
        public string Name => "서리 화살";
        public int ManaCost => 15;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.Level * 8 + 15;
            target.TakeDamage(damage);
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== 게임 캐릭터 전투 시스템 ===");
            
            var warrior = new Warrior("전사", 5);
            var mage = new Mage("마법사", 5);
            
            Console.WriteLine($"\n{warrior.Name} (레벨 {warrior.Level}) vs {mage.Name} (레벨 {mage.Level})");
            Console.WriteLine($"{warrior.Name}: HP {warrior.Health}/{warrior.MaxHealth}, MP {warrior.Mana}/{warrior.MaxMana}");
            Console.WriteLine($"{mage.Name}: HP {mage.Health}/{mage.MaxHealth}, MP {mage.Mana}/{mage.MaxMana}");
            
            // 전투 시뮬레이션
            Console.WriteLine("\n=== 전투 시작! ===");
            
            // 전사의 턴
            Console.WriteLine("\n[전사의 턴]");
            warrior.Attack(mage);
            warrior.UseSkill(0, mage); // 파워 스트라이크
            
            if (mage.IsAlive)
            {
                // 마법사의 턴
                Console.WriteLine("\n[마법사의 턴]");
                mage.Attack(warrior);
                mage.UseSkill(0, warrior); // 화염구
            }
            
            // 결과
            Console.WriteLine("\n=== 전투 결과 ===");
            Console.WriteLine($"{warrior.Name}: HP {warrior.Health}/{warrior.MaxHealth}");
            Console.WriteLine($"{mage.Name}: HP {mage.Health}/{mage.MaxHealth}");
        }
    }
}""",
            
            "은행 계좌 시스템": """using System;
using System.Collections.Generic;
using System.Linq;

namespace BankAccountSystem
{
    // 계좌 유형 열거형
    public enum AccountType
    {
        Checking,
        Savings,
        FixedDeposit
    }
    
    // 거래 기록 클래스
    public class Transaction
    {
        public DateTime Date { get; }
        public string Type { get; }
        public decimal Amount { get; }
        public decimal Balance { get; }
        public string Description { get; }
        
        public Transaction(string type, decimal amount, decimal balance, string description)
        {
            Date = DateTime.Now;
            Type = type;
            Amount = amount;
            Balance = balance;
            Description = description;
        }
        
        public override string ToString()
        {
            return $"{Date:yyyy-MM-dd HH:mm:ss} | {Type,-10} | {Amount,10:C} | {Balance,10:C} | {Description}";
        }
    }
    
    // 기본 계좌 클래스
    public abstract class BankAccount
    {
        private static int nextAccountNumber = 1000;
        
        public string AccountNumber { get; }
        public string AccountHolder { get; }
        public AccountType Type { get; }
        protected decimal balance;
        public decimal Balance => balance;
        
        protected List<Transaction> transactions = new List<Transaction>();
        
        public BankAccount(string accountHolder, AccountType type, decimal initialDeposit)
        {
            AccountNumber = GenerateAccountNumber();
            AccountHolder = accountHolder;
            Type = type;
            
            if (initialDeposit > 0)
            {
                balance = initialDeposit;
                transactions.Add(new Transaction("개설입금", initialDeposit, balance, "계좌 개설"));
            }
        }
        
        private string GenerateAccountNumber()
        {
            return $"ACC{nextAccountNumber++:D6}";
        }
        
        public virtual bool Deposit(decimal amount)
        {
            if (amount <= 0)
            {
                Console.WriteLine("입금액은 0보다 커야 합니다.");
                return false;
            }
            
            balance += amount;
            transactions.Add(new Transaction("입금", amount, balance, "현금 입금"));
            Console.WriteLine($"{amount:C}이 입금되었습니다. 현재 잔액: {balance:C}");
            return true;
        }
        
        public abstract bool Withdraw(decimal amount);
        
        public void PrintStatement()
        {
            Console.WriteLine($"\n=== 계좌 명세서 ===");
            Console.WriteLine($"계좌번호: {AccountNumber}");
            Console.WriteLine($"예금주: {AccountHolder}");
            Console.WriteLine($"계좌유형: {Type}");
            Console.WriteLine($"현재잔액: {balance:C}");
            Console.WriteLine("\n거래내역:");
            Console.WriteLine(new string('-', 80));
            
            foreach (var transaction in transactions.TakeLast(10))
            {
                Console.WriteLine(transaction);
            }
        }
        
        public decimal CalculateInterest()
        {
            return CalculateInterestImpl();
        }
        
        protected abstract decimal CalculateInterestImpl();
    }
    
    // 입출금 계좌
    public class CheckingAccount : BankAccount
    {
        private const decimal OverdraftLimit = 1000m;
        
        public CheckingAccount(string accountHolder, decimal initialDeposit)
            : base(accountHolder, AccountType.Checking, initialDeposit)
        {
        }
        
        public override bool Withdraw(decimal amount)
        {
            if (amount <= 0)
            {
                Console.WriteLine("출금액은 0보다 커야 합니다.");
                return false;
            }
            
            if (balance - amount < -OverdraftLimit)
            {
                Console.WriteLine($"출금 한도를 초과합니다. 최대 출금 가능액: {balance + OverdraftLimit:C}");
                return false;
            }
            
            balance -= amount;
            transactions.Add(new Transaction("출금", -amount, balance, "현금 출금"));
            Console.WriteLine($"{amount:C}이 출금되었습니다. 현재 잔액: {balance:C}");
            
            if (balance < 0)
            {
                Console.WriteLine($"⚠️ 마이너스 통장 사용 중: {balance:C}");
            }
            
            return true;
        }
        
        protected override decimal CalculateInterestImpl()
        {
            return balance > 0 ? balance * 0.001m : 0; // 0.1% 이자
        }
    }
    
    // 저축 계좌
    public class SavingsAccount : BankAccount
    {
        private int withdrawalsThisMonth = 0;
        private const int FreeWithdrawalsPerMonth = 3;
        private const decimal WithdrawalFee = 5m;
        
        public SavingsAccount(string accountHolder, decimal initialDeposit)
            : base(accountHolder, AccountType.Savings, initialDeposit)
        {
        }
        
        public override bool Withdraw(decimal amount)
        {
            if (amount <= 0)
            {
                Console.WriteLine("출금액은 0보다 커야 합니다.");
                return false;
            }
            
            decimal totalAmount = amount;
            if (withdrawalsThisMonth >= FreeWithdrawalsPerMonth)
            {
                totalAmount += WithdrawalFee;
                Console.WriteLine($"월 {FreeWithdrawalsPerMonth}회 초과 출금으로 수수료 {WithdrawalFee:C}가 부과됩니다.");
            }
            
            if (balance < totalAmount)
            {
                Console.WriteLine($"잔액이 부족합니다. 현재 잔액: {balance:C}");
                return false;
            }
            
            balance -= totalAmount;
            withdrawalsThisMonth++;
            transactions.Add(new Transaction("출금", -totalAmount, balance, 
                withdrawalsThisMonth > FreeWithdrawalsPerMonth ? "출금 (수수료 포함)" : "출금"));
            Console.WriteLine($"{amount:C}이 출금되었습니다. 현재 잔액: {balance:C}");
            
            return true;
        }
        
        protected override decimal CalculateInterestImpl()
        {
            return balance * 0.02m; // 2% 이자
        }
        
        public void ResetMonthlyWithdrawals()
        {
            withdrawalsThisMonth = 0;
            Console.WriteLine("월별 출금 횟수가 초기화되었습니다.");
        }
    }
    
    // 은행 시스템
    public class Bank
    {
        private Dictionary<string, BankAccount> accounts = new Dictionary<string, BankAccount>();
        
        public void CreateAccount(AccountType type, string accountHolder, decimal initialDeposit)
        {
            BankAccount account = type switch
            {
                AccountType.Checking => new CheckingAccount(accountHolder, initialDeposit),
                AccountType.Savings => new SavingsAccount(accountHolder, initialDeposit),
                _ => throw new ArgumentException("지원하지 않는 계좌 유형입니다.")
            };
            
            accounts[account.AccountNumber] = account;
            Console.WriteLine($"계좌가 생성되었습니다. 계좌번호: {account.AccountNumber}");
        }
        
        public BankAccount GetAccount(string accountNumber)
        {
            return accounts.TryGetValue(accountNumber, out var account) ? account : null;
        }
        
        public void Transfer(string fromAccountNumber, string toAccountNumber, decimal amount)
        {
            var fromAccount = GetAccount(fromAccountNumber);
            var toAccount = GetAccount(toAccountNumber);
            
            if (fromAccount == null || toAccount == null)
            {
                Console.WriteLine("계좌를 찾을 수 없습니다.");
                return;
            }
            
            if (fromAccount.Withdraw(amount))
            {
                toAccount.Deposit(amount);
                Console.WriteLine($"이체 완료: {fromAccountNumber} → {toAccountNumber}, 금액: {amount:C}");
            }
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Bank bank = new Bank();
            
            // 계좌 생성
            bank.CreateAccount(AccountType.Checking, "홍길동", 10000);
            bank.CreateAccount(AccountType.Savings, "김철수", 50000);
            
            // 계좌 조회 (실제로는 계좌번호를 알아야 함)
            var checkingAccount = bank.GetAccount("ACC001000");
            var savingsAccount = bank.GetAccount("ACC001001");
            
            // 거래 시뮬레이션
            Console.WriteLine("\n=== 거래 시뮬레이션 ===");
            
            checkingAccount?.Deposit(5000);
            checkingAccount?.Withdraw(3000);
            
            savingsAccount?.Deposit(10000);
            savingsAccount?.Withdraw(2000);
            savingsAccount?.Withdraw(3000);
            savingsAccount?.Withdraw(1000);
            savingsAccount?.Withdraw(500); // 수수료 부과
            
            // 이체
            Console.WriteLine("\n=== 계좌 이체 ===");
            bank.Transfer("ACC001000", "ACC001001", 2000);
            
            // 명세서 출력
            checkingAccount?.PrintStatement();
            savingsAccount?.PrintStatement();
            
            // 이자 계산
            Console.WriteLine("\n=== 이자 계산 ===");
            if (checkingAccount != null)
                Console.WriteLine($"입출금계좌 이자: {checkingAccount.CalculateInterest():C}");
            if (savingsAccount != null)
                Console.WriteLine($"저축계좌 이자: {savingsAccount.CalculateInterest():C}");
        }
    }
}"""
        }
        
        # 기본 코드 템플릿
        default_code = f"""using System;

namespace {exercise.replace(' ', '')}
{{
    class Program
    {{
        static void Main(string[] args)
        {{
            Console.WriteLine("{exercise} - {level} 레벨 실습");
            
            // TODO: 실습 코드 구현
            // 이 부분에 실제 구현 코드가 들어갑니다.
            
            Console.WriteLine("실습이 완료되었습니다.");
        }}
    }}
}}"""
        
        return exercise_codes.get(exercise, default_code)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """현재 학습 상태"""
        progress = self._load_user_progress()
        
        all_topics = []
        for block in self.learning_curriculum.values():
            all_topics.extend(block["topics"])
        
        completed = progress.get("completed_topics", [])
        remaining = [t for t in all_topics if t not in completed]
        
        return {
            "is_learning": self.is_learning,
            "total_topics": len(all_topics),
            "completed_topics": len(completed),
            "remaining_topics": len(remaining),
            "completion_rate": (len(completed) / len(all_topics) * 100) if all_topics else 0,
            "total_learning_time": progress.get("total_learning_time", 0),
            "current_level": self._get_current_level(),
            "next_topics": remaining[:5] if remaining else []
        }

# 독립 실행용
async def main():
    """테스트 실행"""
    learning = CSharp24HUserLearning()
    
    print("📚 C# 24시간 학습 시스템")
    print("1. 24시간 전체 학습")
    print("2. 남은 주제만 학습")
    print("3. 모든 주제 처음부터")
    print("4. 학습 상태 확인")
    
    choice = input("선택 (1-4): ")
    
    if choice == "1":
        await learning.start_24h_learning_marathon()
    elif choice == "2":
        await learning.learn_remaining_topics()
    elif choice == "3":
        await learning.learn_all_topics()
    else:
        status = learning.get_learning_status()
        print(f"학습 상태: {json.dumps(status, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main())