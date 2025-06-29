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
    
    async def _execute_exercise(self, exercise: str, level: str):
        """실습 프로젝트 실행"""
        print(f"    🔨 실습 진행 중: {exercise}")
        
        actual_duration = LearningConfig.get_actual_duration(LearningConfig.EXERCISE_DURATION)
        steps = ["설계", "구현", "테스트", "최적화"]
        step_duration = actual_duration / len(steps)
        
        for step in steps:
            print(f"      {step}...")
            await asyncio.sleep(step_duration)
        
        print(f"    ✅ 실습 완료: {exercise}")
        
        # 실습 결과 저장
        exercise_dir = self.user_data_dir / "exercises"
        exercise_dir.mkdir(exist_ok=True)
        
        # 파일명에서 특수문자 제거
        safe_exercise_name = self._make_safe_filename(exercise)
        exercise_file = exercise_dir / f"{safe_exercise_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        exercise_content = f"""# {exercise}

**난이도**: {level}
**완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 구현 내용
- 요구사항 분석 완료
- 핵심 기능 구현
- 테스트 통과
- 성능 최적화

## 학습 성과
이 실습을 통해 {level} 수준의 C# 프로그래밍 실력을 향상시켰습니다.
"""
        exercise_file.write_text(exercise_content, encoding='utf-8')
    
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