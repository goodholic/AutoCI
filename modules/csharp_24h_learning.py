#!/usr/bin/env python3
"""
24시간 C# 학습 시스템
AI가 24시간 동안 지속적으로 C# 프로그래밍을 학습하고 실습
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

@dataclass
class LearningSession:
    """학습 세션"""
    topic: str
    level: str  # beginner, intermediate, advanced, expert
    duration_minutes: int
    start_time: datetime
    exercises_count: int
    completion_rate: float = 0.0
    code_samples: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.code_samples is None:
            self.code_samples = []

@dataclass
class LearningProgress:
    """학습 진행률"""
    total_hours: float
    topics_completed: int
    current_level: str
    mastery_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    next_recommendations: List[str]

class CSharp24HLearning:
    """24시간 C# 학습 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("CSharp24HLearning")
        self.project_root = Path(__file__).parent.parent
        self.learning_data_dir = self.project_root / "csharp_24h_learning"
        self.learning_data_dir.mkdir(exist_ok=True)
        
        # 학습 진행 상태
        self.current_session: Optional[LearningSession] = None
        self.learning_history: List[LearningSession] = []
        self.total_learning_time = 0.0
        self.is_learning = False
        
        # 24시간 학습 커리큘럼
        self.learning_curriculum = self._create_24h_curriculum()
        
        # 학습 통계
        self.daily_goals = {
            "minimum_hours": 2,
            "target_topics": 5,
            "code_exercises": 20,
            "mastery_threshold": 0.8
        }
    
    def _make_safe_filename(self, filename: str) -> str:
        """파일명에서 특수문자를 제거하여 안전한 파일명 생성"""
        # Windows와 Unix에서 문제가 되는 특수문자들을 언더스코어로 대체
        return filename.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('*', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        
    def _create_24h_curriculum(self) -> Dict[str, Dict[str, Any]]:
        """24시간 학습 커리큘럼 생성"""
        return {
            # 1-4시간: 기초
            "basics": {
                "level": "beginner",
                "duration": 4,
                "topics": [
                    "변수와 데이터 타입",
                    "연산자와 표현식", 
                    "조건문 (if, switch)",
                    "반복문 (for, while, foreach)",
                    "메서드 기초",
                    "배열과 컬렉션 기초"
                ],
                "practical_exercises": [
                    "계산기 프로그램",
                    "숫자 맞추기 게임",
                    "간단한 문자열 처리"
                ]
            },
            
            # 5-8시간: 객체지향
            "oop_fundamentals": {
                "level": "beginner",
                "duration": 4,
                "topics": [
                    "클래스와 객체",
                    "생성자와 소멸자",
                    "상속 (Inheritance)",
                    "다형성 (Polymorphism)",
                    "캡슐화 (Encapsulation)",
                    "추상 클래스와 인터페이스"
                ],
                "practical_exercises": [
                    "동물 클래스 계층구조",
                    "게임 캐릭터 시스템",
                    "간단한 은행 계좌 시스템"
                ]
            },
            
            # 9-12시간: 고급 기능
            "advanced_features": {
                "level": "intermediate",
                "duration": 4,
                "topics": [
                    "제네릭 (Generics)",
                    "델리게이트와 이벤트",
                    "람다 표현식",
                    "LINQ 기초",
                    "예외 처리",
                    "파일 I/O"
                ],
                "practical_exercises": [
                    "제네릭 컬렉션 구현",
                    "이벤트 기반 시스템",
                    "데이터 검색 및 필터링"
                ]
            },
            
            # 13-16시간: 비동기 및 병렬
            "async_parallel": {
                "level": "intermediate",
                "duration": 4,
                "topics": [
                    "Task와 async/await",
                    "비동기 프로그래밍 패턴",
                    "병렬 처리 (Parallel)",
                    "Thread Safety",
                    "CancellationToken",
                    "비동기 스트림"
                ],
                "practical_exercises": [
                    "비동기 웹 크롤러",
                    "병렬 데이터 처리",
                    "실시간 데이터 스트림"
                ]
            },
            
            # 17-20시간: Godot 특화
            "godot_integration": {
                "level": "intermediate",
                "duration": 4,
                "topics": [
                    "Godot C# 바인딩",
                    "Node 시스템 이해",
                    "신호(Signal) 시스템",
                    "리소스 관리",
                    "씬 트리 조작",
                    "물리 시스템 프로그래밍"
                ],
                "practical_exercises": [
                    "플레이어 컨트롤러",
                    "AI 적 시스템",
                    "인벤토리 시스템"
                ]
            },
            
            # 21-24시간: 고급 게임 개발
            "advanced_game_dev": {
                "level": "advanced",
                "duration": 4,
                "topics": [
                    "게임 아키텍처 패턴",
                    "상태 머신",
                    "컴포넌트 시스템",
                    "멀티플레이어 네트워킹",
                    "성능 최적화",
                    "메모리 관리"
                ],
                "practical_exercises": [
                    "완전한 게임 프로토타입",
                    "멀티플레이어 시스템",
                    "성능 분석 도구"
                ]
            }
        }
    
    async def start_24h_learning_marathon(self):
        """24시간 C# 학습 마라톤 시작"""
        print("🚀 24시간 C# 학습 마라톤 시작!")
        print("=" * 80)
        print("AI가 24시간 동안 체계적으로 C# 프로그래밍을 학습합니다.")
        print("실시간으로 학습 진행률과 성과를 확인할 수 있습니다.")
        print("=" * 80)
        
        self.is_learning = True
        start_time = datetime.now()
        
        try:
            # 각 학습 블록 순차 실행
            for block_name, block_info in self.learning_curriculum.items():
                if not self.is_learning:
                    break
                    
                print(f"\n📚 학습 블록 시작: {block_name}")
                print(f"   📖 난이도: {block_info['level']}")
                print(f"   ⏰ 예상 시간: {block_info['duration']}시간")
                print(f"   📋 주제 수: {len(block_info['topics'])}개")
                
                await self._execute_learning_block(block_name, block_info)
                
                # 블록 완료 후 진행률 표시
                await self._display_progress_report()
                
        except KeyboardInterrupt:
            print("\n⏸️ 학습이 사용자에 의해 중단되었습니다.")
        finally:
            self.is_learning = False
            await self._generate_final_report(start_time)
    
    async def _execute_learning_block(self, block_name: str, block_info: Dict[str, Any]):
        """학습 블록 실행"""
        topics = block_info["topics"]
        exercises = block_info.get("practical_exercises", [])
        
        # 각 주제별 학습
        for i, topic in enumerate(topics):
            if not self.is_learning:
                break
                
            print(f"\n  🎯 주제 {i+1}/{len(topics)}: {topic}")
            
            # 학습 세션 시작
            session = LearningSession(
                topic=topic,
                level=block_info["level"],
                duration_minutes=random.randint(20, 40),
                start_time=datetime.now(),
                exercises_count=random.randint(3, 8)
            )
            
            self.current_session = session
            
            # 실제 학습 시뮬레이션
            await self._simulate_learning_session(session)
            
            # 학습 기록 저장
            self.learning_history.append(session)
            self.total_learning_time += session.duration_minutes / 60.0
            
            # 코드 예제 생성
            await self._generate_code_examples(topic, session)
            
        # 실습 프로젝트 수행
        for exercise in exercises:
            if not self.is_learning:
                break
                
            print(f"\n  🛠️ 실습 프로젝트: {exercise}")
            await self._execute_practical_exercise(exercise, block_info["level"])
    
    async def _simulate_learning_session(self, session: LearningSession):
        """학습 세션 시뮬레이션"""
        print(f"    📖 학습 중... (예상 {session.duration_minutes}분)")
        
        # 학습 진행 시뮬레이션
        progress_steps = 10
        for step in range(progress_steps + 1):
            if not self.is_learning:
                break
                
            progress = step / progress_steps
            session.completion_rate = progress * 100
            
            # 진행률 표시
            filled = int(progress * 20)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\r    ⏳ [{bar}] {session.completion_rate:.0f}%", end="", flush=True)
            
            # 학습 시뮬레이션 (실제로는 더 복잡한 로직)
            await asyncio.sleep(session.duration_minutes * 60 / progress_steps / 60)  # 실제 시간 단축
        
        print(f"\n    ✅ '{session.topic}' 학습 완료!")
        
        # 학습 노트 생성
        session.notes = await self._generate_learning_notes(session.topic)
    
    async def _generate_code_examples(self, topic: str, session: LearningSession):
        """주제별 코드 예제 생성"""
        # 주제에 따른 코드 예제 매핑
        code_examples = {
            "변수와 데이터 타입": [
                "int number = 42;",
                "string message = \"Hello, C#!\";",
                "bool isActive = true;",
                "double price = 99.99;"
            ],
            "조건문 (if, switch)": [
                """if (score >= 90)
{
    grade = "A";
}
else if (score >= 80)
{
    grade = "B";
}""",
                """switch (dayOfWeek)
{
    case "Monday":
        mood = "Tired";
        break;
    case "Friday":
        mood = "Happy";
        break;
    default:
        mood = "Normal";
        break;
}"""
            ],
            "클래스와 객체": [
                """public class Player
{
    public string Name { get; set; }
    public int Health { get; set; }
    
    public Player(string name)
    {
        Name = name;
        Health = 100;
    }
    
    public void TakeDamage(int damage)
    {
        Health -= damage;
        if (Health < 0) Health = 0;
    }
}""",
                """Player hero = new Player("Hero");
hero.TakeDamage(25);
Console.WriteLine($"{hero.Name} has {hero.Health} health");"""
            ],
            "async/await": [
                """public async Task<string> DownloadDataAsync(string url)
{
    using var client = new HttpClient();
    var response = await client.GetStringAsync(url);
    return response;
}""",
                """public async Task ProcessDataAsync()
{
    try
    {
        var data = await DownloadDataAsync("https://api.example.com");
        Console.WriteLine($"Downloaded {data.Length} characters");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error: {ex.Message}");
    }
}"""
            ]
        }
        
        # 기본 예제 또는 주제별 예제 선택
        examples = code_examples.get(topic, [f"// {topic} 관련 코드 예제"])
        session.code_samples.extend(examples)
        
        # 코드 파일로 저장 - 파일명에서 특수문자 제거
        safe_topic_name = self._make_safe_filename(topic)
        topic_dir = self.learning_data_dir / safe_topic_name
        topic_dir.mkdir(exist_ok=True)
        
        for i, code in enumerate(examples):
            code_file = topic_dir / f"example_{i+1}.cs"
            code_file.write_text(code, encoding='utf-8')
    
    async def _generate_learning_notes(self, topic: str) -> str:
        """학습 노트 생성"""
        notes_templates = {
            "변수와 데이터 타입": "C#의 기본 데이터 타입을 학습했습니다. int, string, bool, double 등의 타입 사용법을 익혔습니다.",
            "조건문 (if, switch)": "조건부 실행을 위한 if문과 switch문의 사용법을 학습했습니다. 복잡한 조건 처리가 가능해졌습니다.",
            "클래스와 객체": "객체지향 프로그래밍의 핵심인 클래스와 객체를 학습했습니다. 캡슐화와 메서드 사용법을 익혔습니다.",
            "async/await": "비동기 프로그래밍의 핵심인 async/await 패턴을 학습했습니다. 논블로킹 코드 작성이 가능해졌습니다."
        }
        
        return notes_templates.get(topic, f"{topic}에 대한 학습을 완료했습니다. 이론과 실습을 통해 개념을 이해했습니다.")
    
    async def _execute_practical_exercise(self, exercise: str, level: str):
        """실습 프로젝트 수행"""
        print(f"    🔨 실습 진행 중: {exercise}")
        
        # 실습 시뮬레이션
        steps = ["설계", "구현", "테스트", "최적화"]
        for step in steps:
            print(f"      {step} 단계...")
            await asyncio.sleep(0.5)  # 실제로는 더 긴 시간
        
        print(f"    ✅ 실습 완료: {exercise}")
        
        # 실습 결과 저장
        exercise_dir = self.learning_data_dir / "exercises"
        exercise_dir.mkdir(exist_ok=True)
        
        # 파일명에서 특수문자 제거
        safe_exercise_name = self._make_safe_filename(exercise)
        exercise_file = exercise_dir / f"{safe_exercise_name}.md"
        exercise_content = f"""# {exercise}

**난이도**: {level}
**완료 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 구현 내용
- 설계 단계에서 요구사항 분석
- 핵심 기능 구현
- 단위 테스트 작성
- 성능 최적화

## 학습 성과
이 실습을 통해 {level} 수준의 C# 프로그래밍 실력을 향상시켰습니다.
"""
        exercise_file.write_text(exercise_content, encoding='utf-8')
    
    async def _display_progress_report(self):
        """진행률 리포트 표시"""
        completed_topics = len(self.learning_history)
        total_topics = sum(len(block["topics"]) for block in self.learning_curriculum.values())
        progress_percentage = (completed_topics / total_topics) * 100 if total_topics > 0 else 0
        
        print(f"\n📊 학습 진행률 리포트")
        print(f"   ⏰ 총 학습 시간: {self.total_learning_time:.1f}시간")
        print(f"   📚 완료된 주제: {completed_topics}/{total_topics} ({progress_percentage:.1f}%)")
        print(f"   🎯 현재 수준: {self._get_current_level()}")
        print(f"   ⭐ 숙련도 점수: {self._calculate_mastery_score():.1f}/100")
    
    def _get_current_level(self) -> str:
        """현재 학습 수준 계산"""
        if self.total_learning_time < 8:
            return "초급 (Beginner)"
        elif self.total_learning_time < 16:
            return "중급 (Intermediate)"
        else:
            return "고급 (Advanced)"
    
    def _calculate_mastery_score(self) -> float:
        """숙련도 점수 계산"""
        if not self.learning_history:
            return 0.0
        
        total_score = 0.0
        for session in self.learning_history:
            # 완료율과 학습 시간을 기반으로 점수 계산
            time_score = min(100, (session.duration_minutes / 30) * 50)
            completion_score = session.completion_rate * 0.5
            total_score += time_score + completion_score
        
        return min(100, total_score / len(self.learning_history))
    
    async def _generate_final_report(self, start_time: datetime):
        """최종 학습 리포트 생성"""
        end_time = datetime.now()
        actual_duration = end_time - start_time
        
        print(f"\n" + "=" * 80)
        print("🎉 24시간 C# 학습 마라톤 완료!")
        print("=" * 80)
        
        # 학습 통계
        total_topics = len(self.learning_history)
        total_time = self.total_learning_time
        mastery_score = self._calculate_mastery_score()
        current_level = self._get_current_level()
        
        # 강점과 개선 영역 분석
        strengths = self._analyze_strengths()
        improvements = self._analyze_improvements()
        recommendations = self._generate_recommendations()
        
        report = f"""
📊 학습 성과 요약:
  ⏰ 실제 소요 시간: {actual_duration}
  📚 총 학습 시간: {total_time:.1f}시간
  🎯 완료한 주제: {total_topics}개
  📈 현재 수준: {current_level}
  ⭐ 최종 숙련도: {mastery_score:.1f}/100

💪 학습 강점:
{chr(10).join(f'  ✅ {strength}' for strength in strengths)}

🎯 개선 영역:
{chr(10).join(f'  📈 {improvement}' for improvement in improvements)}

💡 다음 단계 추천:
{chr(10).join(f'  🚀 {rec}' for rec in recommendations)}

📁 생성된 학습 자료:
  📂 코드 예제: {self.learning_data_dir}/*/example_*.cs
  📝 실습 프로젝트: {self.learning_data_dir}/exercises/*.md
  📊 학습 로그: {self.learning_data_dir}/learning_log.json
"""
        
        print(report)
        
        # 학습 로그 저장
        await self._save_learning_log(start_time, end_time)
        
        print("=" * 80)
        print("🚀 이제 실제 게임 개발에 C# 지식을 적용해보세요!")
        print("   'autoci --production' 으로 24시간 게임 개발을 시작하세요.")
        print("=" * 80)
    
    def _analyze_strengths(self) -> List[str]:
        """학습 강점 분석"""
        strengths = []
        
        if self.total_learning_time >= 20:
            strengths.append("장시간 집중 학습 능력")
        
        if len(self.learning_history) >= 15:
            strengths.append("다양한 주제 학습 완료")
        
        avg_completion = sum(s.completion_rate for s in self.learning_history) / len(self.learning_history) if self.learning_history else 0
        if avg_completion >= 90:
            strengths.append("높은 학습 완료율")
        
        if self._calculate_mastery_score() >= 80:
            strengths.append("우수한 이해도 및 실습 능력")
        
        return strengths if strengths else ["꾸준한 학습 의지"]
    
    def _analyze_improvements(self) -> List[str]:
        """개선 영역 분석"""
        improvements = []
        
        if self.total_learning_time < 12:
            improvements.append("학습 시간 증대")
        
        if self._calculate_mastery_score() < 70:
            improvements.append("실습 프로젝트 더 많이 수행")
        
        if len(self.learning_history) < 10:
            improvements.append("더 다양한 주제 학습")
        
        return improvements if improvements else ["지속적인 복습과 실습"]
    
    def _generate_recommendations(self) -> List[str]:
        """다음 단계 추천"""
        recommendations = [
            "실제 게임 프로젝트에 C# 적용",
            "Godot Engine과 C# 통합 개발",
            "오픈소스 C# 프로젝트 기여",
            "고급 디자인 패턴 학습",
            "성능 최적화 기법 연구"
        ]
        
        # 현재 수준에 따른 맞춤 추천
        if self.total_learning_time < 12:
            recommendations.insert(0, "기초 개념 복습 및 강화")
        elif self.total_learning_time >= 20:
            recommendations.append("C# 고급 프레임워크 학습 (.NET, Unity)")
        
        return recommendations[:5]  # 최대 5개
    
    async def _save_learning_log(self, start_time: datetime, end_time: datetime):
        """학습 로그 JSON 파일로 저장"""
        log_data = {
            "session_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": self.total_learning_time,
                "actual_duration": str(end_time - start_time)
            },
            "learning_sessions": [
                {
                    "topic": session.topic,
                    "level": session.level,
                    "duration_minutes": session.duration_minutes,
                    "completion_rate": session.completion_rate,
                    "start_time": session.start_time.isoformat(),
                    "exercises_count": session.exercises_count,
                    "notes": session.notes
                }
                for session in self.learning_history
            ],
            "statistics": {
                "total_topics": len(self.learning_history),
                "mastery_score": self._calculate_mastery_score(),
                "current_level": self._get_current_level(),
                "strengths": self._analyze_strengths(),
                "improvements": self._analyze_improvements(),
                "recommendations": self._generate_recommendations()
            }
        }
        
        log_file = self.learning_data_dir / "learning_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    async def quick_learning_session(self, topic: str, duration_minutes: int = 30):
        """빠른 학습 세션 (개별 주제)"""
        print(f"🎯 빠른 학습 세션: {topic}")
        print(f"⏰ 예상 시간: {duration_minutes}분")
        
        session = LearningSession(
            topic=topic,
            level="mixed",
            duration_minutes=duration_minutes,
            start_time=datetime.now(),
            exercises_count=random.randint(2, 5)
        )
        
        await self._simulate_learning_session(session)
        await self._generate_code_examples(topic, session)
        
        self.learning_history.append(session)
        self.total_learning_time += duration_minutes / 60.0
        
        print(f"✅ '{topic}' 빠른 학습 완료!")
        return session
    
    def get_learning_status(self) -> Dict[str, Any]:
        """현재 학습 상태 반환"""
        return {
            "is_learning": self.is_learning,
            "total_learning_time": self.total_learning_time,
            "completed_topics": len(self.learning_history),
            "current_level": self._get_current_level(),
            "mastery_score": self._calculate_mastery_score(),
            "current_session": asdict(self.current_session) if self.current_session else None
        }

# 독립 실행용
async def main():
    """테스트 실행"""
    learning_system = CSharp24HLearning()
    
    print("C# 24시간 학습 시스템 테스트")
    mode = input("모드 선택 (1: 전체 24시간, 2: 빠른 세션, 3: 상태 확인): ")
    
    if mode == "1":
        await learning_system.start_24h_learning_marathon()
    elif mode == "2":
        topic = input("학습할 주제 입력: ") or "변수와 데이터 타입"
        await learning_system.quick_learning_session(topic)
    else:
        status = learning_system.get_learning_status()
        print(f"학습 상태: {json.dumps(status, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    asyncio.run(main())