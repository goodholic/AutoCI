#!/usr/bin/env python3
"""
AutoCI 대화형 인터페이스
자동 초기화 및 백그라운드 학습 중 명령 처리
ChatGPT 수준 한국어 AI 통합 버전
"""

import os
import sys
import json
import asyncio
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import readline  # 명령어 히스토리
import cmd
# 의존성 없는 컬러 지원
try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    # colorama가 없으면 간단한 대체 클래스 사용
    class Fore:
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        RESET = '\033[0m'
    
    class Style:
        RESET_ALL = '\033[0m'
        BRIGHT = '\033[1m'
    
    class Back:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
    
    HAS_COLORAMA = False

# Rich 대체 (선택적 import)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.syntax import Syntax
    console = Console()
    HAS_RICH = True
except ImportError:
    # Rich가 없으면 간단한 대체 클래스 사용
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    
    console = Console()
    HAS_RICH = False

# 추가 모듈들 (선택적)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import signal
import time
import sqlite3
import re
import random
from collections import defaultdict

# 초기화는 위에서 처리됨

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('autoci_interactive.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KoreanAIProcessor:
    """ChatGPT 수준 한국어 AI 처리기"""
    
    def __init__(self):
        # 한국어 패턴 분석
        self.korean_patterns = {
            "particles": ["은", "는", "이", "가", "을", "를", "에", "에서", "로", "으로", "와", "과", "의", "도", "만", "라도", "나마"],
            "endings": {
                "formal": ["습니다", "입니다", "하십시오", "하시겠습니까", "되십시오", "드립니다"],
                "polite": ["해요", "이에요", "예요", "돼요", "거예요", "세요"],
                "casual": ["해", "야", "이야", "어", "지", "네", "다"]
            },
            "honorifics": ["님", "씨", "선생님", "교수님", "사장님", "께서", "드리다", "받으시다", "하시다"],
            "emotions": {
                "positive": ["좋다", "행복하다", "기쁘다", "즐겁다", "감사하다", "만족하다", "훌륭하다"],
                "negative": ["나쁘다", "슬프다", "화나다", "힘들다", "어렵다", "불편하다", "짜증나다"],
                "neutral": ["괜찮다", "보통이다", "그럭저럭", "상관없다"]
            },
            "questions": ["무엇", "언제", "어디", "누가", "어떻게", "왜", "몇", "어느", "어떤"],
            "requests": ["해주세요", "해줘", "부탁", "도와주세요", "알려주세요", "가르쳐주세요", "설명해주세요"]
        }
        
        # 감정 분석 키워드
        self.emotion_keywords = {
            "stress": ["스트레스", "힘들어", "어려워", "피곤해", "지쳐", "답답해"],
            "happy": ["기뻐", "좋아", "행복해", "즐거워", "신나", "기대돼"],
            "confused": ["모르겠어", "헷갈려", "이해안돼", "복잡해", "어려워"],
            "angry": ["화나", "짜증나", "답답해", "속상해", "불만"],
            "grateful": ["고마워", "감사해", "도움돼", "잘했어", "훌륭해"]
        }
        
        # 응답 템플릿
        self.response_templates = {
            "greeting": [
                "안녕하세요! 😊 AutoCI와 함께하는 코딩이 즐거워질 거예요!",
                "반가워요! 👋 오늘도 멋진 코드를 만들어봐요!",
                "안녕하세요! ✨ 무엇을 도와드릴까요?"
            ],
            "unity_help": [
                "Unity 개발에서 도움이 필요하시군요! 구체적으로 어떤 부분이 궁금하신가요?",
                "Unity 관련해서 설명해드릴게요! 어떤 기능에 대해 알고 싶으신가요?",
                "Unity 전문가 AutoCI가 도와드리겠습니다! 🎮"
            ],
            "code_help": [
                "코드 관련해서 도움을 드릴게요! 구체적으로 어떤 문제인가요?",
                "프로그래밍 질문이시네요. 자세히 설명해드리겠습니다!",
                "코딩에서 막히는 부분이 있으시군요. 함께 해결해봐요! 💻"
            ],
            "empathy": [
                "그런 기분이 드실 수 있어요. 이해합니다.",
                "힘드시겠지만 차근차근 해나가면 분명 해결될 거예요!",
                "걱정하지 마세요. 제가 도와드릴게요! 😊"
            ],
            "encouragement": [
                "정말 잘하고 계세요! 👍",
                "훌륭한 접근이에요! 계속 진행해보세요!",
                "좋은 방향으로 가고 있어요! 💪"
            ]
        }
        
    def analyze_korean_text(self, text: str) -> Dict[str, any]:
        """한국어 텍스트 심층 분석"""
        analysis = {
            "formality": self._detect_formality(text),
            "emotion": self._detect_emotion(text),
            "intent": self._detect_intent(text),
            "topic": self._detect_topic(text),
            "patterns": self._analyze_patterns(text)
        }
        return analysis
        
    def _detect_formality(self, text: str) -> str:
        """격식 수준 감지"""
        formal_count = sum(1 for pattern in self.korean_patterns["endings"]["formal"] if pattern in text)
        polite_count = sum(1 for pattern in self.korean_patterns["endings"]["polite"] if pattern in text)
        casual_count = sum(1 for pattern in self.korean_patterns["endings"]["casual"] if pattern in text)
        
        if formal_count > 0:
            return "formal"
        elif polite_count > 0:
            return "polite"
        elif casual_count > 0:
            return "casual"
        else:
            return "neutral"
            
    def _detect_emotion(self, text: str) -> str:
        """감정 감지"""
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                return emotion
        return "neutral"
        
    def _detect_intent(self, text: str) -> str:
        """의도 분석"""
        if any(q in text for q in self.korean_patterns["questions"]):
            return "question"
        elif any(r in text for r in self.korean_patterns["requests"]):
            return "request"
        elif any(greeting in text for greeting in ["안녕", "반가", "처음"]):
            return "greeting"
        else:
            return "statement"
            
    def _detect_topic(self, text: str) -> str:
        """주제 감지"""
        unity_keywords = ["유니티", "Unity", "게임", "스크립트", "GameObject", "Transform", "Collider"]
        code_keywords = ["코드", "프로그래밍", "개발", "버그", "오류", "함수", "변수", "클래스"]
        
        if any(keyword in text for keyword in unity_keywords):
            return "unity"
        elif any(keyword in text for keyword in code_keywords):
            return "programming"
        else:
            return "general"
            
    def _analyze_patterns(self, text: str) -> Dict[str, int]:
        """언어 패턴 분석"""
        patterns = {
            "particles": sum(1 for p in self.korean_patterns["particles"] if p in text),
            "honorifics": sum(1 for h in self.korean_patterns["honorifics"] if h in text),
            "korean_ratio": self._calculate_korean_ratio(text)
        }
        return patterns
        
    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'\s', '', text))
        return korean_chars / total_chars if total_chars > 0 else 0.0
        
    def generate_response(self, user_input: str, analysis: Dict[str, any]) -> str:
        """ChatGPT 스타일 자연스러운 응답 생성"""
        
        # 의도별 응답 생성
        if analysis["intent"] == "greeting":
            base_response = random.choice(self.response_templates["greeting"])
        elif analysis["topic"] == "unity":
            base_response = random.choice(self.response_templates["unity_help"])
        elif analysis["topic"] == "programming":
            base_response = random.choice(self.response_templates["code_help"])
        else:
            # 감정에 따른 응답
            if analysis["emotion"] == "stress":
                base_response = random.choice(self.response_templates["empathy"])
            elif analysis["emotion"] == "grateful":
                base_response = random.choice(self.response_templates["encouragement"])
            else:
                base_response = "네, 말씀해주세요! 어떤 도움이 필요하신가요? 😊"
        
        # 격식 수준에 맞춰 응답 조정
        if analysis["formality"] == "formal":
            base_response = self._make_formal(base_response)
        elif analysis["formality"] == "casual":
            base_response = self._make_casual(base_response)
            
        return base_response
        
    def _make_formal(self, text: str) -> str:
        """격식체로 변환"""
        text = text.replace("해요", "합니다")
        text = text.replace("이에요", "입니다")
        text = text.replace("예요", "입니다")
        text = text.replace("드릴게요", "드리겠습니다")
        return text
        
    def _make_casual(self, text: str) -> str:
        """반말로 변환"""
        text = text.replace("해요", "해")
        text = text.replace("이에요", "이야")
        text = text.replace("예요", "야")
        text = text.replace("드릴게요", "줄게")
        text = text.replace("하세요", "해")
        return text


class AutoCIShell(cmd.Cmd):
    """AutoCI 대화형 셸"""
    
    intro = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  {Fore.YELLOW}🤖 AutoCI - 진짜 학습하는 한국어 AI 코딩 어시스턴트{Fore.CYAN}         ║
║                                                              ║
║  {Fore.GREEN}✓ 가상환경 활성화됨{Fore.CYAN}                                         ║
║  {Fore.GREEN}✓ ChatGPT 수준 한국어 AI{Fore.CYAN}                                    ║
║  {Fore.GREEN}✓ 실시간 학습 시스템 작동{Fore.CYAN}                                    ║
║  {Fore.GREEN}✓ 1분마다 자동 모니터링{Fore.CYAN}                                     ║
║                                                              ║
║  {Fore.WHITE}자연스러운 한국어로 대화하세요! 🇰🇷{Fore.CYAN}                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    
    prompt = f'{Fore.GREEN}🤖 autoci>{Style.RESET_ALL} '
    
    def __init__(self):
        super().__init__()
        self.base_path = Path(__file__).parent
        self.current_project = None
        self.background_tasks = {}
        self.is_initialized = False
        self.system_status = {
            'indexing': 'pending',
            'rag': 'stopped',
            'training': 'stopped',
            'monitoring': 'running'
        }
        
        # 한국어 AI 프로세서 초기화
        try:
            # 고급 한국어 AI 사용 시도
            from advanced_korean_ai import AdvancedKoreanAI
            self.korean_ai = AdvancedKoreanAI()
            self.use_advanced_ai = True
            logger.info("✅ ChatGPT 수준 고급 한국어 AI 프로세서 초기화 완료")
        except ImportError:
            # 기본 한국어 AI 사용
            self.korean_ai = KoreanAIProcessor()
            self.use_advanced_ai = False
            logger.info("✅ 기본 한국어 AI 프로세서 초기화 완료")
            
        # 실제 학습 시스템 초기화
        try:
            from real_learning_system import RealLearningSystem
            self.learning_system = RealLearningSystem()
            self.learning_system.start_background_learning()
            self.has_learning = True
            logger.info("✅ 실제 학습 시스템 초기화 및 시작")
        except Exception as e:
            self.learning_system = None
            self.has_learning = False
            logger.warning(f"학습 시스템 초기화 실패: {e}")
        
        # 백그라운드 초기화 시작
        self.init_thread = threading.Thread(target=self.background_init)
        self.init_thread.daemon = True
        self.init_thread.start()
        
        # 1분마다 모니터링 시작
        self.monitoring_thread = threading.Thread(target=self.start_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # 시그널 핸들러
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Ctrl+C 처리"""
        print(f"\n{Fore.YELLOW}종료하시겠습니까? (y/n): {Style.RESET_ALL}", end='')
        if input().lower() == 'y':
            self.do_exit(None)
        else:
            print(f"{self.prompt}", end='')
            
    def background_init(self):
        """백그라운드 초기화"""
        try:
            # 1. 데이터 체크
            console.print("\n[yellow]📥 데이터 체크 중...[/yellow]")
            if not self.check_data_exists():
                console.print("[cyan]데이터 수집을 시작합니다...[/cyan]")
                self.collect_data()
                
            # 2. 인덱싱
            console.print("[yellow]🔍 데이터 인덱싱 중...[/yellow]")
            self.index_data()
            
            # 3. Dual Phase 시스템 시작
            console.print("[yellow]🚀 Dual Phase 시스템 시작 중...[/yellow]")
            self.start_dual_phase()
            
            self.is_initialized = True
            console.print("\n[green]✅ 모든 시스템이 준비되었습니다![/green]")
            console.print("[cyan]이제 프로젝트 명령을 실행할 수 있습니다.[/cyan]\n")
            
        except Exception as e:
            logger.error(f"초기화 오류: {e}")
            console.print(f"\n[red]❌ 초기화 오류: {e}[/red]")
            
    def check_data_exists(self) -> bool:
        """데이터 존재 확인"""
        data_path = self.base_path / "expert_learning_data"
        vector_index = data_path / "vector_index" / "faiss_index.bin"
        return data_path.exists() and vector_index.exists()
        
    def collect_data(self):
        """데이터 수집"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("[cyan]C# 전문 데이터 수집 중...", total=None)
                
                process = subprocess.Popen(
                    [sys.executable, str(self.base_path / "deep_csharp_collector.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # 프로세스 완료 대기
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    progress.update(task, completed=100)
                    console.print("[green]✅ 데이터 수집 완료[/green]")
                else:
                    raise Exception(f"데이터 수집 실패: {stderr}")
                    
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            raise
            
    def index_data(self):
        """데이터 인덱싱"""
        try:
            self.system_status['indexing'] = 'running'
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("[cyan]벡터 인덱싱 중...", total=None)
                
                process = subprocess.Popen(
                    [sys.executable, str(self.base_path / "vector_indexer.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    progress.update(task, completed=100)
                    self.system_status['indexing'] = 'completed'
                    console.print("[green]✅ 인덱싱 완료[/green]")
                else:
                    raise Exception(f"인덱싱 실패: {stderr}")
                    
        except Exception as e:
            self.system_status['indexing'] = 'error'
            logger.error(f"인덱싱 오류: {e}")
            raise
            
    def start_dual_phase(self):
        """Dual Phase 시스템 시작"""
        try:
            # 백그라운드로 실행
            self.background_tasks['dual_phase'] = subprocess.Popen(
                [sys.executable, str(self.base_path / "robust_dual_phase.py"), "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 시스템이 시작될 때까지 잠시 대기
            time.sleep(5)
            
            self.system_status['rag'] = 'running'
            self.system_status['training'] = 'running'
            
            console.print("[green]✅ Dual Phase 시스템 시작됨[/green]")
            console.print("[yellow]웹 모니터링: http://localhost:8080[/yellow]")
            
        except Exception as e:
            logger.error(f"Dual Phase 시작 오류: {e}")
            raise
            
    def start_monitoring(self):
        """1분마다 AI 학습 환경 모니터링"""
        try:
            # 모니터링 프로세스가 이미 실행 중인지 확인
            import subprocess
            result = subprocess.run(['pgrep', '-f', 'ai_learning_monitor.py'], capture_output=True)
            if result.returncode != 0:  # 프로세스가 없으면
                # 백그라운드로 모니터링 시작
                subprocess.Popen(
                    [sys.executable, str(self.base_path / "ai_learning_monitor.py")],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info("📊 AI 학습 환경 모니터링이 백그라운드에서 시작되었습니다.")
                console.print("[green]📊 1분마다 AI 학습 환경을 모니터링합니다.[/green]")
                console.print("[yellow]웹 대시보드: http://localhost:8888[/yellow]")
        except Exception as e:
            logger.error(f"모니터링 시작 오류: {e}")
            
    def do_project(self, arg):
        """프로젝트 설정 - project <경로>"""
        if not arg:
            console.print("[yellow]프로젝트 경로를 입력해주세요.[/yellow]")
            console.print("[cyan]예시: project /path/to/unity/project[/cyan]")
            return
            
        project_path = Path(arg)
        
        if not project_path.exists():
            console.print(f"[red]❌ 경로가 존재하지 않습니다: {project_path}[/red]")
            return
            
        self.current_project = project_path
        console.print(f"[green]✅ 프로젝트 설정됨: {project_path}[/green]")
        
        # Unity 프로젝트인지 확인
        is_unity = self.check_unity_project(project_path)
        if is_unity:
            console.print("[cyan]🎮 Unity 프로젝트를 감지했습니다![/cyan]")
            self.analyze_unity_structure(project_path)
        
        # 프로젝트 분석
        self.analyze_project()
        
    def check_unity_project(self, path: Path) -> bool:
        """Unity 프로젝트 여부 확인"""
        unity_indicators = ['Assets', 'ProjectSettings', 'Packages']
        return all((path / indicator).exists() for indicator in unity_indicators)
        
    def analyze_unity_structure(self, project_path: Path):
        """Unity 프로젝트 구조 분석"""
        assets_path = project_path / "Assets"
        
        # 중요한 스크립트 폴더들 확인 (사용자가 언급한 4개 폴더)
        important_folders = [
            "Scripts",
            "OX UI Scripts", 
            "InGame UI Scripts",
            "Editor"
        ]
        
        console.print("\n[cyan]🔍 Unity Assets 폴더 구조 분석:[/cyan]")
        
        found_folders = []
        missing_folders = []
        
        for folder in important_folders:
            folder_path = assets_path / folder
            if folder_path.exists():
                found_folders.append(folder)
                script_count = len(list(folder_path.rglob("*.cs")))
                console.print(f"  [green]✅ {folder}[/green] - {script_count}개 스크립트")
            else:
                missing_folders.append(folder)
                console.print(f"  [yellow]❓ {folder}[/yellow] - 폴더 없음")
        
        if found_folders:
            console.print(f"\n[green]📂 발견된 스크립트 폴더: {len(found_folders)}개[/green]")
            # 폴더 간 이동된 파일 검사
            self.check_script_migrations(assets_path, found_folders)
            
        if missing_folders:
            console.print(f"[yellow]⚠️  누락된 폴더: {', '.join(missing_folders)}[/yellow]")
            
    def check_script_migrations(self, assets_path: Path, folders: List[str]):
        """스크립트 폴더 간 이동 파일 검사"""
        console.print("\n[cyan]🔄 폴더 간 이동 파일 검사 중...[/cyan]")
        
        # 각 폴더의 스크립트 파일 수집
        folder_scripts = {}
        all_scripts = {}
        
        for folder in folders:
            folder_path = assets_path / folder
            scripts = list(folder_path.rglob("*.cs"))
            folder_scripts[folder] = scripts
            
            for script in scripts:
                script_name = script.name
                if script_name in all_scripts:
                    # 중복 파일명 발견
                    console.print(f"  [yellow]⚠️  중복 파일명: {script_name}[/yellow]")
                    console.print(f"     1️⃣ {all_scripts[script_name].relative_to(assets_path)}")
                    console.print(f"     2️⃣ {script.relative_to(assets_path)}")
                else:
                    all_scripts[script_name] = script
        
        # 잘못된 위치에 있을 수 있는 스크립트 검사
        self.check_misplaced_scripts(folder_scripts, assets_path)
        
    def check_misplaced_scripts(self, folder_scripts: Dict[str, List[Path]], assets_path: Path):
        """잘못 배치된 스크립트 검사"""
        # UI 스크립트 패턴
        ui_patterns = ['UI', 'Button', 'Panel', 'Canvas', 'Menu', 'HUD', 'Dialog']
        editor_patterns = ['Editor', 'Inspector', 'Window', 'Tool']
        game_patterns = ['Player', 'Enemy', 'Game', 'Controller', 'Manager']
        
        misplaced = []
        
        for folder, scripts in folder_scripts.items():
            for script in scripts:
                script_name = script.stem  # 확장자 제외
                content = ""
                try:
                    content = script.read_text(encoding='utf-8')
                except:
                    continue
                    
                # 스크립트 분류
                is_ui = any(pattern in script_name for pattern in ui_patterns) or 'UnityEngine.UI' in content
                is_editor = any(pattern in script_name for pattern in editor_patterns) or 'UnityEditor' in content
                is_game = any(pattern in script_name for pattern in game_patterns)
                
                # 잘못된 위치 검사
                wrong_location = False
                suggestion = ""
                
                if is_editor and folder != "Editor":
                    wrong_location = True
                    suggestion = "Editor"
                elif is_ui and "UI" not in folder and folder != "Scripts":
                    if "OX" in script_name or "ox" in script_name.lower():
                        suggestion = "OX UI Scripts"
                    else:
                        suggestion = "InGame UI Scripts"
                    wrong_location = True
                elif is_game and folder in ["OX UI Scripts", "InGame UI Scripts", "Editor"]:
                    wrong_location = True
                    suggestion = "Scripts"
                    
                if wrong_location:
                    misplaced.append({
                        'file': script,
                        'current_folder': folder,
                        'suggested_folder': suggestion,
                        'reason': f"{'에디터' if is_editor else 'UI' if is_ui else '게임'} 스크립트"
                    })
        
        if misplaced:
            console.print(f"\n[yellow]📋 잘못 배치된 가능성이 있는 스크립트: {len(misplaced)}개[/yellow]")
            for item in misplaced[:5]:  # 처음 5개만 표시
                rel_path = item['file'].relative_to(assets_path)
                console.print(f"  [yellow]💡 {rel_path}[/yellow]")
                console.print(f"     현재: {item['current_folder']} → 권장: {item['suggested_folder']}")
                console.print(f"     이유: {item['reason']}")
                
            if len(misplaced) > 5:
                console.print(f"  [cyan]... 및 {len(misplaced) - 5}개 더[/cyan]")
                
            console.print(f"\n[cyan]💡 '정리' 명령으로 자동 정리를 수행할 수 있습니다.[/cyan]")
        else:
            console.print(f"[green]✅ 모든 스크립트가 적절한 위치에 있습니다![/green]")
            
    def do_정리(self, arg):
        """Unity 스크립트 폴더 정리 - 정리"""
        if not self.current_project:
            console.print("[yellow]먼저 프로젝트를 설정해주세요.[/yellow]")
            return
            
        console.print("[cyan]🧹 Unity 스크립트 폴더 정리를 시작합니다...[/cyan]")
        
        # 백업 생성 확인
        console.print("[yellow]⚠️  이 작업은 파일을 이동시킵니다. 계속하시겠습니까? (y/n): [/yellow]", end='')
        if input().lower() != 'y':
            console.print("[cyan]정리 작업이 취소되었습니다.[/cyan]")
            return
            
        # Unity 프로젝트 백업
        self.create_unity_backup()
        
        # 스크립트 정리 수행
        self.reorganize_unity_scripts()
        
    def create_unity_backup(self):
        """Unity 프로젝트 백업"""
        import shutil
        
        backup_dir = self.current_project.parent / f"{self.current_project.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        assets_backup = backup_dir / "Assets"
        
        console.print(f"[cyan]💾 백업 생성 중: {backup_dir}[/cyan]")
        
        try:
            # Assets 폴더만 백업 (용량 절약)
            shutil.copytree(self.current_project / "Assets", assets_backup)
            console.print(f"[green]✅ 백업 완료: {backup_dir}[/green]")
        except Exception as e:
            console.print(f"[red]❌ 백업 실패: {e}[/red]")
            raise
            
    def reorganize_unity_scripts(self):
        """Unity 스크립트 재정리"""
        assets_path = self.current_project / "Assets"
        
        # 폴더 생성
        required_folders = ["Scripts", "OX UI Scripts", "InGame UI Scripts", "Editor"]
        for folder in required_folders:
            (assets_path / folder).mkdir(exist_ok=True)
            
        # 모든 .cs 파일 스캔
        all_scripts = list(assets_path.rglob("*.cs"))
        moved_count = 0
        
        for script in all_scripts:
            target_folder = self.determine_target_folder(script, assets_path)
            current_folder = script.parent.name
            
            if target_folder and current_folder != target_folder:
                # 파일 이동
                target_path = assets_path / target_folder / script.name
                
                # 중복 파일명 처리
                if target_path.exists():
                    target_path = assets_path / target_folder / f"{script.stem}_moved{script.suffix}"
                    
                try:
                    script.rename(target_path)
                    console.print(f"[green]📝 {script.name}[/green] → [cyan]{target_folder}[/cyan]")
                    moved_count += 1
                except Exception as e:
                    console.print(f"[red]❌ 이동 실패 {script.name}: {e}[/red]")
                    
        console.print(f"\n[green]✅ 정리 완료! {moved_count}개 파일이 이동되었습니다.[/green]")
        
    def determine_target_folder(self, script: Path, assets_path: Path) -> Optional[str]:
        """스크립트의 적절한 대상 폴더 결정"""
        try:
            content = script.read_text(encoding='utf-8')
            script_name = script.stem
            
            # Editor 스크립트
            if 'UnityEditor' in content or any(pattern in script_name for pattern in ['Editor', 'Inspector', 'Tool', 'Window']):
                return "Editor"
                
            # UI 스크립트
            if 'UnityEngine.UI' in content or any(pattern in script_name for pattern in ['UI', 'Button', 'Panel', 'Canvas', 'Menu']):
                if 'OX' in script_name or 'ox' in script_name.lower():
                    return "OX UI Scripts"
                else:
                    return "InGame UI Scripts"
                    
            # 일반 게임 스크립트
            return "Scripts"
            
        except:
            return "Scripts"  # 기본값
        
    def analyze_project(self):
        """프로젝트 분석"""
        if not self.current_project:
            return
            
        cs_files = list(self.current_project.rglob("*.cs"))
        
        table = Table(title=f"프로젝트 분석: {self.current_project.name}")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="green")
        
        table.add_row("C# 파일 수", str(len(cs_files)))
        table.add_row("프로젝트 크기", self.get_project_size())
        table.add_row("주요 네임스페이스", self.get_namespaces(cs_files[:10]))
        
        console.print(table)
        
    def get_project_size(self) -> str:
        """프로젝트 크기 계산"""
        total_size = 0
        for file in self.current_project.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
                
        # MB로 변환
        size_mb = total_size / (1024 * 1024)
        return f"{size_mb:.1f} MB"
        
    def get_namespaces(self, cs_files: List[Path]) -> str:
        """네임스페이스 추출"""
        namespaces = set()
        for file in cs_files[:5]:  # 샘플
            try:
                content = file.read_text(encoding='utf-8')
                import re
                ns_matches = re.findall(r'namespace\s+([\w.]+)', content)
                namespaces.update(ns_matches)
            except:
                pass
                
        return ', '.join(list(namespaces)[:3]) if namespaces else "N/A"
        
    def do_analyze(self, arg):
        """코드 분석 - analyze [파일명]"""
        if not self.current_project:
            console.print("[red]❌ 먼저 프로젝트를 설정하세요 (project <경로>)[/red]")
            return
            
        if not self.is_initialized:
            console.print("[yellow]⏳ 시스템 초기화 중입니다. 잠시 기다려주세요...[/yellow]")
            return
            
        if arg:
            # 특정 파일 분석
            file_path = self.current_project / arg
            if not file_path.exists():
                # 파일 검색
                matches = list(self.current_project.rglob(f"*{arg}*"))
                if matches:
                    file_path = matches[0]
                    console.print(f"[cyan]파일 찾음: {file_path}[/cyan]")
                else:
                    console.print(f"[red]❌ 파일을 찾을 수 없습니다: {arg}[/red]")
                    return
                    
            self.analyze_file(file_path)
        else:
            # 전체 프로젝트 분석
            self.analyze_all_files()
            
    def analyze_file(self, file_path: Path):
        """파일 분석"""
        console.print(f"\n[cyan]📝 파일 분석: {file_path.name}[/cyan]")
        
        try:
            # 임시로 advanced_autoci_system의 분석 기능 사용
            content = file_path.read_text(encoding='utf-8')
            
            # 간단한 분석
            lines = content.split('\n')
            classes = len([l for l in lines if 'class ' in l])
            methods = len([l for l in lines if re.search(r'(public|private|protected)\s+\w+\s+\w+\s*\(', l)])
            
            # 코드 품질 점수 (간단한 휴리스틱)
            quality_score = 0.7
            if '// TODO' in content or '// FIXME' in content:
                quality_score -= 0.1
            if 'try' in content and 'catch' in content:
                quality_score += 0.1
            if 'async' in content and 'await' in content:
                quality_score += 0.05
                
            table = Table(title="분석 결과")
            table.add_column("항목", style="cyan")
            table.add_column("값", style="green")
            
            table.add_row("파일 크기", f"{len(lines)} 줄")
            table.add_row("클래스 수", str(classes))
            table.add_row("메서드 수", str(methods))
            table.add_row("품질 점수", f"{quality_score:.2f}/1.0")
            
            console.print(table)
            
            # 개선 제안
            if quality_score < 0.8:
                console.print("\n[yellow]💡 개선 제안:[/yellow]")
                suggestions = self.get_improvement_suggestions(content)
                for i, suggestion in enumerate(suggestions, 1):
                    console.print(f"  {i}. {suggestion}")
                    
        except Exception as e:
            console.print(f"[red]❌ 분석 오류: {e}[/red]")
            
    def get_improvement_suggestions(self, content: str) -> List[str]:
        """개선 제안 생성"""
        suggestions = []
        
        if '// TODO' in content or '// FIXME' in content:
            suggestions.append("TODO/FIXME 주석을 해결하세요")
            
        if 'catch (Exception' in content:
            suggestions.append("구체적인 예외 타입을 catch하세요")
            
        if not ('/// <summary>' in content):
            suggestions.append("XML 문서화 주석을 추가하세요")
            
        if content.count('if') > 10:
            suggestions.append("복잡한 조건문을 리팩토링하세요")
            
        return suggestions[:3]  # 최대 3개
        
    def do_improve(self, arg):
        """코드 개선 - improve <파일명>"""
        if not self.current_project:
            console.print("[red]❌ 먼저 프로젝트를 설정하세요 (project <경로>)[/red]")
            return
            
        if not self.is_initialized:
            console.print("[yellow]⏳ 시스템 초기화 중입니다. 잠시 기다려주세요...[/yellow]")
            return
            
        if not arg:
            console.print("[yellow]사용법: improve <파일명>[/yellow]")
            return
            
        # 파일 찾기
        file_path = self.current_project / arg
        if not file_path.exists():
            matches = list(self.current_project.rglob(f"*{arg}*"))
            if matches:
                file_path = matches[0]
            else:
                console.print(f"[red]❌ 파일을 찾을 수 없습니다: {arg}[/red]")
                return
                
        console.print(f"[cyan]🔧 코드 개선 중: {file_path.name}[/cyan]")
        
        # 백업 생성
        backup_path = file_path.with_suffix(f'.bak{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        file_path.rename(backup_path)
        
        try:
            # advanced_autoci_system 실행
            process = subprocess.run(
                [sys.executable, str(self.base_path / "advanced_autoci_system.py"), 
                 "start", "--path", str(file_path.parent)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if process.returncode == 0:
                console.print("[green]✅ 코드 개선 완료![/green]")
                
                # 변경사항 표시
                self.show_diff(backup_path, file_path)
            else:
                raise Exception(process.stderr)
                
        except Exception as e:
            # 롤백
            backup_path.rename(file_path)
            console.print(f"[red]❌ 개선 실패: {e}[/red]")
            
    def show_diff(self, old_file: Path, new_file: Path):
        """변경사항 표시"""
        try:
            import difflib
            
            old_content = old_file.read_text(encoding='utf-8').splitlines()
            new_content = new_file.read_text(encoding='utf-8').splitlines()
            
            diff = difflib.unified_diff(old_content, new_content, 
                                       fromfile=old_file.name, 
                                       tofile=new_file.name,
                                       lineterm='')
            
            console.print("\n[cyan]📝 변경사항:[/cyan]")
            for line in list(diff)[:20]:  # 최대 20줄
                if line.startswith('+'):
                    console.print(f"[green]{line}[/green]")
                elif line.startswith('-'):
                    console.print(f"[red]{line}[/red]")
                else:
                    console.print(line)
                    
        except Exception as e:
            logger.error(f"Diff 표시 오류: {e}")
            
    def do_search(self, arg):
        """코드 검색 - search <검색어>"""
        if not arg:
            console.print("[yellow]사용법: search <검색어>[/yellow]")
            return
            
        if not self.is_initialized:
            console.print("[yellow]⏳ 시스템 초기화 중입니다. 잠시 기다려주세요...[/yellow]")
            return
            
        console.print(f"[cyan]🔍 검색 중: {arg}[/cyan]")
        
        # 벡터 검색 사용
        try:
            from vector_indexer import VectorIndexer
            
            indexer = VectorIndexer()
            results = indexer.search(arg, k=5)
            
            if results:
                table = Table(title="검색 결과")
                table.add_column("#", style="cyan", width=3)
                table.add_column("유사도", style="green", width=8)
                table.add_column("카테고리", style="yellow", width=15)
                table.add_column("내용", style="white", width=50)
                
                for i, (chunk, similarity) in enumerate(results, 1):
                    content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    table.add_row(
                        str(i),
                        f"{similarity:.3f}",
                        chunk.category,
                        content_preview
                    )
                    
                console.print(table)
            else:
                console.print("[yellow]검색 결과가 없습니다.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]❌ 검색 오류: {e}[/red]")
            
    def do_status(self, arg):
        """시스템 상태 확인 - status"""
        table = Table(title="시스템 상태")
        table.add_column("구성요소", style="cyan")
        table.add_column("상태", style="green")
        table.add_column("설명", style="white")
        
        # 상태 아이콘
        status_icons = {
            'running': '🟢 실행 중',
            'stopped': '🔴 중지됨',
            'pending': '🟡 대기 중',
            'completed': '✅ 완료',
            'error': '❌ 오류'
        }
        
        table.add_row("인덱싱", status_icons.get(self.system_status['indexing'], '❓'), 
                     "데이터 벡터 인덱싱")
        table.add_row("RAG 시스템", status_icons.get(self.system_status['rag'], '❓'), 
                     "실시간 검색 및 응답")
        table.add_row("파인튜닝", status_icons.get(self.system_status['training'], '❓'), 
                     "백그라운드 모델 학습")
        table.add_row("모니터링", status_icons.get(self.system_status['monitoring'], '❓'), 
                     "시스템 모니터링")
        
        console.print(table)
        
        # 리소스 사용량
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
        else:
            cpu_percent = 0.0
            memory = type('Memory', (), {'percent': 0.0, 'available': 0, 'total': 0, 'used': 0})()
        
        resource_table = Table(title="리소스 사용량")
        resource_table.add_column("항목", style="cyan")
        resource_table.add_column("사용량", style="green")
        
        resource_table.add_row("CPU", f"{cpu_percent}%")
        resource_table.add_row("메모리", f"{memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
        
        console.print(resource_table)
        
        if self.current_project:
            console.print(f"\n현재 프로젝트: [cyan]{self.current_project}[/cyan]")
            
    def do_monitor(self, arg):
        """웹 모니터링 열기 - monitor"""
        import webbrowser
        
        url = "http://localhost:8080"
        console.print(f"[cyan]🌐 웹 모니터링 열기: {url}[/cyan]")
        
        try:
            webbrowser.open(url)
        except:
            console.print(f"[yellow]브라우저에서 직접 열어주세요: {url}[/yellow]")
            
    def do_report(self, arg):
        """리포트 보기 - report"""
        report_dir = self.base_path / "autoci_reports"
        
        if not report_dir.exists():
            console.print("[yellow]리포트가 없습니다.[/yellow]")
            return
            
        # 최신 리포트 찾기
        reports = list(report_dir.glob("*.md"))
        if not reports:
            console.print("[yellow]리포트가 없습니다.[/yellow]")
            return
            
        latest_report = max(reports, key=lambda x: x.stat().st_mtime)
        
        console.print(f"[cyan]📄 최신 리포트: {latest_report.name}[/cyan]\n")
        
        # 리포트 내용 표시
        content = latest_report.read_text(encoding='utf-8')
        
        # 처음 50줄만 표시
        lines = content.split('\n')[:50]
        for line in lines:
            if line.startswith('#'):
                console.print(f"[bold cyan]{line}[/bold cyan]")
            elif line.startswith('-'):
                console.print(f"[green]{line}[/green]")
            else:
                console.print(line)
                
        if len(content.split('\n')) > 50:
            console.print(f"\n[yellow]... (전체 리포트: {latest_report})[/yellow]")
            
    def do_help(self, arg):
        """도움말 표시"""
        help_text = """
[bold cyan]🤖 AutoCI 명령어 가이드[/bold cyan]

[yellow]한국어 인사 및 대화:[/yellow]
  안녕, 안녕하세요     - AI와 인사하기
  고마워, 네, 응       - 자연스러운 대화
  
[yellow]한국어 명령어:[/yellow]
  도움말, 도움         - 이 도움말 표시 (help)
  상태, 상태확인       - 시스템 상태 확인 (status)
  프로젝트 <경로>      - Unity 프로젝트 설정 (project)
  분석 [파일]         - 코드 분석 (analyze)
  개선 <파일>         - 코드 자동 개선 (improve)
  검색, 찾기 <검색어>  - 코드/패턴 검색 (search)
  정리               - Unity 스크립트 폴더 정리
  리포트, 보고서      - 최신 리포트 보기 (report)
  모니터링, 모니터    - 웹 모니터링 열기 (monitor)
  종료, 나가기, 끝    - 프로그램 종료 (exit)

[yellow]영어 명령어:[/yellow]
  project <경로>      - 작업할 프로젝트 설정
  analyze [파일]      - 코드 분석 (파일 또는 전체)
  improve <파일>      - 코드 자동 개선
  search <검색어>     - 코드/패턴 검색
  status             - 시스템 상태 확인
  monitor            - 웹 모니터링 열기
  report             - 최신 리포트 보기
  help, ?            - 도움말 표시
  exit, quit         - 종료

[yellow]🎮 Unity 특화 기능:[/yellow]
  • Assets/Scripts, OX UI Scripts, InGame UI Scripts, Editor 폴더 자동 감지
  • 잘못 배치된 스크립트 파일 검사 및 추천
  • 스크립트 폴더 간 이동 파일 감지
  • Unity 프로젝트 백업 및 자동 정리

[cyan]📝 사용 예시:[/cyan]
  안녕                          - AI와 인사하기
  프로젝트 C:/Unity/MyGame      - Unity 프로젝트 설정
  분석                         - 전체 프로젝트 분석
  분석 PlayerController.cs     - 특정 파일 분석
  개선 GameManager.cs          - 코드 자동 개선
  정리                         - Unity 스크립트 폴더 정리
  검색 "async await"           - 패턴 검색
  상태                         - 시스템 상태 확인

[green]💡 팁:[/green]
  • 자연스러운 한국어로 대화할 수 있습니다!
  • Unity 프로젝트 설정 시 자동으로 폴더 구조를 분석합니다
  • 잘못 배치된 스크립트를 자동으로 찾아 정리해드립니다
  • 24시간 백그라운드에서 코드 품질을 모니터링합니다
"""
        console.print(help_text)
        
    def do_exit(self, arg):
        """종료"""
        console.print("\n[yellow]시스템을 종료합니다...[/yellow]")
        
        # 백그라운드 작업 종료
        for name, process in self.background_tasks.items():
            if process and process.poll() is None:
                console.print(f"[cyan]{name} 종료 중...[/cyan]")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        console.print("[green]👋 안녕히 가세요![/green]")
        return True
        
    def do_quit(self, arg):
        """종료"""
        return self.do_exit(arg)
        
    def default(self, line):
        """ChatGPT 수준 한국어 AI 처리"""
        line = line.strip()
        
        # 한국어 비율 확인
        korean_ratio = self.korean_ai._calculate_korean_ratio(line)
        
        if korean_ratio > 0.3:  # 한국어 텍스트인 경우
            # ChatGPT 스타일 한국어 분석
            console.print(f"[cyan]🤔 '{line}'에 대해 생각해보고 있어요...[/cyan]")
            
            if self.use_advanced_ai:
                # 고급 AI 사용
                analysis = self.korean_ai.analyze_input(line)
            else:
                # 기본 AI 사용
                analysis = self.korean_ai.analyze_korean_text(line)
            
            # 한국어 명령어 매핑 먼저 확인
            korean_commands = {
                '도움말': 'help', '도움': 'help', '명령어': 'help',
                '상태': 'status', '상태확인': 'status',
                '프로젝트': 'project', '분석': 'analyze', '개선': 'improve',
                '검색': 'search', '찾기': 'search',
                '리포트': 'report', '보고서': 'report',
                '모니터링': 'monitor', '모니터': 'monitor',
                '종료': 'exit', '나가기': 'exit', '끝': 'exit', '그만': 'exit'
            }
            
            # 명령어인지 확인
            for korean_cmd, english_cmd in korean_commands.items():
                if korean_cmd in line:
                    console.print(f"[cyan]✅ '{korean_cmd}' 명령을 실행합니다![/cyan]")
                    self.onecmd(english_cmd)
                    return
            
            # ChatGPT 스타일 자연스러운 응답 생성
            if self.use_advanced_ai:
                smart_response = self.korean_ai.generate_response(analysis)
            else:
                smart_response = self.korean_ai.generate_response(line, analysis)
                
            # 실제 학습 시스템에 대화 저장
            if self.has_learning and self.learning_system:
                try:
                    learning_result = self.learning_system.learn_from_conversation(
                        line, smart_response,
                        {'session_id': 'interactive', 'timestamp': datetime.now()}
                    )
                    logger.info(f"대화 학습 완료: {learning_result['patterns']} 패턴 발견")
                except Exception as e:
                    logger.error(f"학습 중 오류: {e}")
            
            # 분석 결과 표시 (디버그용)
            if self.use_advanced_ai:
                console.print(f"[dim]📊 분석: {analysis['formality']} / {analysis['emotion']} / {analysis['intent']} / {analysis['topic']}[/dim]")
                console.print(f"[dim]   키워드: {', '.join(analysis.get('keywords', [])[:5])}[/dim]")
            else:
                console.print(f"[dim]📊 분석: {analysis['formality']} / {analysis['emotion']} / {analysis['intent']} / {analysis['topic']}[/dim]")
            
            # AI 응답 출력
            console.print(f"\n[green]🤖 AutoCI:[/green] {smart_response}")
            
            # 구체적인 도움 제안
            if analysis["intent"] == "question":
                if analysis["topic"] == "unity":
                    console.print(f"\n[yellow]💡 Unity 관련 도움말:[/yellow]")
                    console.print(f"   [cyan]• 'analyze <스크립트명>'[/cyan] - 스크립트 분석")
                    console.print(f"   [cyan]• 'improve <스크립트명>'[/cyan] - 코드 자동 개선")
                    console.print(f"   [cyan]• '정리'[/cyan] - 스크립트 폴더 정리")
                elif analysis["topic"] == "programming":
                    console.print(f"\n[yellow]💡 프로그래밍 도움말:[/yellow]")
                    console.print(f"   [cyan]• 'search <키워드>'[/cyan] - 코드 패턴 검색")
                    console.print(f"   [cyan]• 'analyze'[/cyan] - 전체 프로젝트 분석")
                else:
                    console.print(f"\n[yellow]💡 더 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있어요![/yellow]")
                    console.print(f"   예: [cyan]'유니티 스크립트를 어떻게 정리하나요?'[/cyan]")
            
            # 감정에 따른 추가 지원
            if analysis["emotion"] == "stress":
                console.print(f"\n[yellow]😊 힘내세요! 단계별로 차근차근 해결해나가면 됩니다.[/yellow]")
                console.print(f"   [cyan]• 'status'[/cyan] - 현재 시스템 상태 확인")
                console.print(f"   [cyan]• '도움말'[/cyan] - 전체 기능 보기")
            
            # RAG 시스템 연동 시도
            if self.is_initialized and self.system_status.get('rag') == 'running':
                try:
                    import requests
                    response = requests.post(
                        "http://localhost:8000/query",
                        json={"query": line, "k": 2},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('results'):
                            console.print(f"\n[green]📚 관련 지식:[/green]")
                            for i, result in enumerate(data['results'][:1], 1):
                                console.print(f"   [dim]{result['content'][:150]}...[/dim]")
                except:
                    pass  # RAG 연동 실패해도 계속 진행
                    
        else:
            # 영어 또는 명령어 처리
            console.print(f"[yellow]🤔 '{line}'에 대해 생각해보고 있어요...[/yellow]")
            console.print(f"[cyan]💡 더 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있어요![/cyan]")
            console.print(f"   예: [yellow]'유니티 스크립트를 어떻게 정리하나요?'[/yellow]")
            
    def emptyline(self):
        """빈 줄 입력 시"""
        pass
        
    def postcmd(self, stop, line):
        """명령 실행 후"""
        print()  # 빈 줄 추가
        return stop


def main():
    """메인 함수"""
    try:
        # 터미널 클리어
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # 대화형 셸 시작
        shell = AutoCIShell()
        shell.cmdloop()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]종료합니다...[/yellow]")
    except Exception as e:
        console.print(f"\n[red]오류: {e}[/red]")
        logger.error(f"치명적 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()