#!/usr/bin/env python3
"""
AutoCI Production System - 상용화 수준의 24시간 AI 게임 개발 시스템
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import psutil
import signal
import json

# 향상된 모듈 import
try:
    from modules.enhanced_logging import init_logging, get_logger, LogContextManager
    from modules.enhanced_error_handler import get_enhanced_error_handler, with_error_handling, ErrorSeverity
    from modules.enhanced_monitoring import get_enhanced_monitor, with_metrics, MetricType
    from modules.enhanced_godot_controller import EnhancedGodotController
    from modules.csharp_learning_agent import CSharpLearningAgent
    from modules.ai_model_integration import get_ai_integration
except ImportError as e:
    print(f"필수 모듈 로드 실패: {e}")
    print("설치를 확인해주세요.")
    sys.exit(1)

class ProductionAutoCI:
    """상용화 수준의 AutoCI 시스템"""
    
    def __init__(self):
        # 로깅 초기화
        init_logging()
        self.logger = get_logger("AutoCI")
        self.logger.info("🚀 AutoCI Production System 시작")
        
        # 시스템 컴포넌트
        self.error_handler = get_enhanced_error_handler()
        self.monitor = get_enhanced_monitor()
        
        # 프로젝트 루트
        self.project_root = Path(__file__).parent
        self.setup_directories()
        
        # AI 모델 초기화
        self.ai_model_name = self.select_ai_model()
        self.logger.info(f"🤖 AI 모델 선택: {self.ai_model_name}")
        self.ai_integration = get_ai_integration()
        
        # Godot 컨트롤러
        self.godot_controller = EnhancedGodotController()
        
        # C# 학습 에이전트
        self.csharp_agent = CSharpLearningAgent()
        
        # 프로젝트 관리
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.current_project: Optional[str] = None
        
        # 시스템 상태
        self.running = True
        self.start_time = datetime.now()
        self.shutdown_event = asyncio.Event()
        
        # 통계
        self.stats = {
            "games_created": 0,
            "features_added": 0,
            "bugs_fixed": 0,
            "csharp_concepts_learned": 0,
            "optimization_runs": 0,
            "errors_recovered": 0
        }
        
        # 설정 로드
        self.config = self.load_config()
        
        # 시그널 핸들러
        self._setup_signal_handlers()
    
    def setup_directories(self):
        """디렉토리 구조 설정"""
        directories = [
            "game_projects",
            "csharp_learning",
            "logs",
            "data",
            "config",
            "backups",
            "exports"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """설정 로드"""
        config_path = self.project_root / "config" / "production.json"
        
        default_config = {
            "game_creation_interval": {"min": 7200, "max": 14400},
            "feature_addition_interval": 1800,
            "bug_check_interval": 900,
            "optimization_interval": 3600,
            "backup_interval": 86400,
            "max_concurrent_projects": 3,
            "auto_export": True,
            "enable_metrics": True,
            "enable_alerts": True
        }
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"설정 로드 실패, 기본값 사용: {e}")
        
        return default_config
    
    def select_ai_model(self) -> str:
        """메모리 기반 AI 모델 선택"""
        try:
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            
            if available_memory >= 32:
                return "Qwen2.5-Coder-32B"
            elif available_memory >= 16:
                return "CodeLlama-13B"
            else:
                return "Llama-3.1-8B"
        except:
            return "Llama-3.1-8B"
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        def signal_handler(signum, frame):
            self.logger.info(f"시그널 수신: {signum}")
            self.shutdown_event.set()
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @with_error_handling(component="main", severity=ErrorSeverity.HIGH)
    @with_metrics("autoci.startup")
    async def start(self):
        """시스템 시작"""
        with LogContextManager(self.logger, "AutoCI 시작"):
            # 헬스 체크
            health = await self.monitor.health_check()
            if health["status"] != "healthy":
                self.logger.warning(f"시스템 상태 경고: {health}")
            
            # 백그라운드 작업 시작
            tasks = [
                self.game_creation_loop(),
                self.feature_addition_loop(),
                self.bug_detection_loop(),
                self.optimization_loop(),
                self.learning_loop(),
                self.backup_loop(),
                self.terminal_interface()
            ]
            
            # 메트릭 수집 시작
            await self.monitor.record_metric("app.startup", 1, MetricType.COUNTER)
            
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                self.logger.info("작업 취소됨")
            finally:
                await self.shutdown()
    
    @with_error_handling(component="game_creation", severity=ErrorSeverity.MEDIUM)
    @with_metrics("game.creation")
    async def game_creation_loop(self):
        """게임 생성 루프"""
        while self.running:
            try:
                # 동시 프로젝트 수 체크
                active_projects = len([p for p in self.projects.values() 
                                     if p.get("status") == "active"])
                
                if active_projects >= self.config["max_concurrent_projects"]:
                    await asyncio.sleep(300)  # 5분 대기
                    continue
                
                # 게임 타입 선택
                game_types = ["platformer", "racing", "puzzle", "rpg"]
                game_type = game_types[self.stats["games_created"] % len(game_types)]
                
                with LogContextManager(self.logger, f"{game_type} 게임 생성",
                                     game_type=game_type):
                    # 프로젝트 생성
                    project_name = f"{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    project_path = self.project_root / "game_projects" / project_name
                    
                    # Godot 프로젝트 생성
                    success = await self.godot_controller.create_project(
                        project_name, str(project_path), game_type
                    )
                    
                    if success:
                        # 프로젝트 정보 저장
                        self.projects[project_name] = {
                            "type": game_type,
                            "path": str(project_path),
                            "created": datetime.now(),
                            "status": "active",
                            "features": [],
                            "bugs_fixed": 0,
                            "optimizations": 0
                        }
                        
                        self.current_project = project_name
                        self.stats["games_created"] += 1
                        
                        # 메트릭 기록
                        await self.monitor.record_metric(
                            "business.games.created", 1, MetricType.COUNTER
                        )
                        
                        # AI 코드 생성
                        await self.generate_initial_code(project_name, game_type)
                
                # 대기 시간 계산
                min_interval = self.config["game_creation_interval"]["min"]
                max_interval = self.config["game_creation_interval"]["max"]
                wait_time = min_interval + (self.stats["games_created"] % 3) * \
                           ((max_interval - min_interval) // 3)
                
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                await self.error_handler.handle_error(e, {
                    "component": "game_creation",
                    "task": "create_game"
                })
                await asyncio.sleep(300)
    
    @with_error_handling(component="feature_addition", severity=ErrorSeverity.LOW)
    @with_metrics("feature.addition")
    async def feature_addition_loop(self):
        """기능 추가 루프"""
        await asyncio.sleep(self.config["feature_addition_interval"])
        
        while self.running:
            try:
                if self.current_project and self.current_project in self.projects:
                    project = self.projects[self.current_project]
                    
                    # 게임 타입별 기능 선택
                    features = self.get_features_for_game_type(project["type"])
                    available_features = [f for f in features 
                                        if f not in project["features"]]
                    
                    if available_features:
                        feature = available_features[0]
                        
                        with LogContextManager(self.logger, f"기능 추가: {feature}",
                                             feature=feature, project=self.current_project):
                            # AI로 기능 구현
                            success = await self.implement_feature(feature, project)
                            
                            if success:
                                project["features"].append(feature)
                                self.stats["features_added"] += 1
                                
                                await self.monitor.record_metric(
                                    "business.features.added", 1, MetricType.COUNTER
                                )
                
                await asyncio.sleep(self.config["feature_addition_interval"])
                
            except Exception as e:
                await self.error_handler.handle_error(e, {
                    "component": "feature_addition",
                    "task": "add_feature"
                })
                await asyncio.sleep(300)
    
    @with_error_handling(component="bug_detection", severity=ErrorSeverity.MEDIUM)
    @with_metrics("bug.detection")
    async def bug_detection_loop(self):
        """버그 감지 및 수정 루프"""
        await asyncio.sleep(self.config["bug_check_interval"])
        
        while self.running:
            try:
                if self.current_project and self.current_project in self.projects:
                    project = self.projects[self.current_project]
                    
                    with LogContextManager(self.logger, "버그 검사"):
                        # 프로젝트 분석
                        analysis = await self.godot_controller.analyze_project(
                            project["path"]
                        )
                        
                        # 가상의 버그 감지 (실제로는 정적 분석 도구 사용)
                        bugs_found = await self.detect_bugs(project, analysis)
                        
                        if bugs_found:
                            # AI로 버그 수정
                            fixed = await self.fix_bugs(bugs_found, project)
                            
                            project["bugs_fixed"] += fixed
                            self.stats["bugs_fixed"] += fixed
                            
                            await self.monitor.record_metric(
                                "business.bugs.fixed", fixed, MetricType.COUNTER
                            )
                
                await asyncio.sleep(self.config["bug_check_interval"])
                
            except Exception as e:
                await self.error_handler.handle_error(e, {
                    "component": "bug_detection",
                    "task": "detect_and_fix_bugs"
                })
                await asyncio.sleep(300)
    
    @with_error_handling(component="optimization", severity=ErrorSeverity.LOW)
    @with_metrics("project.optimization")
    async def optimization_loop(self):
        """최적화 루프"""
        await asyncio.sleep(self.config["optimization_interval"])
        
        while self.running:
            try:
                if self.current_project and self.current_project in self.projects:
                    project = self.projects[self.current_project]
                    
                    with LogContextManager(self.logger, "프로젝트 최적화"):
                        # 최적화 수행
                        optimizations = await self.godot_controller.optimize_project(
                            project["path"]
                        )
                        
                        if optimizations:
                            project["optimizations"] += 1
                            self.stats["optimization_runs"] += 1
                            
                            await self.monitor.record_metric(
                                "business.optimizations", 1, MetricType.COUNTER
                            )
                
                await asyncio.sleep(self.config["optimization_interval"])
                
            except Exception as e:
                await self.error_handler.handle_error(e, {
                    "component": "optimization",
                    "task": "optimize_project"
                })
                await asyncio.sleep(300)
    
    @with_error_handling(component="learning", severity=ErrorSeverity.LOW)
    async def learning_loop(self):
        """C# 학습 루프"""
        topics = [
            "async/await patterns",
            "LINQ expressions",
            "delegates and events",
            "generics",
            "reflection",
            "dependency injection",
            "design patterns",
            "performance optimization"
        ]
        
        topic_index = 0
        
        while self.running:
            try:
                topic = topics[topic_index % len(topics)]
                
                with LogContextManager(self.logger, f"C# 학습: {topic}"):
                    # 학습 콘텐츠 생성
                    content = await self.csharp_agent.generate_learning_content(topic)
                    
                    if content:
                        # 학습 자료 저장
                        learning_path = self.project_root / "csharp_learning" / \
                                      f"{topic.replace(' ', '_')}.md"
                        learning_path.write_text(content)
                        
                        self.stats["csharp_concepts_learned"] += 1
                        
                        await self.monitor.record_metric(
                            "business.concepts.learned", 1, MetricType.COUNTER
                        )
                
                topic_index += 1
                await asyncio.sleep(1800)  # 30분마다
                
            except Exception as e:
                await self.error_handler.handle_error(e, {
                    "component": "learning",
                    "task": "learn_csharp"
                })
                await asyncio.sleep(300)
    
    async def backup_loop(self):
        """백업 루프"""
        while self.running:
            try:
                await asyncio.sleep(self.config["backup_interval"])
                
                with LogContextManager(self.logger, "프로젝트 백업"):
                    backup_dir = self.project_root / "backups" / \
                                datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 활성 프로젝트 백업
                    for project_name, project_info in self.projects.items():
                        if project_info["status"] == "active":
                            await self.backup_project(project_name, backup_dir)
                
            except Exception as e:
                self.logger.error(f"백업 실패: {e}")
                await asyncio.sleep(3600)
    
    async def terminal_interface(self):
        """터미널 인터페이스"""
        print("\n🚀 AutoCI Production System")
        print("=" * 60)
        print("상용화 수준의 24시간 AI 게임 개발 시스템")
        print("'help'를 입력하여 명령어를 확인하세요.")
        print("=" * 60 + "\n")
        
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        
        try:
            await asyncio.get_event_loop().connect_read_pipe(
                lambda: protocol, sys.stdin
            )
        except:
            # Windows 환경에서 실패할 수 있음
            self.logger.warning("터미널 인터페이스 초기화 실패")
            return
        
        while self.running:
            try:
                # 비동기 입력 대기
                print("autoci> ", end="", flush=True)
                
                # 입력 대기 또는 종료 이벤트 대기
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(reader.readline()),
                        asyncio.create_task(self.shutdown_event.wait())
                    ],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # 대기 중인 작업 취소
                for task in pending:
                    task.cancel()
                
                if self.shutdown_event.is_set():
                    break
                
                # 완료된 작업에서 결과 가져오기
                for task in done:
                    if not task.cancelled():
                        line = task.result()
                        if line:
                            command = line.decode().strip()
                            await self.handle_command(command)
                
            except Exception as e:
                self.logger.error(f"터미널 인터페이스 오류: {e}")
                await asyncio.sleep(1)
    
    async def handle_command(self, command: str):
        """명령어 처리"""
        parts = command.lower().split()
        if not parts:
            return
        
        cmd = parts[0]
        
        if cmd == "status":
            await self.show_status()
        elif cmd == "projects":
            await self.list_projects()
        elif cmd == "metrics":
            await self.show_metrics()
        elif cmd == "health":
            await self.show_health()
        elif cmd == "errors":
            await self.show_errors()
        elif cmd == "help":
            self.show_help()
        elif cmd in ["exit", "quit"]:
            self.running = False
            self.shutdown_event.set()
        else:
            print(f"알 수 없는 명령어: {cmd}")
    
    async def show_status(self):
        """시스템 상태 표시"""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        print("\n" + "=" * 60)
        print("📊 AutoCI Production System 상태")
        print("=" * 60)
        print(f"⏱️  가동 시간: {uptime:.1f}시간")
        print(f"🎮 생성된 게임: {self.stats['games_created']}개")
        print(f"➕ 추가된 기능: {self.stats['features_added']}개")
        print(f"🐛 수정된 버그: {self.stats['bugs_fixed']}개")
        print(f"📚 학습한 개념: {self.stats['csharp_concepts_learned']}개")
        print(f"⚡ 최적화 실행: {self.stats['optimization_runs']}회")
        print(f"🔧 복구된 에러: {self.stats['errors_recovered']}개")
        
        # 시스템 리소스
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"\n💻 시스템 리소스:")
        print(f"   CPU: {cpu_percent}%")
        print(f"   메모리: {memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
        print(f"   디스크: {disk.percent}% ({disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB)")
        print("=" * 60 + "\n")
    
    async def list_projects(self):
        """프로젝트 목록"""
        print("\n📁 프로젝트 목록:")
        print("-" * 60)
        
        for name, info in self.projects.items():
            status_icon = "🟢" if info["status"] == "active" else "🔴"
            created = info["created"].strftime("%Y-%m-%d %H:%M")
            print(f"{status_icon} {name}")
            print(f"   타입: {info['type']}")
            print(f"   생성: {created}")
            print(f"   기능: {len(info['features'])}개")
            print(f"   버그 수정: {info['bugs_fixed']}개")
            print()
    
    async def show_metrics(self):
        """메트릭스 표시"""
        summary = self.monitor.get_system_status()
        
        print("\n📈 시스템 메트릭스:")
        print("-" * 60)
        print(f"수집된 메트릭: {summary['metrics_collected']}개")
        print(f"활성 알림: {summary['active_alerts']}개")
        print(f"데이터베이스 크기: {summary['database_size']:.2f}MB")
        
        if "last_metrics" in summary:
            print("\n최근 시스템 메트릭:")
            for name, value in summary["last_metrics"].items():
                if value is not None:
                    print(f"  {name}: {value:.2f}")
    
    async def show_health(self):
        """헬스 체크"""
        health = await self.monitor.health_check()
        
        print("\n🏥 시스템 헬스 체크:")
        print("-" * 60)
        print(f"전체 상태: {health['status']}")
        print("\n컴포넌트 상태:")
        for component, status in health["components"].items():
            icon = "✅" if status == "healthy" else "❌"
            print(f"  {icon} {component}: {status}")
    
    async def show_errors(self):
        """에러 리포트"""
        report = self.error_handler.get_error_report()
        
        print("\n🚨 에러 리포트 (최근 24시간):")
        print("-" * 60)
        print(f"총 에러: {report['errors_24h']}개")
        print(f"복구 성공률: {report['recovery_success_rate']:.1f}%")
        
        if report["most_common_errors"]:
            print("\n가장 빈번한 에러:")
            for error_type, count in report["most_common_errors"]:
                print(f"  - {error_type}: {count}회")
    
    def show_help(self):
        """도움말"""
        print("\n📖 명령어 도움말")
        print("=" * 60)
        print("status    - 시스템 상태 확인")
        print("projects  - 프로젝트 목록")
        print("metrics   - 메트릭스 확인")
        print("health    - 헬스 체크")
        print("errors    - 에러 리포트")
        print("help      - 이 도움말")
        print("exit      - 시스템 종료")
        print("=" * 60 + "\n")
    
    def get_features_for_game_type(self, game_type: str) -> List[str]:
        """게임 타입별 기능 목록"""
        features = {
            "platformer": [
                "double jump", "wall jump", "dash", "collectibles",
                "moving platforms", "enemy AI", "checkpoints", "power-ups",
                "parallax background", "particle effects", "sound effects"
            ],
            "racing": [
                "boost system", "drift mechanics", "lap timer", "AI opponents",
                "track obstacles", "vehicle customization", "minimap", "replay",
                "weather effects", "multiplayer", "leaderboard"
            ],
            "puzzle": [
                "hint system", "undo/redo", "level select", "score system",
                "timer", "achievements", "particle effects", "sound feedback",
                "tutorial", "difficulty modes", "save progress"
            ],
            "rpg": [
                "inventory", "dialogue", "quests", "combat", "skill tree",
                "save/load", "NPCs", "leveling", "equipment", "map system",
                "cutscenes"
            ]
        }
        return features.get(game_type, [])
    
    async def generate_initial_code(self, project_name: str, game_type: str):
        """초기 코드 생성"""
        if not self.ai_integration:
            return
        
        context = {
            "game_type": game_type,
            "project_name": project_name,
            "engine": "Godot 4.2",
            "language": "GDScript"
        }
        
        # AI로 코드 생성
        prompt = f"Create initial game structure for {game_type} game in Godot"
        result = await self.ai_integration.generate_code(prompt, context)
        
        if result["success"]:
            project = self.projects[project_name]
            script_path = Path(project["path"]) / "scripts" / "Game.gd"
            script_path.parent.mkdir(exist_ok=True)
            script_path.write_text(result["code"])
    
    async def implement_feature(self, feature: str, project: Dict[str, Any]) -> bool:
        """기능 구현"""
        if not self.ai_integration:
            return False
        
        context = {
            "feature": feature,
            "game_type": project["type"],
            "existing_features": project["features"]
        }
        
        prompt = f"Implement {feature} feature for {project['type']} game"
        result = await self.ai_integration.generate_code(prompt, context)
        
        if result["success"]:
            script_path = Path(project["path"]) / "scripts" / f"{feature.replace(' ', '_')}.gd"
            script_path.write_text(result["code"])
            return True
        
        return False
    
    async def detect_bugs(self, project: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """버그 감지 (시뮬레이션)"""
        # 실제로는 정적 분석 도구 사용
        bugs = []
        
        # 가상의 버그 감지
        if len(project["features"]) > 5 and project["bugs_fixed"] < 2:
            bugs.append({
                "type": "null_reference",
                "file": "Player.gd",
                "line": 42,
                "severity": "medium"
            })
        
        return bugs
    
    async def fix_bugs(self, bugs: List[Dict[str, Any]], project: Dict[str, Any]) -> int:
        """버그 수정"""
        fixed = 0
        
        for bug in bugs:
            # AI로 버그 수정
            if self.ai_integration:
                context = {
                    "bug": bug,
                    "game_type": project["type"]
                }
                
                result = await self.ai_integration.fix_bug(bug, context)
                if result["success"]:
                    fixed += 1
        
        return fixed
    
    async def backup_project(self, project_name: str, backup_dir: Path):
        """프로젝트 백업"""
        project = self.projects[project_name]
        project_path = Path(project["path"])
        
        if project_path.exists():
            backup_path = backup_dir / project_name
            
            # shutil.copytree 대신 tar 사용 (대용량 처리)
            import tarfile
            
            tar_path = backup_path.with_suffix('.tar.gz')
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(project_path, arcname=project_name)
            
            self.logger.info(f"프로젝트 백업 완료: {project_name}")
    
    async def shutdown(self):
        """시스템 종료"""
        self.logger.info("시스템 종료 시작...")
        
        # 현재 작업 저장
        await self.save_state()
        
        # 메트릭 저장
        await self.monitor._export_metrics()
        
        # 에러 통계 저장
        await self.error_handler.save_error_statistics()
        
        # 프로젝트 내보내기 (설정된 경우)
        if self.config.get("auto_export"):
            for project_name, project in self.projects.items():
                if project["status"] == "active":
                    export_path = self.project_root / "exports" / f"{project_name}.zip"
                    await self.godot_controller.export_project(
                        project["path"], str(export_path)
                    )
        
        self.logger.info("시스템 종료 완료")
    
    async def save_state(self):
        """시스템 상태 저장"""
        state = {
            "projects": self.projects,
            "stats": self.stats,
            "current_project": self.current_project,
            "shutdown_time": datetime.now().isoformat()
        }
        
        state_path = self.project_root / "data" / "system_state.json"
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)


async def main():
    """메인 함수"""
    autoci = ProductionAutoCI()
    await autoci.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n시스템이 종료되었습니다.")
    except Exception as e:
        print(f"치명적 오류: {e}")
        sys.exit(1)