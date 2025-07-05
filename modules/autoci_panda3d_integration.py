"""
AutoCI Panda3D 통합 시스템
Panda3D, Socket.IO, PyTorch 기반의 AI 자동 게임 개발 시스템
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import click

# 내부 모듈 임포트
from .panda3d_ai_agent import Panda3DAIAgent
from .ai_model_integration import get_ai_integration
from .socketio_realtime_system import SocketIORealtimeSystem
from .panda3d_automation_controller import Panda3DAutomationController
from .pytorch_deep_learning_module import DeepLearningModule

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoCIPanda3DSystem:
    """AutoCI Panda3D 통합 시스템"""
    
    def __init__(self):
        self.ai_integration = get_ai_integration()
        self.socketio_system = SocketIORealtimeSystem()
        self.agents: Dict[str, Panda3DAIAgent] = {}
        self.is_running = False
        
        # 지원하는 게임 타입
        self.game_types = {
            "platformer": "2D/2.5D 플랫폼 게임",
            "racing": "레이싱 게임",
            "rpg": "롤플레잉 게임", 
            "puzzle": "퍼즐 게임",
            "shooter": "슈팅 게임",
            "adventure": "어드벤처 게임",
            "simulation": "시뮬레이션 게임"
        }
        
        logger.info("🚀 AutoCI Panda3D 시스템 초기화 완료")
    
    async def create_game(self, project_name: str, game_type: str, 
                         development_hours: float = 24.0) -> Dict[str, Any]:
        """
        AI가 자동으로 게임을 개발
        
        Args:
            project_name: 프로젝트 이름
            game_type: 게임 타입
            development_hours: 개발 시간 (기본 24시간)
            
        Returns:
            개발 결과 정보
        """
        if game_type not in self.game_types:
            raise ValueError(f"지원하지 않는 게임 타입: {game_type}")
        
        logger.info(f"🎮 새 게임 개발 시작: {project_name} ({game_type})")
        
        # AI 에이전트 생성
        agent = Panda3DAIAgent(project_name, game_type)
        self.agents[project_name] = agent
        
        # 개발 시작
        try:
            await agent.start_development(target_hours=development_hours)
            
            # 개발 결과 반환
            result = {
                "success": True,
                "project_name": project_name,
                "game_type": game_type,
                "development_time": development_hours,
                "quality_score": agent.game_state.quality_score,
                "completeness": agent.game_state.completeness,
                "features": agent.game_state.features,
                "project_path": agent.panda_controller.get_project_path()
            }
            
        except Exception as e:
            logger.error(f"게임 개발 실패: {e}")
            result = {
                "success": False,
                "error": str(e)
            }
        
        return result
    
    async def modify_game(self, project_name: str, modification_request: str) -> Dict[str, Any]:
        """
        기존 게임 수정
        
        Args:
            project_name: 프로젝트 이름
            modification_request: 수정 요청 내용
            
        Returns:
            수정 결과
        """
        if project_name not in self.agents:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}
        
        agent = self.agents[project_name]
        
        # AI에게 수정 요청
        context = {
            "project_name": project_name,
            "current_features": agent.game_state.features,
            "modification_request": modification_request
        }
        
        result = await self.ai_integration.generate_code(
            prompt=f"Modify the game with this request: {modification_request}",
            context=context,
            task_type="game_dev"
        )
        
        if result["success"]:
            # 코드 적용
            # TODO: 실제 코드 적용 로직
            return {
                "success": True,
                "message": "게임이 수정되었습니다",
                "code": result["code"]
            }
        
        return {"success": False, "error": "수정 실패"}
    
    async def analyze_game(self, project_path: str) -> Dict[str, Any]:
        """
        게임 분석
        
        Args:
            project_path: 프로젝트 경로
            
        Returns:
            분석 결과
        """
        logger.info(f"🔍 게임 분석 시작: {project_path}")
        
        # 프로젝트 파일들 읽기
        project_files = self._read_project_files(project_path)
        
        # AI를 통한 코드 분석
        analysis_results = {}
        
        for file_path, code in project_files.items():
            if file_path.endswith('.py'):
                result = await self.ai_integration.analyze_code(
                    code=code,
                    analysis_type="comprehensive"
                )
                analysis_results[file_path] = result
        
        # 종합 분석
        overall_quality = sum(r.get("quality_score", 0) for r in analysis_results.values()) / len(analysis_results)
        
        return {
            "project_path": project_path,
            "file_count": len(project_files),
            "overall_quality": overall_quality,
            "file_analyses": analysis_results,
            "recommendations": self._generate_recommendations(analysis_results)
        }
    
    def _read_project_files(self, project_path: str) -> Dict[str, str]:
        """프로젝트 파일 읽기"""
        files = {}
        project_dir = Path(project_path)
        
        if not project_dir.exists():
            return files
        
        # Python 파일들 읽기
        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                relative_path = py_file.relative_to(project_dir)
                files[str(relative_path)] = content
            except Exception as e:
                logger.warning(f"파일 읽기 실패: {py_file} - {e}")
        
        return files
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """분석 결과를 바탕으로 개선 권장사항 생성"""
        recommendations = []
        
        # 전체적인 품질 점수 기반 권장사항
        avg_quality = sum(r.get("quality_score", 0) for r in analysis_results.values()) / len(analysis_results)
        
        if avg_quality < 60:
            recommendations.append("전반적인 코드 품질 개선이 필요합니다")
        
        # 공통 문제 파악
        common_issues = {}
        for file_analysis in analysis_results.values():
            for bug in file_analysis.get("bugs", []):
                common_issues[bug] = common_issues.get(bug, 0) + 1
        
        # 가장 많이 발생한 문제들
        for issue, count in sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:5]:
            recommendations.append(f"{issue} (발생 횟수: {count})")
        
        return recommendations
    
    async def start_monitoring(self, port: int = 5001):
        """실시간 모니터링 시작"""
        logger.info(f"📊 실시간 모니터링 시작 (포트: {port})")
        await self.socketio_system.start()
    
    def get_supported_game_types(self) -> Dict[str, str]:
        """지원하는 게임 타입 목록"""
        return self.game_types
    
    def get_active_projects(self) -> List[Dict[str, Any]]:
        """활성 프로젝트 목록"""
        projects = []
        
        for name, agent in self.agents.items():
            projects.append({
                "name": name,
                "type": agent.game_type,
                "status": agent.game_state.current_phase,
                "quality": agent.game_state.quality_score,
                "completeness": agent.game_state.completeness,
                "features": agent.game_state.features
            })
        
        return projects


# CLI 명령어 구현
@click.group()
def cli():
    """AutoCI Panda3D - AI 자동 게임 개발 시스템"""
    pass


@cli.command()
@click.option('--name', prompt='프로젝트 이름', help='게임 프로젝트 이름')
@click.option('--type', 'game_type', 
              type=click.Choice(['platformer', 'racing', 'rpg', 'puzzle', 'shooter', 'adventure', 'simulation']),
              prompt='게임 타입', help='개발할 게임 타입')
@click.option('--hours', default=24.0, help='개발 시간 (기본 24시간)')
def create(name: str, game_type: str, hours: float):
    """새 게임 자동 생성"""
    async def run():
        system = AutoCIPanda3DSystem()
        result = await system.create_game(name, game_type, hours)
        
        if result["success"]:
            click.echo(f"✅ 게임 개발 완료!")
            click.echo(f"   프로젝트: {result['project_name']}")
            click.echo(f"   품질 점수: {result['quality_score']:.1f}/100")
            click.echo(f"   완성도: {result['completeness']:.1f}%")
            click.echo(f"   경로: {result['project_path']}")
        else:
            click.echo(f"❌ 개발 실패: {result['error']}")
    
    asyncio.run(run())


@cli.command()
@click.argument('project_path')
def analyze(project_path: str):
    """게임 프로젝트 분석"""
    async def run():
        system = AutoCIPanda3DSystem()
        result = await system.analyze_game(project_path)
        
        click.echo(f"📊 게임 분석 결과")
        click.echo(f"   전체 품질: {result['overall_quality']:.1f}/100")
        click.echo(f"   파일 수: {result['file_count']}")
        click.echo(f"\n권장사항:")
        for rec in result['recommendations']:
            click.echo(f"   - {rec}")
    
    asyncio.run(run())


@cli.command()
@click.option('--port', default=5001, help='모니터링 포트')
def monitor(port: int):
    """실시간 모니터링 시작"""
    async def run():
        system = AutoCIPanda3DSystem()
        click.echo(f"🌐 실시간 모니터링 시작 (http://localhost:{port})")
        await system.start_monitoring(port)
    
    asyncio.run(run())


@cli.command()
def types():
    """지원하는 게임 타입 목록"""
    system = AutoCIPanda3DSystem()
    game_types = system.get_supported_game_types()
    
    click.echo("🎮 지원하는 게임 타입:")
    for type_id, description in game_types.items():
        click.echo(f"   {type_id}: {description}")


# 프로그래밍 방식 사용 예시
async def example_usage():
    """사용 예시"""
    # 시스템 초기화
    system = AutoCIPanda3DSystem()
    
    # 플랫폼 게임 생성 (24시간 개발)
    result = await system.create_game(
        project_name="SuperPlatformer",
        game_type="platformer",
        development_hours=24.0
    )
    
    if result["success"]:
        print(f"게임 개발 성공!")
        print(f"품질 점수: {result['quality_score']}")
        print(f"프로젝트 경로: {result['project_path']}")
        
        # 게임 수정
        mod_result = await system.modify_game(
            project_name="SuperPlatformer",
            modification_request="Add double jump feature and more particle effects"
        )
        
        if mod_result["success"]:
            print("게임 수정 완료!")
    
    # 게임 분석
    analysis = await system.analyze_game(result['project_path'])
    print(f"코드 품질: {analysis['overall_quality']}")


if __name__ == "__main__":
    # CLI 모드
    if len(sys.argv) > 1:
        cli()
    else:
        # 프로그래밍 모드 예시
        asyncio.run(example_usage())