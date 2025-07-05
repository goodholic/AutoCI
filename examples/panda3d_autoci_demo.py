#!/usr/bin/env python3
"""
AutoCI Panda3D 데모
AI가 자동으로 게임을 개발하는 과정을 시연하는 예제
"""

import asyncio
import sys
import os
from pathlib import Path
import click
import logging
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.autoci_panda3d_integration import AutoCIPanda3DSystem
from modules.panda3d_ai_agent import Panda3DAIAgent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoci_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoCIDemo:
    """AutoCI 데모 클래스"""
    
    def __init__(self):
        self.system = AutoCIPanda3DSystem()
        self.demo_projects = []
    
    async def demo_quick_game(self):
        """빠른 게임 생성 데모 (5분)"""
        print("\n" + "="*50)
        print("🚀 AutoCI 빠른 게임 생성 데모")
        print("="*50)
        print("AI가 5분 동안 간단한 플랫폼 게임을 만듭니다...\n")
        
        project_name = f"QuickPlatformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = await self.system.create_game(
            project_name=project_name,
            game_type="platformer",
            development_hours=0.083  # 5분
        )
        
        if result["success"]:
            print(f"\n✅ 게임 생성 완료!")
            print(f"   프로젝트: {result['project_name']}")
            print(f"   품질 점수: {result['quality_score']:.1f}/100")
            print(f"   완성도: {result['completeness']:.1f}%")
            print(f"   구현된 기능: {', '.join(result['features'])}")
            print(f"   프로젝트 경로: {result['project_path']}")
            
            self.demo_projects.append(result)
        else:
            print(f"❌ 게임 생성 실패: {result['error']}")
        
        return result
    
    async def demo_game_types(self):
        """다양한 게임 타입 데모"""
        print("\n" + "="*50)
        print("🎮 다양한 게임 타입 생성 데모")
        print("="*50)
        
        game_types = ["racing", "puzzle", "shooter"]
        development_time = 0.05  # 3분
        
        for game_type in game_types:
            print(f"\n{game_type.upper()} 게임 생성 중...")
            
            project_name = f"Demo{game_type.title()}_{datetime.now().strftime('%H%M%S')}"
            
            result = await self.system.create_game(
                project_name=project_name,
                game_type=game_type,
                development_hours=development_time
            )
            
            if result["success"]:
                print(f"   ✅ {game_type} 게임 생성 완료 (품질: {result['quality_score']:.1f})")
                self.demo_projects.append(result)
            else:
                print(f"   ❌ {game_type} 게임 생성 실패")
            
            # 짧은 대기
            await asyncio.sleep(1)
    
    async def demo_game_modification(self):
        """게임 수정 데모"""
        print("\n" + "="*50)
        print("🔧 게임 수정 데모")
        print("="*50)
        
        if not self.demo_projects:
            print("먼저 게임을 생성해주세요!")
            return
        
        # 첫 번째 프로젝트 수정
        project = self.demo_projects[0]
        project_name = project["project_name"]
        
        modifications = [
            "Add double jump feature to the player",
            "Create more challenging enemy AI",
            "Add particle effects for jumps and collisions",
            "Implement a power-up system"
        ]
        
        for mod in modifications:
            print(f"\n수정 요청: {mod}")
            
            result = await self.system.modify_game(
                project_name=project_name,
                modification_request=mod
            )
            
            if result["success"]:
                print(f"   ✅ 수정 완료")
            else:
                print(f"   ❌ 수정 실패")
            
            await asyncio.sleep(0.5)
    
    async def demo_game_analysis(self):
        """게임 분석 데모"""
        print("\n" + "="*50)
        print("📊 게임 분석 데모")
        print("="*50)
        
        if not self.demo_projects:
            print("분석할 게임이 없습니다!")
            return
        
        for project in self.demo_projects[:2]:  # 처음 2개만 분석
            print(f"\n프로젝트 분석: {project['project_name']}")
            
            analysis = await self.system.analyze_game(project["project_path"])
            
            print(f"   전체 품질: {analysis['overall_quality']:.1f}/100")
            print(f"   파일 수: {analysis['file_count']}")
            print(f"   권장사항:")
            for rec in analysis['recommendations'][:3]:
                print(f"      - {rec}")
    
    async def demo_realtime_monitoring(self):
        """실시간 모니터링 데모"""
        print("\n" + "="*50)
        print("📡 실시간 모니터링 데모")
        print("="*50)
        print("Socket.IO 서버를 시작합니다...")
        print("웹 브라우저에서 http://localhost:5001 로 접속하세요")
        print("(Ctrl+C로 중지)")
        
        try:
            await self.system.start_monitoring(port=5001)
        except KeyboardInterrupt:
            print("\n모니터링 중지됨")
    
    def show_summary(self):
        """데모 요약 표시"""
        print("\n" + "="*50)
        print("📋 데모 요약")
        print("="*50)
        
        if not self.demo_projects:
            print("생성된 프로젝트가 없습니다.")
            return
        
        print(f"\n총 {len(self.demo_projects)}개의 게임이 생성되었습니다:")
        
        for i, project in enumerate(self.demo_projects, 1):
            print(f"\n{i}. {project['project_name']}")
            print(f"   타입: {project['game_type']}")
            print(f"   품질: {project['quality_score']:.1f}/100")
            print(f"   완성도: {project['completeness']:.1f}%")
            print(f"   경로: {project['project_path']}")
    
    async def run_full_demo(self):
        """전체 데모 실행"""
        print("\n🎉 AutoCI Panda3D 전체 데모를 시작합니다!")
        print("AI가 자동으로 게임을 개발하는 과정을 보여드립니다.\n")
        
        # 1. 빠른 게임 생성
        await self.demo_quick_game()
        await asyncio.sleep(2)
        
        # 2. 다양한 게임 타입
        await self.demo_game_types()
        await asyncio.sleep(2)
        
        # 3. 게임 수정
        await self.demo_game_modification()
        await asyncio.sleep(2)
        
        # 4. 게임 분석
        await self.demo_game_analysis()
        
        # 5. 요약
        self.show_summary()
        
        print("\n✨ 데모가 완료되었습니다!")
        print("생성된 게임들은 각 프로젝트 폴더에서 확인할 수 있습니다.")


@click.group()
def cli():
    """AutoCI Panda3D 데모"""
    pass


@cli.command()
@click.option('--time', default=5, help='개발 시간 (분)')
def quick(time: int):
    """빠른 게임 생성 데모"""
    demo = AutoCIDemo()
    asyncio.run(demo.demo_quick_game())


@cli.command()
def types():
    """다양한 게임 타입 데모"""
    demo = AutoCIDemo()
    asyncio.run(demo.demo_game_types())


@cli.command()
def full():
    """전체 데모 실행"""
    demo = AutoCIDemo()
    asyncio.run(demo.run_full_demo())


@cli.command()
def monitor():
    """실시간 모니터링"""
    demo = AutoCIDemo()
    asyncio.run(demo.demo_realtime_monitoring())


# 프로그래밍 방식 사용 예제
async def example_custom_game():
    """커스텀 게임 생성 예제"""
    system = AutoCIPanda3DSystem()
    
    # RPG 게임 생성 (1시간 개발)
    print("🎮 커스텀 RPG 게임 생성 중...")
    
    result = await system.create_game(
        project_name="MyCustomRPG",
        game_type="rpg",
        development_hours=1.0
    )
    
    if result["success"]:
        print(f"\n✅ RPG 게임 생성 성공!")
        print(f"품질 점수: {result['quality_score']}")
        print(f"구현된 기능들:")
        for feature in result['features']:
            print(f"  - {feature}")
        
        # 추가 기능 요청
        print("\n🔧 멀티플레이어 기능 추가 중...")
        
        mod_result = await system.modify_game(
            project_name="MyCustomRPG",
            modification_request="Add multiplayer support using Socket.IO"
        )
        
        if mod_result["success"]:
            print("✅ 멀티플레이어 기능 추가 완료!")


async def example_batch_creation():
    """배치 게임 생성 예제"""
    system = AutoCIPanda3DSystem()
    game_configs = [
        ("SpaceShooter2025", "shooter", 0.5),
        ("PuzzleMaster", "puzzle", 0.3),
        ("RacingPro", "racing", 0.4),
        ("AdventureQuest", "adventure", 0.6)
    ]
    
    print("🎮 여러 게임 동시 생성 시작...")
    
    # 동시에 여러 게임 생성
    tasks = []
    for name, game_type, hours in game_configs:
        task = system.create_game(
            project_name=name,
            game_type=game_type,
            development_hours=hours
        )
        tasks.append(task)
    
    # 모든 게임 생성 대기
    results = await asyncio.gather(*tasks)
    
    # 결과 출력
    print("\n📊 배치 생성 결과:")
    for i, result in enumerate(results):
        config = game_configs[i]
        if result["success"]:
            print(f"✅ {config[0]}: 품질 {result['quality_score']:.1f}/100")
        else:
            print(f"❌ {config[0]}: 실패")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI 모드
        cli()
    else:
        # 기본 데모 실행
        print("AutoCI Panda3D 데모")
        print("1. 전체 데모 실행")
        print("2. 커스텀 게임 생성")
        print("3. 배치 게임 생성")
        print("4. 실시간 모니터링")
        
        choice = input("\n선택 (1-4): ")
        
        if choice == "1":
            demo = AutoCIDemo()
            asyncio.run(demo.run_full_demo())
        elif choice == "2":
            asyncio.run(example_custom_game())
        elif choice == "3":
            asyncio.run(example_batch_creation())
        elif choice == "4":
            demo = AutoCIDemo()
            asyncio.run(demo.demo_realtime_monitoring())
        else:
            print("잘못된 선택입니다.")