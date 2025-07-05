#!/usr/bin/env python3
"""
AutoCI Panda3D 실행 스크립트
AI 자동 게임 개발 시스템 시작
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# 프로젝트 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from modules.autoci_panda3d_integration import AutoCIPanda3DSystem
from modules.panda3d_ai_agent import Panda3DAIAgent


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="AutoCI Panda3D - AI 자동 게임 개발 시스템")
    parser.add_argument('command', choices=['create', 'analyze', 'monitor', 'demo'],
                       help='실행할 명령')
    parser.add_argument('--name', help='프로젝트 이름')
    parser.add_argument('--type', choices=['platformer', 'racing', 'rpg', 'puzzle', 'shooter', 'adventure', 'simulation'],
                       default='platformer', help='게임 타입')
    parser.add_argument('--hours', type=float, default=24.0, help='개발 시간 (시간 단위)')
    parser.add_argument('--path', help='분석할 프로젝트 경로')
    parser.add_argument('--port', type=int, default=5001, help='모니터링 포트')
    
    args = parser.parse_args()
    
    # AutoCI 시스템 초기화
    system = AutoCIPanda3DSystem()
    
    if args.command == 'create':
        # 게임 생성
        if not args.name:
            print("❌ 프로젝트 이름을 지정해주세요 (--name)")
            return
        
        print(f"🎮 AutoCI가 {args.type} 게임을 만들기 시작합니다...")
        print(f"   프로젝트: {args.name}")
        print(f"   예상 시간: {args.hours}시간")
        print(f"   AI 모델이 자동으로 게임을 개발합니다...\n")
        
        result = await system.create_game(
            project_name=args.name,
            game_type=args.type,
            development_hours=args.hours
        )
        
        if result["success"]:
            print(f"\n✅ 게임 개발 완료!")
            print(f"   품질 점수: {result['quality_score']:.1f}/100")
            print(f"   완성도: {result['completeness']:.1f}%")
            print(f"   구현된 기능: {len(result['features'])}개")
            print(f"   프로젝트 경로: {result['project_path']}")
            print(f"\n게임을 실행하려면:")
            print(f"   cd {result['project_path']}")
            print(f"   python main.py")
        else:
            print(f"❌ 개발 실패: {result['error']}")
    
    elif args.command == 'analyze':
        # 게임 분석
        if not args.path:
            print("❌ 분석할 프로젝트 경로를 지정해주세요 (--path)")
            return
        
        print(f"📊 프로젝트 분석 중: {args.path}")
        
        analysis = await system.analyze_game(args.path)
        
        print(f"\n분석 결과:")
        print(f"   전체 품질: {analysis['overall_quality']:.1f}/100")
        print(f"   파일 수: {analysis['file_count']}")
        print(f"\n권장사항:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    elif args.command == 'monitor':
        # 실시간 모니터링
        print(f"📡 실시간 모니터링 서버 시작...")
        print(f"   포트: {args.port}")
        print(f"   브라우저에서 http://localhost:{args.port} 접속")
        print(f"   Ctrl+C로 중지\n")
        
        try:
            await system.start_monitoring(port=args.port)
        except KeyboardInterrupt:
            print("\n모니터링 중지됨")
    
    elif args.command == 'demo':
        # 데모 모드
        print("🎉 AutoCI Panda3D 데모 모드")
        print("AI가 5분 동안 간단한 게임을 만듭니다...\n")
        
        demo_name = f"DemoGame_{Path.cwd().name}"
        
        result = await system.create_game(
            project_name=demo_name,
            game_type="platformer",
            development_hours=0.083  # 5분
        )
        
        if result["success"]:
            print(f"\n✅ 데모 게임 생성 완료!")
            print(f"   프로젝트: {result['project_path']}")
            print(f"   품질: {result['quality_score']:.1f}/100")
            
            # 게임 실행 안내
            print(f"\n게임을 실행하려면:")
            print(f"   cd {result['project_path']}")
            print(f"   python main.py")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    AutoCI Panda3D v1.0                    ║
    ║         AI 자동 2.5D/3D 게임 개발 시스템                  ║
    ║                                                           ║
    ║  Powered by: Panda3D, PyTorch, Socket.IO, DeepSeek-Coder ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()