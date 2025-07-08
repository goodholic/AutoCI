#!/usr/bin/env python3
"""
Enhanced create command for AutoCI
Supports continuing development on existing projects
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

async def create_or_continue_game(game_type: str, game_name: Optional[str] = None) -> Dict[str, Any]:
    """
    게임을 생성하거나 기존 게임 개발을 이어서 진행
    
    Args:
        game_type: 게임 타입 (platformer, rpg, racing, puzzle)
        game_name: 게임 이름 (선택사항)
    
    Returns:
        결과 딕셔너리
    """
    from modules.game_factory_24h import GameFactory24H
    
    # mvp_games 디렉토리 확인
    mvp_games_dir = Path("mvp_games")
    existing_projects = []
    
    if mvp_games_dir.exists():
        # 같은 타입의 기존 프로젝트 찾기
        for project_dir in mvp_games_dir.iterdir():
            if project_dir.is_dir() and (project_dir / "project.godot").exists():
                # 프로젝트 정보 읽기
                project_info_file = project_dir / "project_info.json"
                if project_info_file.exists():
                    with open(project_info_file, 'r', encoding='utf-8') as f:
                        project_info = json.load(f)
                        if project_info.get('type') == game_type:
                            existing_projects.append({
                                'path': project_dir,
                                'name': project_dir.name,
                                'info': project_info,
                                'modified': datetime.fromtimestamp(project_dir.stat().st_mtime)
                            })
    
    # 기존 프로젝트가 있으면 선택 옵션 제공
    if existing_projects:
        print(f"\n📂 {game_type} 타입의 기존 프로젝트를 발견했습니다!")
        print("=" * 60)
        
        # 최신순으로 정렬
        existing_projects.sort(key=lambda x: x['modified'], reverse=True)
        
        for i, project in enumerate(existing_projects[:5], 1):
            print(f"\n{i}. {project['name']}")
            print(f"   경로: {project['path']}")
            print(f"   마지막 수정: {project['modified'].strftime('%Y-%m-%d %H:%M')}")
            info = project['info']
            print(f"   진행 상태: {info.get('progress', '0')}%")
            features = info.get('features', [])
            if features:
                print(f"   구현된 기능: {', '.join(features[:3])}")
                if len(features) > 3:
                    print(f"                 외 {len(features)-3}개...")
        
        print("\n선택하세요:")
        print("1-5. 기존 프로젝트 이어서 개발하기")
        print("0. 새로운 프로젝트 시작하기")
        print("Enter. 가장 최근 프로젝트 이어서 개발하기")
        
        choice = input("\n선택 (0-5 또는 Enter): ").strip()
        
        if choice == '':
            # Enter 누르면 가장 최근 프로젝트
            selected_project = existing_projects[0]
            print(f"\n✅ '{selected_project['name']}' 프로젝트를 이어서 개발합니다!")
            return await continue_project(selected_project)
        elif choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= min(5, len(existing_projects)):
                selected_project = existing_projects[choice_num - 1]
                print(f"\n✅ '{selected_project['name']}' 프로젝트를 이어서 개발합니다!")
                return await continue_project(selected_project)
            elif choice_num == 0:
                print("\n🆕 새로운 프로젝트를 시작합니다!")
                return await create_new_project(game_type, game_name)
    
    # 기존 프로젝트가 없으면 새로 생성
    return await create_new_project(game_type, game_name)

async def continue_project(project: Dict[str, Any]) -> Dict[str, Any]:
    """기존 프로젝트 이어서 개발"""
    from modules.game_factory_24h import GameFactory24H
    
    factory = GameFactory24H()
    
    # 프로젝트 정보 설정
    factory.current_project = {
        'path': str(project['path']),
        'name': project['name'],
        'type': project['info'].get('type', 'platformer'),
        'features': project['info'].get('features', []),
        'progress': project['info'].get('progress', 0)
    }
    
    print("\n🏭 24시간 게임 개발 공장을 시작합니다...")
    print(f"📂 기존 프로젝트: {project['name']}")
    print(f"🎮 게임 타입: {factory.current_project['type']}")
    print(f"📊 현재 진행률: {factory.current_project['progress']}%")
    print(f"✨ 구현된 기능: {len(factory.current_project['features'])}개")
    
    # 24시간 자동 개발 시작
    await factory.start_factory(
        game_name=project['name'],
        game_type=factory.current_project['type'],
        existing_project=True
    )
    
    return {
        'success': True,
        'project_path': str(project['path']),
        'continued': True
    }

async def create_new_project(game_type: str, game_name: Optional[str] = None) -> Dict[str, Any]:
    """새 프로젝트 생성"""
    from modules.game_factory_24h import GameFactory24H
    
    # 게임 이름 자동 생성
    if not game_name:
        timestamp = datetime.now().strftime('%y%m%d%H%M')
        game_name = f"{timestamp}{game_type.capitalize()}"
    
    factory = GameFactory24H()
    
    print("\n🏭 24시간 게임 개발 공장을 시작합니다...")
    print(f"🆕 새 프로젝트: {game_name}")
    print(f"🎮 게임 타입: {game_type}")
    
    # 24시간 자동 개발 시작
    await factory.start_factory(
        game_name=game_name,
        game_type=game_type,
        existing_project=False
    )
    
    return {
        'success': True,
        'project_name': game_name,
        'continued': False
    }