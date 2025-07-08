#!/usr/bin/env python3
"""
AutoCI Create - Windows Safe Version
모듈 임포트 오류를 처리하는 Windows 전용 버전
"""

import sys
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path

# 안전한 임포트
try:
    from modules.game_session_manager import GameSessionManager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    print("⚠️ 세션 관리자를 사용할 수 없습니다 (선택사항)")

class SimpleGameCreator:
    """간단한 게임 생성기"""
    
    def __init__(self):
        self.game_type = None
        self.game_name = None
        
    async def create_game(self, game_type):
        """게임 생성"""
        self.game_type = game_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.game_name = f"{game_type}_game_{timestamp}"
        
        print(f"\n🎮 {game_type} 게임 생성 중...")
        
        # 게임 디렉토리 생성
        game_dir = Path(f"games/{self.game_name}")
        game_dir.mkdir(parents=True, exist_ok=True)
        
        # 기본 파일들 생성
        await self._create_main_file(game_dir)
        await self._create_config_file(game_dir)
        await self._create_readme_file(game_dir)
        
        # 게임 타입별 추가 파일
        if game_type == "rpg":
            await self._create_rpg_files(game_dir)
        elif game_type == "platformer":
            await self._create_platformer_files(game_dir)
            
        print(f"\n✅ {self.game_name} 생성 완료!")
        print(f"📂 위치: {game_dir.absolute()}")
        
    async def _create_main_file(self, game_dir):
        """메인 파일 생성"""
        content = f'''#!/usr/bin/env python3
"""
{self.game_name} - AutoCI Windows로 생성됨
게임 타입: {self.game_type}
생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys

def main():
    print("🎮 {self.game_name} 시작!")
    print("게임 타입: {self.game_type}")
    print("이 게임은 AutoCI Windows 버전으로 생성되었습니다.")
    
    # TODO: 게임 로직 구현
    if "{self.game_type}" == "rpg":
        print("⚔️ RPG 게임 모드")
    elif "{self.game_type}" == "platformer":
        print("🏃 플랫폼 게임 모드")
    elif "{self.game_type}" == "racing":
        print("🏎️ 레이싱 게임 모드")
    elif "{self.game_type}" == "puzzle":
        print("🧩 퍼즐 게임 모드")

if __name__ == "__main__":
    main()
'''
        
        (game_dir / "main.py").write_text(content, encoding='utf-8')
        print("✓ main.py 생성")
        
    async def _create_config_file(self, game_dir):
        """설정 파일 생성"""
        config = {
            "game_name": self.game_name,
            "game_type": self.game_type,
            "created_at": datetime.now().isoformat(),
            "version": "0.1.0",
            "platform": "windows",
            "engine": "autoci",
            "settings": {
                "resolution": "1280x720",
                "fullscreen": False,
                "vsync": True
            }
        }
        
        (game_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        print("✓ config.json 생성")
        
    async def _create_readme_file(self, game_dir):
        """README 파일 생성"""
        content = f'''# {self.game_name}

AutoCI Windows 버전으로 생성된 {self.game_type} 게임입니다.

## 게임 정보
- **타입**: {self.game_type}
- **생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **버전**: 0.1.0
- **플랫폼**: Windows

## 실행 방법
```bash
python main.py
```

## 게임 특징
- AutoCI로 자동 생성됨
- Windows 최적화

## 개발 상태
- [x] 기본 구조 생성
- [ ] 게임 로직 구현
- [ ] 그래픽 추가
- [ ] 사운드 추가
'''
        
        (game_dir / "README.md").write_text(content, encoding='utf-8')
        print("✓ README.md 생성")
        
    async def _create_rpg_files(self, game_dir):
        """RPG 게임 전용 파일"""
        # 캐릭터 클래스
        character_content = '''class Character:
    def __init__(self, name, hp=100, mp=50):
        self.name = name
        self.hp = hp
        self.mp = mp
        self.level = 1
        
    def attack(self):
        return 10 + self.level * 2
'''
        (game_dir / "character.py").write_text(character_content, encoding='utf-8')
        
        # 인벤토리
        inventory_content = '''class Inventory:
    def __init__(self):
        self.items = []
        self.gold = 0
        
    def add_item(self, item):
        self.items.append(item)
'''
        (game_dir / "inventory.py").write_text(inventory_content, encoding='utf-8')
        
        print("✓ RPG 전용 파일 생성 (character.py, inventory.py)")
        
    async def _create_platformer_files(self, game_dir):
        """플랫폼 게임 전용 파일"""
        # 플레이어 클래스
        player_content = '''class Player:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.velocity_y = 0
        self.on_ground = False
        
    def jump(self):
        if self.on_ground:
            self.velocity_y = -10
'''
        (game_dir / "player.py").write_text(player_content, encoding='utf-8')
        
        print("✓ 플랫폼 게임 전용 파일 생성 (player.py)")

async def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        game_type = sys.argv[1]
    else:
        # 게임 타입 선택
        print("\n🎮 어떤 게임을 만들고 싶으신가요?")
        print("\n선택 가능한 게임 타입:")
        print("  1. platformer - 플랫폼 게임")
        print("  2. racing     - 레이싱 게임")
        print("  3. rpg        - RPG 게임")
        print("  4. puzzle     - 퍼즐 게임")
        print("\n게임 타입을 입력하세요 (번호 또는 이름): ", end='')
        
        choice = input().strip().lower()
        
        game_type_map = {
            '1': 'platformer',
            '2': 'racing',
            '3': 'rpg',
            '4': 'puzzle'
        }
        
        if choice in game_type_map:
            game_type = game_type_map[choice]
        elif choice in ['platformer', 'racing', 'rpg', 'puzzle']:
            game_type = choice
        else:
            print("❌ 잘못된 선택입니다.")
            return
    
    # 게임 생성
    creator = SimpleGameCreator()
    await creator.create_game(game_type)

if __name__ == "__main__":
    asyncio.run(main())