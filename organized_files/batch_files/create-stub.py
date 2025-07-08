#!/usr/bin/env python3
"""
AutoCI Create 명령 임시 실행 스크립트
모듈 임포트 문제를 우회하여 기본 기능 제공
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

def create_game(game_type):
    """간단한 게임 생성 기능"""
    print(f"\n🎮 {game_type} 게임 생성을 시작합니다...")
    print("📌 이것은 간단한 임시 버전입니다.")
    
    # 게임 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    game_name = f"{game_type}_game_{timestamp}"
    game_dir = Path(f"games/{game_name}")
    game_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본 구조 생성
    print(f"📁 게임 디렉토리 생성: {game_dir}")
    
    # 메인 파일 생성
    main_content = f'''#!/usr/bin/env python3
"""
{game_name} - AutoCI로 생성된 {game_type} 게임
생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys
print("🎮 {game_name} 시작!")
print("이 게임은 AutoCI로 자동 생성되었습니다.")

# TODO: 게임 로직 구현
'''
    
    main_file = game_dir / "main.py"
    main_file.write_text(main_content, encoding='utf-8')
    print(f"✓ 메인 파일 생성: {main_file}")
    
    # 설정 파일 생성
    config = {
        "game_name": game_name,
        "game_type": game_type,
        "created_at": datetime.now().isoformat(),
        "version": "0.1.0",
        "status": "development"
    }
    
    config_file = game_dir / "game_config.json"
    config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✓ 설정 파일 생성: {config_file}")
    
    # README 생성
    readme_content = f'''# {game_name}

AutoCI로 자동 생성된 {game_type} 게임입니다.

## 게임 정보
- 타입: {game_type}
- 생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 버전: 0.1.0

## 실행 방법
```
python main.py
```

## 개발 상태
현재 기본 구조만 생성되었습니다.
'''
    
    readme_file = game_dir / "README.md"
    readme_file.write_text(readme_content, encoding='utf-8')
    print(f"✓ README 파일 생성: {readme_file}")
    
    print(f"\n✅ {game_type} 게임이 성공적으로 생성되었습니다!")
    print(f"📂 위치: {game_dir.absolute()}")
    
    return game_dir

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
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
    else:
        game_type = sys.argv[1]
    
    # 게임 생성
    try:
        create_game(game_type)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()