#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# AutoCI 경로 추가
sys.path.append(str(Path(__file__).parent))

async def restart_improvement():
    try:
        from modules.persistent_game_improver import PersistentGameImprover
        
        # 최신 프로젝트 찾기
        mvp_dir = Path("mvp_games")
        if mvp_dir.exists():
            projects = sorted(mvp_dir.glob("rpg_*"), key=lambda x: x.stat().st_mtime)
            if projects:
                latest_project = projects[-1]
                print(f"🎮 재시작할 프로젝트: {latest_project}")
                
                # 개선 시스템 시작
                improver = PersistentGameImprover()
                await improver.start_24h_improvement(latest_project)
            else:
                print("❌ 재시작할 프로젝트를 찾을 수 없습니다")
        else:
            print("❌ mvp_games 디렉토리가 없습니다")
            
    except Exception as e:
        print(f"❌ 재시작 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(restart_improvement())
