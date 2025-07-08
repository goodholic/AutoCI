#!/usr/bin/env python3
"""
게임 개발 세션 관리자
게임 개발 상태를 저장하고 복원하는 시스템
"""

import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class GameSession:
    """게임 개발 세션 정보"""
    session_id: str
    game_type: str
    game_name: str
    created_at: str
    last_modified: str
    status: str  # 'active', 'paused', 'completed'
    progress: Dict[str, Any]
    features: List[str]
    files: List[str]
    metadata: Dict[str, Any]

class GameSessionManager:
    """게임 개발 세션 관리자"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """초기화"""
        if base_dir is None:
            # Platform-specific default path
            if platform.system() == "Windows":
                base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            else:
                base_dir = Path("/mnt/d/AutoCI/AutoCI")
        
        self.base_dir = base_dir
        self.sessions_dir = self.base_dir / "game_sessions"
        self.active_sessions_file = self.sessions_dir / "active_sessions.json"
        
        # 디렉토리 생성
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # 활성 세션 로드
        self.active_sessions = self._load_active_sessions()
        
    def _load_active_sessions(self) -> Dict[str, str]:
        """활성 세션 목록 로드"""
        if self.active_sessions_file.exists():
            try:
                with open(self.active_sessions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"활성 세션 로드 실패: {e}")
        return {}
    
    def _save_active_sessions(self):
        """활성 세션 목록 저장"""
        try:
            with open(self.active_sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.active_sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"활성 세션 저장 실패: {e}")
    
    def create_session(self, game_type: str, game_name: str) -> GameSession:
        """새 게임 개발 세션 생성"""
        session_id = f"{game_type}_{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = GameSession(
            session_id=session_id,
            game_type=game_type,
            game_name=game_name,
            created_at=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat(),
            status='active',
            progress={
                'stage': 'initialization',
                'completed_tasks': [],
                'current_task': None,
                'completion_percentage': 0
            },
            features=[],
            files=[],
            metadata={
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'autoci_version': '5.0'
            }
        )
        
        # 세션 디렉토리 생성
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # 세션 정보 저장
        self.save_session(session)
        
        # 활성 세션으로 등록
        self.active_sessions[game_type] = session_id
        self._save_active_sessions()
        
        logger.info(f"새 게임 개발 세션 생성: {session_id}")
        return session
    
    def save_session(self, session: GameSession):
        """세션 상태 저장"""
        session_dir = self.sessions_dir / session.session_id
        session_file = session_dir / "session.json"
        
        # 마지막 수정 시간 업데이트
        session.last_modified = datetime.now().isoformat()
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, ensure_ascii=False)
            logger.info(f"세션 저장 완료: {session.session_id}")
        except Exception as e:
            logger.error(f"세션 저장 실패: {e}")
    
    def load_session(self, session_id: str) -> Optional[GameSession]:
        """세션 상태 로드"""
        session_file = self.sessions_dir / session_id / "session.json"
        
        if not session_file.exists():
            logger.error(f"세션 파일을 찾을 수 없습니다: {session_id}")
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return GameSession(**data)
        except Exception as e:
            logger.error(f"세션 로드 실패: {e}")
            return None
    
    def get_active_session(self, game_type: str) -> Optional[GameSession]:
        """특정 게임 타입의 활성 세션 가져오기"""
        if game_type in self.active_sessions:
            session_id = self.active_sessions[game_type]
            return self.load_session(session_id)
        return None
    
    def list_sessions(self, status: Optional[str] = None) -> List[GameSession]:
        """모든 세션 목록 가져오기"""
        sessions = []
        
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir() and session_dir.name != "active_sessions.json":
                session = self.load_session(session_dir.name)
                if session:
                    if status is None or session.status == status:
                        sessions.append(session)
        
        # 최근 수정 순으로 정렬
        sessions.sort(key=lambda s: s.last_modified, reverse=True)
        return sessions
    
    def update_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """세션 진행 상황 업데이트"""
        session = self.load_session(session_id)
        if session:
            session.progress.update(progress_data)
            self.save_session(session)
    
    def add_feature(self, session_id: str, feature: str):
        """세션에 기능 추가"""
        session = self.load_session(session_id)
        if session and feature not in session.features:
            session.features.append(feature)
            self.save_session(session)
    
    def add_file(self, session_id: str, file_path: str):
        """세션에 파일 추가"""
        session = self.load_session(session_id)
        if session and file_path not in session.files:
            session.files.append(file_path)
            self.save_session(session)
    
    def pause_session(self, session_id: str):
        """세션 일시 정지"""
        session = self.load_session(session_id)
        if session:
            session.status = 'paused'
            self.save_session(session)
    
    def resume_session(self, session_id: str):
        """세션 재개"""
        session = self.load_session(session_id)
        if session:
            session.status = 'active'
            self.save_session(session)
            
            # 활성 세션으로 등록
            self.active_sessions[session.game_type] = session_id
            self._save_active_sessions()
    
    def complete_session(self, session_id: str):
        """세션 완료"""
        session = self.load_session(session_id)
        if session:
            session.status = 'completed'
            session.progress['completion_percentage'] = 100
            self.save_session(session)
            
            # 활성 세션에서 제거
            if session.game_type in self.active_sessions:
                del self.active_sessions[session.game_type]
                self._save_active_sessions()
    
    def export_session(self, session_id: str, export_path: Path):
        """세션 데이터 내보내기"""
        session_dir = self.sessions_dir / session_id
        if session_dir.exists():
            shutil.make_archive(str(export_path), 'zip', session_dir)
            logger.info(f"세션 내보내기 완료: {export_path}.zip")
    
    def import_session(self, import_path: Path) -> Optional[str]:
        """세션 데이터 가져오기"""
        try:
            # 임시 디렉토리에 압축 해제
            temp_dir = self.sessions_dir / "temp_import"
            shutil.unpack_archive(str(import_path), temp_dir)
            
            # session.json 찾기
            session_file = temp_dir / "session.json"
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                session_id = data['session_id']
                
                # 세션 디렉토리로 이동
                target_dir = self.sessions_dir / session_id
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.move(str(temp_dir), str(target_dir))
                
                logger.info(f"세션 가져오기 완료: {session_id}")
                return session_id
            
        except Exception as e:
            logger.error(f"세션 가져오기 실패: {e}")
        
        finally:
            # 임시 디렉토리 정리
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return None