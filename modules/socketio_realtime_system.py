"""
Socket.IO 실시간 통신 시스템
AutoCI의 실시간 멀티플레이어 및 모니터링 기능 지원
"""

import asyncio
import socketio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SocketIORealtimeSystem:
    """Socket.IO 기반 실시간 통신 시스템"""
    
    def __init__(self, port: int = 5001):
        """
        초기화
        
        Args:
            port: Socket.IO 서버 포트
        """
        self.port = port
        self.sio = socketio.AsyncServer(cors_allowed_origins='*')
        self.app = socketio.ASGIApp(self.sio)
        
        # 연결된 클라이언트 관리
        self.clients: Dict[str, Dict[str, Any]] = {}
        
        # 이벤트 핸들러
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # 게임 룸 관리
        self.game_rooms: Dict[str, Dict[str, Any]] = {}
        
        # 이벤트 설정
        self._setup_events()
        
        logger.info(f"Socket.IO 실시간 시스템 초기화 (포트: {port})")
    
    def _setup_events(self):
        """Socket.IO 이벤트 설정"""
        
        @self.sio.event
        async def connect(sid, environ):
            """클라이언트 연결"""
            logger.info(f"클라이언트 연결: {sid}")
            
            self.clients[sid] = {
                'connected_at': datetime.now(),
                'room': None,
                'player_data': {}
            }
            
            # 연결 확인 메시지
            await self.sio.emit('connected', {
                'sid': sid,
                'message': 'AutoCI 실시간 시스템에 연결되었습니다!'
            }, to=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """클라이언트 연결 해제"""
            logger.info(f"클라이언트 연결 해제: {sid}")
            
            # 게임 룸에서 제거
            if sid in self.clients:
                room = self.clients[sid].get('room')
                if room and room in self.game_rooms:
                    await self._leave_room(sid, room)
                
                del self.clients[sid]
        
        @self.sio.event
        async def join_game(sid, data):
            """게임 룸 참가"""
            room_id = data.get('room_id')
            player_name = data.get('player_name', f'Player_{sid[:8]}')
            
            if not room_id:
                await self.sio.emit('error', {
                    'message': '룸 ID가 필요합니다.'
                }, to=sid)
                return
            
            # 룸 생성 또는 참가
            if room_id not in self.game_rooms:
                self.game_rooms[room_id] = {
                    'created_at': datetime.now(),
                    'players': {},
                    'game_state': {},
                    'max_players': 4
                }
            
            room = self.game_rooms[room_id]
            
            # 최대 인원 체크
            if len(room['players']) >= room['max_players']:
                await self.sio.emit('error', {
                    'message': '룸이 가득 찼습니다.'
                }, to=sid)
                return
            
            # 룸 참가
            await self._join_room(sid, room_id, player_name)
        
        @self.sio.event
        async def leave_game(sid, data):
            """게임 룸 나가기"""
            room_id = data.get('room_id')
            
            if room_id and room_id in self.game_rooms:
                await self._leave_room(sid, room_id)
        
        @self.sio.event
        async def game_action(sid, data):
            """게임 액션 처리"""
            room_id = self.clients[sid].get('room')
            
            if not room_id or room_id not in self.game_rooms:
                return
            
            # 게임 액션 브로드캐스트
            await self.sio.emit('player_action', {
                'player_id': sid,
                'action': data.get('action'),
                'data': data.get('data'),
                'timestamp': datetime.now().isoformat()
            }, room=room_id, skip_sid=sid)
        
        @self.sio.event
        async def update_game_state(sid, data):
            """게임 상태 업데이트 (호스트만)"""
            room_id = self.clients[sid].get('room')
            
            if not room_id or room_id not in self.game_rooms:
                return
            
            room = self.game_rooms[room_id]
            
            # 호스트 권한 체크 (첫 번째 플레이어)
            if list(room['players'].keys())[0] != sid:
                return
            
            # 게임 상태 업데이트
            room['game_state'].update(data.get('state', {}))
            
            # 모든 플레이어에게 브로드캐스트
            await self.sio.emit('game_state_updated', {
                'state': room['game_state'],
                'timestamp': datetime.now().isoformat()
            }, room=room_id)
        
        @self.sio.event
        async def chat_message(sid, data):
            """채팅 메시지"""
            room_id = self.clients[sid].get('room')
            
            if not room_id:
                return
            
            player_name = self.clients[sid].get('player_data', {}).get('name', 'Unknown')
            
            # 채팅 브로드캐스트
            await self.sio.emit('chat_message', {
                'player_id': sid,
                'player_name': player_name,
                'message': data.get('message'),
                'timestamp': datetime.now().isoformat()
            }, room=room_id)
    
    async def _join_room(self, sid: str, room_id: str, player_name: str):
        """룸 참가 처리"""
        # Socket.IO 룸 참가
        self.sio.enter_room(sid, room_id)
        
        # 클라이언트 정보 업데이트
        self.clients[sid]['room'] = room_id
        self.clients[sid]['player_data'] = {
            'name': player_name,
            'joined_at': datetime.now()
        }
        
        # 룸 정보 업데이트
        room = self.game_rooms[room_id]
        room['players'][sid] = {
            'name': player_name,
            'joined_at': datetime.now(),
            'ready': False
        }
        
        # 참가 알림
        await self.sio.emit('player_joined', {
            'player_id': sid,
            'player_name': player_name,
            'total_players': len(room['players'])
        }, room=room_id)
        
        # 참가자에게 룸 정보 전송
        await self.sio.emit('room_info', {
            'room_id': room_id,
            'players': {
                pid: {'name': pdata['name'], 'ready': pdata['ready']}
                for pid, pdata in room['players'].items()
            },
            'game_state': room['game_state']
        }, to=sid)
        
        logger.info(f"플레이어 '{player_name}' 룸 '{room_id}' 참가")
    
    async def _leave_room(self, sid: str, room_id: str):
        """룸 나가기 처리"""
        # Socket.IO 룸 나가기
        self.sio.leave_room(sid, room_id)
        
        # 룸에서 플레이어 제거
        room = self.game_rooms[room_id]
        player_name = room['players'].get(sid, {}).get('name', 'Unknown')
        
        if sid in room['players']:
            del room['players'][sid]
        
        # 나가기 알림
        await self.sio.emit('player_left', {
            'player_id': sid,
            'player_name': player_name,
            'total_players': len(room['players'])
        }, room=room_id)
        
        # 빈 룸 제거
        if len(room['players']) == 0:
            del self.game_rooms[room_id]
            logger.info(f"빈 룸 '{room_id}' 제거")
        
        # 클라이언트 정보 업데이트
        self.clients[sid]['room'] = None
        
        logger.info(f"플레이어 '{player_name}' 룸 '{room_id}' 나감")
    
    async def broadcast_to_room(self, room_id: str, event: str, data: Any):
        """특정 룸에 이벤트 브로드캐스트"""
        if room_id in self.game_rooms:
            await self.sio.emit(event, data, room=room_id)
    
    async def send_to_player(self, player_id: str, event: str, data: Any):
        """특정 플레이어에게 이벤트 전송"""
        if player_id in self.clients:
            await self.sio.emit(event, data, to=player_id)
    
    async def update_game_metrics(self, metrics: Dict[str, Any]):
        """게임 메트릭 업데이트 (모니터링용)"""
        await self.sio.emit('game_metrics', {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    
    async def notify_ai_action(self, action: str, details: Dict[str, Any]):
        """AI 액션 알림 (개발 과정 시각화)"""
        await self.sio.emit('ai_action', {
            'action': action,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """룸 정보 조회"""
        if room_id in self.game_rooms:
            room = self.game_rooms[room_id]
            return {
                'room_id': room_id,
                'players': len(room['players']),
                'max_players': room['max_players'],
                'created_at': room['created_at'].isoformat(),
                'game_state': room['game_state']
            }
        return None
    
    def get_active_rooms(self) -> List[Dict[str, Any]]:
        """활성 룸 목록 조회"""
        return [
            self.get_room_info(room_id)
            for room_id in self.game_rooms
        ]
    
    async def start(self):
        """Socket.IO 서버 시작"""
        logger.info(f"🌐 Socket.IO 서버 시작 (포트: {self.port})")
        
        # 별도 스레드에서 서버 실행
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def stop(self):
        """Socket.IO 서버 중지"""
        logger.info("Socket.IO 서버 중지")


# 클라이언트 예제 코드
def create_test_client():
    """테스트용 Socket.IO 클라이언트 생성"""
    import socketio as socketio_client
    
    sio_client = socketio_client.Client()
    
    @sio_client.event
    def connect():
        print("서버에 연결되었습니다!")
        # 게임 룸 참가
        sio_client.emit('join_game', {
            'room_id': 'test_room',
            'player_name': 'TestPlayer'
        })
    
    @sio_client.event
    def connected(data):
        print(f"연결 확인: {data}")
    
    @sio_client.event
    def room_info(data):
        print(f"룸 정보: {data}")
    
    @sio_client.event
    def player_joined(data):
        print(f"플레이어 참가: {data}")
    
    @sio_client.event
    def game_state_updated(data):
        print(f"게임 상태 업데이트: {data}")
    
    @sio_client.event
    def disconnect():
        print("서버 연결이 끊어졌습니다.")
    
    return sio_client


if __name__ == "__main__":
    # 테스트 실행
    async def test_server():
        server = SocketIORealtimeSystem()
        await server.start()
    
    asyncio.run(test_server())