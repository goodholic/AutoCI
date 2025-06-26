#!/usr/bin/env python3
"""
Enhanced AutoCI 한국어 통합 시스템
기존 AutoCI 기능 + 고급 한국어 대화형 AI 통합
"""

import sys
import os
import cmd
import json
import re
import random
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import time

# 한국어 대화형 AI 모듈 import
try:
    from korean_conversational_ai import KoreanConversationalAI
except ImportError:
    # 인라인 클래스 정의 (fallback)
    class KoreanConversationalAI:
        def chat(self, text):
            return f"한국어 AI 모듈이 로드되지 않았습니다. 기본 응답: {text}"

class EnhancedAutoCI(cmd.Cmd):
    """한국어 대화형 AutoCI 통합 시스템"""
    
    intro = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║  🤖 Enhanced AutoCI v3.0 - 한국어 대화형 통합 시스템             ║
║                                                                  ║
║  ✨ 자연스러운 한국어 대화 처리                                   ║
║  🔧 24시간 자동 코드 수정 시스템                                  ║
║  🎮 Unity 전문 스크립트 관리                                     ║
║  🧠 AI 기반 의도 분석 및 맥락 이해                                ║
║                                                                  ║
║  💬 평소처럼 자연스럽게 대화하세요!                                ║
║     "유니티 파일 정리해줘", "코드 에러 찾아줘" 등                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    prompt = '🤖 Enhanced AutoCI> '
    
    def __init__(self):
        super().__init__()
        self.current_project = None
        self.korean_ai = KoreanConversationalAI()
        self.conversation_mode = True
        self.unity_folders = [
            "Scripts",
            "OX UI Scripts", 
            "InGame UI Scripts",
            "Editor"
        ]
        self.setup_system()
        
    def setup_system(self):
        """시스템 초기 설정"""
        self.session_start = datetime.now()
        self.commands_executed = 0
        self.auto_organize_enabled = True
        
        # 한국어 명령어 매핑 확장
        self.korean_commands = {
            # 기본 명령어
            '도움말': 'help', '도움': 'help', '명령어': 'help',
            '상태': 'status', '상태확인': 'status', '시스템상태': 'status',
            '프로젝트': 'project', '프로젝트설정': 'project',
            '분석': 'analyze', '코드분석': 'analyze', '검사': 'analyze',
            '개선': 'improve', '수정': 'improve', '코드개선': 'improve',
            '정리': 'organize', '파일정리': 'organize', '폴더정리': 'organize',
            '검색': 'search', '찾기': 'search', '찾아줘': 'search',
            '종료': 'exit', '나가기': 'exit', '끝': 'exit', '그만': 'exit',
            
            # Unity 특화 명령어
            '유니티정리': 'unity_organize',
            '스크립트정리': 'script_organize', 
            '스크립트이동': 'script_move',
            '에셋정리': 'asset_organize',
            
            # AI 기능
            '학습시작': 'start_learning',
            '학습상태': 'learning_status',
            '모니터링': 'monitor',
            '백업': 'backup',
            
            # 대화 모드
            '대화모드': 'conversation_mode',
            '명령모드': 'command_mode'
        }
        
        # 자연어 패턴 매핑
        self.natural_patterns = {
            r'.*?파일.*?정리.*?': 'organize',
            r'.*?스크립트.*?정리.*?': 'script_organize',
            r'.*?유니티.*?정리.*?': 'unity_organize',
            r'.*?코드.*?분석.*?': 'analyze',
            r'.*?버그.*?찾.*?': 'analyze',
            r'.*?에러.*?수정.*?': 'improve',
            r'.*?성능.*?개선.*?': 'improve',
            r'.*?프로젝트.*?설정.*?': 'project',
            r'.*?백업.*?': 'backup',
            r'.*?모니터.*?': 'monitor'
        }
    
    def default(self, line):
        """자연어 입력 처리 - 대화형 AI 통합"""
        line = line.strip()
        
        if not line:
            return
            
        # 한국어 명령어 직접 매핑 확인
        if line in self.korean_commands:
            command = self.korean_commands[line]
            print(f"✅ '{line}' → '{command}' 명령을 실행합니다...")
            self.onecmd(command)
            return
            
        # 자연어 패턴 매칭
        for pattern, command in self.natural_patterns.items():
            if re.search(pattern, line):
                print(f"🔍 자연어 패턴 감지: '{line}' → '{command}' 실행")
                
                # 파라미터 추출
                if command == 'project' and ('경로' in line or 'path' in line.lower()):
                    # 경로 추출 시도
                    path_match = re.search(r'[A-Za-z]?:?[/\\][^\s]+', line)
                    if path_match:
                        self.onecmd(f"{command} {path_match.group()}")
                        return
                        
                self.onecmd(command)
                return
        
        # 대화형 AI 처리
        if self.conversation_mode:
            print("🧠 AI가 생각하고 있습니다...")
            ai_response = self.korean_ai.chat(line)
            print(f"🤖 AI: {ai_response}")
            
            # AI 응답에서 실행 가능한 명령어 추출
            self.extract_and_execute_commands(line, ai_response)
        else:
            print(f"❓ '{line}'를 이해하지 못했습니다.")
            print("💡 '도움말'을 입력하여 사용 가능한 명령어를 확인하세요.")
    
    def extract_and_execute_commands(self, user_input, ai_response):
        """AI 응답에서 실행 가능한 명령어 추출 및 실행"""
        # 사용자 입력에서 실제 작업 의도 파악
        action_keywords = {
            'organize': ['정리', '청소', '정돈'],
            'analyze': ['분석', '검사', '확인', '점검'],
            'improve': ['개선', '수정', '고쳐', '향상'],
            'backup': ['백업', '저장', '보관'],
            'search': ['찾기', '검색', '찾아']
        }
        
        for command, keywords in action_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                print(f"🎯 작업 의도 감지: {command} 작업을 실행합니다...")
                
                # 사용자 확인
                confirm = input(f"   💡 '{command}' 작업을 실행하시겠습니까? (y/n): ").strip().lower()
                if confirm in ['y', 'yes', '네', '예', '응']:
                    self.onecmd(command)
                break
    
    def do_conversation_mode(self, arg):
        """대화 모드 활성화"""
        self.conversation_mode = True
        print("💬 대화 모드가 활성화되었습니다!")
        print("   이제 자연스러운 한국어로 대화할 수 있습니다.")
        
    def do_command_mode(self, arg):
        """명령 모드 활성화"""
        self.conversation_mode = False
        print("⚡ 명령 모드가 활성화되었습니다!")
        print("   정확한 명령어를 입력해주세요.")
    
    def do_help(self, arg):
        """도움말 표시 (한국어)"""
        help_text = """
🤖 Enhanced AutoCI 한국어 명령어 가이드

🗣️ 자연어 대화 (추천!):
  "유니티 파일 정리해줘"     - Unity 스크립트 폴더 정리
  "코드 에러 찾아줘"         - 프로젝트 전체 코드 분석
  "성능 개선해줘"           - 코드 자동 최적화
  "프로젝트 백업해줘"       - 프로젝트 안전 백업
  "스크립트 정리 필요해"     - Scripts 폴더 구조 최적화

📋 한국어 명령어:
  기본 명령어:
    도움말, 상태확인, 프로젝트 <경로>
    분석, 개선 <파일>, 정리, 검색 <키워드>
    
  Unity 전용:
    유니티정리      - Unity 프로젝트 전체 정리
    스크립트정리    - Scripts 폴더 정리
    에셋정리       - Assets 폴더 구조 최적화
    
  AI 기능:
    학습시작       - AI 백그라운드 학습 시작
    학습상태       - AI 학습 진행상황 확인
    모니터링       - 시스템 실시간 모니터링

🎮 Unity 특화 기능:
  ✅ Assets/Scripts, OX UI Scripts, InGame UI Scripts, Editor 자동 관리
  ✅ 잘못 배치된 스크립트 파일 감지 및 이동
  ✅ Unity 프로젝트 구조 최적화
  ✅ 실시간 파일 변경 모니터링

💡 사용 팁:
  • 자연스러운 한국어로 말하세요! AI가 의도를 파악합니다.
  • '대화모드'/'명령모드'로 인터페이스 변경 가능
  • 복잡한 작업은 단계별로 안내해드립니다.

🚀 고급 기능:
  • 24시간 자동 코드 개선
  • AI 기반 버그 예측 및 수정
  • 실시간 성능 최적화 제안
"""
        print(help_text)
    
    def do_status(self, arg):
        """시스템 상태 확인 (한국어)"""
        current_time = datetime.now()
        uptime = current_time - self.session_start
        
        print(f"\n📊 Enhanced AutoCI 시스템 상태:")
        print(f"  🟢 한국어 대화형 AI - {'활성화' if self.conversation_mode else '비활성화'}")
        print(f"  🟢 Unity 스크립트 관리 - 활성화")
        print(f"  🟢 자동 파일 정리 - {'활성화' if self.auto_organize_enabled else '비활성화'}")
        print(f"  ⏰ 시스템 가동시간 - {str(uptime).split('.')[0]}")
        print(f"  📈 실행된 명령 수 - {self.commands_executed}개")
        
        if self.current_project:
            print(f"\n📂 현재 프로젝트:")
            print(f"  📁 경로: {self.current_project}")
            if self.check_unity_project(Path(self.current_project)):
                print(f"  🎮 타입: Unity 프로젝트")
                self.show_unity_status()
            else:
                print(f"  💻 타입: 일반 프로젝트")
        else:
            print(f"\n📂 프로젝트: 설정되지 않음")
            
        # AI 학습 상태
        print(f"\n🧠 AI 학습 상태:")
        self.check_learning_status()
        print()
    
    def do_exit(self, arg):
        """프로그램 종료"""
        print("\n🤖 Enhanced AutoCI를 종료합니다...")
        print("👋 언제든 다시 와주세요! 좋은 코딩 되세요!")
        return True


if __name__ == "__main__":
    try:
        enhanced_autoci = EnhancedAutoCI()
        enhanced_autoci.cmdloop()
    except KeyboardInterrupt:
        print("\n\n🤖 Enhanced AutoCI가 중단되었습니다. 안녕히 가세요! 👋")
    except Exception as e:
        print(f"\n❌ 시스템 오류가 발생했습니다: {e}")
        print("💡 시스템을 재시작해주세요.") 