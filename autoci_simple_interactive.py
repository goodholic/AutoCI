#!/usr/bin/env python3
"""
AutoCI 간단한 대화형 인터페이스 (의존성 최소화)
한국어 응답 테스트용
"""

import os
import sys
import cmd
from pathlib import Path

class AutoCISimpleShell(cmd.Cmd):
    """AutoCI 간단한 대화형 셸"""
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🤖 AutoCI - 24시간 자동 코드 수정 시스템 (테스트 모드)        ║
║                                                              ║
║  ✅ 한국어 대화 지원 활성화                                   ║
║  ✅ Unity 스크립트 분석 준비됨                                ║
║                                                              ║
║  💬 "안녕"이라고 말해보세요!                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    prompt = '🤖 autoci> '
    
    def __init__(self):
        super().__init__()
        self.current_project = None
        
    def default(self, line):
        """알 수 없는 명령 처리 - 한국어 지원"""
        line = line.strip()
        
        # 한국어 인사말 및 일반적인 표현 처리
        korean_greetings = {
            '안녕': '안녕하세요! 👋 AutoCI 시스템에 오신 것을 환영합니다!\n저는 24시간 코드를 자동으로 개선해드리는 AI입니다. 어떤 도움이 필요하신가요?',
            '안녕하세요': '안녕하세요! 😊 반갑습니다! AutoCI와 함께 코드 품질을 향상시켜보세요!',
            '반가워': '저도 반가워요! 🤗 코딩 작업에서 어떤 도움이 필요하신지 말씀해주세요.',
            '고마워': '천만에요! 😊 언제든지 도움이 필요하시면 말씀해주세요!',
            '고맙습니다': '별말씀을요! 🙏 더 필요한 것이 있으면 언제든 말씀해주세요.',
            '잘했어': '감사합니다! 😄 더 나은 서비스를 위해 계속 발전하고 있어요!',
            '좋아': '기뻐요! 👍 계속해서 좋은 코드를 만들어나가요!',
            '네': '네! 무엇을 도와드릴까요? 🤔',
            '응': '네, 말씀하세요! ✨',
            '음': '어떤 생각을 하고 계신가요? 코드 관련해서 궁금한 것이 있으시면 언제든 물어보세요! 💭',
            '하이': '하이! 👋 반가워요! 오늘 어떤 코드 작업을 도와드릴까요?',
            '헬로': '헬로! 😄 환영합니다! Unity 프로젝트나 C# 코드 개선에 도움이 필요하시면 말씀해주세요!'
        }
        
        # 한국어 명령어 매핑
        korean_commands = {
            '도움말': 'help',
            '도움': 'help',
            '명령어': 'help',
            '상태': 'status',
            '상태확인': 'status',
            '프로젝트': 'project',
            '분석': 'analyze',
            '개선': 'improve',
            '검색': 'search',
            '찾기': 'search',
            '리포트': 'report',
            '보고서': 'report',
            '모니터링': 'monitor',
            '모니터': 'monitor',
            '종료': 'exit',
            '나가기': 'exit',
            '끝': 'exit',
            '그만': 'exit',
            '정리': 'organize'
        }
        
        # 인사말 처리
        if line.lower() in korean_greetings:
            print(f"\n🎉 {korean_greetings[line.lower()]}")
            print(f"\n💡 주요 명령어:")
            print(f"   • 프로젝트 <경로> - Unity 프로젝트 설정")
            print(f"   • 분석 - 코드 분석")
            print(f"   • 개선 <파일> - 코드 자동 개선")
            print(f"   • 정리 - Unity 스크립트 폴더 정리")
            print(f"   • 도움말 - 전체 명령어 보기")
            print()
            return
            
        # 한국어 명령어 변환
        if line in korean_commands:
            english_cmd = korean_commands[line]
            print(f"✅ '{line}' → '{english_cmd}' 명령을 실행합니다...")
            self.onecmd(english_cmd)
            return
            
        # 질문이나 대화형 입력 감지
        conversation_patterns = ['어떻게', '뭐야', '무엇', '왜', '언제', '어디서', '누가', '어느', '몇', '?', '？']
        if any(pattern in line for pattern in conversation_patterns):
            print(f"🤔 '{line}'에 대해 생각해보고 있어요...")
            print("💡 더 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있어요!")
            print("   예: '유니티 스크립트를 어떻게 정리하나요?'")
            return
        
        # 기본 응답
        print(f"😅 '{line}'는 아직 이해하지 못하겠어요.")
        print(f"💡 '도움말' 또는 'help'를 입력하시면 사용 가능한 명령어를 볼 수 있어요!")
        print()
            
    def do_help(self, arg):
        """도움말 표시"""
        help_text = """
🤖 AutoCI 명령어 가이드

🗣️ 한국어 인사 및 대화:
  안녕, 안녕하세요, 하이     - AI와 인사하기
  고마워, 네, 응, 좋아       - 자연스러운 대화
  
📋 한국어 명령어:
  도움말, 도움         - 이 도움말 표시
  상태, 상태확인       - 시스템 상태 확인
  프로젝트 <경로>      - Unity 프로젝트 설정
  분석 [파일]         - 코드 분석
  개선 <파일>         - 코드 자동 개선
  정리               - Unity 스크립트 폴더 정리
  검색, 찾기 <검색어>  - 코드/패턴 검색
  종료, 나가기, 끝    - 프로그램 종료

🎮 Unity 특화 기능:
  • Assets/Scripts, OX UI Scripts, InGame UI Scripts, Editor 폴더 관리
  • 잘못 배치된 스크립트 파일 자동 감지
  • 스크립트 폴더 간 이동 파일 검사
  • Unity 프로젝트 백업 및 자동 정리

📝 사용 예시:
  안녕                    - AI와 인사하기
  프로젝트 C:/Unity/Game  - Unity 프로젝트 설정
  분석                   - 전체 프로젝트 분석
  정리                   - 스크립트 폴더 정리

💡 팁: 자연스러운 한국어로 대화할 수 있습니다!
"""
        print(help_text)
        
    def do_status(self, arg):
        """시스템 상태 확인"""
        print("\n📊 시스템 상태:")
        print("  🟢 한국어 대화 - 활성화")
        print("  🟢 Unity 분석 - 준비됨")
        print("  🟡 RAG 시스템 - 대기 중")
        print("  🟡 백그라운드 학습 - 대기 중")
        
        if self.current_project:
            print(f"\n📂 현재 프로젝트: {self.current_project}")
        else:
            print("\n📂 프로젝트: 설정되지 않음")
        print()
        
    def do_project(self, arg):
        """프로젝트 설정"""
        if not arg:
            print("📁 프로젝트 경로를 입력해주세요.")
            print("   예시: 프로젝트 C:/Unity/MyGame")
            return
            
        project_path = Path(arg)
        
        if not project_path.exists():
            print(f"❌ 경로가 존재하지 않습니다: {project_path}")
            return
            
        self.current_project = project_path
        print(f"✅ 프로젝트 설정됨: {project_path}")
        
        # Unity 프로젝트 확인
        if self.check_unity_project(project_path):
            print("🎮 Unity 프로젝트를 감지했습니다!")
            self.analyze_unity_structure(project_path)
        else:
            print("📁 일반 프로젝트로 인식됩니다.")
        print()
        
    def check_unity_project(self, path: Path) -> bool:
        """Unity 프로젝트 여부 확인"""
        unity_indicators = ['Assets', 'ProjectSettings', 'Packages']
        return all((path / indicator).exists() for indicator in unity_indicators)
        
    def analyze_unity_structure(self, project_path: Path):
        """Unity 프로젝트 구조 분석"""
        assets_path = project_path / "Assets"
        
        important_folders = [
            "Scripts",
            "OX UI Scripts", 
            "InGame UI Scripts",
            "Editor"
        ]
        
        print("\n🔍 Unity Assets 폴더 구조 분석:")
        
        found_folders = []
        missing_folders = []
        
        for folder in important_folders:
            folder_path = assets_path / folder
            if folder_path.exists():
                found_folders.append(folder)
                script_count = len(list(folder_path.rglob("*.cs")))
                print(f"  ✅ {folder} - {script_count}개 스크립트")
            else:
                missing_folders.append(folder)
                print(f"  ❓ {folder} - 폴더 없음")
        
        if found_folders:
            print(f"\n📂 발견된 스크립트 폴더: {len(found_folders)}개")
            
        if missing_folders:
            print(f"⚠️  누락된 폴더: {', '.join(missing_folders)}")
            
        print("💡 '정리' 명령으로 스크립트 폴더를 자동 정리할 수 있습니다.")
        
    def do_analyze(self, arg):
        """코드 분석"""
        if not self.current_project:
            print("❌ 먼저 프로젝트를 설정해주세요.")
            print("   사용법: 프로젝트 <경로>")
            return
            
        print("🔍 코드 분석을 시작합니다...")
        print("✅ 분석 완료! (테스트 모드)")
        print()
        
    def do_organize(self, arg):
        """Unity 스크립트 정리"""
        if not self.current_project:
            print("❌ 먼저 프로젝트를 설정해주세요.")
            return
            
        print("🧹 Unity 스크립트 폴더 정리를 시작합니다...")
        print("✅ 정리 완료! (테스트 모드)")
        print()
        
    def do_exit(self, arg):
        """종료"""
        print("\n👋 AutoCI를 이용해주셔서 감사합니다!")
        print("🚀 더 나은 코드와 함께 돌아오세요!")
        return True
        
    def do_quit(self, arg):
        """종료"""
        return self.do_exit(arg)
        
    def emptyline(self):
        """빈 줄 입력 시"""
        pass
        
    def postcmd(self, stop, line):
        """명령 실행 후"""
        return stop


def main():
    """메인 함수"""
    try:
        # 대화형 셸 시작
        shell = AutoCISimpleShell()
        shell.cmdloop()
        
    except KeyboardInterrupt:
        print("\n\n👋 종료합니다...")
    except Exception as e:
        print(f"\n❌ 오류: {e}")


if __name__ == "__main__":
    main()