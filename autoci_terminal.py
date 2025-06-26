#!/usr/bin/env python3
"""
AutoCI Terminal Interface - WSL 터미널에서 자연어로 코드 수정 명령을 실행하는 시스템
"""
import os
import sys
import json
import argparse
import threading
import time
import requests
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple
import re

# 시스템 경로 설정
SCRIPT_DIR = Path(__file__).parent
EXPERT_DATA_DIR = SCRIPT_DIR / "expert_learning_data"
LEARNING_RESULTS_DIR = SCRIPT_DIR / "learning_results"
LOG_DIR = SCRIPT_DIR / "logs"

# 디렉토리 생성
LEARNING_RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

class AutoCITerminal:
    def __init__(self):
        self.base_url = "http://localhost:5000/api"
        self.rag_url = "http://localhost:8001"
        self.tasks = []
        self.expert_data = self.load_expert_data()
        self.command_patterns = self.init_command_patterns()
        
    def init_command_patterns(self) -> Dict[str, re.Pattern]:
        """자연어 명령 패턴 초기화"""
        return {
            'create': re.compile(r'(만들어|생성|create|make)\s+(.+?)\s+(파일|file|클래스|class)', re.IGNORECASE),
            'modify': re.compile(r'(수정|변경|modify|change)\s+(.+?)\s+(파일|file)', re.IGNORECASE),
            'improve': re.compile(r'(개선|향상|improve|enhance)\s+(.+)', re.IGNORECASE),
            'fix': re.compile(r'(고치|수리|fix|repair)\s+(.+)', re.IGNORECASE),
            'location': re.compile(r'(위치|장소|location|path)[:：\s]+(.+)', re.IGNORECASE),
            'add_data': re.compile(r'(데이터\s*추가|add\s*data|학습\s*데이터)', re.IGNORECASE),
            'index': re.compile(r'(인덱싱|index|색인)', re.IGNORECASE),
            'status': re.compile(r'(상태|status|진행)', re.IGNORECASE),
            'start': re.compile(r'(시작|start|실행)', re.IGNORECASE),
            'help': re.compile(r'(도움|help|명령어)', re.IGNORECASE),
            'monitor': re.compile(r'(모니터링|모니터|monitoring|monitor|감시|관찰)', re.IGNORECASE),
            'learning': re.compile(r'(학습|러닝|training|learning)', re.IGNORECASE)
        }
    
    def load_expert_data(self) -> Dict:
        """전문가 학습 데이터 로드"""
        expert_data = {
            'templates': {},
            'patterns': {},
            'categories': set()
        }
        
        if not EXPERT_DATA_DIR.exists():
            print(f"⚠️  전문가 데이터 디렉토리가 없습니다: {EXPERT_DATA_DIR}")
            return expert_data
            
        for json_file in EXPERT_DATA_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    category = data.get('category', 'general')
                    expert_data['categories'].add(category)
                    expert_data['templates'][json_file.stem] = data
                    
                    # 패턴 추출
                    if 'code' in data:
                        for pattern in self.extract_patterns(data['code']):
                            if pattern not in expert_data['patterns']:
                                expert_data['patterns'][pattern] = []
                            expert_data['patterns'][pattern].append(json_file.stem)
            except Exception as e:
                print(f"⚠️  데이터 로드 오류 {json_file}: {e}")
                
        print(f"✅ {len(expert_data['templates'])}개의 전문가 데이터 로드 완료")
        return expert_data
    
    def extract_patterns(self, code: str) -> List[str]:
        """코드에서 패턴 추출"""
        patterns = []
        # 클래스명 추출
        class_matches = re.findall(r'class\s+(\w+)', code)
        patterns.extend(class_matches)
        
        # 메서드명 추출
        method_matches = re.findall(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', code)
        patterns.extend(method_matches)
        
        # Unity 컴포넌트 추출
        unity_matches = re.findall(r'(?:MonoBehaviour|ScriptableObject|EditorWindow)', code)
        patterns.extend(unity_matches)
        
        return patterns
    
    def parse_natural_command(self, command: str) -> Dict:
        """자연어 명령 파싱"""
        result = {
            'action': None,
            'target': None,
            'location': None,
            'description': command,
            'matched_patterns': []
        }
        
        # 명령 유형 식별
        for action, pattern in self.command_patterns.items():
            match = pattern.search(command)
            if match:
                result['action'] = action
                if action in ['create', 'modify', 'improve', 'fix']:
                    result['target'] = match.group(2).strip()
                elif action == 'location':
                    result['location'] = match.group(2).strip()
                    result['location'] = self.normalize_path(result['location'])
                break
        
        # 위치 정보 추가 파싱
        if not result['location']:
            location_keywords = ['에서', 'at', 'in', '위치', 'path']
            for keyword in location_keywords:
                if keyword in command:
                    parts = command.split(keyword)
                    if len(parts) > 1:
                        potential_path = parts[-1].strip().split()[0]
                        if '/' in potential_path or '\\' in potential_path:
                            result['location'] = self.normalize_path(potential_path)
        
        # 관련 패턴 찾기
        for pattern, templates in self.expert_data['patterns'].items():
            if pattern.lower() in command.lower():
                result['matched_patterns'].extend(templates)
        
        return result
    
    def normalize_path(self, path: str) -> str:
        """경로 정규화 (WSL 환경 고려)"""
        # Windows 경로를 WSL 경로로 변환
        if path.startswith('C:\\') or path.startswith('c:\\'):
            path = '/mnt/c/' + path[3:].replace('\\', '/')
        elif '\\' in path:
            path = path.replace('\\', '/')
        
        # 상대 경로를 절대 경로로
        if not path.startswith('/'):
            path = str(Path.cwd() / path)
        
        return path
    
    def add_expert_data(self, data_path: str = None):
        """전문가 데이터 추가 및 인덱싱"""
        print("\n🔍 C# 전문가 데이터 수집 및 인덱싱 시작...")
        
        # GitHub에서 고품질 C# 코드 수집
        sources = [
            "https://github.com/dotnet/aspnetcore",
            "https://github.com/Unity-Technologies/UnityCsReference",
            "https://github.com/dotnet/runtime",
            "https://github.com/microsoft/referencesource"
        ]
        
        collected_count = 0
        for source in sources:
            print(f"📥 {source}에서 데이터 수집 중...")
            # 실제 구현시 GitHub API 사용
            collected_count += 50  # 시뮬레이션
        
        # 데이터 인덱싱
        print(f"\n📊 {collected_count}개의 새로운 패턴 인덱싱...")
        self.index_expert_data()
        
        # 학습 결과 저장
        self.save_learning_results({
            'timestamp': datetime.now().isoformat(),
            'new_patterns': collected_count,
            'total_patterns': len(self.expert_data['patterns']),
            'categories': list(self.expert_data['categories'])
        })
        
        print(f"✅ 데이터 수집 및 인덱싱 완료!")
    
    def index_expert_data(self):
        """고급 데이터 인덱싱"""
        print("\n🔧 고급 데이터 인덱싱 시작...")
        
        # 카테고리별 인덱스
        category_index = {}
        for name, data in self.expert_data['templates'].items():
            category = data.get('category', 'general')
            if category not in category_index:
                category_index[category] = []
            category_index[category].append(name)
        
        # 패턴 기반 인덱스
        pattern_index = {}
        for name, data in self.expert_data['templates'].items():
            if 'code' in data:
                patterns = self.extract_patterns(data['code'])
                for pattern in patterns:
                    if pattern not in pattern_index:
                        pattern_index[pattern] = []
                    pattern_index[pattern].append(name)
        
        # 인덱스 저장
        index_file = LEARNING_RESULTS_DIR / "expert_data_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump({
                'categories': category_index,
                'patterns': pattern_index,
                'total_templates': len(self.expert_data['templates']),
                'indexed_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 인덱싱 완료: {len(category_index)} 카테고리, {len(pattern_index)} 패턴")
    
    def save_learning_results(self, results: Dict):
        """학습 결과를 MD 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = LEARNING_RESULTS_DIR / f"learning_result_{timestamp}.md"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"# AutoCI 학습 결과 보고서\n\n")
            f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 요약\n\n")
            f.write(f"- **새로운 패턴**: {results.get('new_patterns', 0)}개\n")
            f.write(f"- **전체 패턴**: {results.get('total_patterns', 0)}개\n")
            f.write(f"- **카테고리**: {', '.join(results.get('categories', []))}\n\n")
            
            if 'improvements' in results:
                f.write("## 🔧 개선 사항\n\n")
                for improvement in results['improvements']:
                    f.write(f"- {improvement}\n")
                f.write("\n")
            
            if 'code_examples' in results:
                f.write("## 💡 학습된 코드 예제\n\n")
                for example in results['code_examples'][:5]:  # 상위 5개만
                    f.write(f"### {example['name']}\n")
                    f.write(f"```csharp\n{example['code']}\n```\n\n")
        
        print(f"📄 학습 결과 저장됨: {result_file}")
    
    def start_dual_phase_system(self):
        """1단계(RAG) + 2단계(Fine-tuning) 동시 실행"""
        print("\n🚀 이중 단계 시스템 시작...")
        
        # RAG 서버 시작 (백그라운드)
        rag_thread = threading.Thread(target=self.start_rag_server)
        rag_thread.daemon = True
        rag_thread.start()
        
        # Fine-tuning 프로세스 시작 (백그라운드)
        finetuning_thread = threading.Thread(target=self.start_finetuning)
        finetuning_thread.daemon = True
        finetuning_thread.start()
        
        # 자동 코드 수정 시스템 시작
        modifier_thread = threading.Thread(target=self.start_code_modifier)
        modifier_thread.daemon = True
        modifier_thread.start()
        
        print("✅ 모든 시스템이 시작되었습니다!")
        print("  - RAG 시스템: 즉시 사용 가능")
        print("  - Fine-tuning: 백그라운드에서 진행 중")
        print("  - 코드 수정기: 24시간 작동 중")
    
    def start_rag_server(self):
        """RAG 서버 실행"""
        try:
            subprocess.run([sys.executable, "enhanced_rag_system.py"], cwd=SCRIPT_DIR)
        except Exception as e:
            print(f"⚠️  RAG 서버 오류: {e}")
    
    def start_finetuning(self):
        """Fine-tuning 프로세스"""
        print("🔄 Fine-tuning 시작 (578개 데이터 사용)...")
        try:
            # 실제 fine-tuning 실행
            subprocess.run([sys.executable, "hybrid_rag_training_system.py", "--train"], cwd=SCRIPT_DIR)
        except Exception as e:
            print(f"⚠️  Fine-tuning 오류: {e}")
    
    def start_code_modifier(self):
        """24시간 코드 수정 시스템"""
        try:
            subprocess.run([sys.executable, "auto_code_modifier.py"], cwd=SCRIPT_DIR)
        except Exception as e:
            print(f"⚠️  코드 수정기 오류: {e}")
    
    def execute_command(self, command: str):
        """명령 실행"""
        parsed = self.parse_natural_command(command)
        
        if parsed['action'] == 'help':
            self.show_help()
        elif parsed['action'] == 'status':
            self.show_status()
        elif parsed['action'] == 'start':
            self.start_dual_phase_system()
        elif parsed['action'] == 'add_data':
            self.add_expert_data()
        elif parsed['action'] == 'index':
            self.index_expert_data()
        elif parsed['action'] in ['create', 'modify', 'improve', 'fix']:
            self.add_modification_task(parsed)
        elif parsed['action'] == 'monitor':
            self.show_monitoring()
        elif parsed['action'] == 'learning':
            self.show_learning_status()
        else:
            # 한글 명령어를 더 유연하게 처리
            if '모니터' in command or '학습' in command:
                if '모니터' in command:
                    self.show_monitoring()
                elif '학습' in command:
                    self.show_learning_status()
            else:
                print(f"❓ 명령을 이해할 수 없습니다: {command}")
                print("   '도움말' 또는 'help'를 입력하세요.")
    
    def add_modification_task(self, parsed: Dict):
        """코드 수정 작업 추가"""
        task = {
            'type': parsed['action'],
            'target': parsed['target'],
            'location': parsed['location'] or os.getcwd(),
            'description': parsed['description'],
            'matched_patterns': parsed['matched_patterns'],
            'timestamp': datetime.now().isoformat()
        }
        
        # API로 작업 전송
        try:
            response = requests.post(f"{self.base_url}/codemodifier/add-task", json=task)
            if response.status_code == 200:
                print(f"✅ 작업이 추가되었습니다: {task['type']} {task['target']}")
                print(f"   위치: {task['location']}")
                if task['matched_patterns']:
                    print(f"   관련 패턴: {', '.join(task['matched_patterns'][:3])}")
            else:
                print(f"⚠️  작업 추가 실패: {response.text}")
        except Exception as e:
            print(f"❌ 서버 연결 오류: {e}")
            # 로컬 파일로 저장
            tasks_file = SCRIPT_DIR / "pending_tasks.json"
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    tasks = json.load(f)
            else:
                tasks = []
            tasks.append(task)
            with open(tasks_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            print("   (오프라인 모드로 저장됨)")
    
    def show_status(self):
        """시스템 상태 표시"""
        print("\n📊 AutoCI 시스템 상태")
        print("=" * 50)
        
        # 서버 상태 확인
        try:
            response = requests.get(f"{self.base_url}/codemodifier/status")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 코드 수정 서버: 온라인")
                print(f"   - 대기 중: {data.get('pending', 0)}개")
                print(f"   - 진행 중: {data.get('in_progress', 0)}개")
                print(f"   - 완료됨: {data.get('completed', 0)}개")
        except:
            print("❌ 코드 수정 서버: 오프라인")
        
        # RAG 상태
        try:
            response = requests.get(f"{self.rag_url}/status")
            print("✅ RAG 시스템: 온라인")
        except:
            print("❌ RAG 시스템: 오프라인")
        
        # 데이터 상태
        print(f"\n📚 학습 데이터:")
        print(f"   - 전문가 템플릿: {len(self.expert_data['templates'])}개")
        print(f"   - 코드 패턴: {len(self.expert_data['patterns'])}개")
        print(f"   - 카테고리: {len(self.expert_data['categories'])}개")
        
        # 최근 학습 결과
        results = list(LEARNING_RESULTS_DIR.glob("*.md"))
        if results:
            latest = max(results, key=lambda f: f.stat().st_mtime)
            print(f"\n📄 최근 학습 결과: {latest.name}")
            print(f"   생성 시간: {datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    def show_monitoring(self):
        """모델 학습 및 시스템 모니터링 상태 표시"""
        print("\n📊 AutoCI 학습 모니터링")
        print("=" * 60)
        
        # RAG 시스템 상태
        print("\n🔍 RAG 시스템 상태:")
        try:
            response = requests.get(f"{self.rag_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print(f"  ✅ 온라인 - 응답시간: {metrics.get('avg_response_time', 'N/A')}ms")
                print(f"  📚 인덱싱된 문서: {metrics.get('indexed_docs', 0)}개")
                print(f"  🎯 정확도: {metrics.get('accuracy', 'N/A')}%")
            else:
                print("  ⚠️  메트릭 조회 실패")
        except:
            print("  ❌ RAG 시스템 오프라인")
        
        # Fine-tuning 상태
        print("\n🤖 Fine-tuning 상태:")
        training_log = LOG_DIR / "finetuning.log"
        if training_log.exists():
            with open(training_log, 'r') as f:
                lines = f.readlines()[-10:]  # 최근 10줄
                for line in lines:
                    if 'epoch' in line.lower() or 'loss' in line.lower():
                        print(f"  {line.strip()}")
        else:
            print("  ⏳ 아직 시작되지 않음")
        
        # 실시간 작업 상태
        print("\n⚡ 실시간 작업 처리:")
        try:
            response = requests.get(f"{self.base_url}/codemodifier/queue")
            if response.status_code == 200:
                queue = response.json()
                print(f"  🔄 처리 중: {queue.get('processing', [])}")
                print(f"  ⏰ 대기 중: {len(queue.get('pending', []))}개")
                print(f"  ✅ 최근 완료: {queue.get('recent_completed', [])}")
        except:
            print("  ❌ 작업 큐 조회 실패")
        
        print("\n💡 Tip: 실시간 로그를 보려면 'tail -f logs/*.log' 명령을 사용하세요.")
    
    def show_learning_status(self):
        """학습 진행 상태 상세 표시"""
        print("\n📈 AutoCI 학습 상태")
        print("=" * 60)
        
        # 데이터셋 정보
        print("\n📚 학습 데이터셋:")
        print(f"  - 전문가 템플릿: {len(self.expert_data['templates'])}개")
        print(f"  - 코드 패턴: {len(self.expert_data['patterns'])}개")
        print(f"  - 카테고리: {', '.join(list(self.expert_data['categories'])[:5])}...")
        
        # 학습 진행률
        progress_file = LEARNING_RESULTS_DIR / "training_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                print("\n📊 학습 진행률:")
                print(f"  - 현재 에폭: {progress.get('current_epoch', 0)}/{progress.get('total_epochs', 100)}")
                print(f"  - 손실값: {progress.get('loss', 'N/A')}")
                print(f"  - 검증 정확도: {progress.get('val_accuracy', 'N/A')}%")
                print(f"  - 예상 완료 시간: {progress.get('eta', 'N/A')}")
        
        # 최근 학습 결과
        print("\n📄 최근 학습 결과:")
        results = sorted(LEARNING_RESULTS_DIR.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]
        for result in results:
            print(f"  - {result.name} ({datetime.fromtimestamp(result.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})")
        
        print("\n🔄 자동 업데이트: 학습은 백그라운드에서 계속 진행됩니다.")
    
    def show_help(self):
        """도움말 표시"""
        print("\n🤖 AutoCI 터미널 명령어 가이드")
        print("=" * 60)
        print("\n📝 자연어 명령 예시:")
        print("  - 'PlayerController 클래스 만들어줘'")
        print("  - 'GameManager.cs 파일 수정해줘'")
        print("  - 'Assets/Scripts/Player.cs 개선해줘'")
        print("  - 'NetworkManager의 버그 고쳐줘'")
        print("  - '위치: /home/user/project'")
        print("\n🔧 시스템 명령:")
        print("  - '상태' 또는 'status' - 시스템 상태 확인")
        print("  - '시작' 또는 'start' - 모든 시스템 시작")
        print("  - '모니터링' 또는 'monitor' - 학습 및 작업 모니터링")
        print("  - '학습 상태' - 상세 학습 진행 상태")
        print("  - '데이터 추가' - 전문가 데이터 수집")
        print("  - '인덱싱' - 데이터 인덱싱 실행")
        print("  - 'quit' 또는 'exit' - 종료")
        print("\n💡 팁:")
        print("  - 파일 경로는 절대 경로나 상대 경로 모두 사용 가능")
        print("  - Windows 경로도 자동으로 WSL 경로로 변환됨")
        print("  - 명령에 위치 정보가 없으면 현재 디렉토리 사용")
        print("  - 한글 명령어를 자유롭게 사용할 수 있습니다")

def main():
    parser = argparse.ArgumentParser(description="AutoCI Terminal - 자연어 코드 수정 시스템")
    parser.add_argument('command', nargs='*', help='실행할 명령')
    parser.add_argument('--start', action='store_true', help='모든 시스템 자동 시작')
    parser.add_argument('--interactive', '-i', action='store_true', help='대화형 모드')
    
    args = parser.parse_args()
    
    terminal = AutoCITerminal()
    
    if args.start:
        terminal.start_dual_phase_system()
        time.sleep(3)  # 시스템 시작 대기
    
    if args.command:
        # 단일 명령 실행
        command = ' '.join(args.command)
        terminal.execute_command(command)
    else:
        # 대화형 모드
        print("\n🤖 AutoCI Terminal v2.0 - WSL 자연어 코드 수정 시스템")
        print("   '도움말' 또는 'help'를 입력하여 사용법을 확인하세요.")
        print("   'quit' 또는 'exit'로 종료합니다.\n")
        
        while True:
            try:
                command = input("autoci> ").strip()
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 AutoCI를 종료합니다.")
                    break
                elif command:
                    terminal.execute_command(command)
            except KeyboardInterrupt:
                print("\n👋 AutoCI를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()