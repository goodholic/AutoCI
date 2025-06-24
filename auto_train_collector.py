import json
import os
import glob
import re
from datetime import datetime
import asyncio

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog 라이브러리가 설치되지 않았습니다. 파일 감시 기능을 사용할 수 없습니다.")

class UnityCodeCollector(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    def __init__(self, output_file="auto_training_data.json"):
        self.output_file = output_file
        self.training_data = []
        self.load_existing_data()
    
    def load_existing_data(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
    
    def save_data(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
    
    def extract_class_info(self, content):
        """C# 코드에서 클래스 정보 추출"""
        class_pattern = r'public\s+class\s+(\w+)'
        method_pattern = r'(public|private|protected)\s+\w+\s+(\w+)\s*\([^)]*\)'
        
        classes = re.findall(class_pattern, content)
        methods = re.findall(method_pattern, content)
        
        return classes, methods
    
    def generate_instruction(self, filename, content):
        """파일 내용을 기반으로 instruction 생성"""
        classes, methods = self.extract_class_info(content)
        
        instructions = []
        
        # 클래스 기반 instruction
        if classes:
            for class_name in classes:
                instructions.append({
                    "instruction": f"Unity에서 {class_name} 스크립트를 만들어줘",
                    "input": "",
                    "output": content
                })
        
        # 파일명 기반 instruction
        base_name = os.path.splitext(os.path.basename(filename))[0]
        instructions.append({
            "instruction": f"{base_name} 기능을 구현하는 Unity 스크립트 작성해줘",
            "input": "",
            "output": content
        })
        
        return instructions
    
    def on_created(self, event):
        if event.src_path.endswith('.cs'):
            self.process_file(event.src_path)
    
    def on_modified(self, event):
        if event.src_path.endswith('.cs'):
            self.process_file(event.src_path)
    
    def process_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Unity 스크립트인지 확인
            if 'using UnityEngine;' in content or 'using Unity' in content:
                instructions = self.generate_instruction(filepath, content)
                
                for instruction in instructions:
                    # 중복 체크
                    if not any(item['output'] == instruction['output'] for item in self.training_data):
                        self.training_data.append(instruction)
                        print(f"Added training data from: {filepath}")
                
                self.save_data()
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

class AutoTrainer:
    def __init__(self, watch_directory, model_path):
        self.watch_directory = watch_directory
        self.model_path = model_path
        self.collector = UnityCodeCollector()
        self.min_samples = 10  # 최소 학습 샘플 수
        self.last_train_count = 0
    
    async def check_and_train(self):
        """주기적으로 새로운 데이터 확인하고 학습"""
        while True:
            current_count = len(self.collector.training_data)
            
            # 새로운 데이터가 충분히 쌓였으면 학습
            if current_count - self.last_train_count >= self.min_samples:
                print(f"\n새로운 학습 데이터 {current_count - self.last_train_count}개 발견!")
                print("자동 학습을 시작합니다...")
                
                # 학습 실행
                await self.train_model()
                self.last_train_count = current_count
            
            # 30분마다 체크
            await asyncio.sleep(1800)
    
    async def train_model(self):
        """모델 학습 실행"""
        import subprocess
        
        # 학습 스크립트 실행
        cmd = ["python", "fine_tune.py", "--data", "auto_training_data.json"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print("✅ 모델 학습 완료!")
            
            # 서버 재시작을 위한 신호 파일 생성
            with open("model_updated.signal", "w") as f:
                f.write(str(datetime.now()))
        else:
            print(f"❌ 학습 실패: {stderr.decode()}")
    
    def start(self):
        """파일 감시 및 자동 학습 시작"""
        if not WATCHDOG_AVAILABLE:
            print("watchdog 라이브러리가 설치되지 않아 파일 감시를 시작할 수 없습니다.")
            print("수동으로 학습 데이터를 수집하려면 다음을 실행하세요:")
            print("python auto_train_collector.py --collect")
            return
            
        # 파일 감시 시작
        observer = Observer()
        observer.schedule(self.collector, self.watch_directory, recursive=True)
        observer.start()
        
        print(f"📁 감시 중: {self.watch_directory}")
        print("Unity C# 스크립트가 생성/수정되면 자동으로 학습 데이터로 수집됩니다.")
        
        # 비동기 학습 체크 시작
        try:
            asyncio.run(self.check_and_train())
        except KeyboardInterrupt:
            observer.stop()
            print("\n자동 학습 시스템 종료")
        observer.join()

# 실행
if __name__ == "__main__":
    # 현재 디렉토리를 감시
    unity_scripts_path = "/mnt/c/Users/super/Desktop/Unity Project(25년도 제작)/26.AutoCI/AutoCI"
    model_path = "/mnt/c/Users/super/Desktop/Unity Project(25년도 제작)/26.AutoCI/AutoCI/CodeLlama-7b-Instruct-hf"
    
    trainer = AutoTrainer(unity_scripts_path, model_path)
    trainer.start()