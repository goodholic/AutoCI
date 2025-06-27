#!/usr/bin/env python3
"""
WSL용 안정적인 24시간 지속 학습 시스템
중지되지 않고 계속 실행되는 견고한 버전
"""

import os
import sys
import time
import json
import signal
import random
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class LearningStats:
    """학습 통계"""
    total_training_hours: float = 0.0
    data_collected: int = 0
    models_trained: int = 0
    accuracy_improvements: int = 0
    last_training_time: str = ""
    current_accuracy: float = 0.0
    learning_rate: float = 0.001
    batch_size: int = 32
    uptime_minutes: int = 0
    
class Stable24HLearning:
    """안정적인 24시간 지속 학습 시스템"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.log_file = self.base_dir / "stable_learning.log"
        self.stats_file = self.base_dir / "stable_learning_stats.json"
        self.pid_file = self.base_dir / "stable_learning.pid"
        
        # 로깅 설정 (더 견고하게)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Stable24H')
        
        # 통계 로드
        self.stats = self.load_stats()
        
        # 종료 신호 처리
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.running = True
        self.start_time = time.time()
        
        # PID 저장
        self.save_pid()
    
    def load_stats(self) -> LearningStats:
        """학습 통계 로드"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return LearningStats(**data)
            except Exception as e:
                print(f"⚠️ 통계 로드 실패: {e}")
        return LearningStats()
    
    def save_stats(self):
        """학습 통계 저장"""
        try:
            # 업타임 계산
            uptime_seconds = time.time() - self.start_time
            self.stats.uptime_minutes = int(uptime_seconds / 60)
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.stats), f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"통계 저장 실패: {e}")
    
    def save_pid(self):
        """PID 저장"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            self.logger.error(f"PID 저장 실패: {e}")
    
    def signal_handler(self, signum, frame):
        """종료 신호 처리"""
        self.logger.info(f"🛑 종료 신호 받음: {signum}")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """정리 작업"""
        self.save_stats()
        if self.pid_file.exists():
            self.pid_file.unlink()
        self.logger.info("✅ 시스템 정리 완료")
    
    def collect_data(self):
        """데이터 수집 시뮬레이션"""
        try:
            # GitHub 데이터 수집
            github_data = random.randint(20, 100)
            self.stats.data_collected += github_data
            self.logger.info(f"📚 GitHub에서 {github_data}개 코드 수집")
            
            # Stack Overflow 데이터 수집
            so_data = random.randint(10, 50)
            self.stats.data_collected += so_data
            self.logger.info(f"💬 Stack Overflow에서 {so_data}개 Q&A 수집")
            
            self.logger.info(f"📊 총 수집 데이터: {self.stats.data_collected}개")
            
        except Exception as e:
            self.logger.error(f"데이터 수집 오류: {e}")
    
    def train_model(self):
        """모델 훈련 시뮬레이션"""
        try:
            self.logger.info("🧠 신경망 훈련 시작...")
            
            # 훈련 시뮬레이션
            old_accuracy = self.stats.current_accuracy
            improvement = random.uniform(0.01, 0.1)  # 0.01% ~ 0.1% 개선
            self.stats.current_accuracy = min(99.9, self.stats.current_accuracy + improvement)
            
            if self.stats.current_accuracy > old_accuracy:
                self.stats.accuracy_improvements += 1
            
            self.stats.models_trained += 1
            self.stats.last_training_time = datetime.now().isoformat()
            
            self.logger.info(f"✅ 훈련 완료! 정확도: {old_accuracy:.2f}% → {self.stats.current_accuracy:.2f}%")
            
        except Exception as e:
            self.logger.error(f"모델 훈련 오류: {e}")
    
    def optimize_hyperparameters(self):
        """하이퍼파라미터 최적화"""
        try:
            self.logger.info("⚡ 하이퍼파라미터 최적화...")
            
            # 학습률 조정
            if self.stats.accuracy_improvements > 5:
                self.stats.learning_rate *= 0.9  # 감소
            else:
                self.stats.learning_rate *= 1.05  # 증가
                
            # 배치 크기 조정
            batch_options = [16, 32, 64, 128]
            self.stats.batch_size = random.choice(batch_options)
            
            self.logger.info(f"📈 학습률: {self.stats.learning_rate:.6f}, 배치: {self.stats.batch_size}")
            
        except Exception as e:
            self.logger.error(f"최적화 오류: {e}")
    
    def display_status(self):
        """현재 상태 표시"""
        try:
            uptime_hours = self.stats.uptime_minutes / 60
            
            print(f"\n🧠 안정적 24시간 학습 상태")
            print(f"⏰ 실행 시간: {self.stats.uptime_minutes}분 ({uptime_hours:.1f}시간)")
            print(f"📚 수집 데이터: {self.stats.data_collected:,}개")
            print(f"🤖 훈련 모델: {self.stats.models_trained}개")
            print(f"📈 정확도: {self.stats.current_accuracy:.2f}%")
            print(f"🏆 개선 횟수: {self.stats.accuracy_improvements}회")
            print(f"⚙️ 학습률: {self.stats.learning_rate:.6f}")
            print(f"📦 배치 크기: {self.stats.batch_size}")
            print(f"🕐 마지막 훈련: {self.stats.last_training_time}")
            
            self.logger.info(f"📊 상태 업데이트 - 데이터:{self.stats.data_collected}, 모델:{self.stats.models_trained}, 정확도:{self.stats.current_accuracy:.2f}%")
            
        except Exception as e:
            self.logger.error(f"상태 표시 오류: {e}")
    
    def run_24h_learning(self):
        """24시간 지속 학습 메인 루프"""
        self.logger.info("🚀 안정적 24시간 학습 시작!")
        
        # 초기 데이터 수집 및 훈련
        self.collect_data()
        self.train_model()
        self.display_status()
        
        # 카운터들
        minute_counter = 0
        data_collection_counter = 0
        training_counter = 0
        optimization_counter = 0
        status_counter = 0
        
        self.logger.info("🔄 메인 학습 루프 시작...")
        
        while self.running:
            try:
                # 1분 대기
                time.sleep(60)
                minute_counter += 1
                
                # 매분 실행: 통계 저장
                self.save_stats()
                
                # 5분마다: 데이터 수집
                data_collection_counter += 1
                if data_collection_counter >= 5:
                    data_collection_counter = 0
                    self.collect_data()
                
                # 10분마다: 모델 훈련
                training_counter += 1
                if training_counter >= 10:
                    training_counter = 0
                    self.train_model()
                
                # 30분마다: 하이퍼파라미터 최적화
                optimization_counter += 1
                if optimization_counter >= 30:
                    optimization_counter = 0
                    self.optimize_hyperparameters()
                
                # 60분마다: 상태 표시
                status_counter += 1
                if status_counter >= 60:
                    status_counter = 0
                    self.display_status()
                
                # 하트비트 (매 10분)
                if minute_counter % 10 == 0:
                    self.logger.info(f"💗 시스템 정상 작동 중 - {minute_counter}분 경과")
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ 사용자 중지 요청")
                break
            except Exception as e:
                self.logger.error(f"학습 루프 오류: {e}")
                time.sleep(30)  # 30초 대기 후 재시도
        
        self.cleanup()

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            # 상태 확인
            stats_file = Path(__file__).parent / "stable_learning_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                uptime_hours = stats.get('uptime_minutes', 0) / 60
                print(f"🧠 안정적 24시간 학습 상태")
                print(f"⏰ 실행 시간: {stats.get('uptime_minutes', 0)}분 ({uptime_hours:.1f}시간)")
                print(f"📚 수집 데이터: {stats.get('data_collected', 0):,}개")
                print(f"🤖 훈련 모델: {stats.get('models_trained', 0)}개")
                print(f"📈 정확도: {stats.get('current_accuracy', 0):.2f}%")
                print(f"🏆 개선 횟수: {stats.get('accuracy_improvements', 0)}회")
                print(f"⚙️ 학습률: {stats.get('learning_rate', 0):.6f}")
                print(f"📦 배치 크기: {stats.get('batch_size', 32)}")
                print(f"🕐 마지막 훈련: {stats.get('last_training_time', 'N/A')}")
            else:
                print("⚠️ 학습 통계를 찾을 수 없습니다.")
            return
        elif sys.argv[1] == "stop":
            # 중지
            pid_file = Path(__file__).parent / "stable_learning.pid"
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    os.kill(pid, signal.SIGTERM)
                    print(f"✅ 프로세스 {pid} 종료 신호 전송")
                    time.sleep(2)
                    try:
                        os.kill(pid, 0)  # 프로세스 존재 확인
                        os.kill(pid, signal.SIGKILL)  # 강제 종료
                        print("🔥 강제 종료됨")
                    except OSError:
                        print("✅ 정상 종료됨")
                except Exception as e:
                    print(f"❌ 종료 오류: {e}")
            else:
                print("⚠️ 실행 중인 프로세스를 찾을 수 없습니다.")
            return
    
    # 기본: 24시간 학습 시작
    try:
        learner = Stable24HLearning()
        learner.run_24h_learning()
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 