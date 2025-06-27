#!/usr/bin/env python3
"""
AutoCI 학습 스케줄러 및 최적화 시스템
적응형 학습율, 동적 배치 크기, 성능 기반 스케줄링
"""

import os
import sys
import time
import json
import sqlite3
import threading
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveLearningScheduler:
    """적응형 학습 스케줄러"""
    
    def __init__(self, initial_lr: float = 0.001, min_lr: float = 1e-6, max_lr: float = 0.01):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # 성능 추적
        self.performance_history = []
        self.loss_history = []
        self.lr_history = []
        
        # 적응형 파라미터
        self.patience = 5  # 성능 개선 없을 때 대기 횟수
        self.factor = 0.5  # 학습율 감소 비율
        self.improvement_threshold = 0.01  # 개선 임계값
        self.no_improvement_count = 0
        
        # 학습율 증가/감소 전략
        self.lr_strategy = "adaptive"  # adaptive, cosine, plateau
        
        logger.info(f"🎯 적응형 학습 스케줄러 초기화: LR={initial_lr}")
    
    def update_performance(self, loss: float, accuracy: float, epoch: int):
        """성능 업데이트 및 학습율 조정"""
        
        # 성능 기록
        performance_score = accuracy - (loss * 0.1)  # 정확도 우선, 손실 패널티
        self.performance_history.append(performance_score)
        self.loss_history.append(loss)
        self.lr_history.append(self.current_lr)
        
        # 최근 성능만 유지 (메모리 절약)
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
            self.loss_history = self.loss_history[-50:]
            self.lr_history = self.lr_history[-50:]
        
        # 학습율 조정
        self._adjust_learning_rate(performance_score, epoch)
        
        logger.info(f"📈 성능 업데이트: 손실={loss:.4f}, 정확도={accuracy:.4f}, LR={self.current_lr:.6f}")
    
    def _adjust_learning_rate(self, current_performance: float, epoch: int):
        """학습율 조정 로직"""
        
        if len(self.performance_history) < 2:
            return
        
        if self.lr_strategy == "adaptive":
            self._adaptive_lr_adjustment(current_performance)
        elif self.lr_strategy == "cosine":
            self._cosine_lr_adjustment(epoch)
        elif self.lr_strategy == "plateau":
            self._plateau_lr_adjustment(current_performance)
    
    def _adaptive_lr_adjustment(self, current_performance: float):
        """적응형 학습율 조정"""
        
        if len(self.performance_history) < 2:
            return
        
        recent_avg = sum(self.performance_history[-3:]) / min(3, len(self.performance_history))
        older_avg = sum(self.performance_history[-6:-3]) / max(1, min(3, len(self.performance_history) - 3))
        
        improvement = recent_avg - older_avg
        
        if improvement > self.improvement_threshold:
            # 성능 개선 중 - 학습율 약간 증가
            self.current_lr = min(self.current_lr * 1.05, self.max_lr)
            self.no_improvement_count = 0
            logger.info(f"🔼 성능 개선 감지, 학습율 증가: {self.current_lr:.6f}")
            
        elif improvement < -self.improvement_threshold:
            # 성능 악화 - 학습율 감소
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.no_improvement_count = 0
                logger.info(f"🔽 성능 악화 감지, 학습율 감소: {self.current_lr:.6f}")
        
        else:
            # 성능 정체
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience * 2:
                # 학습율을 리셋하여 local minimum 탈출 시도
                self.current_lr = self.initial_lr
                self.no_improvement_count = 0
                logger.info(f"🔄 성능 정체, 학습율 리셋: {self.current_lr:.6f}")
    
    def _cosine_lr_adjustment(self, epoch: int):
        """코사인 어닐링 학습율 조정"""
        T_max = 100  # 주기
        eta_min = self.min_lr
        
        self.current_lr = eta_min + (self.initial_lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        self.current_lr = max(self.current_lr, self.min_lr)
    
    def _plateau_lr_adjustment(self, current_performance: float):
        """플래토 기반 학습율 조정"""
        if len(self.performance_history) < 5:
            return
        
        # 최근 5회 성능의 최대값과 비교
        recent_max = max(self.performance_history[-5:])
        
        if current_performance < recent_max - self.improvement_threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        
        if self.no_improvement_count >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.no_improvement_count = 0
            logger.info(f"🎯 플래토 감지, 학습율 감소: {self.current_lr:.6f}")
    
    def get_current_lr(self) -> float:
        """현재 학습율 반환"""
        return self.current_lr
    
    def get_lr_stats(self) -> Dict:
        """학습율 통계 반환"""
        return {
            "current_lr": self.current_lr,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "no_improvement_count": self.no_improvement_count,
            "performance_trend": self._calculate_trend(),
            "lr_history": self.lr_history[-10:],  # 최근 10개
            "performance_history": self.performance_history[-10:]
        }
    
    def _calculate_trend(self) -> str:
        """성능 트렌드 계산"""
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent = self.performance_history[-3:]
        if len(set(recent)) == 1:
            return "stable"
        
        if recent[-1] > recent[0]:
            return "improving"
        elif recent[-1] < recent[0]:
            return "declining"
        else:
            return "stable"

class DynamicBatchSizer:
    """동적 배치 크기 조정기"""
    
    def __init__(self, initial_batch_size: int = 16, min_batch_size: int = 4, max_batch_size: int = 128):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        # 성능 추적
        self.throughput_history = []  # 처리량 (samples/sec)
        self.memory_usage_history = []
        self.loss_convergence_history = []
        
        # 조정 파라미터
        self.adjustment_factor = 1.2
        self.memory_threshold = 0.8  # 메모리 사용률 임계값
        
        logger.info(f"📦 동적 배치 크기 조정기 초기화: {initial_batch_size}")
    
    def update_metrics(self, processing_time: float, samples_processed: int, 
                      memory_usage: float, loss_improvement: float):
        """성능 메트릭 업데이트"""
        
        # 처리량 계산
        throughput = samples_processed / max(processing_time, 0.001)
        self.throughput_history.append(throughput)
        self.memory_usage_history.append(memory_usage)
        self.loss_convergence_history.append(loss_improvement)
        
        # 최근 기록만 유지
        if len(self.throughput_history) > 20:
            self.throughput_history = self.throughput_history[-20:]
            self.memory_usage_history = self.memory_usage_history[-20:]
            self.loss_convergence_history = self.loss_convergence_history[-20:]
        
        # 배치 크기 조정
        self._adjust_batch_size(throughput, memory_usage, loss_improvement)
    
    def _adjust_batch_size(self, current_throughput: float, memory_usage: float, loss_improvement: float):
        """배치 크기 조정 로직"""
        
        if len(self.throughput_history) < 3:
            return
        
        # 최근 처리량 트렌드
        recent_throughput = sum(self.throughput_history[-3:]) / 3
        older_throughput = sum(self.throughput_history[-6:-3]) / max(1, min(3, len(self.throughput_history) - 3))
        
        throughput_improvement = recent_throughput - older_throughput
        
        # 조정 결정
        should_increase = (
            throughput_improvement > 0 and 
            memory_usage < self.memory_threshold and 
            loss_improvement > 0 and
            self.current_batch_size < self.max_batch_size
        )
        
        should_decrease = (
            memory_usage > self.memory_threshold or 
            throughput_improvement < -0.1 or
            loss_improvement < -0.01
        ) and self.current_batch_size > self.min_batch_size
        
        if should_increase:
            new_batch_size = min(int(self.current_batch_size * self.adjustment_factor), self.max_batch_size)
            if new_batch_size != self.current_batch_size:
                self.current_batch_size = new_batch_size
                logger.info(f"📈 배치 크기 증가: {self.current_batch_size}")
        
        elif should_decrease:
            new_batch_size = max(int(self.current_batch_size / self.adjustment_factor), self.min_batch_size)
            if new_batch_size != self.current_batch_size:
                self.current_batch_size = new_batch_size
                logger.info(f"📉 배치 크기 감소: {self.current_batch_size}")
    
    def get_current_batch_size(self) -> int:
        """현재 배치 크기 반환"""
        return self.current_batch_size
    
    def get_batch_stats(self) -> Dict:
        """배치 크기 통계 반환"""
        return {
            "current_batch_size": self.current_batch_size,
            "initial_batch_size": self.initial_batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "avg_throughput": sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0,
            "avg_memory_usage": sum(self.memory_usage_history) / len(self.memory_usage_history) if self.memory_usage_history else 0,
            "throughput_history": self.throughput_history[-5:],
            "memory_usage_history": self.memory_usage_history[-5:]
        }

class PerformanceBasedScheduler:
    """성능 기반 학습 스케줄러"""
    
    def __init__(self):
        self.priority_queue = []  # (priority, task_type, task_data)
        self.task_performance = {}  # task_type -> performance_score
        self.execution_history = {}  # task_type -> [execution_times]
        
        # 작업 우선순위 가중치
        self.task_weights = {
            "high_feedback_learning": 1.0,    # 높은 피드백 점수 데이터
            "error_correction": 0.9,          # 오류 수정 학습
            "new_pattern_learning": 0.8,      # 새로운 패턴 학습
            "reinforcement_learning": 0.7,    # 강화 학습
            "general_learning": 0.5,          # 일반 학습
            "maintenance": 0.3                # 유지보수 작업
        }
        
        logger.info("🎯 성능 기반 스케줄러 초기화")
    
    def add_task(self, task_type: str, task_data: Dict, urgency: float = 0.5):
        """작업 추가"""
        
        # 우선순위 계산
        base_priority = self.task_weights.get(task_type, 0.5)
        performance_bonus = self.task_performance.get(task_type, 0.5)
        urgency_factor = urgency
        
        # 시간 가중치 (오래된 작업일수록 우선순위 증가)
        time_weight = min(1.0, (time.time() - task_data.get('created_time', time.time())) / 3600)
        
        final_priority = (base_priority + performance_bonus + urgency_factor + time_weight) / 4
        
        self.priority_queue.append((final_priority, task_type, task_data))
        self.priority_queue.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"📋 작업 추가: {task_type} (우선순위: {final_priority:.3f})")
    
    def get_next_task(self) -> Optional[Tuple[str, Dict]]:
        """다음 실행할 작업 반환"""
        if not self.priority_queue:
            return None
        
        priority, task_type, task_data = self.priority_queue.pop(0)
        return task_type, task_data
    
    def update_task_performance(self, task_type: str, performance_score: float, execution_time: float):
        """작업 성능 업데이트"""
        
        # 성능 점수 업데이트 (지수 이동 평균)
        if task_type in self.task_performance:
            self.task_performance[task_type] = 0.7 * self.task_performance[task_type] + 0.3 * performance_score
        else:
            self.task_performance[task_type] = performance_score
        
        # 실행 시간 기록
        if task_type not in self.execution_history:
            self.execution_history[task_type] = []
        
        self.execution_history[task_type].append(execution_time)
        if len(self.execution_history[task_type]) > 10:
            self.execution_history[task_type] = self.execution_history[task_type][-10:]
        
        logger.info(f"📊 성능 업데이트: {task_type} -> {performance_score:.3f} (실행시간: {execution_time:.2f}s)")
    
    def get_scheduler_stats(self) -> Dict:
        """스케줄러 통계 반환"""
        return {
            "queue_length": len(self.priority_queue),
            "task_performance": self.task_performance.copy(),
            "avg_execution_times": {
                task_type: sum(times) / len(times) 
                for task_type, times in self.execution_history.items()
            },
            "pending_tasks": [
                {"type": task_type, "priority": priority}
                for priority, task_type, _ in self.priority_queue[:5]
            ]
        }

class LearningOptimizerManager:
    """학습 최적화 관리자"""
    
    def __init__(self, db_path: str = "continuous_learning.db"):
        self.db_path = db_path
        self.lr_scheduler = AdaptiveLearningScheduler()
        self.batch_sizer = DynamicBatchSizer()
        self.task_scheduler = PerformanceBasedScheduler()
        
        # 최적화 상태
        self.optimization_enabled = True
        self.auto_tune_mode = True
        
        # 성능 메트릭
        self.global_performance_score = 0.5
        self.optimization_history = []
        
        logger.info("🎛️ 학습 최적화 관리자 초기화 완료")
    
    def optimize_learning_session(self, learning_data: List[Dict], model=None, optimizer=None) -> Dict:
        """학습 세션 최적화"""
        
        start_time = time.time()
        
        # 현재 설정
        current_lr = self.lr_scheduler.get_current_lr()
        current_batch_size = self.batch_sizer.get_current_batch_size()
        
        # 학습 실행
        if TORCH_AVAILABLE and model and optimizer:
            results = self._pytorch_optimized_training(learning_data, model, optimizer, current_lr, current_batch_size)
        else:
            results = self._simulated_optimized_training(learning_data, current_batch_size)
        
        # 성능 업데이트
        execution_time = time.time() - start_time
        self._update_optimizers(results, execution_time, len(learning_data))
        
        return {
            "optimization_applied": True,
            "learning_rate": current_lr,
            "batch_size": current_batch_size,
            "execution_time": execution_time,
            "performance_score": results.get("accuracy", 0.5),
            "loss": results.get("loss", 0.0),
            "samples_processed": len(learning_data)
        }
    
    def _pytorch_optimized_training(self, learning_data: List[Dict], model, optimizer, lr: float, batch_size: int) -> Dict:
        """PyTorch 최적화 학습"""
        
        # 학습율 적용
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        total_loss = 0.0
        total_accuracy = 0.0
        batch_count = 0
        
        # 배치로 나누어 학습
        for i in range(0, len(learning_data), batch_size):
            batch = learning_data[i:i + batch_size]
            
            # 여기서 실제 PyTorch 학습 로직 구현
            # (간소화된 버전)
            batch_loss = 0.5 + (len(batch) * 0.01)  # 시뮬레이션
            batch_accuracy = 0.7 + (batch_size * 0.001)  # 시뮬레이션
            
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_accuracy = total_accuracy / batch_count if batch_count > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "batches_processed": batch_count
        }
    
    def _simulated_optimized_training(self, learning_data: List[Dict], batch_size: int) -> Dict:
        """시뮬레이션 최적화 학습"""
        
        # 시뮬레이션 로직
        data_quality = sum(d.get("quality", 0.5) for d in learning_data) / len(learning_data)
        batch_efficiency = min(1.0, batch_size / 32)
        
        simulated_loss = max(0.1, 1.0 - data_quality * batch_efficiency)
        simulated_accuracy = min(0.95, data_quality * batch_efficiency + 0.3)
        
        return {
            "loss": simulated_loss,
            "accuracy": simulated_accuracy,
            "batches_processed": len(learning_data) // batch_size + 1
        }
    
    def _update_optimizers(self, results: Dict, execution_time: float, data_size: int):
        """최적화기들 업데이트"""
        
        loss = results.get("loss", 0.0)
        accuracy = results.get("accuracy", 0.5)
        
        # 학습율 스케줄러 업데이트
        self.lr_scheduler.update_performance(loss, accuracy, len(self.optimization_history))
        
        # 메모리 사용률 추정 (시뮬레이션)
        estimated_memory_usage = min(0.9, data_size / 1000 * 0.1)
        loss_improvement = self.global_performance_score - loss
        
        # 배치 크기 조정기 업데이트
        self.batch_sizer.update_metrics(
            processing_time=execution_time,
            samples_processed=data_size,
            memory_usage=estimated_memory_usage,
            loss_improvement=loss_improvement
        )
        
        # 전역 성능 점수 업데이트
        self.global_performance_score = 0.8 * self.global_performance_score + 0.2 * accuracy
        
        # 최적화 기록
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "learning_rate": self.lr_scheduler.get_current_lr(),
            "batch_size": self.batch_sizer.get_current_batch_size(),
            "performance_score": self.global_performance_score,
            "loss": loss,
            "accuracy": accuracy,
            "execution_time": execution_time
        }
        
        self.optimization_history.append(optimization_record)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def schedule_learning_task(self, task_type: str, task_data: Dict, urgency: float = 0.5):
        """학습 작업 스케줄링"""
        task_data["created_time"] = time.time()
        self.task_scheduler.add_task(task_type, task_data, urgency)
    
    def get_next_scheduled_task(self) -> Optional[Tuple[str, Dict]]:
        """다음 예정된 작업 가져오기"""
        return self.task_scheduler.get_next_task()
    
    def report_task_completion(self, task_type: str, performance_score: float, execution_time: float):
        """작업 완료 보고"""
        self.task_scheduler.update_task_performance(task_type, performance_score, execution_time)
    
    def get_optimization_status(self) -> Dict:
        """최적화 상태 보고"""
        return {
            "global_performance_score": self.global_performance_score,
            "optimization_enabled": self.optimization_enabled,
            "auto_tune_mode": self.auto_tune_mode,
            "learning_rate_stats": self.lr_scheduler.get_lr_stats(),
            "batch_size_stats": self.batch_sizer.get_batch_stats(),
            "scheduler_stats": self.task_scheduler.get_scheduler_stats(),
            "recent_optimizations": self.optimization_history[-5:] if self.optimization_history else []
        }
    
    def save_optimization_config(self, config_path: str = "optimization_config.json"):
        """최적화 설정 저장"""
        config = {
            "learning_rate": {
                "current": self.lr_scheduler.get_current_lr(),
                "initial": self.lr_scheduler.initial_lr,
                "min": self.lr_scheduler.min_lr,
                "max": self.lr_scheduler.max_lr,
                "strategy": self.lr_scheduler.lr_strategy
            },
            "batch_size": {
                "current": self.batch_sizer.get_current_batch_size(),
                "initial": self.batch_sizer.initial_batch_size,
                "min": self.batch_sizer.min_batch_size,
                "max": self.batch_sizer.max_batch_size
            },
            "global_performance": self.global_performance_score,
            "optimization_enabled": self.optimization_enabled
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 최적화 설정 저장: {config_path}")

def test_learning_optimizer():
    """학습 최적화기 테스트"""
    print("🧪 학습 스케줄러 및 최적화 시스템 테스트")
    print("=" * 60)
    
    # 최적화 관리자 초기화
    optimizer_manager = LearningOptimizerManager()
    
    # 테스트 학습 데이터
    test_learning_data = [
        {"input": "Unity GameObject 생성", "quality": 0.8},
        {"input": "C# async/await 사용법", "quality": 0.9},
        {"input": "코루틴 메모리 누수", "quality": 0.7},
        {"input": "Physics2D 충돌 감지", "quality": 0.6}
    ] * 10  # 40개 데이터
    
    print(f"📚 {len(test_learning_data)}개 테스트 데이터로 최적화 테스트")
    
    # 여러 번의 최적화 세션 실행
    for session in range(5):
        print(f"\n🔄 최적화 세션 {session + 1}/5")
        
        # 학습 작업 스케줄링
        optimizer_manager.schedule_learning_task(
            task_type="general_learning",
            task_data={"data": test_learning_data, "session": session},
            urgency=0.5 + session * 0.1
        )
        
        # 예정된 작업 실행
        task_info = optimizer_manager.get_next_scheduled_task()
        if task_info:
            task_type, task_data = task_info
            
            # 최적화된 학습 실행
            start_time = time.time()
            results = optimizer_manager.optimize_learning_session(test_learning_data)
            execution_time = time.time() - start_time
            
            # 작업 완료 보고
            optimizer_manager.report_task_completion(
                task_type, 
                results["performance_score"], 
                execution_time
            )
            
            print(f"✅ 세션 완료: LR={results['learning_rate']:.6f}, "
                  f"배치={results['batch_size']}, 성능={results['performance_score']:.3f}")
        
        time.sleep(1)  # 시뮬레이션 대기
    
    # 최종 상태 보고
    print(f"\n📊 최적화 시스템 최종 상태:")
    status = optimizer_manager.get_optimization_status()
    
    print(f"전역 성능 점수: {status['global_performance_score']:.3f}")
    print(f"현재 학습율: {status['learning_rate_stats']['current_lr']:.6f}")
    print(f"현재 배치 크기: {status['batch_size_stats']['current_batch_size']}")
    print(f"성능 트렌드: {status['learning_rate_stats']['performance_trend']}")
    
    # 설정 저장
    optimizer_manager.save_optimization_config()
    
    print("\n🎉 학습 최적화 시스템 테스트 완료!")

if __name__ == "__main__":
    test_learning_optimizer()