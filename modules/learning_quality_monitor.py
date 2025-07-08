#!/usr/bin/env python3
"""
학습 품질 모니터링 시스템
과적합 방지, 학습 진행 추적, 품질 평가를 담당
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
try:
    import matplotlib
    matplotlib.use('Agg')  # 비대화형 백엔드 사용
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("Matplotlib이 설치되지 않았습니다. 플롯 생성이 비활성화됩니다.")
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """학습 메트릭 데이터"""
    timestamp: str
    epoch: int
    iteration: int
    train_loss: float
    validation_loss: Optional[float] = None
    learning_rate: float = 0.0
    quality_score: float = 0.0
    overfitting_score: float = 0.0  # 0-1, 높을수록 과적합
    
    def to_dict(self) -> Dict:
        return asdict(self)

class LearningQualityMonitor:
    """학습 품질 모니터링 및 과적합 방지 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.monitor_dir = self.project_root / "continuous_learning" / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 저장 경로
        self.metrics_file = self.monitor_dir / "learning_metrics.jsonl"
        self.plots_dir = self.monitor_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # 과적합 감지 설정
        self.overfitting_threshold = 0.1  # validation_loss가 train_loss보다 10% 이상 높으면 경고
        self.early_stopping_patience = 5  # validation_loss가 5번 연속 증가하면 중단
        self.min_delta = 0.001  # 최소 개선 폭
        
        # 메트릭 히스토리
        self.train_loss_history = deque(maxlen=100)
        self.val_loss_history = deque(maxlen=100)
        self.quality_score_history = deque(maxlen=100)
        
        # 조기 종료 관련
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # 학습률 스케줄러 설정
        self.initial_learning_rate = 2e-5
        self.min_learning_rate = 1e-6
        self.lr_decay_factor = 0.95
        
        # 품질 평가 기준
        self.quality_criteria = {
            "coherence": 0.3,      # 응답의 일관성
            "relevance": 0.3,      # 질문과의 관련성
            "completeness": 0.2,   # 답변의 완성도
            "accuracy": 0.2        # 기술적 정확성
        }
        
        # 실시간 모니터링 상태
        self.monitoring_active = False
        self.start_time = None
        self.total_iterations = 0
        
    def start_monitoring(self, model_name: str, dataset_size: int):
        """모니터링 시작"""
        self.monitoring_active = True
        self.start_time = datetime.now()
        self.total_iterations = 0
        self.should_stop = False
        
        # 모니터링 세션 정보 저장
        session_info = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_name": model_name,
            "dataset_size": dataset_size,
            "start_time": self.start_time.isoformat(),
            "initial_learning_rate": self.initial_learning_rate,
            "overfitting_threshold": self.overfitting_threshold
        }
        
        session_file = self.monitor_dir / "current_session.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2)
        
        logger.info(f"학습 모니터링 시작: {model_name} (데이터셋 크기: {dataset_size})")
    
    def update_metrics(self, epoch: int, iteration: int, 
                      train_loss: float, validation_loss: Optional[float] = None,
                      learning_rate: Optional[float] = None) -> LearningMetrics:
        """메트릭 업데이트 및 과적합 체크"""
        self.total_iterations += 1
        
        # 현재 학습률 (제공되지 않으면 계산)
        if learning_rate is None:
            learning_rate = self.calculate_learning_rate(epoch)
        
        # 과적합 점수 계산
        overfitting_score = 0.0
        if validation_loss is not None:
            overfitting_score = self.calculate_overfitting_score(train_loss, validation_loss)
            
            # 조기 종료 체크
            self.check_early_stopping(validation_loss)
        
        # 품질 점수 계산 (간단한 휴리스틱)
        quality_score = self.estimate_quality_score(train_loss, validation_loss)
        
        # 메트릭 생성
        metrics = LearningMetrics(
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            iteration=iteration,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            quality_score=quality_score,
            overfitting_score=overfitting_score
        )
        
        # 히스토리 업데이트
        self.train_loss_history.append(train_loss)
        if validation_loss is not None:
            self.val_loss_history.append(validation_loss)
        self.quality_score_history.append(quality_score)
        
        # 메트릭 저장
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
        
        # 주기적으로 플롯 생성 (100 iterations마다)
        if self.total_iterations % 100 == 0:
            self.generate_plots()
        
        # 경고 및 권장사항 출력
        self.print_recommendations(metrics)
        
        return metrics
    
    def calculate_overfitting_score(self, train_loss: float, val_loss: float) -> float:
        """과적합 점수 계산 (0-1)"""
        if train_loss == 0:
            return 0.0
        
        # validation loss가 train loss보다 얼마나 높은지
        gap_ratio = (val_loss - train_loss) / train_loss
        
        # 0-1 범위로 정규화
        overfitting_score = max(0.0, min(1.0, gap_ratio))
        
        return overfitting_score
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """조기 종료 체크"""
        if val_loss < self.best_val_loss - self.min_delta:
            # 개선됨
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            # 개선 안됨
            self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                self.should_stop = True
                logger.warning(f"조기 종료 트리거됨! Validation loss가 {self.patience_counter}번 연속 개선되지 않음")
                return True
        
        return False
    
    def calculate_learning_rate(self, epoch: int) -> float:
        """에포크에 따른 학습률 계산 (감쇠 적용)"""
        lr = self.initial_learning_rate * (self.lr_decay_factor ** epoch)
        return max(lr, self.min_learning_rate)
    
    def estimate_quality_score(self, train_loss: float, val_loss: Optional[float]) -> float:
        """학습 품질 점수 추정 (0-1)"""
        # 단순 휴리스틱: loss가 낮을수록 높은 점수
        base_score = 1.0 / (1.0 + train_loss)
        
        # validation loss가 있으면 과적합 페널티 적용
        if val_loss is not None:
            overfitting_penalty = max(0, (val_loss - train_loss) / (train_loss + 1e-8))
            base_score *= (1.0 - min(overfitting_penalty, 0.5))
        
        return min(1.0, base_score)
    
    def print_recommendations(self, metrics: LearningMetrics):
        """학습 상태에 따른 권장사항 출력"""
        recommendations = []
        
        # 과적합 경고
        if metrics.overfitting_score > 0.3:
            recommendations.append("⚠️ 과적합 징후 감지! 다음을 권장합니다:")
            recommendations.append("  - 학습률 감소 (현재: {:.2e})".format(metrics.learning_rate))
            recommendations.append("  - 드롭아웃 비율 증가")
            recommendations.append("  - 데이터 증강 적용")
            recommendations.append("  - 정규화 강화")
        
        # 학습 정체 경고
        if len(self.train_loss_history) > 10:
            recent_losses = list(self.train_loss_history)[-10:]
            loss_variance = np.var(recent_losses)
            if loss_variance < 0.0001:
                recommendations.append("⚠️ 학습이 정체되었습니다! 다음을 시도하세요:")
                recommendations.append("  - 학습률 조정")
                recommendations.append("  - 다른 옵티마이저 시도")
                recommendations.append("  - 배치 크기 변경")
        
        # 조기 종료 경고
        if self.should_stop:
            recommendations.append("🛑 조기 종료를 권장합니다! 더 이상의 학습은 과적합을 유발할 수 있습니다.")
        
        # 권장사항 출력
        if recommendations:
            logger.warning("\n".join(recommendations))
    
    def generate_plots(self):
        """학습 진행 상황 플롯 생성"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib이 없어 플롯을 생성할 수 없습니다")
            return
            
        if len(self.train_loss_history) < 2:
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Loss 곡선
        ax1 = axes[0, 0]
        iterations = range(len(self.train_loss_history))
        ax1.plot(iterations, self.train_loss_history, 'b-', label='Train Loss', alpha=0.7)
        if self.val_loss_history:
            val_iterations = range(len(self.val_loss_history))
            ax1.plot(val_iterations, self.val_loss_history, 'r-', label='Validation Loss', alpha=0.7)
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 품질 점수 추이
        ax2 = axes[0, 1]
        quality_iterations = range(len(self.quality_score_history))
        ax2.plot(quality_iterations, self.quality_score_history, 'g-', alpha=0.7)
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Learning Quality')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 3. 과적합 분석
        if self.val_loss_history and len(self.val_loss_history) > 0:
            ax3 = axes[1, 0]
            train_losses = list(self.train_loss_history)[:len(self.val_loss_history)]
            val_losses = list(self.val_loss_history)
            
            if train_losses and val_losses:
                overfitting_scores = [
                    self.calculate_overfitting_score(t, v) 
                    for t, v in zip(train_losses, val_losses)
                ]
                ax3.plot(range(len(overfitting_scores)), overfitting_scores, 'r-', alpha=0.7)
                ax3.axhline(y=0.3, color='r', linestyle='--', label='Warning Threshold')
                ax3.set_xlabel('Iterations')
                ax3.set_ylabel('Overfitting Score')
                ax3.set_title('Overfitting Detection')
                ax3.set_ylim(0, 1)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. 학습 시간 및 속도
        ax4 = axes[1, 1]
        if self.start_time:
            elapsed_time = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
            speed = self.total_iterations / (elapsed_time + 1e-8)  # iterations per hour
            
            info_text = f"""학습 정보:
총 반복: {self.total_iterations}
경과 시간: {elapsed_time:.2f} 시간
학습 속도: {speed:.1f} iterations/hour
현재 Train Loss: {self.train_loss_history[-1]:.4f}
"""
            if self.val_loss_history:
                info_text += f"현재 Val Loss: {self.val_loss_history[-1]:.4f}\n"
                info_text += f"과적합 점수: {overfitting_scores[-1]:.3f}" if 'overfitting_scores' in locals() else ""
            
            ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='center')
            ax4.axis('off')
        
        # 플롯 저장
        plot_file = self.plots_dir / f"learning_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 최신 플롯으로 심볼릭 링크 생성 (Linux/WSL only)
        latest_plot = self.plots_dir / "latest_progress.png"
        if latest_plot.exists():
            latest_plot.unlink()
        try:
            latest_plot.symlink_to(plot_file.name)
        except:
            # Windows에서는 복사
            import shutil
            shutil.copy(plot_file, latest_plot)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """학습 요약 정보 반환"""
        summary = {
            "total_iterations": self.total_iterations,
            "should_stop": self.should_stop,
            "current_train_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "current_val_loss": self.val_loss_history[-1] if self.val_loss_history else None,
            "best_val_loss": self.best_val_loss if self.best_val_loss < float('inf') else None,
            "average_quality_score": np.mean(self.quality_score_history) if self.quality_score_history else 0,
            "overfitting_detected": any(s > 0.3 for s in self.quality_score_history[-10:]) if len(self.quality_score_history) > 10 else False,
            "elapsed_time": str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
        }
        
        return summary
    
    def export_report(self) -> Path:
        """학습 보고서 생성"""
        report_path = self.monitor_dir / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 전체 메트릭 로드
        all_metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_metrics.append(json.loads(line.strip()))
                    except:
                        continue
        
        # 보고서 생성
        report = {
            "summary": self.get_learning_summary(),
            "metrics_count": len(all_metrics),
            "final_metrics": all_metrics[-1] if all_metrics else None,
            "best_metrics": min(all_metrics, key=lambda x: x.get('validation_loss', float('inf'))) if all_metrics else None,
            "recommendations": self.generate_final_recommendations()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"학습 보고서 생성: {report_path}")
        return report_path
    
    def generate_final_recommendations(self) -> List[str]:
        """최종 권장사항 생성"""
        recommendations = []
        
        summary = self.get_learning_summary()
        
        if summary["overfitting_detected"]:
            recommendations.append("과적합이 감지되었습니다. 다음 학습 시 데이터 증강과 정규화를 강화하세요.")
        
        if summary["should_stop"]:
            recommendations.append("조기 종료가 발생했습니다. 하이퍼파라미터 조정이 필요할 수 있습니다.")
        
        avg_quality = summary["average_quality_score"]
        if avg_quality < 0.5:
            recommendations.append(f"평균 품질 점수가 낮습니다 ({avg_quality:.2f}). 데이터셋 품질을 개선하세요.")
        elif avg_quality > 0.8:
            recommendations.append(f"우수한 학습 품질을 보였습니다 ({avg_quality:.2f}). 현재 설정을 유지하세요.")
        
        return recommendations

# 싱글톤 인스턴스
_quality_monitor = None

def get_quality_monitor() -> LearningQualityMonitor:
    """품질 모니터 싱글톤 인스턴스 반환"""
    global _quality_monitor
    if _quality_monitor is None:
        _quality_monitor = LearningQualityMonitor()
    return _quality_monitor