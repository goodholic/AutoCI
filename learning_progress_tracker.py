#!/usr/bin/env python3
"""
AutoCI 학습 진행률 추적 시스템
실시간 학습 진행 상황, 성능 변화, 모델 개선 추적
"""

import os
import sys
import time
import json
import sqlite3
import threading
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningMetric:
    """학습 메트릭"""
    timestamp: str
    epoch: int
    loss: float
    accuracy: float
    learning_rate: float
    batch_size: int
    data_points: int
    training_time: float
    model_version: str
    performance_score: float

@dataclass
class ProgressSummary:
    """진행률 요약"""
    total_epochs: int
    total_training_time: float
    total_data_points: int
    best_accuracy: float
    best_loss: float
    current_performance: float
    improvement_rate: float
    estimated_completion: str
    learning_efficiency: float

class LearningProgressDatabase:
    """학습 진행률 데이터베이스"""
    
    def __init__(self, db_path: str = "learning_progress.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 학습 메트릭 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    epoch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    learning_rate REAL,
                    batch_size INTEGER,
                    data_points INTEGER,
                    training_time REAL,
                    model_version TEXT,
                    performance_score REAL
                )
            ''')
            
            # 진행률 스냅샷 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_epochs INTEGER,
                    total_training_time REAL,
                    total_data_points INTEGER,
                    best_accuracy REAL,
                    best_loss REAL,
                    current_performance REAL,
                    improvement_rate REAL,
                    learning_efficiency REAL
                )
            ''')
            
            # 이정표 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS milestones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    milestone_type TEXT,
                    description TEXT,
                    metric_value REAL,
                    target_value REAL,
                    achieved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
    
    def record_metric(self, metric: LearningMetric):
        """학습 메트릭 기록"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO learning_metrics 
                (timestamp, epoch, loss, accuracy, learning_rate, batch_size, 
                 data_points, training_time, model_version, performance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp, metric.epoch, metric.loss, metric.accuracy,
                metric.learning_rate, metric.batch_size, metric.data_points,
                metric.training_time, metric.model_version, metric.performance_score
            ))
            conn.commit()
    
    def record_progress_snapshot(self, summary: ProgressSummary):
        """진행률 스냅샷 기록"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO progress_snapshots 
                (timestamp, total_epochs, total_training_time, total_data_points,
                 best_accuracy, best_loss, current_performance, improvement_rate, learning_efficiency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), summary.total_epochs, summary.total_training_time,
                summary.total_data_points, summary.best_accuracy, summary.best_loss,
                summary.current_performance, summary.improvement_rate, summary.learning_efficiency
            ))
            conn.commit()
    
    def get_recent_metrics(self, hours: int = 24) -> List[LearningMetric]:
        """최근 메트릭 조회"""
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, epoch, loss, accuracy, learning_rate, batch_size,
                       data_points, training_time, model_version, performance_score
                FROM learning_metrics 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            return [LearningMetric(*row) for row in rows]
    
    def get_all_metrics(self) -> List[LearningMetric]:
        """모든 메트릭 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, epoch, loss, accuracy, learning_rate, batch_size,
                       data_points, training_time, model_version, performance_score
                FROM learning_metrics 
                ORDER BY timestamp ASC
            ''')
            
            rows = cursor.fetchall()
            return [LearningMetric(*row) for row in rows]
    
    def add_milestone(self, milestone_type: str, description: str, 
                     metric_value: float, target_value: float):
        """이정표 추가"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO milestones 
                (timestamp, milestone_type, description, metric_value, target_value, achieved)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), milestone_type, description,
                metric_value, target_value, metric_value >= target_value
            ))
            conn.commit()

class PerformanceAnalyzer:
    """성능 분석기"""
    
    def __init__(self, db: LearningProgressDatabase):
        self.db = db
        
    def analyze_learning_trend(self, hours: int = 24) -> Dict:
        """학습 트렌드 분석"""
        metrics = self.db.get_recent_metrics(hours)
        
        if len(metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # 시간순 정렬
        metrics.sort(key=lambda x: x.timestamp)
        
        # 트렌드 계산
        accuracies = [m.accuracy for m in metrics]
        losses = [m.loss for m in metrics]
        
        accuracy_trend = self._calculate_trend(accuracies)
        loss_trend = self._calculate_trend(losses)
        
        # 개선률 계산
        if len(metrics) >= 2:
            recent_accuracy = np.mean(accuracies[-5:]) if len(accuracies) >= 5 else accuracies[-1]
            older_accuracy = np.mean(accuracies[:5]) if len(accuracies) >= 10 else accuracies[0]
            improvement_rate = (recent_accuracy - older_accuracy) / max(older_accuracy, 0.001)
        else:
            improvement_rate = 0.0
        
        return {
            "trend": "improving" if accuracy_trend > 0 and loss_trend < 0 else "degrading" if accuracy_trend < 0 or loss_trend > 0 else "stable",
            "accuracy_trend": accuracy_trend,
            "loss_trend": loss_trend,
            "improvement_rate": improvement_rate,
            "best_accuracy": max(accuracies),
            "best_loss": min(losses),
            "avg_accuracy": np.mean(accuracies),
            "avg_loss": np.mean(losses),
            "learning_speed": len(metrics) / hours  # metrics per hour
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """값들의 트렌드 계산"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        return z[0]  # 기울기
    
    def calculate_learning_efficiency(self, metrics: List[LearningMetric]) -> float:
        """학습 효율성 계산"""
        if not metrics:
            return 0.0
        
        # 시간당 성능 향상률
        total_time = sum(m.training_time for m in metrics)
        if total_time <= 0:
            return 0.0
        
        performance_gain = max(m.performance_score for m in metrics) - min(m.performance_score for m in metrics)
        efficiency = performance_gain / (total_time / 3600)  # per hour
        
        return min(efficiency, 1.0)  # 최대 1.0으로 제한
    
    def predict_performance(self, target_accuracy: float) -> Dict:
        """성능 예측"""
        metrics = self.db.get_all_metrics()
        
        if len(metrics) < 10:
            return {"prediction": "insufficient_data"}
        
        # 최근 트렌드 기반 예측
        recent_metrics = metrics[-20:]  # 최근 20개
        accuracies = [m.accuracy for m in recent_metrics]
        timestamps = [datetime.fromisoformat(m.timestamp) for m in recent_metrics]
        
        # 선형 회귀로 트렌드 계산
        if len(accuracies) >= 5:
            x = np.arange(len(accuracies))
            z = np.polyfit(x, accuracies, 1)
            slope = z[0]
            
            if slope > 0:
                current_accuracy = accuracies[-1]
                remaining_improvement = target_accuracy - current_accuracy
                estimated_steps = remaining_improvement / slope
                
                # 시간 간격 계산
                time_intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                                for i in range(len(timestamps)-1)]
                avg_interval = np.mean(time_intervals) if time_intervals else 3600
                
                estimated_time = estimated_steps * avg_interval
                estimated_completion = datetime.now() + timedelta(seconds=estimated_time)
                
                return {
                    "prediction": "achievable",
                    "estimated_completion": estimated_completion.isoformat(),
                    "estimated_days": estimated_time / 86400,
                    "current_accuracy": current_accuracy,
                    "target_accuracy": target_accuracy,
                    "improvement_rate": slope
                }
        
        return {"prediction": "uncertain"}

class VisualizationGenerator:
    """시각화 생성기"""
    
    def __init__(self, db: LearningProgressDatabase):
        self.db = db
        self.output_dir = "progress_charts"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_learning_curve(self, save_path: str = None) -> str:
        """학습 곡선 생성"""
        metrics = self.db.get_all_metrics()
        
        if not metrics:
            return None
        
        # 데이터 준비
        timestamps = [datetime.fromisoformat(m.timestamp) for m in metrics]
        accuracies = [m.accuracy for m in metrics]
        losses = [m.loss for m in metrics]
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 정확도 그래프
        ax1.plot(timestamps, accuracies, 'b-', linewidth=2, label='Accuracy')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('AutoCI Learning Progress', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 손실 그래프
        ax2.plot(timestamps, losses, 'r-', linewidth=2, label='Loss')
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 시간 축 포맷
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 저장
        if not save_path:
            save_path = os.path.join(self.output_dir, f"learning_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_performance_dashboard(self, save_path: str = None) -> str:
        """성능 대시보드 생성"""
        metrics = self.db.get_all_metrics()
        
        if not metrics:
            return None
        
        # 데이터 준비
        timestamps = [datetime.fromisoformat(m.timestamp) for m in metrics]
        accuracies = [m.accuracy for m in metrics]
        losses = [m.loss for m in metrics]
        learning_rates = [m.learning_rate for m in metrics]
        batch_sizes = [m.batch_size for m in metrics]
        
        # 4개 서브플롯 생성
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 정확도/손실
        ax1_twin = ax1.twinx()
        ax1.plot(timestamps, accuracies, 'b-', linewidth=2, label='Accuracy')
        ax1_twin.plot(timestamps, losses, 'r-', linewidth=2, label='Loss')
        ax1.set_ylabel('Accuracy', color='b', fontsize=10)
        ax1_twin.set_ylabel('Loss', color='r', fontsize=10)
        ax1.set_title('Accuracy & Loss Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 학습률
        ax2.plot(timestamps, learning_rates, 'g-', linewidth=2)
        ax2.set_ylabel('Learning Rate', fontsize=10)
        ax2.set_title('Learning Rate Changes', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 배치 크기
        ax3.plot(timestamps, batch_sizes, 'm-', linewidth=2)
        ax3.set_ylabel('Batch Size', fontsize=10)
        ax3.set_title('Batch Size Changes', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 성능 점수
        performance_scores = [m.performance_score for m in metrics]
        ax4.plot(timestamps, performance_scores, 'orange', linewidth=2)
        ax4.set_ylabel('Performance Score', fontsize=10)
        ax4.set_title('Overall Performance Score', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 시간 축 포맷
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 저장
        if not save_path:
            save_path = os.path.join(self.output_dir, f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_efficiency_chart(self, save_path: str = None) -> str:
        """효율성 차트 생성"""
        metrics = self.db.get_all_metrics()
        
        if len(metrics) < 10:
            return None
        
        # 시간별 효율성 계산
        window_size = 10
        timestamps = []
        efficiencies = []
        
        for i in range(window_size, len(metrics)):
            window_metrics = metrics[i-window_size:i]
            analyzer = PerformanceAnalyzer(self.db)
            efficiency = analyzer.calculate_learning_efficiency(window_metrics)
            
            timestamps.append(datetime.fromisoformat(metrics[i].timestamp))
            efficiencies.append(efficiency)
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, efficiencies, 'purple', linewidth=2, marker='o', markersize=4)
        plt.ylabel('Learning Efficiency', fontsize=12)
        plt.xlabel('Time', fontsize=12)
        plt.title('Learning Efficiency Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 평균선 추가
        avg_efficiency = np.mean(efficiencies)
        plt.axhline(y=avg_efficiency, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_efficiency:.3f}')
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 저장
        if not save_path:
            save_path = os.path.join(self.output_dir, f"efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

class LearningProgressTracker:
    """학습 진행률 추적기"""
    
    def __init__(self, db_path: str = "learning_progress.db"):
        self.db = LearningProgressDatabase(db_path)
        self.analyzer = PerformanceAnalyzer(self.db)
        self.visualizer = VisualizationGenerator(self.db)
        
        # 이정표 설정
        self.milestones = [
            {"type": "accuracy", "description": "70% 정확도 달성", "target": 0.7},
            {"type": "accuracy", "description": "80% 정확도 달성", "target": 0.8},
            {"type": "accuracy", "description": "90% 정확도 달성", "target": 0.9},
            {"type": "loss", "description": "0.3 이하 손실 달성", "target": 0.3},
            {"type": "loss", "description": "0.1 이하 손실 달성", "target": 0.1}
        ]
        
        logger.info("📈 학습 진행률 추적기 초기화 완료")
    
    def record_learning_step(self, epoch: int, loss: float, accuracy: float,
                           learning_rate: float, batch_size: int, data_points: int,
                           training_time: float, model_version: str = "default"):
        """학습 단계 기록"""
        
        # 성능 점수 계산 (정확도 기반, 손실 패널티)
        performance_score = accuracy - (loss * 0.1)
        
        metric = LearningMetric(
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            batch_size=batch_size,
            data_points=data_points,
            training_time=training_time,
            model_version=model_version,
            performance_score=performance_score
        )
        
        self.db.record_metric(metric)
        
        # 이정표 확인
        self._check_milestones(metric)
        
        logger.info(f"📊 학습 단계 기록: 에포크={epoch}, 손실={loss:.4f}, 정확도={accuracy:.4f}")
    
    def _check_milestones(self, metric: LearningMetric):
        """이정표 확인"""
        for milestone in self.milestones:
            if milestone["type"] == "accuracy":
                if metric.accuracy >= milestone["target"]:
                    self.db.add_milestone(
                        milestone["type"],
                        milestone["description"],
                        metric.accuracy,
                        milestone["target"]
                    )
                    logger.info(f"🎯 이정표 달성: {milestone['description']}")
            
            elif milestone["type"] == "loss":
                if metric.loss <= milestone["target"]:
                    self.db.add_milestone(
                        milestone["type"],
                        milestone["description"],
                        metric.loss,
                        milestone["target"]
                    )
                    logger.info(f"🎯 이정표 달성: {milestone['description']}")
    
    def get_current_progress(self) -> ProgressSummary:
        """현재 진행률 조회"""
        all_metrics = self.db.get_all_metrics()
        
        if not all_metrics:
            return ProgressSummary(
                total_epochs=0,
                total_training_time=0.0,
                total_data_points=0,
                best_accuracy=0.0,
                best_loss=999.0,
                current_performance=0.0,
                improvement_rate=0.0,
                estimated_completion="unknown",
                learning_efficiency=0.0
            )
        
        # 통계 계산
        total_epochs = len(all_metrics)
        total_training_time = sum(m.training_time for m in all_metrics)
        total_data_points = sum(m.data_points for m in all_metrics)
        best_accuracy = max(m.accuracy for m in all_metrics)
        best_loss = min(m.loss for m in all_metrics)
        current_performance = all_metrics[-1].performance_score
        
        # 개선률 계산
        trend_analysis = self.analyzer.analyze_learning_trend(24)
        improvement_rate = trend_analysis.get("improvement_rate", 0.0)
        
        # 학습 효율성
        learning_efficiency = self.analyzer.calculate_learning_efficiency(all_metrics[-20:])
        
        # 완료 예상 시간
        prediction = self.analyzer.predict_performance(0.9)  # 90% 정확도 목표
        estimated_completion = prediction.get("estimated_completion", "unknown")
        
        summary = ProgressSummary(
            total_epochs=total_epochs,
            total_training_time=total_training_time / 3600,  # 시간 단위
            total_data_points=total_data_points,
            best_accuracy=best_accuracy,
            best_loss=best_loss,
            current_performance=current_performance,
            improvement_rate=improvement_rate,
            estimated_completion=estimated_completion,
            learning_efficiency=learning_efficiency
        )
        
        # 스냅샷 저장
        self.db.record_progress_snapshot(summary)
        
        return summary
    
    def generate_progress_report(self) -> Dict:
        """진행률 보고서 생성"""
        progress = self.get_current_progress()
        trend_analysis = self.analyzer.analyze_learning_trend(24)
        
        # 시각화 생성
        learning_curve_path = self.visualizer.generate_learning_curve()
        dashboard_path = self.visualizer.generate_performance_dashboard()
        efficiency_path = self.visualizer.generate_efficiency_chart()
        
        report = {
            "progress_summary": asdict(progress),
            "trend_analysis": trend_analysis,
            "visualizations": {
                "learning_curve": learning_curve_path,
                "dashboard": dashboard_path,
                "efficiency_chart": efficiency_path
            },
            "recommendations": self._generate_recommendations(progress, trend_analysis),
            "generated_at": datetime.now().isoformat()
        }
        
        # 보고서 파일 저장
        report_file = f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 진행률 보고서 생성: {report_file}")
        
        return report
    
    def _generate_recommendations(self, progress: ProgressSummary, trend: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 성능 기반 권장사항
        if progress.current_performance < 0.6:
            recommendations.append("모델 성능이 낮습니다. 학습률 조정을 고려해보세요.")
        
        if progress.improvement_rate < 0.01:
            recommendations.append("개선률이 낮습니다. 데이터 품질 점검 또는 모델 구조 변경을 고려해보세요.")
        
        if progress.learning_efficiency < 0.3:
            recommendations.append("학습 효율성이 낮습니다. 배치 크기나 최적화 알고리즘을 조정해보세요.")
        
        # 트렌드 기반 권장사항
        if trend.get("trend") == "degrading":
            recommendations.append("성능이 하락하고 있습니다. 과적합 가능성을 확인해보세요.")
        
        if trend.get("learning_speed", 0) < 1:
            recommendations.append("학습 속도가 느립니다. 더 빈번한 학습 스케줄을 고려해보세요.")
        
        if not recommendations:
            recommendations.append("현재 학습이 잘 진행되고 있습니다. 계속 유지하세요!")
        
        return recommendations
    
    def start_real_time_tracking(self, interval: int = 300):
        """실시간 추적 시작 (5분마다)"""
        logger.info(f"🔄 실시간 진행률 추적 시작 (간격: {interval}초)")
        
        def tracking_loop():
            while True:
                try:
                    progress = self.get_current_progress()
                    logger.info(f"📈 현재 진행률: "
                              f"에포크={progress.total_epochs}, "
                              f"최고정확도={progress.best_accuracy:.3f}, "
                              f"효율성={progress.learning_efficiency:.3f}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"실시간 추적 오류: {e}")
                    time.sleep(60)
        
        tracking_thread = threading.Thread(target=tracking_loop, daemon=True)
        tracking_thread.start()

def test_progress_tracker():
    """진행률 추적기 테스트"""
    print("🧪 학습 진행률 추적 시스템 테스트")
    print("=" * 60)
    
    # 추적기 초기화
    tracker = LearningProgressTracker("test_progress.db")
    
    # 시뮬레이션 학습 데이터
    print("📚 시뮬레이션 학습 데이터 생성 중...")
    
    import random
    
    for epoch in range(1, 21):  # 20 에포크
        # 시뮬레이션: 점진적 개선
        base_accuracy = 0.5 + (epoch * 0.02) + random.uniform(-0.05, 0.05)
        base_loss = 1.0 - (epoch * 0.03) + random.uniform(-0.1, 0.1)
        
        base_accuracy = max(0.0, min(1.0, base_accuracy))
        base_loss = max(0.1, base_loss)
        
        lr = 0.001 * (0.95 ** (epoch // 5))  # 학습률 감소
        batch_size = 16 + (epoch // 10) * 8  # 배치 크기 증가
        
        tracker.record_learning_step(
            epoch=epoch,
            loss=base_loss,
            accuracy=base_accuracy,
            learning_rate=lr,
            batch_size=batch_size,
            data_points=random.randint(50, 200),
            training_time=random.uniform(30, 120),
            model_version="test_v1.0"
        )
        
        print(f"에포크 {epoch:2d}: 정확도={base_accuracy:.3f}, 손실={base_loss:.3f}")
        
        time.sleep(0.1)  # 시뮬레이션 딜레이
    
    # 진행률 보고서 생성
    print("\n📋 진행률 보고서 생성 중...")
    report = tracker.generate_progress_report()
    
    print(f"\n📊 진행률 요약:")
    summary = report["progress_summary"]
    print(f"  총 에포크: {summary['total_epochs']}")
    print(f"  총 학습 시간: {summary['total_training_time']:.2f}시간")
    print(f"  최고 정확도: {summary['best_accuracy']:.3f}")
    print(f"  최저 손실: {summary['best_loss']:.3f}")
    print(f"  학습 효율성: {summary['learning_efficiency']:.3f}")
    
    print(f"\n🎯 권장사항:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n📈 생성된 시각화:")
    for name, path in report["visualizations"].items():
        if path:
            print(f"  {name}: {path}")
    
    print("\n🎉 진행률 추적 시스템 테스트 완료!")

if __name__ == "__main__":
    test_progress_tracker()