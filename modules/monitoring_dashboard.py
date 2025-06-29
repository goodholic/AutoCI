#!/usr/bin/env python3
"""
AutoCI 실시간 모니터링 대시보드
시스템 상태, AI 작업, 성능 메트릭을 실시간으로 표시
"""

import asyncio
import time
import os
import psutil
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    gpu_percent: Optional[float] = None
    temperature: Optional[float] = None

@dataclass
class AITaskStatus:
    """AI 작업 상태"""
    task_id: str
    task_type: str
    status: str  # running, completed, failed
    progress: float
    start_time: float
    estimated_completion: Optional[float] = None
    details: Dict[str, Any] = None

class MonitoringDashboard:
    """실시간 모니터링 대시보드"""
    
    def __init__(self):
        self.logger = logging.getLogger("MonitoringDashboard")
        self.is_running = False
        self.metrics_history: List[SystemMetrics] = []
        self.ai_tasks: List[AITaskStatus] = []
        self.max_history = 100  # 최대 메트릭 기록 수
        
        # 가상의 AI 작업들 (데모용)
        self.demo_tasks = [
            "게임 프로젝트 생성",
            "씬 구성 최적화", 
            "리소스 자동 생성",
            "코드 품질 검사",
            "성능 벤치마크",
            "멀티플레이어 테스트"
        ]
        
    async def start_real_time_monitoring(self):
        """실시간 모니터링 시작"""
        print("📊 AutoCI 실시간 모니터링 대시보드")
        print("=" * 80)
        print("시스템 상태와 AI 작업을 실시간으로 모니터링합니다.")
        print("종료하려면 Ctrl+C를 누르세요.")
        print("=" * 80)
        
        self.is_running = True
        
        try:
            # 모니터링 작업들을 동시에 실행
            await asyncio.gather(
                self._collect_system_metrics(),
                self._monitor_ai_tasks(),
                self._display_dashboard(),
                self._simulate_ai_activities()  # 데모용 AI 활동 시뮬레이션
            )
        except KeyboardInterrupt:
            print("\n\n🛑 모니터링이 중지되었습니다.")
            self.is_running = False
    
    async def _collect_system_metrics(self):
        """시스템 메트릭 수집"""
        while self.is_running:
            try:
                # CPU, 메모리, 디스크 사용률
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 네트워크 I/O
                network = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
                
                # GPU 사용률 (시뮬레이션)
                gpu_percent = self._get_gpu_usage()
                
                # 온도 (시뮬레이션)
                temperature = self._get_system_temperature()
                
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_percent=disk.percent,
                    network_io=network_io,
                    gpu_percent=gpu_percent,
                    temperature=temperature
                )
                
                # 기록 추가
                self.metrics_history.append(metrics)
                
                # 기록 수 제한
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
            except Exception as e:
                self.logger.error(f"메트릭 수집 오류: {e}")
            
            await asyncio.sleep(2)  # 2초마다 수집
    
    def _get_gpu_usage(self) -> Optional[float]:
        """GPU 사용률 가져오기 (시뮬레이션)"""
        try:
            # nvidia-ml-py3가 설치되어 있다면 실제 GPU 정보 사용
            # 여기서는 시뮬레이션
            return random.uniform(20, 80)
        except:
            return None
    
    def _get_system_temperature(self) -> Optional[float]:
        """시스템 온도 가져오기 (시뮬레이션)"""
        try:
            # psutil로 온도 정보 시도
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
            
            # 시뮬레이션 값
            return random.uniform(35, 65)
        except:
            return None
    
    async def _monitor_ai_tasks(self):
        """AI 작업 모니터링"""
        while self.is_running:
            # 완료된 작업 제거
            self.ai_tasks = [task for task in self.ai_tasks 
                           if task.status != "completed" or 
                           time.time() - task.start_time < 30]  # 30초간 완료 상태 유지
            
            # 실행 중인 작업들의 진행률 업데이트
            for task in self.ai_tasks:
                if task.status == "running":
                    # 진행률 업데이트 (시뮬레이션)
                    elapsed = time.time() - task.start_time
                    if elapsed < 10:  # 10초 작업이라고 가정
                        task.progress = min(95, (elapsed / 10) * 100)
                    else:
                        task.status = "completed"
                        task.progress = 100
            
            await asyncio.sleep(1)
    
    async def _simulate_ai_activities(self):
        """AI 활동 시뮬레이션 (데모용)"""
        while self.is_running:
            # 랜덤하게 새 AI 작업 시작
            if random.random() < 0.3:  # 30% 확률
                task_type = random.choice(self.demo_tasks)
                task_id = f"ai_{int(time.time())}_{random.randint(1000, 9999)}"
                
                new_task = AITaskStatus(
                    task_id=task_id,
                    task_type=task_type,
                    status="running",
                    progress=0.0,
                    start_time=time.time(),
                    details={"priority": random.choice(["high", "medium", "low"])}
                )
                
                self.ai_tasks.append(new_task)
            
            await asyncio.sleep(5)  # 5초마다 체크
    
    async def _display_dashboard(self):
        """대시보드 화면 표시"""
        while self.is_running:
            # 화면 지우기
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # 헤더
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"📊 AutoCI 실시간 모니터링 대시보드 - {current_time}")
            print("=" * 80)
            
            # 시스템 메트릭 표시
            await self._display_system_metrics()
            
            # AI 작업 상태 표시
            await self._display_ai_tasks()
            
            # 성능 차트 표시
            await self._display_performance_charts()
            
            # 시스템 알림 표시
            await self._display_system_alerts()
            
            # 하단 정보
            print("=" * 80)
            print("🔄 2초마다 자동 업데이트 | Ctrl+C로 종료")
            
            await asyncio.sleep(2)
    
    async def _display_system_metrics(self):
        """시스템 메트릭 표시"""
        if not self.metrics_history:
            print("📊 시스템 메트릭: 수집 중...")
            return
        
        latest = self.metrics_history[-1]
        
        print("📊 시스템 상태:")
        print(f"  🖥️  CPU:      {self._create_bar(latest.cpu_percent, 100)} {latest.cpu_percent:.1f}%")
        print(f"  💾 메모리:    {self._create_bar(latest.memory_percent, 100)} {latest.memory_percent:.1f}%")
        print(f"  💿 디스크:    {self._create_bar(latest.disk_percent, 100)} {latest.disk_percent:.1f}%")
        
        if latest.gpu_percent:
            print(f"  🎮 GPU:      {self._create_bar(latest.gpu_percent, 100)} {latest.gpu_percent:.1f}%")
        
        if latest.temperature:
            temp_color = "🟢" if latest.temperature < 50 else "🟡" if latest.temperature < 70 else "🔴"
            print(f"  🌡️  온도:     {temp_color} {latest.temperature:.1f}°C")
        
        # 네트워크 I/O
        network = latest.network_io
        print(f"  🌐 네트워크:  ↑ {self._format_bytes(network['bytes_sent'])} "
              f"↓ {self._format_bytes(network['bytes_recv'])}")
        print()
    
    async def _display_ai_tasks(self):
        """AI 작업 상태 표시"""
        print("🤖 AI 작업 상태:")
        
        if not self.ai_tasks:
            print("  💤 현재 실행 중인 AI 작업이 없습니다.")
        else:
            running_tasks = [task for task in self.ai_tasks if task.status == "running"]
            completed_tasks = [task for task in self.ai_tasks if task.status == "completed"]
            
            print(f"  🔄 실행 중: {len(running_tasks)}개 | ✅ 완료: {len(completed_tasks)}개")
            
            # 실행 중인 작업들 표시
            for task in running_tasks[:5]:  # 최대 5개만 표시
                progress_bar = self._create_bar(task.progress, 100)
                elapsed = time.time() - task.start_time
                print(f"    🔧 {task.task_type:20} {progress_bar} {task.progress:.1f}% ({elapsed:.1f}s)")
            
            # 최근 완료된 작업들 표시 (최대 3개)
            for task in completed_tasks[-3:]:
                print(f"    ✅ {task.task_type:20} 완료 ({task.task_id[-4:]})")
        print()
    
    async def _display_performance_charts(self):
        """성능 차트 표시"""
        if len(self.metrics_history) < 2:
            return
        
        print("📈 성능 추이 (최근 20개 데이터):")
        
        # 최근 20개 데이터만 사용
        recent_metrics = self.metrics_history[-20:]
        
        # CPU 차트
        cpu_values = [m.cpu_percent for m in recent_metrics]
        cpu_chart = self._create_sparkline(cpu_values, 100)
        print(f"  🖥️  CPU:    {cpu_chart} (avg: {sum(cpu_values)/len(cpu_values):.1f}%)")
        
        # 메모리 차트
        memory_values = [m.memory_percent for m in recent_metrics]
        memory_chart = self._create_sparkline(memory_values, 100)
        print(f"  💾 메모리:  {memory_chart} (avg: {sum(memory_values)/len(memory_values):.1f}%)")
        
        # GPU 차트 (있는 경우)
        gpu_values = [m.gpu_percent for m in recent_metrics if m.gpu_percent is not None]
        if gpu_values:
            gpu_chart = self._create_sparkline(gpu_values, 100)
            print(f"  🎮 GPU:    {gpu_chart} (avg: {sum(gpu_values)/len(gpu_values):.1f}%)")
        
        print()
    
    async def _display_system_alerts(self):
        """시스템 알림 표시"""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        alerts = []
        
        # CPU 사용률 경고
        if latest.cpu_percent > 80:
            alerts.append("🔴 높은 CPU 사용률!")
        elif latest.cpu_percent > 60:
            alerts.append("🟡 보통 CPU 사용률")
        
        # 메모리 사용률 경고
        if latest.memory_percent > 85:
            alerts.append("🔴 높은 메모리 사용률!")
        elif latest.memory_percent > 70:
            alerts.append("🟡 보통 메모리 사용률")
        
        # 온도 경고
        if latest.temperature and latest.temperature > 70:
            alerts.append("🔴 높은 시스템 온도!")
        elif latest.temperature and latest.temperature > 60:
            alerts.append("🟡 보통 시스템 온도")
        
        # AI 작업 통계
        running_count = len([task for task in self.ai_tasks if task.status == "running"])
        if running_count > 5:
            alerts.append(f"⚡ 높은 AI 작업 부하 ({running_count}개)")
        elif running_count > 0:
            alerts.append(f"🟢 AI 작업 정상 실행 중 ({running_count}개)")
        
        if alerts:
            print("🚨 시스템 알림:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("🟢 시스템 상태: 모든 지표가 정상 범위입니다.")
    
    def _create_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """진행률 바 생성"""
        if max_value == 0:
            return "█" * width
        
        filled = int((value / max_value) * width)
        filled = max(0, min(width, filled))
        
        # 색상 결정
        if value / max_value < 0.5:
            fill_char = "█"
        elif value / max_value < 0.8:
            fill_char = "█"
        else:
            fill_char = "█"
        
        empty_char = "░"
        
        return f"[{fill_char * filled}{empty_char * (width - filled)}]"
    
    def _create_sparkline(self, values: List[float], max_value: float) -> str:
        """스파크라인 차트 생성"""
        if not values:
            return ""
        
        # 스파크라인 문자들
        spark_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        
        # 값들을 정규화
        if max_value == 0:
            normalized = [0] * len(values)
        else:
            normalized = [min(1.0, v / max_value) for v in values]
        
        # 스파크라인 생성
        sparkline = ""
        for value in normalized:
            char_index = int(value * (len(spark_chars) - 1))
            sparkline += spark_chars[char_index]
        
        return sparkline
    
    def _format_bytes(self, bytes_value: int) -> str:
        """바이트를 읽기 쉬운 형태로 포맷"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """모니터링 리포트 생성"""
        if not self.metrics_history:
            return {}
        
        # 통계 계산
        recent_metrics = self.metrics_history[-10:]  # 최근 10개
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        
        # AI 작업 통계
        total_tasks = len(self.ai_tasks)
        running_tasks = len([t for t in self.ai_tasks if t.status == "running"])
        completed_tasks = len([t for t in self.ai_tasks if t.status == "completed"])
        
        report = {
            "생성_시간": datetime.now().isoformat(),
            "모니터링_기간": f"{len(self.metrics_history)} 데이터 포인트",
            "시스템_평균": {
                "CPU": f"{avg_cpu:.1f}%",
                "메모리": f"{avg_memory:.1f}%",
                "디스크": f"{avg_disk:.1f}%"
            },
            "AI_작업_통계": {
                "총_작업": total_tasks,
                "실행_중": running_tasks,
                "완료됨": completed_tasks
            },
            "시스템_상태": "정상" if avg_cpu < 80 and avg_memory < 85 else "주의"
        }
        
        return report

# 독립 실행용
async def main():
    """테스트 실행"""
    dashboard = MonitoringDashboard()
    await dashboard.start_real_time_monitoring()

if __name__ == "__main__":
    asyncio.run(main())