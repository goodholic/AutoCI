#!/usr/bin/env python3
"""
AutoCI Performance Profiler - 성능 벤치마크 및 프로파일링 시스템
"""

import os
import sys
import time
import asyncio
import psutil
import json
import statistics
import cProfile
import pstats
import memory_profiler
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    iterations: int
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass 
class PerformanceReport:
    """성능 리포트"""
    test_name: str
    results: List[BenchmarkResult]
    summary: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime

class PerformanceProfiler:
    """성능 프로파일러"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("performance_reports")
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmark_results: List[BenchmarkResult] = []
        
        # 성능 임계값
        self.thresholds = {
            "max_duration": 5.0,  # 초
            "max_memory": 500.0,  # MB
            "max_cpu": 80.0,      # %
            "min_throughput": 100.0  # ops/sec
        }
    
    def benchmark(self, name: str, iterations: int = 10):
        """벤치마크 데코레이터"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                return await self._run_async_benchmark(name, func, iterations, args, kwargs)
            
            def sync_wrapper(*args, **kwargs):
                return self._run_sync_benchmark(name, func, iterations, args, kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _run_async_benchmark(self, name: str, func: Callable, iterations: int, args, kwargs):
        """비동기 함수 벤치마크"""
        results = []
        
        for i in range(iterations):
            # 시스템 상태 측정 시작
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent()
            start_time = time.perf_counter()
            
            # 함수 실행
            result = await func(*args, **kwargs)
            
            # 시스템 상태 측정 종료
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            # 결과 기록
            benchmark_result = BenchmarkResult(
                name=f"{name}_iteration_{i+1}",
                duration=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=max(end_cpu - start_cpu, 0),
                iterations=1,
                timestamp=datetime.now(),
                metadata={"iteration": i+1, "total_iterations": iterations}
            )
            
            results.append(benchmark_result)
            self.benchmark_results.append(benchmark_result)
            
            # 약간의 쿨다운
            await asyncio.sleep(0.1)
        
        # 요약 통계 생성
        self._generate_summary_stats(name, results)
        
        return result
    
    def _run_sync_benchmark(self, name: str, func: Callable, iterations: int, args, kwargs):
        """동기 함수 벤치마크"""
        results = []
        
        for i in range(iterations):
            # 시스템 상태 측정 시작
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_cpu = process.cpu_percent()
            start_time = time.perf_counter()
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 시스템 상태 측정 종료
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            # 결과 기록
            benchmark_result = BenchmarkResult(
                name=f"{name}_iteration_{i+1}",
                duration=end_time - start_time,
                memory_usage=end_memory - start_memory,
                cpu_usage=max(end_cpu - start_cpu, 0),
                iterations=1,
                timestamp=datetime.now(),
                metadata={"iteration": i+1, "total_iterations": iterations}
            )
            
            results.append(benchmark_result)
            self.benchmark_results.append(benchmark_result)
            
            # 약간의 쿨다운
            time.sleep(0.1)
        
        # 요약 통계 생성
        self._generate_summary_stats(name, results)
        
        return result
    
    def _generate_summary_stats(self, name: str, results: List[BenchmarkResult]):
        """요약 통계 생성"""
        durations = [r.duration for r in results]
        memory_usages = [r.memory_usage for r in results]
        cpu_usages = [r.cpu_usage for r in results]
        
        summary = {
            "name": name,
            "iterations": len(results),
            "duration": {
                "mean": statistics.mean(durations),
                "median": statistics.median(durations),
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
                "min": min(durations),
                "max": max(durations)
            },
            "memory": {
                "mean": statistics.mean(memory_usages),
                "median": statistics.median(memory_usages),
                "max": max(memory_usages)
            },
            "cpu": {
                "mean": statistics.mean(cpu_usages),
                "median": statistics.median(cpu_usages),
                "max": max(cpu_usages)
            },
            "throughput": len(results) / sum(durations) if sum(durations) > 0 else 0
        }
        
        print(f"\n📊 {name} 벤치마크 결과:")
        print(f"  평균 실행시간: {summary['duration']['mean']:.4f}초")
        print(f"  처리량: {summary['throughput']:.2f} ops/sec")
        print(f"  최대 메모리 사용: {summary['memory']['max']:.2f}MB")
        print(f"  평균 CPU 사용: {summary['cpu']['mean']:.2f}%")
        
        # 성능 경고 확인
        self._check_performance_warnings(name, summary)
    
    def _check_performance_warnings(self, name: str, summary: Dict[str, Any]):
        """성능 경고 확인"""
        warnings = []
        
        if summary['duration']['mean'] > self.thresholds['max_duration']:
            warnings.append(f"⚠️  평균 실행시간이 임계값을 초과했습니다: {summary['duration']['mean']:.4f}초 > {self.thresholds['max_duration']}초")
        
        if summary['memory']['max'] > self.thresholds['max_memory']:
            warnings.append(f"⚠️  최대 메모리 사용량이 임계값을 초과했습니다: {summary['memory']['max']:.2f}MB > {self.thresholds['max_memory']}MB")
        
        if summary['cpu']['max'] > self.thresholds['max_cpu']:
            warnings.append(f"⚠️  최대 CPU 사용률이 임계값을 초과했습니다: {summary['cpu']['max']:.2f}% > {self.thresholds['max_cpu']}%")
        
        if summary['throughput'] < self.thresholds['min_throughput']:
            warnings.append(f"⚠️  처리량이 임계값보다 낮습니다: {summary['throughput']:.2f} ops/sec < {self.thresholds['min_throughput']} ops/sec")
        
        for warning in warnings:
            print(warning)
    
    def profile_memory(self, func, *args, **kwargs):
        """메모리 프로파일링"""
        print(f"\n🧠 메모리 프로파일링: {func.__name__}")
        
        @memory_profiler.profile
        def profiled_func():
            return func(*args, **kwargs)
        
        return profiled_func()
    
    def profile_cpu(self, func, *args, **kwargs):
        """CPU 프로파일링"""
        print(f"\n⚡ CPU 프로파일링: {func.__name__}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # 프로파일 결과 저장
        profile_file = self.output_dir / f"cpu_profile_{func.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
        profiler.dump_stats(str(profile_file))
        
        # 상위 10개 함수 출력
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return result
    
    def run_stress_test(self, func, duration_seconds: int = 60, concurrent_tasks: int = 10):
        """스트레스 테스트"""
        print(f"\n🔥 스트레스 테스트: {func.__name__} ({duration_seconds}초, {concurrent_tasks}개 동시 작업)")
        
        async def stress_task():
            start_time = time.time()
            task_count = 0
            
            while time.time() - start_time < duration_seconds:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
                task_count += 1
                await asyncio.sleep(0.01)  # 약간의 쿨다운
            
            return task_count
        
        async def run_stress():
            # 시스템 상태 모니터링 시작
            process = psutil.Process()
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024
            
            # 동시 작업 실행
            tasks = [stress_task() for _ in range(concurrent_tasks)]
            results = await asyncio.gather(*tasks)
            
            # 시스템 상태 측정 종료
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            # 결과 분석
            total_operations = sum(results)
            elapsed_time = end_time - start_time
            ops_per_second = total_operations / elapsed_time
            memory_increase = end_memory - start_memory
            
            print(f"  총 작업 수: {total_operations}")
            print(f"  처리량: {ops_per_second:.2f} ops/sec")
            print(f"  메모리 증가: {memory_increase:.2f}MB")
            
            # 스트레스 테스트 결과 저장
            stress_result = BenchmarkResult(
                name=f"stress_test_{func.__name__}",
                duration=elapsed_time,
                memory_usage=memory_increase,
                cpu_usage=psutil.cpu_percent(),
                iterations=total_operations,
                timestamp=datetime.now(),
                metadata={
                    "concurrent_tasks": concurrent_tasks,
                    "ops_per_second": ops_per_second,
                    "duration_seconds": duration_seconds
                }
            )
            
            self.benchmark_results.append(stress_result)
            
            return stress_result
        
        if asyncio.get_event_loop().is_running():
            # 이미 이벤트 루프가 실행 중인 경우
            return asyncio.create_task(run_stress())
        else:
            return asyncio.run(run_stress())
    
    def generate_performance_report(self, report_name: str = None) -> PerformanceReport:
        """성능 리포트 생성"""
        if not report_name:
            report_name = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n📋 성능 리포트 생성: {report_name}")
        
        # 전체 요약 통계
        all_durations = [r.duration for r in self.benchmark_results]
        all_memory = [r.memory_usage for r in self.benchmark_results]
        all_cpu = [r.cpu_usage for r in self.benchmark_results]
        
        summary = {
            "total_benchmarks": len(self.benchmark_results),
            "avg_duration": statistics.mean(all_durations) if all_durations else 0,
            "max_duration": max(all_durations) if all_durations else 0,
            "avg_memory": statistics.mean(all_memory) if all_memory else 0,
            "max_memory": max(all_memory) if all_memory else 0,
            "avg_cpu": statistics.mean(all_cpu) if all_cpu else 0,
            "max_cpu": max(all_cpu) if all_cpu else 0
        }
        
        # 추천사항 생성
        recommendations = self._generate_recommendations(summary)
        
        # 리포트 객체 생성
        report = PerformanceReport(
            test_name=report_name,
            results=self.benchmark_results.copy(),
            summary=summary,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        # 리포트 저장
        self._save_report(report)
        
        # 시각화 생성
        self._generate_visualizations(report)
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, float]) -> List[str]:
        """성능 최적화 추천사항 생성"""
        recommendations = []
        
        if summary['max_duration'] > self.thresholds['max_duration']:
            recommendations.append("실행 시간이 긴 함수들에 대해 알고리즘 최적화 검토가 필요합니다.")
            recommendations.append("비동기 처리나 병렬 처리 도입을 고려하세요.")
        
        if summary['max_memory'] > self.thresholds['max_memory']:
            recommendations.append("메모리 사용량이 높습니다. 메모리 풀링이나 객체 재사용을 고려하세요.")
            recommendations.append("대용량 데이터 처리 시 스트리밍 방식을 도입하세요.")
        
        if summary['max_cpu'] > self.thresholds['max_cpu']:
            recommendations.append("CPU 사용률이 높습니다. 작업을 더 작은 단위로 분할하세요.")
            recommendations.append("I/O 바운드 작업과 CPU 바운드 작업을 분리하세요.")
        
        if summary['avg_duration'] > 1.0:
            recommendations.append("평균 응답 시간이 1초를 초과합니다. 캐싱 전략을 도입하세요.")
        
        if not recommendations:
            recommendations.append("성능이 양호합니다. 현재 최적화 상태를 유지하세요.")
        
        return recommendations
    
    def _save_report(self, report: PerformanceReport):
        """리포트 저장"""
        # JSON 형식으로 저장
        json_file = self.output_dir / f"{report.test_name}.json"
        report_dict = asdict(report)
        
        # datetime 객체를 문자열로 변환
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, default=serialize_datetime, ensure_ascii=False)
        
        # Markdown 형식으로 저장
        md_file = self.output_dir / f"{report.test_name}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Performance Report: {report.test_name}\n\n")
            f.write(f"**Generated:** {report.generated_at.isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Total Benchmarks: {report.summary['total_benchmarks']}\n")
            f.write(f"- Average Duration: {report.summary['avg_duration']:.4f}s\n")
            f.write(f"- Max Duration: {report.summary['max_duration']:.4f}s\n")
            f.write(f"- Average Memory: {report.summary['avg_memory']:.2f}MB\n")
            f.write(f"- Max Memory: {report.summary['max_memory']:.2f}MB\n")
            f.write(f"- Average CPU: {report.summary['avg_cpu']:.2f}%\n")
            f.write(f"- Max CPU: {report.summary['max_cpu']:.2f}%\n\n")
            
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            f.write("\n")
            
            f.write("## Detailed Results\n\n")
            for result in report.results:
                f.write(f"### {result.name}\n")
                f.write(f"- Duration: {result.duration:.4f}s\n")
                f.write(f"- Memory: {result.memory_usage:.2f}MB\n")
                f.write(f"- CPU: {result.cpu_usage:.2f}%\n")
                f.write(f"- Timestamp: {result.timestamp.isoformat()}\n\n")
        
        print(f"📊 리포트 저장 완료:")
        print(f"  - JSON: {json_file}")
        print(f"  - Markdown: {md_file}")
    
    def _generate_visualizations(self, report: PerformanceReport):
        """시각화 생성"""
        try:
            # 데이터 준비
            df = pd.DataFrame([asdict(result) for result in report.results])
            
            # 시간별 성능 트렌드
            plt.figure(figsize=(15, 10))
            
            # 실행 시간 차트
            plt.subplot(2, 2, 1)
            plt.plot(df['timestamp'], df['duration'])
            plt.title('Execution Time Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Duration (seconds)')
            plt.xticks(rotation=45)
            
            # 메모리 사용량 차트
            plt.subplot(2, 2, 2)
            plt.plot(df['timestamp'], df['memory_usage'])
            plt.title('Memory Usage Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Memory (MB)')
            plt.xticks(rotation=45)
            
            # CPU 사용률 차트
            plt.subplot(2, 2, 3)
            plt.plot(df['timestamp'], df['cpu_usage'])
            plt.title('CPU Usage Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('CPU (%)')
            plt.xticks(rotation=45)
            
            # 성능 분포 히스토그램
            plt.subplot(2, 2, 4)
            plt.hist(df['duration'], bins=20, alpha=0.7)
            plt.title('Duration Distribution')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            
            # 차트 저장
            chart_file = self.output_dir / f"{report.test_name}_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 시각화 저장: {chart_file}")
            
        except ImportError:
            print("⚠️  matplotlib 또는 pandas가 설치되지 않아 시각화를 생성할 수 없습니다.")
        except Exception as e:
            print(f"⚠️  시각화 생성 중 오류: {e}")
    
    def compare_reports(self, report1: PerformanceReport, report2: PerformanceReport):
        """성능 리포트 비교"""
        print(f"\n🔍 성능 리포트 비교: {report1.test_name} vs {report2.test_name}")
        
        # 주요 메트릭 비교
        metrics = ['avg_duration', 'max_duration', 'avg_memory', 'max_memory', 'avg_cpu', 'max_cpu']
        
        print("메트릭 비교:")
        for metric in metrics:
            val1 = report1.summary[metric]
            val2 = report2.summary[metric]
            change = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
            
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"  {metric}: {val1:.4f} → {val2:.4f} ({direction} {change:+.2f}%)")
    
    def clear_results(self):
        """결과 초기화"""
        self.benchmark_results.clear()
        print("📝 벤치마크 결과가 초기화되었습니다.")

# 사용 예제 및 테스트 함수들
async def example_async_function():
    """예제 비동기 함수"""
    await asyncio.sleep(0.1)
    return "async result"

def example_sync_function():
    """예제 동기 함수"""
    time.sleep(0.05)
    return "sync result"

def memory_intensive_function():
    """메모리 집약적 함수"""
    data = [list(range(1000)) for _ in range(1000)]
    return len(data)

def cpu_intensive_function():
    """CPU 집약적 함수"""
    total = 0
    for i in range(1000000):
        total += i * i
    return total

def main():
    """메인 실행 함수"""
    print("🚀 AutoCI Performance Profiler")
    print("=" * 60)
    
    profiler = PerformanceProfiler()
    
    # 벤치마크 테스트
    print("\n📊 벤치마크 테스트 실행...")
    
    # 동기 함수 벤치마크
    profiler.benchmark("sync_function", 5)(example_sync_function)()
    
    # 비동기 함수 벤치마크
    async def async_test():
        await profiler.benchmark("async_function", 5)(example_async_function)()
    
    asyncio.run(async_test())
    
    # 메모리 프로파일링
    print("\n🧠 메모리 프로파일링...")
    profiler.profile_memory(memory_intensive_function)
    
    # CPU 프로파일링  
    print("\n⚡ CPU 프로파일링...")
    profiler.profile_cpu(cpu_intensive_function)
    
    # 스트레스 테스트
    print("\n🔥 스트레스 테스트...")
    asyncio.run(profiler.run_stress_test(example_async_function, duration_seconds=5, concurrent_tasks=3))
    
    # 성능 리포트 생성
    report = profiler.generate_performance_report("autoci_performance_test")
    
    print(f"\n✅ 성능 분석 완료!")
    print(f"📁 결과 저장 위치: {profiler.output_dir}")

if __name__ == "__main__":
    main()