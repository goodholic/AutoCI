#!/usr/bin/env python3
"""
신경망 학습 대시보드
실시간 학습 진행 상황 시각화
"""

import os
import sys
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import http.server
import socketserver
import threading

# 웹 대시보드 HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 AutoCI 신경망 학습 대시보드</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e6ed;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: #1a1f3a;
            border: 1px solid #2d3561;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .card-title {
            font-size: 1.4em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-value {
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .card-description {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #2d3561;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 15px;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        
        .progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .neural-status {
            background: #0a0e27;
            border: 2px solid #667eea;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .status-indicator {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }
        
        .status-active {
            background: #48bb78;
            box-shadow: 0 0 20px #48bb78;
        }
        
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .learning-chart {
            background: #1a1f3a;
            border: 1px solid #2d3561;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            height: 400px;
        }
        
        .pattern-list {
            background: #1a1f3a;
            border: 1px solid #2d3561;
            border-radius: 15px;
            padding: 25px;
        }
        
        .pattern-item {
            padding: 15px;
            border-bottom: 1px solid #2d3561;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }
        
        .pattern-item:hover {
            background: rgba(102, 126, 234, 0.1);
        }
        
        .pattern-item:last-child {
            border-bottom: none;
        }
        
        .emoji {
            font-size: 1.5em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            background: rgba(102, 126, 234, 0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            opacity: 0.6;
            font-size: 0.9em;
        }
        
        .refresh-info {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #667eea;
            padding: 15px 25px;
            border-radius: 30px;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.5);
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 AutoCI 신경망 학습 대시보드</h1>
            <div class="subtitle">24시간 C# 전문 AI 신경망 학습 진행 상황</div>
        </header>
        
        <div class="neural-status">
            <h2>
                <span class="status-indicator status-active"></span>
                신경망 학습 진행 중
            </h2>
            <p>마지막 업데이트: <span id="last-update">-</span></p>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">
                    <span class="emoji">🔥</span>
                    총 학습 단계
                </div>
                <div class="card-value" id="total-steps">0</div>
                <div class="card-description">신경망 가중치 업데이트 횟수</div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <span class="emoji">📚</span>
                    학습된 C# 코드
                </div>
                <div class="card-value" id="total-samples">0</div>
                <div class="card-description">GitHub/StackOverflow 수집</div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <span class="emoji">📉</span>
                    현재 Loss
                </div>
                <div class="card-value" id="current-loss">0.000</div>
                <div class="card-description">낮을수록 좋음</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="loss-bar" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <span class="emoji">🧠</span>
                    모델 파라미터
                </div>
                <div class="card-value" id="model-params">0</div>
                <div class="card-description">신경망 크기</div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <span class="emoji">⚡</span>
                    학습률
                </div>
                <div class="card-value" id="learning-rate">0.0001</div>
                <div class="card-description">현재 학습 속도</div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    <span class="emoji">🎯</span>
                    정확도
                </div>
                <div class="card-value" id="accuracy">0%</div>
                <div class="card-description">C# 코드 이해도</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="accuracy-bar" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="learning-chart">
            <h3 style="margin-bottom: 20px;">
                <span class="emoji">📊</span>
                Loss 추이 (실시간)
            </h3>
            <canvas id="loss-chart"></canvas>
        </div>
        
        <div class="pattern-list">
            <h3 style="margin-bottom: 20px;">
                <span class="emoji">🎯</span>
                최근 학습한 C# 패턴
            </h3>
            <div id="pattern-container">
                <div class="pattern-item">
                    <span>데이터를 불러오는 중...</span>
                </div>
            </div>
        </div>
        
        <div class="card" style="grid-column: 1 / -1;">
            <div class="card-title">
                <span class="emoji">📊</span>
                학습 통계
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div>Unity 패턴</div>
                    <div style="font-size: 1.5em; font-weight: bold;" id="unity-patterns">0</div>
                </div>
                <div class="stat-item">
                    <div>async/await</div>
                    <div style="font-size: 1.5em; font-weight: bold;" id="async-patterns">0</div>
                </div>
                <div class="stat-item">
                    <div>LINQ 사용</div>
                    <div style="font-size: 1.5em; font-weight: bold;" id="linq-patterns">0</div>
                </div>
                <div class="stat-item">
                    <div>디자인 패턴</div>
                    <div style="font-size: 1.5em; font-weight: bold;" id="design-patterns">0</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            AutoCI Neural v3.0 - 24시간 자동 학습하는 ChatGPT 수준 C# AI
        </div>
        
        <div class="refresh-info">
            ⏱️ 10초마다 자동 갱신
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // 차트 초기화
        const ctx = document.getElementById('loss-chart').getContext('2d');
        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#e0e6ed' },
                        grid: { color: '#2d3561' }
                    },
                    y: {
                        ticks: { color: '#e0e6ed' },
                        grid: { color: '#2d3561' }
                    }
                }
            }
        });
        
        // 데이터 업데이트 함수
        async function updateDashboard() {
            try {
                const response = await fetch('/api/neural_stats');
                const data = await response.json();
                
                // 기본 메트릭 업데이트
                document.getElementById('total-steps').textContent = 
                    data.total_steps.toLocaleString();
                document.getElementById('total-samples').textContent = 
                    data.total_samples.toLocaleString();
                document.getElementById('current-loss').textContent = 
                    data.current_loss.toFixed(4);
                document.getElementById('model-params').textContent = 
                    (data.model_parameters / 1000000).toFixed(1) + 'M';
                document.getElementById('learning-rate').textContent = 
                    data.learning_rate.toExponential(2);
                
                // 정확도
                const accuracy = (1 - data.current_loss) * 100;
                document.getElementById('accuracy').textContent = accuracy.toFixed(1) + '%';
                document.getElementById('accuracy-bar').style.width = accuracy + '%';
                
                // Loss 진행률 (역방향)
                const lossProgress = Math.max(0, 100 - (data.current_loss * 100));
                document.getElementById('loss-bar').style.width = lossProgress + '%';
                
                // 차트 업데이트
                if (data.recent_history) {
                    lossChart.data.labels = data.recent_history.map(h => h.time);
                    lossChart.data.datasets[0].data = data.recent_history.map(h => h.loss);
                    lossChart.update('none');
                }
                
                // 패턴 업데이트
                if (data.patterns) {
                    const patternContainer = document.getElementById('pattern-container');
                    patternContainer.innerHTML = data.patterns.map(pattern => `
                        <div class="pattern-item">
                            <span>✅ ${pattern.type}</span>
                            <span style="opacity: 0.6">${pattern.count}회</span>
                        </div>
                    `).join('');
                }
                
                // 통계 업데이트
                document.getElementById('unity-patterns').textContent = 
                    data.stats?.unity_patterns || 0;
                document.getElementById('async-patterns').textContent = 
                    data.stats?.async_patterns || 0;
                document.getElementById('linq-patterns').textContent = 
                    data.stats?.linq_patterns || 0;
                document.getElementById('design-patterns').textContent = 
                    data.stats?.design_patterns || 0;
                
                // 마지막 업데이트 시간
                document.getElementById('last-update').textContent = 
                    new Date().toLocaleString();
                    
            } catch (error) {
                console.error('데이터 로드 실패:', error);
            }
        }
        
        // 초기 로드 및 자동 갱신
        updateDashboard();
        setInterval(updateDashboard, 10000);  // 10초마다 갱신
    </script>
</body>
</html>
"""


class NeuralLearningDashboard:
    """신경망 학습 대시보드 서버"""
    
    def __init__(self, port: int = 8889):
        self.port = port
        self.base_path = Path(__file__).parent
        self.db_path = self.base_path / "neural_learning_data" / "neural_learning.db"
        
    def start_server(self):
        """웹 서버 시작"""
        handler = self.create_handler()
        
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            print(f"🌐 신경망 학습 대시보드 시작: http://localhost:{self.port}")
            print("Ctrl+C로 종료")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n대시보드를 종료합니다.")
                
    def create_handler(self):
        """요청 핸들러 생성"""
        dashboard = self
        
        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(DASHBOARD_HTML.encode())
                    
                elif self.path == '/api/neural_stats':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    stats = dashboard.get_neural_stats()
                    self.wfile.write(json.dumps(stats).encode())
                    
                else:
                    super().do_GET()
                    
            def log_message(self, format, *args):
                # 로그 출력 억제
                pass
                
        return Handler
        
    def get_neural_stats(self) -> Dict:
        """신경망 학습 통계 조회"""
        stats = {
            'total_steps': 0,
            'total_samples': 0,
            'current_loss': 0.0,
            'model_parameters': 500000000,  # 500M
            'learning_rate': 0.0001,
            'recent_history': [],
            'patterns': [],
            'stats': {}
        }
        
        try:
            # 상태 파일에서 읽기
            status_file = self.base_path / "neural_learning_status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    if 'learning_stats' in status_data:
                        stats.update(status_data['learning_stats'])
                        
            # 데이터베이스에서 추가 정보
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                
                # 총 샘플 수
                result = conn.execute('SELECT COUNT(*) FROM learning_samples').fetchone()
                if result:
                    stats['total_samples'] = result[0]
                    
                # 최근 패턴
                patterns = conn.execute('''
                    SELECT pattern_type, frequency 
                    FROM code_patterns 
                    ORDER BY last_seen DESC 
                    LIMIT 10
                ''').fetchall()
                
                stats['patterns'] = [
                    {'type': p[0], 'count': p[1]} for p in patterns
                ]
                
                # 패턴별 통계
                unity_count = conn.execute(
                    "SELECT COUNT(*) FROM learning_samples WHERE code_snippet LIKE '%MonoBehaviour%'"
                ).fetchone()[0]
                
                async_count = conn.execute(
                    "SELECT COUNT(*) FROM learning_samples WHERE code_snippet LIKE '%async%'"
                ).fetchone()[0]
                
                linq_count = conn.execute(
                    "SELECT COUNT(*) FROM learning_samples WHERE code_snippet LIKE '%System.Linq%'"
                ).fetchone()[0]
                
                stats['stats'] = {
                    'unity_patterns': unity_count,
                    'async_patterns': async_count,
                    'linq_patterns': linq_count,
                    'design_patterns': len(patterns)
                }
                
                conn.close()
                
        except Exception as e:
            print(f"통계 조회 오류: {e}")
            
        return stats


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='신경망 학습 대시보드')
    parser.add_argument('--port', type=int, default=8889, help='포트 번호')
    args = parser.parse_args()
    
    dashboard = NeuralLearningDashboard(port=args.port)
    dashboard.start_server()


if __name__ == "__main__":
    main()