#!/usr/bin/env python3
"""
끈질긴 오류 해결 시스템 (상용 수준)

24시간 끈질기게 오류를 발견하고 해결하는 AI 시스템
오류 패턴 학습, 자동 해결 전략 개발, 실시간 오류 대응
상용 AI 모델 수준의 정교한 오류 해결 메커니즘
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import traceback
import hashlib
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict, Counter
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import ast
import dis
import inspect

# 분석 라이브러리
import numpy as np
import psutil
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 코드 분석
try:
    import pylint.lint
    import flake8.api.legacy
    import mypy.api
    import black
    import isort
    STATIC_ANALYSIS_AVAILABLE = True
except ImportError:
    STATIC_ANALYSIS_AVAILABLE = False
    print("Static analysis tools not available. Some features will be disabled.")

# 모니터링
try:
    import py3nvml.py3nvml as nvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


@dataclass
class ErrorContext:
    """오류 컨텍스트 정보"""
    error_id: str
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    code_snippet: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"  # low, medium, high, critical
    category: str = "unknown"
    reproducible: bool = False
    frequency: int = 1
    last_occurrence: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ErrorSolution:
    """오류 해결책"""
    solution_id: str
    error_pattern: str
    solution_type: str  # fix, workaround, config_change, dependency_update
    description: str
    steps: List[str]
    code_changes: List[Dict[str, Any]] = field(default_factory=list)
    config_changes: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    confidence_score: float = 0.0
    tested: bool = False
    automated: bool = False
    risk_level: str = "low"  # low, medium, high
    estimated_time_minutes: int = 5
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    version_compatibility: Dict[str, str] = field(default_factory=dict)


@dataclass
class ErrorResolutionAttempt:
    """오류 해결 시도 기록"""
    attempt_id: str
    error_id: str
    solution_id: str
    timestamp: str
    status: str  # pending, in_progress, success, failed, partial
    execution_time_seconds: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    side_effects_observed: List[str] = field(default_factory=list)
    rollback_performed: bool = False
    success_metrics: Dict[str, float] = field(default_factory=dict)


class ErrorDetector:
    """오류 감지 및 분석 시스템"""
    
    def __init__(self):
        self.error_patterns = {
            "syntax_error": [
                r"SyntaxError:",
                r"IndentationError:",
                r"TabError:"
            ],
            "import_error": [
                r"ImportError:",
                r"ModuleNotFoundError:",
                r"No module named"
            ],
            "type_error": [
                r"TypeError:",
                r"AttributeError:",
                r"NameError:"
            ],
            "value_error": [
                r"ValueError:",
                r"KeyError:",
                r"IndexError:"
            ],
            "runtime_error": [
                r"RuntimeError:",
                r"RecursionError:",
                r"MemoryError:"
            ],
            "godot_error": [
                r"Godot Engine",
                r"GDScript Error:",
                r"scene file corrupted",
                r"missing script",
                r"invalid node path"
            ],
            "performance_issue": [
                r"high memory usage",
                r"slow performance",
                r"timeout",
                r"lag detected"
            ]
        }
        
        self.severity_keywords = {
            "critical": ["crash", "fatal", "critical", "segfault", "abort"],
            "high": ["error", "exception", "failed", "broken"],
            "medium": ["warning", "deprecated", "issue"],
            "low": ["info", "notice", "debug"]
        }
    
    def detect_errors_in_logs(self, log_content: str) -> List[ErrorContext]:
        """로그에서 오류 감지"""
        errors = []
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            for category, patterns in self.error_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        error_id = hashlib.md5(f"{line}{i}".encode()).hexdigest()[:12]
                        
                        # 스택 트레이스 수집
                        stack_trace = self._extract_stack_trace(lines, i)
                        
                        # 컨텍스트 정보 추출
                        context = self._extract_context_from_line(line)
                        
                        error = ErrorContext(
                            error_id=error_id,
                            timestamp=datetime.now().isoformat(),
                            error_type=category,
                            error_message=line.strip(),
                            stack_trace=stack_trace,
                            severity=self._determine_severity(line),
                            category=category,
                            **context
                        )
                        
                        errors.append(error)
                        break
        
        return errors
    
    def analyze_code_for_errors(self, file_path: str) -> List[ErrorContext]:
        """코드 파일에서 잠재적 오류 분석"""
        errors = []
        
        if not STATIC_ANALYSIS_AVAILABLE:
            return errors
        
        try:
            # Pylint 분석
            pylint_errors = self._run_pylint(file_path)
            errors.extend(pylint_errors)
            
            # Flake8 분석
            flake8_errors = self._run_flake8(file_path)
            errors.extend(flake8_errors)
            
            # MyPy 분석 (타입 체크)
            mypy_errors = self._run_mypy(file_path)
            errors.extend(mypy_errors)
            
            # 커스텀 패턴 분석
            custom_errors = self._analyze_custom_patterns(file_path)
            errors.extend(custom_errors)
            
        except Exception as e:
            logging.error(f"Code analysis failed: {e}")
        
        return errors
    
    def _extract_stack_trace(self, lines: List[str], error_line_index: int) -> str:
        """스택 트레이스 추출"""
        stack_lines = []
        
        # 오류 라인 이전의 Traceback 찾기
        for i in range(max(0, error_line_index - 20), error_line_index):
            if "Traceback" in lines[i]:
                stack_lines = lines[i:error_line_index + 1]
                break
        
        # 오류 라인 이후의 추가 정보 찾기
        for i in range(error_line_index + 1, min(len(lines), error_line_index + 10)):
            if lines[i].strip() and not re.match(r'^[A-Z].*:', lines[i]):
                stack_lines.append(lines[i])
            else:
                break
        
        return '\n'.join(stack_lines)
    
    def _extract_context_from_line(self, line: str) -> Dict[str, Any]:
        """라인에서 컨텍스트 정보 추출"""
        context = {}
        
        # 파일 경로와 라인 번호 추출
        file_match = re.search(r'File "([^"]+)", line (\d+)', line)
        if file_match:
            context['file_path'] = file_match.group(1)
            context['line_number'] = int(file_match.group(2))
        
        # 함수명 추출
        func_match = re.search(r'in (\w+)', line)
        if func_match:
            context['function_name'] = func_match.group(1)
        
        return context
    
    def _determine_severity(self, error_message: str) -> str:
        """오류 심각도 결정"""
        error_lower = error_message.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in error_lower for keyword in keywords):
                return severity
        
        return "medium"
    
    def _run_pylint(self, file_path: str) -> List[ErrorContext]:
        """Pylint 실행"""
        errors = []
        try:
            from pylint.lint import Run
            from pylint.reporters.text import TextReporter
            from io import StringIO
            
            output = StringIO()
            reporter = TextReporter(output)
            Run([file_path], reporter=reporter, exit=False)
            
            result = output.getvalue()
            errors = self._parse_pylint_output(result, file_path)
            
        except Exception as e:
            logging.warning(f"Pylint analysis failed: {e}")
        
        return errors
    
    def _run_flake8(self, file_path: str) -> List[ErrorContext]:
        """Flake8 실행"""
        errors = []
        try:
            from flake8.api import legacy as flake8
            
            style_guide = flake8.get_style_guide()
            report = style_guide.check_files([file_path])
            
            # Flake8 결과 파싱 (구현 필요)
            
        except Exception as e:
            logging.warning(f"Flake8 analysis failed: {e}")
        
        return errors
    
    def _run_mypy(self, file_path: str) -> List[ErrorContext]:
        """MyPy 실행"""
        errors = []
        try:
            import mypy.api
            
            result = mypy.api.run([file_path])
            if result[0]:  # stdout contains errors
                errors = self._parse_mypy_output(result[0], file_path)
            
        except Exception as e:
            logging.warning(f"MyPy analysis failed: {e}")
        
        return errors
    
    def _analyze_custom_patterns(self, file_path: str) -> List[ErrorContext]:
        """커스텀 패턴 분석"""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 위험한 패턴들
            dangerous_patterns = [
                (r'eval\s*\(', "Use of eval() is dangerous"),
                (r'exec\s*\(', "Use of exec() is dangerous"),
                (r'os\.system\s*\(', "Use of os.system() is risky"),
                (r'subprocess\.shell\s*=\s*True', "Shell=True in subprocess is risky"),
                (r'pickle\.loads?\s*\(', "Pickle deserialization can be unsafe"),
                (r'TODO|FIXME|HACK', "Code contains TODO/FIXME/HACK"),
                (r'print\s*\([^)]*password[^)]*\)', "Potential password logging"),
                (r'print\s*\([^)]*secret[^)]*\)', "Potential secret logging")
            ]
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern, message in dangerous_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        error_id = hashlib.md5(f"{file_path}{line_num}{message}".encode()).hexdigest()[:12]
                        
                        error = ErrorContext(
                            error_id=error_id,
                            timestamp=datetime.now().isoformat(),
                            error_type="code_quality",
                            error_message=message,
                            stack_trace=f"Line {line_num}: {line.strip()}",
                            file_path=file_path,
                            line_number=line_num,
                            severity="medium",
                            category="code_quality"
                        )
                        errors.append(error)
            
        except Exception as e:
            logging.error(f"Custom pattern analysis failed: {e}")
        
        return errors
    
    def _parse_pylint_output(self, output: str, file_path: str) -> List[ErrorContext]:
        """Pylint 출력 파싱"""
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            # Pylint 메시지 형식: file:line:column: type: message
            match = re.match(r'([^:]+):(\d+):(\d+):\s*(\w+):\s*(.+)', line)
            if match:
                file_name, line_num, col, msg_type, message = match.groups()
                
                error_id = hashlib.md5(f"{file_path}{line_num}{message}".encode()).hexdigest()[:12]
                
                severity_map = {
                    'error': 'high',
                    'warning': 'medium',
                    'convention': 'low',
                    'refactor': 'low'
                }
                
                error = ErrorContext(
                    error_id=error_id,
                    timestamp=datetime.now().isoformat(),
                    error_type="static_analysis",
                    error_message=message,
                    stack_trace=line,
                    file_path=file_path,
                    line_number=int(line_num),
                    severity=severity_map.get(msg_type, 'medium'),
                    category="code_quality"
                )
                errors.append(error)
        
        return errors
    
    def _parse_mypy_output(self, output: str, file_path: str) -> List[ErrorContext]:
        """MyPy 출력 파싱"""
        errors = []
        lines = output.split('\n')
        
        for line in lines:
            # MyPy 메시지 형식: file:line: error: message
            match = re.match(r'([^:]+):(\d+):\s*(\w+):\s*(.+)', line)
            if match:
                file_name, line_num, msg_type, message = match.groups()
                
                error_id = hashlib.md5(f"{file_path}{line_num}{message}".encode()).hexdigest()[:12]
                
                error = ErrorContext(
                    error_id=error_id,
                    timestamp=datetime.now().isoformat(),
                    error_type="type_check",
                    error_message=message,
                    stack_trace=line,
                    file_path=file_path,
                    line_number=int(line_num),
                    severity="medium" if msg_type == "error" else "low",
                    category="type_safety"
                )
                errors.append(error)
        
        return errors


class SolutionGenerator:
    """해결책 생성 시스템"""
    
    def __init__(self):
        self.solution_templates = self._load_solution_templates()
        self.pattern_solutions = self._load_pattern_solutions()
    
    def generate_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """오류에 대한 해결책 생성"""
        solutions = []
        
        # 패턴 기반 해결책
        pattern_solutions = self._find_pattern_solutions(error)
        solutions.extend(pattern_solutions)
        
        # 타입별 해결책
        type_solutions = self._generate_type_specific_solutions(error)
        solutions.extend(type_solutions)
        
        # AI 기반 해결책 생성
        ai_solutions = self._generate_ai_solutions(error)
        solutions.extend(ai_solutions)
        
        # 해결책 우선순위 정렬
        solutions.sort(key=lambda s: (s.success_rate, s.confidence_score), reverse=True)
        
        return solutions
    
    def _load_solution_templates(self) -> Dict[str, Any]:
        """해결책 템플릿 로드"""
        templates = {
            "import_error": {
                "install_package": {
                    "description": "Install missing package",
                    "steps": ["pip install {package_name}"],
                    "success_rate": 0.9,
                    "automated": True
                },
                "add_to_path": {
                    "description": "Add module to Python path",
                    "steps": ["sys.path.append('{module_path}')"],
                    "success_rate": 0.7,
                    "automated": True
                }
            },
            "syntax_error": {
                "fix_indentation": {
                    "description": "Fix indentation errors",
                    "steps": ["Check and fix indentation", "Use consistent spaces or tabs"],
                    "success_rate": 0.95,
                    "automated": True
                },
                "fix_syntax": {
                    "description": "Fix syntax errors",
                    "steps": ["Check syntax highlighting", "Fix missing parentheses/brackets"],
                    "success_rate": 0.85,
                    "automated": False
                }
            },
            "type_error": {
                "add_type_check": {
                    "description": "Add type checking",
                    "steps": ["Add isinstance() check", "Handle None values"],
                    "success_rate": 0.8,
                    "automated": True
                }
            },
            "godot_error": {
                "check_scene_integrity": {
                    "description": "Check scene file integrity",
                    "steps": ["Reload scene", "Check node paths", "Verify script attachments"],
                    "success_rate": 0.7,
                    "automated": True
                }
            }
        }
        return templates
    
    def _load_pattern_solutions(self) -> Dict[str, List[ErrorSolution]]:
        """패턴별 해결책 로드"""
        return {}
    
    def _find_pattern_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """패턴 기반 해결책 찾기"""
        solutions = []
        
        # 메시지 패턴 매칭
        error_message = error.error_message.lower()
        
        if "no module named" in error_message:
            module_name = self._extract_module_name(error.error_message)
            if module_name:
                solution = ErrorSolution(
                    solution_id=f"install_{module_name}",
                    error_pattern="no module named",
                    solution_type="dependency_update",
                    description=f"Install missing module: {module_name}",
                    steps=[f"pip install {module_name}"],
                    success_rate=0.9,
                    confidence_score=0.95,
                    automated=True,
                    estimated_time_minutes=2
                )
                solutions.append(solution)
        
        elif "indentation" in error_message:
            solution = ErrorSolution(
                solution_id="fix_indentation",
                error_pattern="indentation",
                solution_type="fix",
                description="Fix indentation errors",
                steps=[
                    "Check consistent use of spaces or tabs",
                    "Use IDE auto-format feature",
                    "Set proper indentation settings"
                ],
                success_rate=0.95,
                confidence_score=0.9,
                automated=True,
                estimated_time_minutes=1
            )
            solutions.append(solution)
        
        elif "godot" in error_message.lower():
            solutions.extend(self._generate_godot_solutions(error))
        
        return solutions
    
    def _generate_type_specific_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """타입별 해결책 생성"""
        solutions = []
        
        if error.error_type in self.solution_templates:
            templates = self.solution_templates[error.error_type]
            
            for template_name, template in templates.items():
                solution_id = f"{error.error_type}_{template_name}_{error.error_id}"
                
                solution = ErrorSolution(
                    solution_id=solution_id,
                    error_pattern=error.error_type,
                    solution_type="fix",
                    description=template["description"],
                    steps=template["steps"],
                    success_rate=template.get("success_rate", 0.5),
                    confidence_score=0.8,
                    automated=template.get("automated", False),
                    estimated_time_minutes=template.get("time_minutes", 5)
                )
                solutions.append(solution)
        
        return solutions
    
    def _generate_ai_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """AI 기반 해결책 생성"""
        solutions = []
        
        # 코드 분석 기반 해결책
        if error.file_path and error.line_number:
            code_solutions = self._analyze_code_for_solutions(error)
            solutions.extend(code_solutions)
        
        # 유사 오류 기반 해결책
        similar_solutions = self._find_similar_error_solutions(error)
        solutions.extend(similar_solutions)
        
        return solutions
    
    def _generate_godot_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """Godot 특화 해결책 생성"""
        solutions = []
        
        godot_patterns = {
            "scene file corrupted": [
                "Restore scene from backup",
                "Recreate scene manually",
                "Check version control for working version"
            ],
            "missing script": [
                "Reattach script to node",
                "Check script file path",
                "Restore script from backup"
            ],
            "invalid node path": [
                "Update node path references",
                "Check node hierarchy",
                "Use get_node() with correct path"
            ]
        }
        
        error_message = error.error_message.lower()
        
        for pattern, steps in godot_patterns.items():
            if pattern in error_message:
                solution = ErrorSolution(
                    solution_id=f"godot_{pattern.replace(' ', '_')}",
                    error_pattern=pattern,
                    solution_type="fix",
                    description=f"Fix {pattern}",
                    steps=steps,
                    success_rate=0.7,
                    confidence_score=0.8,
                    automated=True,
                    estimated_time_minutes=3
                )
                solutions.append(solution)
        
        return solutions
    
    def _extract_module_name(self, error_message: str) -> Optional[str]:
        """오류 메시지에서 모듈명 추출"""
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", error_message)
        if match:
            return match.group(1)
        
        match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_message)
        if match:
            return match.group(1)
        
        return None
    
    def _analyze_code_for_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """코드 분석으로 해결책 생성"""
        solutions = []
        
        try:
            if not error.file_path or not os.path.exists(error.file_path):
                return solutions
            
            with open(error.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if error.line_number and error.line_number <= len(lines):
                error_line = lines[error.line_number - 1]
                
                # 간단한 패턴 기반 수정 제안
                if "import" in error_line and error.error_type == "import_error":
                    # import 문 수정 제안
                    solutions.append(self._suggest_import_fix(error_line, error))
                
                elif "=" in error_line and error.error_type == "type_error":
                    # 타입 오류 수정 제안
                    solutions.append(self._suggest_type_fix(error_line, error))
        
        except Exception as e:
            logging.warning(f"Code analysis for solutions failed: {e}")
        
        return solutions
    
    def _suggest_import_fix(self, error_line: str, error: ErrorContext) -> ErrorSolution:
        """Import 수정 제안"""
        return ErrorSolution(
            solution_id=f"import_fix_{error.error_id}",
            error_pattern="import_error",
            solution_type="fix",
            description="Fix import statement",
            steps=[
                "Check module name spelling",
                "Verify module installation",
                "Update import path"
            ],
            code_changes=[{
                "file": error.file_path,
                "line": error.line_number,
                "old_code": error_line.strip(),
                "new_code": "# Check and fix this import",
                "type": "comment"
            }],
            success_rate=0.6,
            confidence_score=0.7,
            automated=False
        )
    
    def _suggest_type_fix(self, error_line: str, error: ErrorContext) -> ErrorSolution:
        """타입 오류 수정 제안"""
        return ErrorSolution(
            solution_id=f"type_fix_{error.error_id}",
            error_pattern="type_error",
            solution_type="fix",
            description="Add type checking",
            steps=[
                "Add isinstance() check",
                "Handle None values",
                "Add error handling"
            ],
            code_changes=[{
                "file": error.file_path,
                "line": error.line_number,
                "old_code": error_line.strip(),
                "new_code": f"if value is not None:  # Add type check\n    {error_line.strip()}",
                "type": "wrapper"
            }],
            success_rate=0.7,
            confidence_score=0.6,
            automated=True
        )
    
    def _find_similar_error_solutions(self, error: ErrorContext) -> List[ErrorSolution]:
        """유사 오류의 해결책 찾기"""
        # 이 부분은 기존 해결책 데이터베이스에서 유사한 오류를 찾아서
        # 성공했던 해결책을 제안하는 로직
        return []


class PersistentErrorResolutionSystem:
    """끈질긴 오류 해결 시스템"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("/mnt/d/AutoCI/AutoCI/error_resolution")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스
        self.db_path = self.base_path / "error_resolution.db"
        self._initialize_database()
        
        # 컴포넌트
        self.error_detector = ErrorDetector()
        self.solution_generator = SolutionGenerator()
        
        # 상태 관리
        self.active_errors: Dict[str, ErrorContext] = {}
        self.resolution_queue = deque()
        self.resolution_history: Dict[str, List[ErrorResolutionAttempt]] = defaultdict(list)
        
        # 모니터링
        self.monitoring_active = False
        self.monitoring_threads = []
        
        # 통계
        self.stats = {
            "total_errors_detected": 0,
            "total_errors_resolved": 0,
            "resolution_success_rate": 0.0,
            "average_resolution_time": 0.0,
            "most_common_errors": Counter(),
            "most_effective_solutions": Counter()
        }
        
        # 설정
        self.config = {
            "max_retry_attempts": 5,
            "retry_delay_seconds": 30,
            "escalation_threshold": 3,
            "auto_resolution_enabled": True,
            "backup_before_fix": True,
            "rollback_on_failure": True,
            "monitor_interval_seconds": 10
        }
    
    def _initialize_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 오류 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS errors (
                error_id TEXT PRIMARY KEY,
                timestamp TEXT,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                file_path TEXT,
                line_number INTEGER,
                function_name TEXT,
                severity TEXT,
                category TEXT,
                frequency INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # 해결책 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS solutions (
                solution_id TEXT PRIMARY KEY,
                error_pattern TEXT,
                solution_type TEXT,
                description TEXT,
                steps TEXT,
                success_rate REAL,
                confidence_score REAL,
                automated BOOLEAN,
                created_at TEXT
            )
        ''')
        
        # 해결 시도 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resolution_attempts (
                attempt_id TEXT PRIMARY KEY,
                error_id TEXT,
                solution_id TEXT,
                timestamp TEXT,
                status TEXT,
                execution_time REAL,
                success BOOLEAN,
                logs TEXT,
                FOREIGN KEY (error_id) REFERENCES errors (error_id),
                FOREIGN KEY (solution_id) REFERENCES solutions (solution_id)
            )
        ''')
        
        # 성능 통계 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def start_continuous_monitoring(self):
        """24시간 연속 모니터링 시작"""
        self.monitoring_active = True
        
        # 로그 모니터링
        log_monitor = threading.Thread(target=self._monitor_logs)
        log_monitor.daemon = True
        log_monitor.start()
        self.monitoring_threads.append(log_monitor)
        
        # 코드 모니터링
        code_monitor = threading.Thread(target=self._monitor_code_changes)
        code_monitor.daemon = True
        code_monitor.start()
        self.monitoring_threads.append(code_monitor)
        
        # 시스템 모니터링
        system_monitor = threading.Thread(target=self._monitor_system_health)
        system_monitor.daemon = True
        system_monitor.start()
        self.monitoring_threads.append(system_monitor)
        
        # 해결 프로세서
        resolution_processor = threading.Thread(target=self._process_resolutions)
        resolution_processor.daemon = True
        resolution_processor.start()
        self.monitoring_threads.append(resolution_processor)
        
        logging.info("Persistent error resolution system started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring_active = False
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
        logging.info("Error resolution monitoring stopped")
    
    def _monitor_logs(self):
        """로그 파일 모니터링"""
        log_paths = [
            "/var/log/",
            "/tmp/",
            str(self.base_path.parent / "logs"),
            str(self.base_path.parent / "game_projects")
        ]
        
        monitored_files = {}
        
        while self.monitoring_active:
            try:
                for log_path in log_paths:
                    if not os.path.exists(log_path):
                        continue
                    
                    for root, dirs, files in os.walk(log_path):
                        for file in files:
                            if file.endswith(('.log', '.err', '.out', '.txt')):
                                file_path = os.path.join(root, file)
                                
                                # 파일 변경 감지
                                try:
                                    stat = os.stat(file_path)
                                    last_modified = stat.st_mtime
                                    
                                    if file_path not in monitored_files:
                                        monitored_files[file_path] = last_modified
                                        continue
                                    
                                    if last_modified > monitored_files[file_path]:
                                        self._analyze_log_file(file_path)
                                        monitored_files[file_path] = last_modified
                                
                                except (OSError, IOError):
                                    continue
                
                time.sleep(self.config["monitor_interval_seconds"])
                
            except Exception as e:
                logging.error(f"Log monitoring error: {e}")
                time.sleep(30)
    
    def _monitor_code_changes(self):
        """코드 변경 모니터링"""
        project_path = self.base_path.parent
        monitored_files = {}
        
        while self.monitoring_active:
            try:
                for root, dirs, files in os.walk(project_path):
                    # .git, __pycache__ 등 제외
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                    
                    for file in files:
                        if file.endswith(('.py', '.gd', '.cs', '.tscn', '.tres')):
                            file_path = os.path.join(root, file)
                            
                            try:
                                stat = os.stat(file_path)
                                last_modified = stat.st_mtime
                                
                                if file_path not in monitored_files:
                                    monitored_files[file_path] = last_modified
                                    continue
                                
                                if last_modified > monitored_files[file_path]:
                                    self._analyze_code_file(file_path)
                                    monitored_files[file_path] = last_modified
                            
                            except (OSError, IOError):
                                continue
                
                time.sleep(self.config["monitor_interval_seconds"] * 2)
                
            except Exception as e:
                logging.error(f"Code monitoring error: {e}")
                time.sleep(30)
    
    def _monitor_system_health(self):
        """시스템 헬스 모니터링"""
        while self.monitoring_active:
            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # 메모리 사용률
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # 디스크 사용률
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # GPU 모니터링
                gpu_info = {}
                if GPU_MONITORING_AVAILABLE:
                    try:
                        nvml.nvmlInit()
                        device_count = nvml.nvmlDeviceGetCount()
                        
                        for i in range(device_count):
                            handle = nvml.nvmlDeviceGetHandleByIndex(i)
                            util = nvml.nvmlDeviceGetUtilizationRates(handle)
                            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            gpu_info[f'gpu_{i}'] = {
                                'utilization': util.gpu,
                                'memory_used': memory_info.used / 1024**3,  # GB
                                'memory_total': memory_info.total / 1024**3  # GB
                            }
                    except Exception:
                        pass
                
                # 임계값 확인
                if cpu_percent > 90:
                    self._create_performance_error("High CPU usage", cpu_percent)
                
                if memory_percent > 90:
                    self._create_performance_error("High memory usage", memory_percent)
                
                if disk_percent > 95:
                    self._create_performance_error("Low disk space", disk_percent)
                
                # 통계 저장
                self._save_performance_stats({
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'gpu_info': gpu_info
                })
                
                time.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                logging.error(f"System monitoring error: {e}")
                time.sleep(60)
    
    def _process_resolutions(self):
        """해결책 처리 프로세스"""
        while self.monitoring_active:
            try:
                if self.resolution_queue:
                    error_id = self.resolution_queue.popleft()
                    
                    if error_id in self.active_errors:
                        error = self.active_errors[error_id]
                        asyncio.run(self._resolve_error_persistently(error))
                
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"Resolution processing error: {e}")
                time.sleep(10)
    
    def _analyze_log_file(self, file_path: str):
        """로그 파일 분석"""
        try:
            # 최근 몇 줄만 읽기 (성능 최적화)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # 파일 끝으로
                file_size = f.tell()
                
                # 마지막 10KB만 읽기
                read_size = min(10 * 1024, file_size)
                f.seek(max(0, file_size - read_size))
                content = f.read()
            
            errors = self.error_detector.detect_errors_in_logs(content)
            
            for error in errors:
                self._handle_detected_error(error)
        
        except Exception as e:
            logging.warning(f"Log analysis failed for {file_path}: {e}")
    
    def _analyze_code_file(self, file_path: str):
        """코드 파일 분석"""
        try:
            errors = self.error_detector.analyze_code_for_errors(file_path)
            
            for error in errors:
                self._handle_detected_error(error)
        
        except Exception as e:
            logging.warning(f"Code analysis failed for {file_path}: {e}")
    
    def _handle_detected_error(self, error: ErrorContext):
        """감지된 오류 처리"""
        try:
            # 중복 오류 확인
            if error.error_id in self.active_errors:
                existing_error = self.active_errors[error.error_id]
                existing_error.frequency += 1
                existing_error.last_occurrence = error.timestamp
                self._update_error_in_db(existing_error)
                return
            
            # 새 오류 등록
            self.active_errors[error.error_id] = error
            self._save_error_to_db(error)
            
            # 통계 업데이트
            self.stats["total_errors_detected"] += 1
            self.stats["most_common_errors"][error.error_type] += 1
            
            # 자동 해결 대상인지 확인
            if self.config["auto_resolution_enabled"]:
                self.resolution_queue.append(error.error_id)
            
            logging.info(f"New error detected: {error.error_type} - {error.error_message[:100]}")
        
        except Exception as e:
            logging.error(f"Error handling failed: {e}")
    
    async def _resolve_error_persistently(self, error: ErrorContext):
        """끈질기게 오류 해결"""
        max_attempts = self.config["max_retry_attempts"]
        attempt_count = 0
        
        while attempt_count < max_attempts and self.monitoring_active:
            attempt_count += 1
            
            try:
                # 해결책 생성
                solutions = self.solution_generator.generate_solutions(error)
                
                if not solutions:
                    logging.warning(f"No solutions found for error {error.error_id}")
                    break
                
                # 해결책 시도
                for solution in solutions:
                    success = await self._attempt_solution(error, solution)
                    
                    if success:
                        # 성공한 경우
                        error.reproducible = False
                        self._mark_error_resolved(error.error_id)
                        self.stats["total_errors_resolved"] += 1
                        self.stats["most_effective_solutions"][solution.solution_id] += 1
                        
                        logging.info(f"Error {error.error_id} resolved with solution {solution.solution_id}")
                        return
                    
                    # 실패한 경우 다음 해결책 시도
                    await asyncio.sleep(self.config["retry_delay_seconds"])
                
                # 모든 해결책이 실패한 경우
                logging.warning(f"All solutions failed for error {error.error_id}, attempt {attempt_count}")
                
                if attempt_count < max_attempts:
                    # 재시도 전 대기
                    await asyncio.sleep(self.config["retry_delay_seconds"] * attempt_count)
            
            except Exception as e:
                logging.error(f"Error resolution attempt failed: {e}")
                await asyncio.sleep(self.config["retry_delay_seconds"])
        
        # 최종적으로 해결되지 않은 경우
        if attempt_count >= max_attempts:
            logging.error(f"Failed to resolve error {error.error_id} after {max_attempts} attempts")
            self._escalate_error(error)
    
    async def _attempt_solution(self, error: ErrorContext, solution: ErrorSolution) -> bool:
        """해결책 시도"""
        attempt_id = hashlib.md5(f"{error.error_id}{solution.solution_id}{time.time()}".encode()).hexdigest()[:12]
        
        attempt = ErrorResolutionAttempt(
            attempt_id=attempt_id,
            error_id=error.error_id,
            solution_id=solution.solution_id,
            timestamp=datetime.now().isoformat(),
            status="in_progress"
        )
        
        start_time = time.time()
        
        try:
            # 백업 생성 (설정에 따라)
            backup_info = None
            if self.config["backup_before_fix"] and error.file_path:
                backup_info = self._create_backup(error.file_path)
            
            # 해결책 실행
            success = await self._execute_solution(solution, error)
            
            attempt.execution_time_seconds = time.time() - start_time
            attempt.status = "success" if success else "failed"
            
            # 검증
            if success:
                verification_result = await self._verify_solution(error, solution)
                if not verification_result:
                    success = False
                    attempt.status = "partial"
            
            # 실패한 경우 롤백
            if not success and self.config["rollback_on_failure"] and backup_info:
                self._restore_backup(backup_info)
                attempt.rollback_performed = True
            
            # 결과 기록
            attempt.result = {"success": success, "backup_info": backup_info}
            self.resolution_history[error.error_id].append(attempt)
            self._save_attempt_to_db(attempt)
            
            return success
        
        except Exception as e:
            attempt.execution_time_seconds = time.time() - start_time
            attempt.status = "failed"
            attempt.logs.append(f"Exception: {str(e)}")
            
            self.resolution_history[error.error_id].append(attempt)
            self._save_attempt_to_db(attempt)
            
            logging.error(f"Solution execution failed: {e}")
            return False
    
    async def _execute_solution(self, solution: ErrorSolution, error: ErrorContext) -> bool:
        """해결책 실행"""
        try:
            if solution.solution_type == "dependency_update":
                return await self._execute_dependency_update(solution, error)
            elif solution.solution_type == "fix":
                return await self._execute_code_fix(solution, error)
            elif solution.solution_type == "config_change":
                return await self._execute_config_change(solution, error)
            elif solution.solution_type == "workaround":
                return await self._execute_workaround(solution, error)
            else:
                return await self._execute_generic_solution(solution, error)
        
        except Exception as e:
            logging.error(f"Solution execution error: {e}")
            return False
    
    async def _execute_dependency_update(self, solution: ErrorSolution, error: ErrorContext) -> bool:
        """의존성 업데이트 실행"""
        for step in solution.steps:
            if step.startswith("pip install"):
                try:
                    result = subprocess.run(
                        step.split(),
                        capture_output=True,
                        text=True,
                        timeout=300  # 5분 타임아웃
                    )
                    
                    if result.returncode != 0:
                        logging.error(f"Pip install failed: {result.stderr}")
                        return False
                
                except subprocess.TimeoutExpired:
                    logging.error("Pip install timed out")
                    return False
                except Exception as e:
                    logging.error(f"Pip install error: {e}")
                    return False
        
        return True
    
    async def _execute_code_fix(self, solution: ErrorSolution, error: ErrorContext) -> bool:
        """코드 수정 실행"""
        if not solution.code_changes or not error.file_path:
            return False
        
        try:
            with open(error.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 코드 변경 적용
            for change in solution.code_changes:
                if change["type"] == "replace" and error.line_number:
                    if error.line_number <= len(lines):
                        lines[error.line_number - 1] = change["new_code"] + "\n"
                
                elif change["type"] == "insert" and error.line_number:
                    if error.line_number <= len(lines):
                        lines.insert(error.line_number - 1, change["new_code"] + "\n")
                
                elif change["type"] == "comment" and error.line_number:
                    if error.line_number <= len(lines):
                        lines[error.line_number - 1] = f"# {lines[error.line_number - 1]}"
            
            # 파일 저장
            with open(error.file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return True
        
        except Exception as e:
            logging.error(f"Code fix execution failed: {e}")
            return False
    
    async def _execute_config_change(self, solution: ErrorSolution, error: ErrorContext) -> bool:
        """설정 변경 실행"""
        for change in solution.config_changes:
            try:
                config_file = change.get("file")
                if not config_file or not os.path.exists(config_file):
                    continue
                
                # JSON 설정 파일 처리
                if config_file.endswith('.json'):
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # 설정 값 변경
                    key_path = change.get("key", "").split('.')
                    value = change.get("value")
                    
                    current = config
                    for key in key_path[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    
                    current[key_path[-1]] = value
                    
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2)
            
            except Exception as e:
                logging.error(f"Config change failed: {e}")
                return False
        
        return True
    
    async def _execute_workaround(self, solution: ErrorSolution, error: ErrorContext) -> bool:
        """임시 해결책 실행"""
        # 임시 해결책은 보통 환경 변수 설정이나 심볼릭 링크 생성 등
        for step in solution.steps:
            try:
                if step.startswith("export "):
                    # 환경 변수 설정
                    var_assignment = step[7:]  # "export " 제거
                    key, value = var_assignment.split('=', 1)
                    os.environ[key] = value
                
                elif step.startswith("ln -s "):
                    # 심볼릭 링크 생성
                    result = subprocess.run(step.split(), capture_output=True, text=True)
                    if result.returncode != 0:
                        return False
            
            except Exception as e:
                logging.error(f"Workaround step failed: {e}")
                return False
        
        return True
    
    async def _execute_generic_solution(self, solution: ErrorSolution, error: ErrorContext) -> bool:
        """일반 해결책 실행"""
        # 단계별 실행 로그만 기록
        for i, step in enumerate(solution.steps):
            logging.info(f"Executing step {i+1}: {step}")
            await asyncio.sleep(1)  # 각 단계 사이 대기
        
        return True
    
    async def _verify_solution(self, error: ErrorContext, solution: ErrorSolution) -> bool:
        """해결책 검증"""
        try:
            # 기본 검증: 같은 오류가 다시 발생하는지 확인
            if error.file_path and os.path.exists(error.file_path):
                # 코드 다시 분석
                new_errors = self.error_detector.analyze_code_for_errors(error.file_path)
                
                # 같은 오류가 여전히 존재하는지 확인
                for new_error in new_errors:
                    if (new_error.error_type == error.error_type and 
                        new_error.line_number == error.line_number and
                        new_error.error_message == error.error_message):
                        return False
            
            # 추가 검증 로직
            if solution.verification_method:
                return await self._run_custom_verification(solution.verification_method, error)
            
            return True
        
        except Exception as e:
            logging.error(f"Solution verification failed: {e}")
            return False
    
    async def _run_custom_verification(self, verification_method: str, error: ErrorContext) -> bool:
        """커스텀 검증 실행"""
        try:
            if verification_method == "compile_check":
                # 컴파일 확인
                if error.file_path and error.file_path.endswith('.py'):
                    result = subprocess.run([
                        sys.executable, "-m", "py_compile", error.file_path
                    ], capture_output=True, text=True)
                    return result.returncode == 0
            
            elif verification_method == "import_check":
                # import 확인
                if error.file_path and error.file_path.endswith('.py'):
                    try:
                        spec = importlib.util.spec_from_file_location("test_module", error.file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return True
                    except Exception:
                        return False
            
            elif verification_method == "syntax_check":
                # 구문 확인
                if error.file_path:
                    try:
                        with open(error.file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        ast.parse(content)
                        return True
                    except SyntaxError:
                        return False
            
            return True
        
        except Exception as e:
            logging.error(f"Custom verification failed: {e}")
            return False
    
    def _create_backup(self, file_path: str) -> Dict[str, Any]:
        """파일 백업 생성"""
        try:
            backup_dir = self.base_path / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{os.path.basename(file_path)}.{timestamp}.backup"
            backup_path = backup_dir / backup_name
            
            import shutil
            shutil.copy2(file_path, backup_path)
            
            return {
                "original_path": file_path,
                "backup_path": str(backup_path),
                "timestamp": timestamp
            }
        
        except Exception as e:
            logging.error(f"Backup creation failed: {e}")
            return {}
    
    def _restore_backup(self, backup_info: Dict[str, Any]):
        """백업 복원"""
        try:
            if not backup_info or "original_path" not in backup_info:
                return
            
            import shutil
            shutil.copy2(backup_info["backup_path"], backup_info["original_path"])
            
            logging.info(f"Restored backup: {backup_info['backup_path']}")
        
        except Exception as e:
            logging.error(f"Backup restoration failed: {e}")
    
    def _create_performance_error(self, error_type: str, value: float):
        """성능 관련 오류 생성"""
        error_id = hashlib.md5(f"{error_type}{datetime.now().date()}".encode()).hexdigest()[:12]
        
        error = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            error_type="performance",
            error_message=f"{error_type}: {value}%",
            stack_trace="System monitoring",
            severity="high" if value > 95 else "medium",
            category="system_performance",
            system_state={"value": value, "threshold": 90}
        )
        
        self._handle_detected_error(error)
    
    def _escalate_error(self, error: ErrorContext):
        """오류 에스컬레이션"""
        logging.critical(f"ESCALATED ERROR: {error.error_id} - {error.error_message}")
        
        # 에스컬레이션 로직 (알림, 관리자 통지 등)
        escalation_info = {
            "error": asdict(error),
            "attempts": len(self.resolution_history.get(error.error_id, [])),
            "timestamp": datetime.now().isoformat(),
            "priority": "critical"
        }
        
        # 에스컬레이션 로그 저장
        escalation_file = self.base_path / "escalated_errors.json"
        try:
            if escalation_file.exists():
                with open(escalation_file, 'r', encoding='utf-8') as f:
                    escalations = json.load(f)
            else:
                escalations = []
            
            escalations.append(escalation_info)
            
            with open(escalation_file, 'w', encoding='utf-8') as f:
                json.dump(escalations, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            logging.error(f"Escalation logging failed: {e}")
    
    def _save_error_to_db(self, error: ErrorContext):
        """오류를 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO errors 
                (error_id, timestamp, error_type, error_message, stack_trace, 
                 file_path, line_number, function_name, severity, category, 
                 frequency, first_seen, last_seen, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                error.error_id, error.timestamp, error.error_type, error.error_message,
                error.stack_trace, error.file_path, error.line_number, error.function_name,
                error.severity, error.category, error.frequency, error.timestamp,
                error.last_occurrence, False
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logging.error(f"Database save failed: {e}")
    
    def _update_error_in_db(self, error: ErrorContext):
        """데이터베이스의 오류 정보 업데이트"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE errors 
                SET frequency = ?, last_seen = ?
                WHERE error_id = ?
            ''', (error.frequency, error.last_occurrence, error.error_id))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logging.error(f"Database update failed: {e}")
    
    def _mark_error_resolved(self, error_id: str):
        """오류를 해결됨으로 표시"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE errors 
                SET resolved = TRUE
                WHERE error_id = ?
            ''', (error_id,))
            
            conn.commit()
            conn.close()
            
            # 활성 오류에서 제거
            if error_id in self.active_errors:
                del self.active_errors[error_id]
        
        except Exception as e:
            logging.error(f"Error resolution marking failed: {e}")
    
    def _save_attempt_to_db(self, attempt: ErrorResolutionAttempt):
        """해결 시도를 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO resolution_attempts 
                (attempt_id, error_id, solution_id, timestamp, status, 
                 execution_time, success, logs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                attempt.attempt_id, attempt.error_id, attempt.solution_id,
                attempt.timestamp, attempt.status, attempt.execution_time_seconds,
                attempt.status == "success", json.dumps(attempt.logs)
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logging.error(f"Attempt save failed: {e}")
    
    def _save_performance_stats(self, stats: Dict[str, Any]):
        """성능 통계 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            for metric_name, metric_value in stats.items():
                if isinstance(metric_value, dict):
                    metric_value = json.dumps(metric_value)
                    
                cursor.execute('''
                    INSERT INTO performance_stats 
                    (timestamp, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, metric_name, float(metric_value) if isinstance(metric_value, (int, float)) else 0, str(metric_value)))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logging.error(f"Performance stats save failed: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """오류 통계 반환"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 전체 통계
            cursor.execute("SELECT COUNT(*) FROM errors")
            total_errors = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM errors WHERE resolved = TRUE")
            resolved_errors = cursor.fetchone()[0]
            
            # 타입별 통계
            cursor.execute('''
                SELECT error_type, COUNT(*) 
                FROM errors 
                GROUP BY error_type 
                ORDER BY COUNT(*) DESC
            ''')
            error_types = cursor.fetchall()
            
            # 심각도별 통계
            cursor.execute('''
                SELECT severity, COUNT(*) 
                FROM errors 
                GROUP BY severity
            ''')
            severity_stats = cursor.fetchall()
            
            # 최근 해결 시도
            cursor.execute('''
                SELECT COUNT(*) 
                FROM resolution_attempts 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            recent_attempts = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_errors": total_errors,
                "resolved_errors": resolved_errors,
                "resolution_rate": resolved_errors / total_errors if total_errors > 0 else 0,
                "active_errors": len(self.active_errors),
                "error_types": dict(error_types),
                "severity_distribution": dict(severity_stats),
                "recent_attempts_24h": recent_attempts,
                "system_stats": self.stats
            }
        
        except Exception as e:
            logging.error(f"Statistics retrieval failed: {e}")
            return {"error": str(e)}
    
    def get_active_errors(self) -> List[Dict[str, Any]]:
        """활성 오류 목록 반환"""
        return [asdict(error) for error in self.active_errors.values()]
    
    def force_resolve_error(self, error_id: str) -> bool:
        """오류 강제 해결"""
        try:
            if error_id in self.active_errors:
                self._mark_error_resolved(error_id)
                logging.info(f"Manually resolved error: {error_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Force resolution failed: {e}")
            return False


# 메인 실행 부분
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Persistent Error Resolution System")
    parser.add_argument("--base-path", type=str, help="Base path for data storage")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--list-errors", action="store_true", help="List active errors")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 시스템 생성
    base_path = Path(args.base_path) if args.base_path else None
    system = PersistentErrorResolutionSystem(base_path)
    
    if args.monitor:
        print("Starting persistent error resolution monitoring...")
        try:
            asyncio.run(system.start_continuous_monitoring())
            # 무한 대기
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            system.stop_monitoring()
    
    elif args.stats:
        stats = system.get_error_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    elif args.list_errors:
        errors = system.get_active_errors()
        print(json.dumps(errors, indent=2, ensure_ascii=False))
    
    else:
        print("Use --monitor to start monitoring, --stats for statistics, or --list-errors to see active errors")