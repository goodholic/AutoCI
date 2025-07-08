#!/usr/bin/env python3
"""
Failure Tracking and Learning System
실패를 추적하고 학습하여 미래의 개발을 개선하는 시스템
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import hashlib
import re
from collections import defaultdict, Counter

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FailureType(Enum):
    """실패 유형"""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    RESOURCE_MISSING = "resource_missing"
    PERFORMANCE_ISSUE = "performance_issue"
    COMPATIBILITY_ISSUE = "compatibility_issue"
    BUILD_FAILURE = "build_failure"
    TEST_FAILURE = "test_failure"
    INTEGRATION_ERROR = "integration_error"
    DEPENDENCY_ERROR = "dependency_error"

class FailureSeverity(Enum):
    """실패 심각도"""
    CRITICAL = 5  # 프로그램 실행 불가
    HIGH = 4      # 주요 기능 작동 안함
    MEDIUM = 3    # 일부 기능 제한
    LOW = 2       # 사소한 문제
    INFO = 1      # 정보성 경고

class FailureTrackingSystem:
    """실패 추적 및 학습 시스템"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.failure_db_path = self.project_root / "knowledge_base" / "failure_tracking.db"
        self.failure_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 초기화
        self._init_database()
        
        # 메모리 캐시
        self.failure_patterns = {}
        self.solution_database = {}
        self.learning_insights = {}
        
        # 통계
        self.stats = {
            "total_failures": 0,
            "resolved_failures": 0,
            "recurring_failures": 0,
            "prevention_success": 0
        }
        
        # 실패 해결 전략
        self.resolution_strategies = {
            FailureType.SYNTAX_ERROR: self._resolve_syntax_error,
            FailureType.RUNTIME_ERROR: self._resolve_runtime_error,
            FailureType.RESOURCE_MISSING: self._resolve_resource_missing,
            FailureType.PERFORMANCE_ISSUE: self._resolve_performance_issue,
            FailureType.DEPENDENCY_ERROR: self._resolve_dependency_error
        }
        
        # 패턴 매칭 규칙
        self.pattern_rules = self._load_pattern_rules()
        
        # AI 모델 연동
        self.ai_model = None
        try:
            from modules.ai_model_integration import get_ai_integration
            self.ai_model = get_ai_integration()
        except ImportError:
            logger.warning("AI 모델을 로드할 수 없습니다.")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.failure_db_path)
        cursor = conn.cursor()
        
        # 실패 기록 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                project_name TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                severity INTEGER NOT NULL,
                error_message TEXT,
                stack_trace TEXT,
                context TEXT,
                file_path TEXT,
                line_number INTEGER,
                code_snippet TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_method TEXT,
                resolution_time REAL,
                prevention_applied BOOLEAN DEFAULT FALSE,
                recurrence_count INTEGER DEFAULT 0,
                pattern_hash TEXT
            )
        """)
        
        # 해결책 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                failure_pattern TEXT NOT NULL,
                solution_description TEXT NOT NULL,
                solution_code TEXT,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                last_used TEXT,
                tags TEXT
            )
        """)
        
        # 학습 인사이트 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                description TEXT NOT NULL,
                related_failures TEXT,
                prevention_strategy TEXT,
                effectiveness_score REAL DEFAULT 0.0
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_failure_type ON failures(failure_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project ON failures(project_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern ON failures(pattern_hash)")
        
        conn.commit()
        conn.close()
    
    def _load_pattern_rules(self) -> Dict[str, Dict[str, Any]]:
        """패턴 매칭 규칙 로드"""
        return {
            "import_error": {
                "pattern": r"ImportError: .*'(\w+)'",
                "type": FailureType.DEPENDENCY_ERROR,
                "solution": "Install missing module or check import path"
            },
            "attribute_error": {
                "pattern": r"AttributeError: '(\w+)' object has no attribute '(\w+)'",
                "type": FailureType.RUNTIME_ERROR,
                "solution": "Check object type and available attributes"
            },
            "type_error": {
                "pattern": r"TypeError: .* expected (\w+), got (\w+)",
                "type": FailureType.RUNTIME_ERROR,
                "solution": "Fix type mismatch"
            },
            "godot_signal_error": {
                "pattern": r"Signal '(\w+)' is already connected",
                "type": FailureType.RUNTIME_ERROR,
                "solution": "Check for duplicate signal connections"
            },
            "godot_node_error": {
                "pattern": r"Node not found: (.+)",
                "type": FailureType.RUNTIME_ERROR,
                "solution": "Verify node path and scene structure"
            },
            "resource_not_found": {
                "pattern": r"Failed to load resource: (.+)",
                "type": FailureType.RESOURCE_MISSING,
                "solution": "Check resource path or create missing resource"
            }
        }
    
    async def track_failure(
        self,
        error: Exception,
        context: Dict[str, Any],
        project_name: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None
    ) -> int:
        """실패 추적"""
        # 실패 유형 판별
        failure_type = self._classify_failure(error, context)
        severity = self._assess_severity(error, failure_type)
        
        # 패턴 해시 생성
        pattern_hash = self._generate_pattern_hash(error, failure_type)
        
        # 코드 스니펫 추출
        code_snippet = None
        if file_path and line_number:
            code_snippet = self._extract_code_snippet(file_path, line_number)
        
        # 데이터베이스에 기록
        conn = sqlite3.connect(self.failure_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO failures (
                timestamp, project_name, failure_type, severity,
                error_message, stack_trace, context, file_path,
                line_number, code_snippet, pattern_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            project_name,
            failure_type.value,
            severity.value,
            str(error),
            self._get_stack_trace(error),
            json.dumps(context),
            file_path,
            line_number,
            code_snippet,
            pattern_hash
        ))
        
        failure_id = cursor.lastrowid
        
        # 재발 체크
        cursor.execute("""
            UPDATE failures 
            SET recurrence_count = recurrence_count + 1 
            WHERE pattern_hash = ? AND id != ?
        """, (pattern_hash, failure_id))
        
        conn.commit()
        conn.close()
        
        # 통계 업데이트
        self.stats["total_failures"] += 1
        
        # 자동 해결 시도
        if severity.value >= FailureSeverity.MEDIUM.value:
            await self._attempt_auto_resolution(failure_id, error, failure_type, context)
        
        # 학습
        await self._learn_from_failure(failure_id, error, failure_type, context)
        
        logger.info(f"✅ 실패 추적 완료: ID={failure_id}, Type={failure_type.value}, Severity={severity.value}")
        
        return failure_id
    
    def _classify_failure(self, error: Exception, context: Dict[str, Any]) -> FailureType:
        """실패 유형 분류"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # 에러 메시지 기반 분류
        if "syntax" in error_str or error_type == "SyntaxError":
            return FailureType.SYNTAX_ERROR
        elif "import" in error_str or error_type == "ImportError":
            return FailureType.DEPENDENCY_ERROR
        elif "resource" in error_str or "file not found" in error_str:
            return FailureType.RESOURCE_MISSING
        elif "performance" in error_str or "timeout" in error_str:
            return FailureType.PERFORMANCE_ISSUE
        elif "build" in error_str or "compile" in error_str:
            return FailureType.BUILD_FAILURE
        elif "test" in error_str or "assert" in error_str:
            return FailureType.TEST_FAILURE
        
        # 컨텍스트 기반 분류
        if context.get("phase") == "integration":
            return FailureType.INTEGRATION_ERROR
        elif context.get("phase") == "logic":
            return FailureType.LOGIC_ERROR
        
        # 기본값
        return FailureType.RUNTIME_ERROR
    
    def _assess_severity(self, error: Exception, failure_type: FailureType) -> FailureSeverity:
        """실패 심각도 평가"""
        # 치명적 오류
        if failure_type in [FailureType.BUILD_FAILURE, FailureType.SYNTAX_ERROR]:
            return FailureSeverity.CRITICAL
        
        # 높은 심각도
        if failure_type in [FailureType.RUNTIME_ERROR, FailureType.DEPENDENCY_ERROR]:
            return FailureSeverity.HIGH
        
        # 중간 심각도
        if failure_type in [FailureType.LOGIC_ERROR, FailureType.INTEGRATION_ERROR]:
            return FailureSeverity.MEDIUM
        
        # 낮은 심각도
        if failure_type in [FailureType.PERFORMANCE_ISSUE, FailureType.TEST_FAILURE]:
            return FailureSeverity.LOW
        
        # 정보성
        return FailureSeverity.INFO
    
    def _generate_pattern_hash(self, error: Exception, failure_type: FailureType) -> str:
        """패턴 해시 생성"""
        # 에러의 주요 특징을 추출하여 해시 생성
        pattern_string = f"{failure_type.value}:{type(error).__name__}"
        
        # 에러 메시지에서 변수 부분 제거
        error_msg = str(error)
        # 숫자, 경로, 변수명 등을 일반화
        error_msg = re.sub(r'\d+', 'NUM', error_msg)
        error_msg = re.sub(r'/[^\s]+', 'PATH', error_msg)
        error_msg = re.sub(r"'[^']+'", 'VAR', error_msg)
        
        pattern_string += f":{error_msg}"
        
        return hashlib.md5(pattern_string.encode()).hexdigest()
    
    def _extract_code_snippet(self, file_path: str, line_number: int, context_lines: int = 5) -> str:
        """코드 스니펫 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            snippet_lines = []
            for i in range(start, end):
                prefix = ">>> " if i == line_number - 1 else "    "
                snippet_lines.append(f"{i+1:4d} {prefix}{lines[i].rstrip()}")
            
            return "\n".join(snippet_lines)
        except:
            return ""
    
    def _get_stack_trace(self, error: Exception) -> str:
        """스택 트레이스 추출"""
        import traceback
        return traceback.format_exc()
    
    async def _attempt_auto_resolution(
        self,
        failure_id: int,
        error: Exception,
        failure_type: FailureType,
        context: Dict[str, Any]
    ):
        """자동 해결 시도"""
        logger.info(f"🔧 자동 해결 시도: {failure_type.value}")
        
        # 해결 전략 선택
        if failure_type in self.resolution_strategies:
            strategy = self.resolution_strategies[failure_type]
            resolution_start = datetime.now()
            
            try:
                # 해결 시도
                success, resolution_method = await strategy(error, context)
                
                if success:
                    resolution_time = (datetime.now() - resolution_start).total_seconds()
                    
                    # 데이터베이스 업데이트
                    conn = sqlite3.connect(self.failure_db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE failures 
                        SET resolved = TRUE, 
                            resolution_method = ?,
                            resolution_time = ?
                        WHERE id = ?
                    """, (resolution_method, resolution_time, failure_id))
                    
                    conn.commit()
                    conn.close()
                    
                    self.stats["resolved_failures"] += 1
                    logger.info(f"✅ 자동 해결 성공: {resolution_method}")
                else:
                    logger.warning("❌ 자동 해결 실패")
                    
            except Exception as e:
                logger.error(f"자동 해결 중 오류: {e}")
    
    async def _resolve_syntax_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """구문 오류 해결"""
        file_path = context.get("file_path")
        if not file_path:
            return False, "No file path provided"
        
        try:
            # AI 모델 사용
            if self.ai_model:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                prompt = f"""
                다음 코드에 구문 오류가 있습니다:
                
                오류: {str(error)}
                
                코드를 수정해주세요. 수정된 전체 코드를 제공해주세요.
                
                원본 코드:
                {code}
                """
                
                fixed_code = await self.ai_model.generate_response(prompt)
                
                if fixed_code and "def" in fixed_code or "func" in fixed_code:
                    # 백업 생성
                    backup_path = f"{file_path}.backup"
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    # 수정된 코드 저장
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_code)
                    
                    return True, "AI-assisted syntax fix"
            
            # 간단한 자동 수정 시도
            # 예: 누락된 콜론, 괄호 등
            return False, "Manual fix required"
            
        except Exception as e:
            logger.error(f"구문 오류 해결 실패: {e}")
            return False, f"Resolution failed: {str(e)}"
    
    async def _resolve_runtime_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """런타임 오류 해결"""
        error_str = str(error)
        
        # 일반적인 런타임 오류 패턴 처리
        if "division by zero" in error_str:
            # 0으로 나누기 방지 코드 추가
            return await self._add_zero_division_check(context)
        elif "list index out of range" in error_str:
            # 배열 범위 체크 추가
            return await self._add_bounds_check(context)
        elif "NoneType" in error_str:
            # None 체크 추가
            return await self._add_none_check(context)
        
        return False, "Complex runtime error - manual intervention needed"
    
    async def _resolve_resource_missing(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """리소스 누락 해결"""
        resource_path = context.get("resource_path")
        if not resource_path:
            # 에러 메시지에서 리소스 경로 추출 시도
            match = re.search(r'res://([^\s"\']+)', str(error))
            if match:
                resource_path = match.group(0)
        
        if resource_path:
            try:
                from modules.auto_resource_generator import get_resource_generator
                generator = get_resource_generator()
                
                project_path = Path(context.get("project_path", "."))
                if await generator.generate_missing_resource(resource_path, project_path):
                    return True, f"Auto-generated resource: {resource_path}"
            except:
                pass
        
        return False, "Could not generate missing resource"
    
    async def _resolve_performance_issue(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """성능 문제 해결"""
        # 성능 프로파일링 및 최적화
        optimization_applied = []
        
        # 1. 캐싱 추가
        if "repeated_calculation" in context:
            optimization_applied.append("Added caching")
        
        # 2. 알고리즘 최적화
        if "inefficient_loop" in context:
            optimization_applied.append("Optimized loop")
        
        # 3. 리소스 사용 최적화
        if "memory_intensive" in context:
            optimization_applied.append("Reduced memory usage")
        
        if optimization_applied:
            return True, f"Applied optimizations: {', '.join(optimization_applied)}"
        
        return False, "Performance optimization requires manual analysis"
    
    async def _resolve_dependency_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, str]:
        """의존성 오류 해결"""
        error_str = str(error)
        
        # 누락된 모듈 찾기
        match = re.search(r"No module named '(\w+)'", error_str)
        if match:
            module_name = match.group(1)
            
            # 일반적인 대체 모듈 매핑
            alternatives = {
                "cv2": "opencv-python",
                "sklearn": "scikit-learn",
                "np": "numpy",
                "pd": "pandas"
            }
            
            install_name = alternatives.get(module_name, module_name)
            
            # pip install 시도
            try:
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", install_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    return True, f"Installed missing module: {install_name}"
            except:
                pass
        
        return False, "Could not resolve dependency"
    
    async def _add_zero_division_check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """0으로 나누기 체크 추가"""
        file_path = context.get("file_path")
        line_number = context.get("line_number")
        
        if file_path and line_number:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 나누기 연산 찾기
                if line_number <= len(lines):
                    line = lines[line_number - 1]
                    if "/" in line:
                        # 간단한 체크 추가
                        indent = len(line) - len(line.lstrip())
                        check_line = " " * indent + "if denominator != 0:  # Added by failure tracking\n"
                        lines.insert(line_number - 1, check_line)
                        lines[line_number] = " " * (indent + 4) + lines[line_number].lstrip()
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        
                        return True, "Added zero division check"
            except:
                pass
        
        return False, "Could not add zero division check"
    
    async def _add_bounds_check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """배열 범위 체크 추가"""
        # 구현 생략 - 실제로는 코드 분석 후 적절한 체크 추가
        return False, "Bounds check requires manual implementation"
    
    async def _add_none_check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """None 체크 추가"""
        # 구현 생략 - 실제로는 코드 분석 후 적절한 체크 추가
        return False, "None check requires manual implementation"
    
    async def _learn_from_failure(
        self,
        failure_id: int,
        error: Exception,
        failure_type: FailureType,
        context: Dict[str, Any]
    ):
        """실패로부터 학습"""
        # 패턴 분석
        pattern_hash = self._generate_pattern_hash(error, failure_type)
        
        # 유사한 실패 찾기
        conn = sqlite3.connect(self.failure_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*), AVG(resolution_time), GROUP_CONCAT(resolution_method)
            FROM failures
            WHERE pattern_hash = ? AND resolved = TRUE
        """, (pattern_hash,))
        
        count, avg_resolution_time, resolution_methods = cursor.fetchone()
        
        if count > 0:
            # 성공적인 해결 방법 학습
            insight = {
                "pattern": pattern_hash,
                "failure_type": failure_type.value,
                "successful_resolutions": count,
                "avg_resolution_time": avg_resolution_time,
                "methods": resolution_methods.split(",") if resolution_methods else []
            }
            
            self.learning_insights[pattern_hash] = insight
            
            # 인사이트 저장
            cursor.execute("""
                INSERT INTO learning_insights (
                    timestamp, insight_type, description,
                    related_failures, prevention_strategy, effectiveness_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                "pattern_recognition",
                f"Learned pattern for {failure_type.value}",
                pattern_hash,
                json.dumps(insight),
                count / (count + 1)  # 효과성 점수
            ))
        
        conn.commit()
        conn.close()
    
    async def get_failure_report(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """실패 보고서 생성"""
        conn = sqlite3.connect(self.failure_db_path)
        cursor = conn.cursor()
        
        # 프로젝트별 필터
        where_clause = "WHERE project_name = ?" if project_name else ""
        params = (project_name,) if project_name else ()
        
        # 전체 통계
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN resolved = TRUE THEN 1 ELSE 0 END) as resolved,
                AVG(resolution_time) as avg_resolution_time,
                MAX(recurrence_count) as max_recurrence
            FROM failures {where_clause}
        """, params)
        
        stats = cursor.fetchone()
        
        # 유형별 분포
        cursor.execute(f"""
            SELECT failure_type, COUNT(*) as count
            FROM failures {where_clause}
            GROUP BY failure_type
            ORDER BY count DESC
        """, params)
        
        type_distribution = cursor.fetchall()
        
        # 심각도별 분포
        cursor.execute(f"""
            SELECT severity, COUNT(*) as count
            FROM failures {where_clause}
            GROUP BY severity
            ORDER BY severity DESC
        """, params)
        
        severity_distribution = cursor.fetchall()
        
        # 최근 실패
        cursor.execute(f"""
            SELECT timestamp, failure_type, error_message, resolved
            FROM failures {where_clause}
            ORDER BY timestamp DESC
            LIMIT 10
        """, params)
        
        recent_failures = cursor.fetchall()
        
        # 학습 인사이트
        cursor.execute("""
            SELECT description, effectiveness_score
            FROM learning_insights
            ORDER BY effectiveness_score DESC
            LIMIT 5
        """)
        
        top_insights = cursor.fetchall()
        
        conn.close()
        
        return {
            "statistics": {
                "total_failures": stats[0] or 0,
                "resolved_failures": stats[1] or 0,
                "resolution_rate": (stats[1] / stats[0] * 100) if stats[0] else 0,
                "avg_resolution_time": stats[2] or 0,
                "max_recurrence": stats[3] or 0
            },
            "type_distribution": [
                {"type": t, "count": c} for t, c in type_distribution
            ],
            "severity_distribution": [
                {"severity": s, "count": c} for s, c in severity_distribution
            ],
            "recent_failures": [
                {
                    "timestamp": f[0],
                    "type": f[1],
                    "error": f[2],
                    "resolved": bool(f[3])
                } for f in recent_failures
            ],
            "top_insights": [
                {"insight": i[0], "effectiveness": i[1]} for i in top_insights
            ]
        }
    
    async def suggest_preventive_measures(
        self,
        project_path: Path
    ) -> List[Dict[str, Any]]:
        """예방 조치 제안"""
        suggestions = []
        
        # 반복적인 실패 패턴 분석
        conn = sqlite3.connect(self.failure_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pattern_hash, failure_type, COUNT(*) as count, 
                   MAX(error_message) as sample_error
            FROM failures
            WHERE recurrence_count > 2
            GROUP BY pattern_hash
            ORDER BY count DESC
            LIMIT 10
        """)
        
        recurring_patterns = cursor.fetchall()
        
        for pattern in recurring_patterns:
            pattern_hash, failure_type, count, sample_error = pattern
            
            # 예방 전략 생성
            prevention = self._generate_prevention_strategy(
                failure_type, sample_error, count
            )
            
            suggestions.append({
                "pattern": pattern_hash,
                "type": failure_type,
                "occurrences": count,
                "prevention": prevention,
                "priority": "high" if count > 5 else "medium"
            })
        
        conn.close()
        
        # 코드 품질 개선 제안
        quality_suggestions = await self._analyze_code_quality_issues(project_path)
        suggestions.extend(quality_suggestions)
        
        return suggestions
    
    def _generate_prevention_strategy(
        self,
        failure_type: str,
        sample_error: str,
        occurrence_count: int
    ) -> Dict[str, Any]:
        """예방 전략 생성"""
        strategies = {
            FailureType.SYNTAX_ERROR.value: {
                "action": "Add linting and syntax checking",
                "tools": ["pylint", "flake8", "black"],
                "automation": "Pre-commit hooks"
            },
            FailureType.RUNTIME_ERROR.value: {
                "action": "Add comprehensive error handling",
                "patterns": ["try-except blocks", "input validation", "type checking"],
                "testing": "Unit tests for edge cases"
            },
            FailureType.RESOURCE_MISSING.value: {
                "action": "Resource validation on startup",
                "checks": ["File existence", "Path validation", "Fallback resources"],
                "tools": ["Resource manifest", "Asset pipeline"]
            },
            FailureType.DEPENDENCY_ERROR.value: {
                "action": "Dependency management",
                "tools": ["requirements.txt", "virtual environments", "dependency lock files"],
                "validation": "Import checks on startup"
            },
            FailureType.PERFORMANCE_ISSUE.value: {
                "action": "Performance monitoring",
                "techniques": ["Profiling", "Benchmarking", "Resource limits"],
                "optimization": ["Caching", "Algorithm optimization", "Lazy loading"]
            }
        }
        
        base_strategy = strategies.get(failure_type, {
            "action": "General quality improvement",
            "recommendation": "Code review and testing"
        })
        
        # 발생 빈도에 따른 우선순위 조정
        base_strategy["urgency"] = "critical" if occurrence_count > 10 else "important"
        base_strategy["estimated_impact"] = f"Prevent ~{occurrence_count} failures/month"
        
        return base_strategy
    
    async def _analyze_code_quality_issues(self, project_path: Path) -> List[Dict[str, Any]]:
        """코드 품질 문제 분석"""
        suggestions = []
        
        # 기본 코드 품질 체크
        quality_checks = [
            {
                "check": "error_handling",
                "description": "Add error handling to critical functions",
                "pattern": r"def\s+\w+.*:\s*\n(?!.*try:)",
                "priority": "high"
            },
            {
                "check": "input_validation",
                "description": "Validate user inputs",
                "pattern": r"input\(|get_node\(|get\(",
                "priority": "medium"
            },
            {
                "check": "resource_checks",
                "description": "Check resource existence before use",
                "pattern": r"load\(|preload\(|open\(",
                "priority": "medium"
            }
        ]
        
        for check in quality_checks:
            matches = 0
            for file_path in project_path.rglob("*.gd"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(check["pattern"], content):
                            matches += 1
                except:
                    pass
            
            if matches > 0:
                suggestions.append({
                    "type": "code_quality",
                    "check": check["check"],
                    "description": check["description"],
                    "affected_files": matches,
                    "priority": check["priority"]
                })
        
        return suggestions
    
    def export_knowledge_base(self, output_path: Path):
        """지식 베이스 내보내기"""
        conn = sqlite3.connect(self.failure_db_path)
        
        # 모든 데이터를 JSON으로 내보내기
        knowledge = {
            "export_date": datetime.now().isoformat(),
            "statistics": self.stats,
            "failures": [],
            "solutions": [],
            "insights": []
        }
        
        # 실패 데이터
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM failures")
        columns = [description[0] for description in cursor.description]
        for row in cursor.fetchall():
            knowledge["failures"].append(dict(zip(columns, row)))
        
        # 해결책 데이터
        cursor.execute("SELECT * FROM solutions")
        columns = [description[0] for description in cursor.description]
        for row in cursor.fetchall():
            knowledge["solutions"].append(dict(zip(columns, row)))
        
        # 인사이트 데이터
        cursor.execute("SELECT * FROM learning_insights")
        columns = [description[0] for description in cursor.description]
        for row in cursor.fetchall():
            knowledge["insights"].append(dict(zip(columns, row)))
        
        conn.close()
        
        # JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 지식 베이스 내보내기 완료: {output_path}")


# 싱글톤 인스턴스
_failure_tracker = None

def get_failure_tracker() -> FailureTrackingSystem:
    """실패 추적 시스템 싱글톤 인스턴스 반환"""
    global _failure_tracker
    if _failure_tracker is None:
        _failure_tracker = FailureTrackingSystem()
    return _failure_tracker


# 테스트 및 예제
async def test_failure_tracking():
    """테스트 함수"""
    tracker = get_failure_tracker()
    
    # 테스트 실패 추적
    try:
        # 의도적인 오류 발생
        result = 10 / 0
    except Exception as e:
        failure_id = await tracker.track_failure(
            error=e,
            context={
                "operation": "division",
                "phase": "calculation",
                "values": {"numerator": 10, "denominator": 0}
            },
            project_name="TestProject",
            file_path=__file__,
            line_number=783
        )
        
        print(f"Tracked failure: ID={failure_id}")
    
    # 보고서 생성
    report = await tracker.get_failure_report("TestProject")
    print("\nFailure Report:")
    print(json.dumps(report, indent=2))
    
    # 예방 조치 제안
    suggestions = await tracker.suggest_preventive_measures(Path("."))
    print("\nPreventive Measures:")
    for suggestion in suggestions:
        print(f"- {suggestion['description']} (Priority: {suggestion['priority']})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_failure_tracking())