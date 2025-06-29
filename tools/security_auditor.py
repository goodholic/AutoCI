#!/usr/bin/env python3
"""
AutoCI Security Auditor - 보안 감사 및 취약점 검사 시스템
"""

import os
import sys
import ast
import re
import json
import hashlib
import secrets
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

@dataclass
class SecurityIssue:
    """보안 이슈"""
    severity: str  # critical, high, medium, low, info
    category: str  # injection, auth, crypto, etc.
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None

@dataclass
class SecurityReport:
    """보안 리포트"""
    scan_time: datetime
    total_files_scanned: int
    issues_found: List[SecurityIssue]
    summary: Dict[str, int]
    recommendations: List[str]
    compliance_status: Dict[str, bool]

class SecurityAuditor:
    """보안 감사기"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.output_dir = self.project_root / "security_reports"
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("SecurityAuditor")
        
        # 보안 패턴 정의
        self.security_patterns = {
            # SQL Injection 패턴
            "sql_injection": [
                r"cursor\.execute\s*\(\s*[\"'][^\"']*%[sd][^\"']*[\"']\s*%",
                r"\.format\s*\([^)]*\)\s*INTO\s+\w+",
                r"\"[^\"]*\"\s*\+\s*\w+.*?(SELECT|INSERT|UPDATE|DELETE)",
                r"f[\"'][^\"']*{[^}]*}[^\"']*(SELECT|INSERT|UPDATE|DELETE)"
            ],
            
            # Command Injection 패턴
            "command_injection": [
                r"os\.system\s*\([^)]*\+",
                r"subprocess\.(call|run|Popen)\s*\([^)]*\+",
                r"os\.popen\s*\([^)]*\+",
                r"eval\s*\([^)]*input",
                r"exec\s*\([^)]*input"
            ],
            
            # Path Traversal 패턴
            "path_traversal": [
                r"open\s*\([^)]*\.\./",
                r"Path\s*\([^)]*\.\./",
                r"os\.path\.join\s*\([^)]*input[^)]*\)",
                r"\.\./"
            ],
            
            # Hardcoded Secrets 패턴
            "hardcoded_secrets": [
                r"password\s*=\s*[\"'][^\"']{8,}[\"']",
                r"api_key\s*=\s*[\"'][^\"']{20,}[\"']",
                r"secret\s*=\s*[\"'][^\"']{10,}[\"']",
                r"token\s*=\s*[\"'][^\"']{20,}[\"']",
                r"private_key\s*=\s*[\"'][^\"']{100,}[\"']"
            ],
            
            # Weak Crypto 패턴
            "weak_crypto": [
                r"hashlib\.md5\s*\(",
                r"hashlib\.sha1\s*\(",
                r"random\.random\s*\(",
                r"random\.randint\s*\(",
                r"DES\.|RC4\.|ECB"
            ],
            
            # Insecure Deserialization 패턴
            "insecure_deserialization": [
                r"pickle\.loads?\s*\(",
                r"yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader",
                r"eval\s*\(",
                r"exec\s*\("
            ],
            
            # Debug Information 패턴
            "debug_info": [
                r"print\s*\([^)]*password",
                r"print\s*\([^)]*secret",
                r"print\s*\([^)]*token",
                r"logging\.debug\s*\([^)]*password",
                r"DEBUG\s*=\s*True"
            ],
            
            # Unsafe File Operations 패턴
            "unsafe_file_ops": [
                r"open\s*\([^)]*[\"']w[\"'].*mode\s*=\s*[\"']777[\"']",
                r"os\.chmod\s*\([^)]*0o777",
                r"tempfile\.mktemp\s*\(",
                r"os\.tempnam\s*\("
            ]
        }
        
        # 안전하지 않은 함수들
        self.unsafe_functions = {
            "eval", "exec", "compile", "__import__",
            "getattr", "setattr", "delattr", "hasattr",
            "vars", "dir", "globals", "locals"
        }
        
        # 보안 헤더 체크
        self.required_security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000",
            "Content-Security-Policy": "default-src 'self'"
        }
    
    def run_comprehensive_audit(self) -> SecurityReport:
        """종합 보안 감사 실행"""
        print("🔒 보안 감사 시작...")
        
        scan_start = datetime.now()
        issues = []
        files_scanned = 0
        
        # Python 파일 스캔
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            if self._should_scan_file(py_file):
                file_issues = self.scan_python_file(py_file)
                issues.extend(file_issues)
                files_scanned += 1
        
        # 설정 파일 스캔
        config_files = list(self.project_root.rglob("*.json")) + \
                      list(self.project_root.rglob("*.yaml")) + \
                      list(self.project_root.rglob("*.yml")) + \
                      list(self.project_root.rglob("*.ini")) + \
                      list(self.project_root.rglob("*.cfg"))
        
        for config_file in config_files:
            if self._should_scan_file(config_file):
                file_issues = self.scan_config_file(config_file)
                issues.extend(file_issues)
                files_scanned += 1
        
        # 의존성 취약점 검사
        dependency_issues = self.check_dependency_vulnerabilities()
        issues.extend(dependency_issues)
        
        # 권한 검사
        permission_issues = self.check_file_permissions()
        issues.extend(permission_issues)
        
        # 리포트 생성
        report = self._generate_report(scan_start, files_scanned, issues)
        
        # 리포트 저장
        self._save_report(report)
        
        return report
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """파일 스캔 여부 결정"""
        # 제외할 디렉토리
        exclude_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
        
        for part in file_path.parts:
            if part in exclude_dirs:
                return False
        
        # 너무 큰 파일 제외 (10MB 이상)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return False
        except:
            return False
        
        return True
    
    def scan_python_file(self, file_path: Path) -> List[SecurityIssue]:
        """Python 파일 스캔"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # AST를 사용한 고급 분석
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast(tree, file_path, lines))
            except SyntaxError:
                # 구문 오류가 있는 파일은 패턴 매칭으로만 검사
                pass
            
            # 패턴 매칭 검사
            for line_num, line in enumerate(lines, 1):
                for category, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            issue = SecurityIssue(
                                severity=self._get_severity(category),
                                category=category,
                                title=f"Potential {category.replace('_', ' ').title()}",
                                description=f"Line {line_num} contains a pattern that may indicate {category}",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                code_snippet=line.strip(),
                                recommendation=self._get_recommendation(category),
                                cwe_id=self._get_cwe_id(category)
                            )
                            issues.append(issue)
            
            # 추가 보안 검사
            issues.extend(self._check_imports(content, file_path))
            issues.extend(self._check_crypto_usage(content, file_path))
            
        except Exception as e:
            self.logger.error(f"Error scanning {file_path}: {e}")
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[SecurityIssue]:
        """AST를 사용한 코드 분석"""
        issues = []
        
        for node in ast.walk(tree):
            # 위험한 함수 호출 체크
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.unsafe_functions:
                        issues.append(SecurityIssue(
                            severity="high",
                            category="code_injection",
                            title=f"Dangerous function call: {node.func.id}",
                            description=f"Use of {node.func.id} function can lead to code injection",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=node.lineno,
                            code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                            recommendation=f"Avoid using {node.func.id} with user input",
                            cwe_id="CWE-94"
                        ))
            
            # 하드코딩된 비밀번호 체크
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(keyword in target.id.lower() for keyword in ['password', 'secret', 'key', 'token']):
                            if isinstance(node.value, ast.Str):
                                issues.append(SecurityIssue(
                                    severity="high",
                                    category="hardcoded_secrets",
                                    title="Hardcoded secret detected",
                                    description=f"Variable {target.id} contains a hardcoded secret",
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=node.lineno,
                                    code_snippet=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
                                    recommendation="Use environment variables or secure configuration",
                                    cwe_id="CWE-798"
                                ))
        
        return issues
    
    def scan_config_file(self, file_path: Path) -> List[SecurityIssue]:
        """설정 파일 스캔"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # 민감한 정보 패턴 검사
            sensitive_patterns = [
                (r'password\s*[:=]\s*["\']?[^"\'\s]{8,}', "Hardcoded password"),
                (r'api_key\s*[:=]\s*["\']?[^"\'\s]{20,}', "Hardcoded API key"),
                (r'secret\s*[:=]\s*["\']?[^"\'\s]{10,}', "Hardcoded secret"),
                (r'private_key\s*[:=]', "Private key in config"),
                (r'debug\s*[:=]\s*true', "Debug mode enabled")
            ]
            
            for line_num, line in enumerate(lines, 1):
                for pattern, description in sensitive_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity="medium",
                            category="config_security",
                            title="Sensitive information in config",
                            description=description,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation="Move sensitive data to environment variables",
                            cwe_id="CWE-200"
                        ))
            
            # JSON/YAML 특정 검사
            if file_path.suffix in ['.json', '.yaml', '.yml']:
                issues.extend(self._check_config_structure(content, file_path))
            
        except Exception as e:
            self.logger.error(f"Error scanning config file {file_path}: {e}")
        
        return issues
    
    def check_dependency_vulnerabilities(self) -> List[SecurityIssue]:
        """의존성 취약점 검사"""
        issues = []
        
        # requirements.txt 검사
        req_files = list(self.project_root.glob("*requirements*.txt"))
        req_files.extend(list(self.project_root.glob("Pipfile*")))
        
        for req_file in req_files:
            try:
                content = req_file.read_text()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 버전이 고정되지 않은 의존성 체크
                        if '==' not in line and '>=' not in line and line.count('=') == 0:
                            issues.append(SecurityIssue(
                                severity="medium",
                                category="dependency_management",
                                title="Unpinned dependency version",
                                description=f"Dependency {line} does not have a pinned version",
                                file_path=str(req_file.relative_to(self.project_root)),
                                line_number=line_num,
                                code_snippet=line,
                                recommendation="Pin dependency versions for security and reproducibility",
                                cwe_id="CWE-1104"
                            ))
                        
                        # 알려진 취약한 패키지들
                        vulnerable_packages = [
                            'pycrypto',  # pycryptodome으로 대체 권장
                            'requests[security]',  # 구버전에 취약점
                            'flask<1.0',  # 구버전에 취약점
                            'django<2.2'  # 구버전에 취약점
                        ]
                        
                        for vuln_pkg in vulnerable_packages:
                            if vuln_pkg.split('<')[0].split('[')[0] in line.lower():
                                issues.append(SecurityIssue(
                                    severity="high",
                                    category="vulnerable_dependency",
                                    title="Potentially vulnerable dependency",
                                    description=f"Package {line} may have known vulnerabilities",
                                    file_path=str(req_file.relative_to(self.project_root)),
                                    line_number=line_num,
                                    code_snippet=line,
                                    recommendation="Update to latest secure version",
                                    cwe_id="CWE-1035"
                                ))
                
                # safety 도구 실행 시도
                try:
                    result = subprocess.run(['safety', 'check', '-r', str(req_file)], 
                                          capture_output=True, text=True, timeout=30)
                    if result.returncode != 0 and "vulnerabilities found" in result.stdout:
                        issues.append(SecurityIssue(
                            severity="high",
                            category="vulnerable_dependency",
                            title="Vulnerabilities found by safety tool",
                            description="Dependencies contain known vulnerabilities",
                            file_path=str(req_file.relative_to(self.project_root)),
                            line_number=0,
                            code_snippet=result.stdout[:200],
                            recommendation="Update vulnerable dependencies",
                            cwe_id="CWE-1035"
                        ))
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # safety 도구가 없거나 타임아웃
                    pass
                
            except Exception as e:
                self.logger.error(f"Error checking dependencies in {req_file}: {e}")
        
        return issues
    
    def check_file_permissions(self) -> List[SecurityIssue]:
        """파일 권한 검사"""
        issues = []
        
        # 민감한 파일들의 권한 체크
        sensitive_patterns = [
            "**/*secret*",
            "**/*key*",
            "**/*password*",
            "**/*.pem",
            "**/*.key",
            "**/config/*"
        ]
        
        for pattern in sensitive_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        # Unix 계열에서만 권한 체크
                        if hasattr(os, 'stat'):
                            stat_info = file_path.stat()
                            mode = stat_info.st_mode
                            
                            # 다른 사용자가 읽을 수 있는지 체크
                            if mode & 0o044:  # others read permission
                                issues.append(SecurityIssue(
                                    severity="medium",
                                    category="file_permissions",
                                    title="Sensitive file with loose permissions",
                                    description=f"File {file_path.name} is readable by others",
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=0,
                                    code_snippet=f"Permission: {oct(mode)[-3:]}",
                                    recommendation="Set restrictive permissions (600 or 644)",
                                    cwe_id="CWE-732"
                                ))
                    except Exception as e:
                        self.logger.error(f"Error checking permissions for {file_path}: {e}")
        
        return issues
    
    def _check_imports(self, content: str, file_path: Path) -> List[SecurityIssue]:
        """import 문 검사"""
        issues = []
        
        dangerous_imports = {
            'pickle': 'Use safer alternatives like json',
            'subprocess': 'Be careful with user input',
            'os.system': 'Use subprocess with proper input validation',
            'eval': 'Never use with user input',
            'exec': 'Never use with user input'
        }
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for dangerous_module, recommendation in dangerous_imports.items():
                if f'import {dangerous_module}' in line or f'from {dangerous_module}' in line:
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="dangerous_import",
                        title=f"Potentially dangerous import: {dangerous_module}",
                        description=f"Import of {dangerous_module} requires careful security review",
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation=recommendation,
                        cwe_id="CWE-94"
                    ))
        
        return issues
    
    def _check_crypto_usage(self, content: str, file_path: Path) -> List[SecurityIssue]:
        """암호화 사용 검사"""
        issues = []
        
        # 약한 암호화 알고리즘
        weak_crypto = {
            'md5': 'Use SHA-256 or better',
            'sha1': 'Use SHA-256 or better', 
            'DES': 'Use AES',
            'RC4': 'Use AES',
            'ECB': 'Use CBC or GCM mode'
        }
        
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            for weak_alg, recommendation in weak_crypto.items():
                if weak_alg.lower() in line.lower():
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="weak_crypto",
                        title=f"Weak cryptographic algorithm: {weak_alg}",
                        description=f"Use of {weak_alg} is not recommended for security",
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation=recommendation,
                        cwe_id="CWE-327"
                    ))
        
        return issues
    
    def _check_config_structure(self, content: str, file_path: Path) -> List[SecurityIssue]:
        """설정 파일 구조 검사"""
        issues = []
        
        try:
            if file_path.suffix == '.json':
                config = json.loads(content)
            elif file_path.suffix in ['.yaml', '.yml']:
                import yaml
                config = yaml.safe_load(content)
            else:
                return issues
            
            # 재귀적으로 설정 검사
            self._check_config_values(config, file_path, issues)
            
        except Exception as e:
            self.logger.error(f"Error parsing config {file_path}: {e}")
        
        return issues
    
    def _check_config_values(self, config: Any, file_path: Path, issues: List[SecurityIssue], path: str = ""):
        """설정 값 재귀 검사"""
        if isinstance(config, dict):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key
                
                # 키 이름 체크
                if any(secret_key in key.lower() for secret_key in ['password', 'secret', 'key', 'token']):
                    if isinstance(value, str) and len(value) > 5:
                        issues.append(SecurityIssue(
                            severity="high",
                            category="config_secrets",
                            title=f"Secret in config: {key}",
                            description=f"Configuration key '{key}' appears to contain a secret",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=0,
                            code_snippet=f"{key}: {'*' * len(str(value))}",
                            recommendation="Move secrets to environment variables",
                            cwe_id="CWE-798"
                        ))
                
                # Debug 모드 체크
                if key.lower() == 'debug' and value is True:
                    issues.append(SecurityIssue(
                        severity="medium",
                        category="debug_mode",
                        title="Debug mode enabled",
                        description="Debug mode should not be enabled in production",
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=0,
                        code_snippet=f"{key}: {value}",
                        recommendation="Disable debug mode in production",
                        cwe_id="CWE-489"
                    ))
                
                # 재귀 호출
                self._check_config_values(value, file_path, issues, current_path)
        
        elif isinstance(config, list):
            for i, item in enumerate(config):
                self._check_config_values(item, file_path, issues, f"{path}[{i}]")
    
    def _get_severity(self, category: str) -> str:
        """카테고리별 심각도 반환"""
        severity_map = {
            "sql_injection": "critical",
            "command_injection": "critical",
            "hardcoded_secrets": "high",
            "path_traversal": "high",
            "weak_crypto": "medium",
            "insecure_deserialization": "high",
            "debug_info": "low",
            "unsafe_file_ops": "medium"
        }
        return severity_map.get(category, "medium")
    
    def _get_recommendation(self, category: str) -> str:
        """카테고리별 권장사항 반환"""
        recommendations = {
            "sql_injection": "Use parameterized queries or ORM",
            "command_injection": "Validate and sanitize all user input",
            "hardcoded_secrets": "Use environment variables or secure vaults",
            "path_traversal": "Validate file paths and use safe path operations",
            "weak_crypto": "Use strong cryptographic algorithms (AES, SHA-256+)",
            "insecure_deserialization": "Use safe serialization formats like JSON",
            "debug_info": "Remove debug information from production code",
            "unsafe_file_ops": "Use secure file operations with proper permissions"
        }
        return recommendations.get(category, "Review and fix security issue")
    
    def _get_cwe_id(self, category: str) -> str:
        """카테고리별 CWE ID 반환"""
        cwe_map = {
            "sql_injection": "CWE-89",
            "command_injection": "CWE-78",
            "hardcoded_secrets": "CWE-798",
            "path_traversal": "CWE-22",
            "weak_crypto": "CWE-327",
            "insecure_deserialization": "CWE-502",
            "debug_info": "CWE-489",
            "unsafe_file_ops": "CWE-732"
        }
        return cwe_map.get(category, "CWE-1035")
    
    def _generate_report(self, scan_start: datetime, files_scanned: int, issues: List[SecurityIssue]) -> SecurityReport:
        """보안 리포트 생성"""
        # 요약 통계
        summary = {
            "critical": len([i for i in issues if i.severity == "critical"]),
            "high": len([i for i in issues if i.severity == "high"]),
            "medium": len([i for i in issues if i.severity == "medium"]),
            "low": len([i for i in issues if i.severity == "low"]),
            "info": len([i for i in issues if i.severity == "info"])
        }
        
        # 카테고리별 통계
        categories = {}
        for issue in issues:
            categories[issue.category] = categories.get(issue.category, 0) + 1
        
        # 권장사항 생성
        recommendations = self._generate_security_recommendations(issues, summary)
        
        # 컴플라이언스 상태
        compliance = {
            "OWASP_Top_10": summary["critical"] == 0 and summary["high"] < 5,
            "CWE_Top_25": summary["critical"] == 0,
            "Security_Best_Practices": len(issues) < 10
        }
        
        return SecurityReport(
            scan_time=scan_start,
            total_files_scanned=files_scanned,
            issues_found=issues,
            summary=summary,
            recommendations=recommendations,
            compliance_status=compliance
        )
    
    def _generate_security_recommendations(self, issues: List[SecurityIssue], summary: Dict[str, int]) -> List[str]:
        """보안 권장사항 생성"""
        recommendations = []
        
        if summary["critical"] > 0:
            recommendations.append("🚨 즉시 조치 필요: Critical 보안 이슈가 발견되었습니다.")
        
        if summary["high"] > 0:
            recommendations.append("⚠️ 우선 조치: High 보안 이슈들을 빠르게 해결하세요.")
        
        # 카테고리별 권장사항
        categories = {}
        for issue in issues:
            categories[issue.category] = categories.get(issue.category, 0) + 1
        
        if categories.get("hardcoded_secrets", 0) > 0:
            recommendations.append("🔐 모든 하드코딩된 비밀정보를 환경변수로 이동하세요.")
        
        if categories.get("sql_injection", 0) > 0:
            recommendations.append("💉 SQL 인젝션 방지를 위해 매개변수화된 쿼리를 사용하세요.")
        
        if categories.get("weak_crypto", 0) > 0:
            recommendations.append("🔒 약한 암호화 알고리즘을 강력한 것으로 교체하세요.")
        
        # 일반적인 권장사항
        recommendations.extend([
            "🛡️ 정기적인 보안 감사를 수행하세요.",
            "📚 팀 전체에 보안 교육을 실시하세요.",
            "🔄 의존성을 정기적으로 업데이트하세요.",
            "📋 보안 코딩 가이드라인을 수립하세요."
        ])
        
        return recommendations
    
    def _save_report(self, report: SecurityReport):
        """보안 리포트 저장"""
        timestamp = report.scan_time.strftime("%Y%m%d_%H%M%S")
        
        # JSON 리포트
        json_file = self.output_dir / f"security_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, default=str, ensure_ascii=False)
        
        # HTML 리포트
        html_file = self.output_dir / f"security_report_{timestamp}.html"
        self._generate_html_report(report, html_file)
        
        # CSV 리포트 (이슈 목록)
        csv_file = self.output_dir / f"security_issues_{timestamp}.csv"
        self._generate_csv_report(report, csv_file)
        
        print(f"📊 보안 리포트 저장 완료:")
        print(f"  - JSON: {json_file}")
        print(f"  - HTML: {html_file}")
        print(f"  - CSV: {csv_file}")
    
    def _generate_html_report(self, report: SecurityReport, html_file: Path):
        """HTML 리포트 생성"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AutoCI Security Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ padding: 15px; background: #e9ecef; border-radius: 5px; text-align: center; }}
        .critical {{ background: #dc3545; color: white; }}
        .high {{ background: #fd7e14; color: white; }}
        .medium {{ background: #ffc107; }}
        .low {{ background: #28a745; color: white; }}
        .issue {{ margin: 10px 0; padding: 15px; border-left: 4px solid #dee2e6; }}
        .issue.critical {{ border-left-color: #dc3545; }}
        .issue.high {{ border-left-color: #fd7e14; }}
        .issue.medium {{ border-left-color: #ffc107; }}
        .issue.low {{ border-left-color: #28a745; }}
        .code {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 3px; font-family: monospace; }}
        .recommendations {{ background: #d1ecf1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 AutoCI Security Report</h1>
        <p><strong>Scan Time:</strong> {report.scan_time}</p>
        <p><strong>Files Scanned:</strong> {report.total_files_scanned}</p>
        <p><strong>Total Issues:</strong> {len(report.issues_found)}</p>
    </div>
    
    <div class="summary">
        <div class="metric critical">
            <h3>{report.summary['critical']}</h3>
            <p>Critical</p>
        </div>
        <div class="metric high">
            <h3>{report.summary['high']}</h3>
            <p>High</p>
        </div>
        <div class="metric medium">
            <h3>{report.summary['medium']}</h3>
            <p>Medium</p>
        </div>
        <div class="metric low">
            <h3>{report.summary['low']}</h3>
            <p>Low</p>
        </div>
    </div>
    
    <div class="recommendations">
        <h2>🛡️ Recommendations</h2>
        <ul>
"""
        
        for rec in report.recommendations:
            html_content += f"<li>{rec}</li>\n"
        
        html_content += """
        </ul>
    </div>
    
    <h2>📋 Security Issues</h2>
"""
        
        for issue in report.issues_found:
            html_content += f"""
    <div class="issue {issue.severity}">
        <h3>{issue.title}</h3>
        <p><strong>Severity:</strong> {issue.severity.upper()}</p>
        <p><strong>Category:</strong> {issue.category}</p>
        <p><strong>File:</strong> {issue.file_path}:{issue.line_number}</p>
        <p><strong>Description:</strong> {issue.description}</p>
        <div class="code">{issue.code_snippet}</div>
        <p><strong>Recommendation:</strong> {issue.recommendation}</p>
        {f'<p><strong>CWE:</strong> {issue.cwe_id}</p>' if issue.cwe_id else ''}
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        html_file.write_text(html_content, encoding='utf-8')
    
    def _generate_csv_report(self, report: SecurityReport, csv_file: Path):
        """CSV 리포트 생성"""
        import csv
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Severity', 'Category', 'Title', 'File', 'Line', 'Description', 'Recommendation', 'CWE'])
            
            for issue in report.issues_found:
                writer.writerow([
                    issue.severity,
                    issue.category,
                    issue.title,
                    issue.file_path,
                    issue.line_number,
                    issue.description,
                    issue.recommendation,
                    issue.cwe_id or ''
                ])

def main():
    """메인 실행 함수"""
    print("🔒 AutoCI Security Auditor")
    print("=" * 60)
    
    auditor = SecurityAuditor()
    
    try:
        # 종합 보안 감사 실행
        report = auditor.run_comprehensive_audit()
        
        # 결과 출력
        print(f"\n📊 보안 감사 완료!")
        print(f"스캔한 파일: {report.total_files_scanned}개")
        print(f"발견된 이슈: {len(report.issues_found)}개")
        print(f"  - Critical: {report.summary['critical']}개")
        print(f"  - High: {report.summary['high']}개")
        print(f"  - Medium: {report.summary['medium']}개") 
        print(f"  - Low: {report.summary['low']}개")
        
        print(f"\n🛡️ 컴플라이언스 상태:")
        for standard, status in report.compliance_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {standard}")
        
        print(f"\n📁 리포트 저장 위치: {auditor.output_dir}")
        
        return 0 if report.summary['critical'] == 0 else 1
        
    except Exception as e:
        print(f"❌ 보안 감사 실패: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())