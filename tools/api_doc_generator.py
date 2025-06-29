#!/usr/bin/env python3
"""
AutoCI API Documentation Generator - API 문서 자동 생성 시스템
"""

import os
import sys
import ast
import inspect
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

class APIDocumentationGenerator:
    """API 문서 자동 생성기"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs" / "api"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # 문서화할 모듈 목록
        self.modules_to_document = [
            "modules.enhanced_error_handler",
            "modules.enhanced_monitoring", 
            "modules.enhanced_logging",
            "modules.enhanced_godot_controller",
            "modules.csharp_learning_agent",
            "modules.ai_model_integration",
            "autoci_production"
        ]
        
        # 타입 힌트 매핑
        self.type_hints = {
            "str": "string",
            "int": "integer", 
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "Dict": "object",
            "List": "array",
            "Optional": "optional",
            "Union": "union",
            "Any": "any"
        }
    
    def generate_full_documentation(self):
        """전체 API 문서 생성"""
        print("📚 API 문서 생성 시작...")
        
        api_docs = {
            "title": "AutoCI API Documentation",
            "version": "1.0.0", 
            "description": "상용화 수준의 24시간 AI 게임 개발 시스템 API",
            "generated": datetime.now().isoformat(),
            "modules": {}
        }
        
        for module_name in self.modules_to_document:
            try:
                module_doc = self.generate_module_documentation(module_name)
                if module_doc:
                    api_docs["modules"][module_name] = module_doc
                    print(f"✅ {module_name} 문서화 완료")
            except Exception as e:
                print(f"❌ {module_name} 문서화 실패: {e}")
        
        # JSON API 문서 생성
        self.save_json_documentation(api_docs)
        
        # Markdown 문서 생성
        self.generate_markdown_documentation(api_docs)
        
        # OpenAPI 스펙 생성
        self.generate_openapi_spec(api_docs)
        
        print(f"📖 API 문서 생성 완료: {self.docs_dir}")
    
    def generate_module_documentation(self, module_name: str) -> Dict[str, Any]:
        """모듈 문서 생성"""
        try:
            # 모듈 import
            module = self.import_module(module_name)
            if not module:
                return None
            
            module_doc = {
                "name": module_name,
                "description": self.get_module_description(module),
                "classes": {},
                "functions": {},
                "constants": {}
            }
            
            # 클래스 문서화
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if not name.startswith("_") and obj.__module__ == module.__name__:
                    module_doc["classes"][name] = self.document_class(obj)
            
            # 함수 문서화  
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith("_") and obj.__module__ == module.__name__:
                    module_doc["functions"][name] = self.document_function(obj)
            
            # 상수 문서화
            for name, obj in inspect.getmembers(module):
                if (not name.startswith("_") and 
                    not inspect.isclass(obj) and 
                    not inspect.isfunction(obj) and
                    not inspect.ismodule(obj) and
                    isinstance(obj, (str, int, float, bool, list, dict))):
                    module_doc["constants"][name] = {
                        "type": type(obj).__name__,
                        "value": obj if not isinstance(obj, (list, dict)) else str(obj)[:100],
                        "description": f"{name} constant"
                    }
            
            return module_doc
            
        except Exception as e:
            print(f"모듈 {module_name} 문서화 중 오류: {e}")
            return None
    
    def import_module(self, module_name: str):
        """모듈 안전하게 import"""
        try:
            # sys.path에 프로젝트 루트 추가
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            __import__(module_name)
            return sys.modules[module_name]
        except ImportError as e:
            print(f"모듈 {module_name} import 실패: {e}")
            return None
    
    def get_module_description(self, module) -> str:
        """모듈 설명 추출"""
        if module.__doc__:
            return module.__doc__.strip()
        return f"Module: {module.__name__}"
    
    def document_class(self, cls) -> Dict[str, Any]:
        """클래스 문서화"""
        class_doc = {
            "name": cls.__name__,
            "description": self.clean_docstring(cls.__doc__),
            "methods": {},
            "properties": {},
            "inheritance": [base.__name__ for base in cls.__bases__ if base.__name__ != "object"]
        }
        
        # 메서드 문서화
        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith("_"):
                class_doc["methods"][name] = self.document_function(method)
        
        # 함수 (정적 메서드 등) 문서화
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                class_doc["methods"][name] = self.document_function(method)
        
        # 프로퍼티 문서화
        for name, prop in inspect.getmembers(cls):
            if isinstance(prop, property):
                class_doc["properties"][name] = {
                    "type": "property",
                    "description": self.clean_docstring(prop.__doc__),
                    "getter": prop.fget is not None,
                    "setter": prop.fset is not None,
                    "deleter": prop.fdel is not None
                }
        
        return class_doc
    
    def document_function(self, func) -> Dict[str, Any]:
        """함수/메서드 문서화"""
        try:
            signature = inspect.signature(func)
            
            func_doc = {
                "name": func.__name__,
                "description": self.clean_docstring(func.__doc__),
                "parameters": {},
                "returns": {},
                "async": inspect.iscoroutinefunction(func),
                "signature": str(signature)
            }
            
            # 매개변수 문서화
            for param_name, param in signature.parameters.items():
                param_doc = {
                    "type": self.get_type_hint(param.annotation),
                    "required": param.default == inspect.Parameter.empty,
                    "default": None if param.default == inspect.Parameter.empty else str(param.default)
                }
                
                # docstring에서 매개변수 설명 추출
                param_description = self.extract_param_description(func.__doc__, param_name)
                if param_description:
                    param_doc["description"] = param_description
                
                func_doc["parameters"][param_name] = param_doc
            
            # 반환값 문서화
            if signature.return_annotation != inspect.Signature.empty:
                func_doc["returns"] = {
                    "type": self.get_type_hint(signature.return_annotation),
                    "description": self.extract_return_description(func.__doc__)
                }
            
            return func_doc
            
        except Exception as e:
            return {
                "name": func.__name__,
                "description": self.clean_docstring(func.__doc__),
                "error": f"Documentation generation failed: {e}"
            }
    
    def clean_docstring(self, docstring: str) -> str:
        """docstring 정리"""
        if not docstring:
            return ""
        
        lines = docstring.strip().split('\n')
        if len(lines) == 1:
            return lines[0].strip()
        
        # 첫 번째 줄과 나머지 분리
        first_line = lines[0].strip()
        rest_lines = [line.strip() for line in lines[1:] if line.strip()]
        
        if rest_lines:
            return f"{first_line}\n\n{' '.join(rest_lines)}"
        return first_line
    
    def get_type_hint(self, annotation) -> str:
        """타입 힌트를 문자열로 변환"""
        if annotation == inspect.Parameter.empty:
            return "any"
        
        if hasattr(annotation, "__name__"):
            return self.type_hints.get(annotation.__name__, annotation.__name__)
        
        # typing 모듈의 복잡한 타입들 처리
        annotation_str = str(annotation)
        for typing_type, simple_type in self.type_hints.items():
            if typing_type in annotation_str:
                return simple_type
        
        return annotation_str.replace("typing.", "").replace("<class '", "").replace("'>", "")
    
    def extract_param_description(self, docstring: str, param_name: str) -> str:
        """docstring에서 매개변수 설명 추출"""
        if not docstring:
            return ""
        
        # Args: 섹션에서 매개변수 설명 찾기
        args_pattern = r"Args?:.*?(?=Returns?:|Raises?:|Note:|$)"
        args_match = re.search(args_pattern, docstring, re.DOTALL | re.IGNORECASE)
        
        if args_match:
            args_section = args_match.group()
            param_pattern = rf"{param_name}\s*[:\(]([^:\n]+)"
            param_match = re.search(param_pattern, args_section, re.IGNORECASE)
            
            if param_match:
                return param_match.group(1).strip()
        
        return ""
    
    def extract_return_description(self, docstring: str) -> str:
        """docstring에서 반환값 설명 추출"""
        if not docstring:
            return ""
        
        returns_pattern = r"Returns?:\s*([^:\n]+)"
        returns_match = re.search(returns_pattern, docstring, re.IGNORECASE)
        
        if returns_match:
            return returns_match.group(1).strip()
        
        return ""
    
    def save_json_documentation(self, api_docs: Dict[str, Any]):
        """JSON 형식으로 API 문서 저장"""
        json_file = self.docs_dir / "api_documentation.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(api_docs, f, indent=2, ensure_ascii=False)
        
        print(f"📄 JSON 문서 생성: {json_file}")
    
    def generate_markdown_documentation(self, api_docs: Dict[str, Any]):
        """Markdown 형식으로 API 문서 생성"""
        md_content = []
        
        # 헤더
        md_content.append(f"# {api_docs['title']}")
        md_content.append(f"\n**Version:** {api_docs['version']}")
        md_content.append(f"**Generated:** {api_docs['generated']}")
        md_content.append(f"\n{api_docs['description']}\n")
        
        # 목차
        md_content.append("## Table of Contents\n")
        for module_name in api_docs['modules']:
            md_content.append(f"- [{module_name}](#{module_name.replace('.', '-')})")
        md_content.append("")
        
        # 각 모듈 문서화
        for module_name, module_doc in api_docs['modules'].items():
            md_content.append(f"## {module_name}")
            md_content.append(f"\n{module_doc['description']}\n")
            
            # 클래스들
            if module_doc['classes']:
                md_content.append("### Classes\n")
                
                for class_name, class_doc in module_doc['classes'].items():
                    md_content.append(f"#### {class_name}")
                    md_content.append(f"\n{class_doc['description']}\n")
                    
                    if class_doc['inheritance']:
                        md_content.append(f"**Inherits from:** {', '.join(class_doc['inheritance'])}\n")
                    
                    # 메서드들
                    if class_doc['methods']:
                        md_content.append("**Methods:**\n")
                        
                        for method_name, method_doc in class_doc['methods'].items():
                            md_content.append(f"##### {method_name}")
                            
                            if method_doc.get('async'):
                                md_content.append("*async method*")
                            
                            md_content.append(f"\n```python\n{method_doc['signature']}\n```")
                            md_content.append(f"\n{method_doc['description']}\n")
                            
                            # 매개변수
                            if method_doc['parameters']:
                                md_content.append("**Parameters:**")
                                for param_name, param_doc in method_doc['parameters'].items():
                                    required = "required" if param_doc['required'] else "optional"
                                    default = f" (default: {param_doc['default']})" if param_doc['default'] else ""
                                    description = param_doc.get('description', '')
                                    md_content.append(f"- `{param_name}` ({param_doc['type']}, {required}){default}: {description}")
                                md_content.append("")
                            
                            # 반환값
                            if method_doc['returns']:
                                returns = method_doc['returns']
                                md_content.append(f"**Returns:** {returns['type']} - {returns.get('description', '')}\n")
                    
                    # 프로퍼티들
                    if class_doc['properties']:
                        md_content.append("**Properties:**\n")
                        
                        for prop_name, prop_doc in class_doc['properties'].items():
                            access = []
                            if prop_doc['getter']: access.append("get")
                            if prop_doc['setter']: access.append("set")
                            access_str = "/".join(access) if access else "read-only"
                            
                            md_content.append(f"- `{prop_name}` ({access_str}): {prop_doc['description']}")
                        md_content.append("")
            
            # 함수들
            if module_doc['functions']:
                md_content.append("### Functions\n")
                
                for func_name, func_doc in module_doc['functions'].items():
                    md_content.append(f"#### {func_name}")
                    
                    if func_doc.get('async'):
                        md_content.append("*async function*")
                    
                    md_content.append(f"\n```python\n{func_doc['signature']}\n```")
                    md_content.append(f"\n{func_doc['description']}\n")
            
            # 상수들
            if module_doc['constants']:
                md_content.append("### Constants\n")
                
                for const_name, const_doc in module_doc['constants'].items():
                    md_content.append(f"- `{const_name}` ({const_doc['type']}): {const_doc['description']}")
                md_content.append("")
        
        # 파일 저장
        md_file = self.docs_dir / "API_REFERENCE.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        
        print(f"📝 Markdown 문서 생성: {md_file}")
    
    def generate_openapi_spec(self, api_docs: Dict[str, Any]):
        """OpenAPI 3.0 스펙 생성"""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": api_docs['title'],
                "version": api_docs['version'],
                "description": api_docs['description']
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {}
            }
        }
        
        # API 엔드포인트들 생성 (실제 REST API가 있다면)
        openapi_spec["paths"] = {
            "/api/v1/status": {
                "get": {
                    "summary": "Get system status",
                    "responses": {
                        "200": {
                            "description": "System status information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "uptime": {"type": "number"},
                                            "games_created": {"type": "integer"},
                                            "features_added": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/v1/projects": {
                "get": {
                    "summary": "List all projects",
                    "responses": {
                        "200": {
                            "description": "List of projects",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Project"}
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "summary": "Create new project",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CreateProjectRequest"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Project created",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Project"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        # 스키마 정의
        openapi_spec["components"]["schemas"] = {
            "Project": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["platformer", "racing", "puzzle", "rpg"]},
                    "status": {"type": "string"},
                    "created": {"type": "string", "format": "date-time"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "bugs_fixed": {"type": "integer"}
                }
            },
            "CreateProjectRequest": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["platformer", "racing", "puzzle", "rpg"]}
                }
            }
        }
        
        # OpenAPI 스펙 저장
        openapi_file = self.docs_dir / "openapi.json"
        with open(openapi_file, 'w', encoding='utf-8') as f:
            json.dump(openapi_spec, f, indent=2)
        
        print(f"🔌 OpenAPI 스펙 생성: {openapi_file}")
    
    def generate_usage_examples(self):
        """사용 예제 생성"""
        examples = {
            "basic_usage": {
                "title": "Basic Usage",
                "description": "AutoCI 시스템의 기본 사용법",
                "code": '''
# 개발 모드로 시작
python autoci.py

# 프로덕션 모드로 시작  
AUTOCI_MODE=production python autoci.py

# 상태 확인
autoci> status

# 프로젝트 목록
autoci> projects

# 메트릭스 확인
autoci> metrics
'''
            },
            "api_usage": {
                "title": "API Usage",
                "description": "Python API를 통한 시스템 제어",
                "code": '''
from autoci_production import ProductionAutoCI
import asyncio

async def main():
    # 시스템 초기화
    autoci = ProductionAutoCI()
    
    # 게임 프로젝트 생성
    await autoci.godot_controller.create_project(
        "my_game", "/path/to/project", "platformer"
    )
    
    # 메트릭 기록
    await autoci.monitor.record_metric(
        "custom.metric", 42.0, MetricType.GAUGE
    )

asyncio.run(main())
'''
            },
            "error_handling": {
                "title": "Error Handling",
                "description": "에러 처리 및 복구 사용법",
                "code": '''
from modules.enhanced_error_handler import with_error_handling, ErrorSeverity

@with_error_handling(component="my_component", severity=ErrorSeverity.HIGH)
async def risky_operation():
    # 위험한 작업 수행
    result = await some_operation()
    return result

# 에러가 발생하면 자동으로 복구 시도
result = await risky_operation()
'''
            }
        }
        
        examples_file = self.docs_dir / "EXAMPLES.md"
        
        content = ["# AutoCI Usage Examples\n"]
        
        for example_id, example in examples.items():
            content.append(f"## {example['title']}")
            content.append(f"\n{example['description']}\n")
            content.append(f"```python{example['code']}```\n")
        
        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"📚 사용 예제 생성: {examples_file}")

def main():
    """메인 실행 함수"""
    print("🔧 AutoCI API Documentation Generator")
    print("=" * 60)
    
    generator = APIDocumentationGenerator()
    
    try:
        # 전체 문서 생성
        generator.generate_full_documentation()
        
        # 사용 예제 생성
        generator.generate_usage_examples()
        
        print("\n✅ API 문서 생성 완료!")
        print(f"📁 문서 위치: {generator.docs_dir}")
        print("\n생성된 파일:")
        print("- api_documentation.json (전체 API 정보)")
        print("- API_REFERENCE.md (Markdown 문서)")
        print("- openapi.json (OpenAPI 3.0 스펙)")
        print("- EXAMPLES.md (사용 예제)")
        
    except Exception as e:
        print(f"❌ 문서 생성 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())