#!/usr/bin/env python3
"""
Godot Engine AI-Driven Improvement System
AI를 활용한 Godot 엔진 개선 시스템
- 24시간 학습 데이터 기반 엔진 개선
- C#, Korean, Networking, Nakama 통합 최적화
- 자동 패치 생성 및 적용
"""

import os
import sys
import json
import asyncio
import logging
import shutil
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import difflib

class GodotEngineImprover:
    """Godot 엔진 AI 기반 개선 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger("GodotEngineImprover")
        self.project_root = Path(__file__).parent.parent
        
        # 경로 설정
        self.user_learning_data_dir = self.project_root / "user_learning_data"
        self.continuous_learning_dir = self.project_root / "continuous_learning_demo"
        self.godot_source_dir = self.project_root / "godot_modified" / "godot-source"
        self.patches_dir = self.project_root / "godot_ai_patches"
        self.build_dir = self.project_root / "godot_ai_build"
        
        # 개선 카테고리
        self.improvement_categories = {
            "csharp_binding": {
                "priority": 10,
                "description": "C# 바인딩 최적화",
                "source_paths": ["modules/mono", "modules/mono/glue"]
            },
            "korean_support": {
                "priority": 9,
                "description": "한국어 지원 개선",
                "source_paths": ["core/string", "editor/translations", "scene/gui"]
            },
            "networking": {
                "priority": 8,
                "description": "네트워킹 성능 최적화",
                "source_paths": ["core/io", "modules/websocket", "modules/multiplayer"]
            },
            "nakama_integration": {
                "priority": 7,
                "description": "Nakama 통합 개선",
                "source_paths": ["modules/multiplayer", "core/io/net_socket.cpp"]
            },
            "ai_control": {
                "priority": 10,
                "description": "AI 제어 API 추가",
                "source_paths": ["core/object", "scene/main", "editor"]
            }
        }
        
        # 학습 데이터 매핑
        self.learning_topics_mapping = {
            "async_await": ["csharp_binding", "ai_control"],
            "Task": ["csharp_binding", "networking"],
            "병렬_처리": ["csharp_binding", "networking"],
            "Thread_Safety": ["csharp_binding", "networking"],
            "CancellationToken": ["csharp_binding"],
            "Godot_Node": ["ai_control"],
            "Signal_시스템": ["ai_control", "csharp_binding"],
            "네트워킹": ["networking", "nakama_integration"],
            "최적화": ["csharp_binding", "networking", "ai_control"],
            "한글": ["korean_support"],
            "문자열": ["korean_support", "csharp_binding"]
        }
        
        # 초기화
        self._ensure_directories()
        
    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        for dir_path in [self.patches_dir, self.build_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    async def load_learning_data(self) -> Dict[str, Any]:
        """24시간 학습 데이터 로드"""
        self.logger.info("학습 데이터 로드 중...")
        learning_data = {
            "topics": {},
            "exercises": {},
            "summaries": {},
            "improvements": []
        }
        
        try:
            # 사용자 학습 데이터 로드
            for session_dir in self.user_learning_data_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith("202"):
                    for json_file in session_dir.glob("*.json"):
                        topic = json_file.stem
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            learning_data["topics"][topic] = data
                            
                            # 개선 사항 추출
                            if "mastery_score" in data and data["mastery_score"] >= 80:
                                self._extract_improvements(topic, data, learning_data["improvements"])
            
            # 연습 문제 데이터 로드
            exercises_dir = self.user_learning_data_dir / "exercises"
            if exercises_dir.exists():
                for exercise_file in exercises_dir.glob("*.md"):
                    with open(exercise_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        learning_data["exercises"][exercise_file.stem] = content
            
            # 요약 데이터 로드
            for summary_file in self.user_learning_data_dir.glob("learning_summary_*.json"):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    learning_data["summaries"][summary_file.stem] = summary_data
                    
            # 연속 학습 데이터 로드
            if self.continuous_learning_dir.exists():
                for session_dir in self.continuous_learning_dir.iterdir():
                    if session_dir.is_dir():
                        session_summary = session_dir / "session_summary.json"
                        if session_summary.exists():
                            with open(session_summary, 'r', encoding='utf-8') as f:
                                continuous_data = json.load(f)
                                self._merge_continuous_learning(continuous_data, learning_data)
                        
            self.logger.info(f"총 {len(learning_data['topics'])}개 주제 로드 완료")
            return learning_data
            
        except Exception as e:
            self.logger.error(f"학습 데이터 로드 실패: {e}")
            return learning_data
    
    def _extract_improvements(self, topic: str, data: Dict[str, Any], improvements: List[Dict]):
        """학습 데이터에서 개선 사항 추출"""
        if topic in self.learning_topics_mapping:
            for category in self.learning_topics_mapping[topic]:
                improvements.append({
                    "topic": topic,
                    "category": category,
                    "priority": self.improvement_categories[category]["priority"],
                    "mastery_score": data.get("mastery_score", 0),
                    "notes": data.get("notes", "")
                })
    
    def _merge_continuous_learning(self, continuous_data: Dict, learning_data: Dict):
        """연속 학습 데이터 병합"""
        if "generated_qa" in continuous_data:
            for qa in continuous_data["generated_qa"]:
                if "topic" in qa:
                    topic = qa["topic"]
                    if topic not in learning_data["topics"]:
                        learning_data["topics"][topic] = qa
                    learning_data["improvements"].append({
                        "topic": topic,
                        "category": "ai_control",
                        "priority": 8,
                        "mastery_score": qa.get("difficulty", 50) * 2,
                        "notes": qa.get("answer", "")
                    })
    
    async def analyze_engine_source(self) -> Dict[str, Any]:
        """Godot 엔진 소스 코드 분석"""
        self.logger.info("Godot 엔진 소스 코드 분석 중...")
        analysis_result = {
            "files_analyzed": 0,
            "improvement_opportunities": [],
            "code_metrics": {},
            "category_analysis": {}
        }
        
        if not self.godot_source_dir.exists():
            self.logger.error(f"Godot 소스 디렉토리를 찾을 수 없습니다: {self.godot_source_dir}")
            return analysis_result
            
        try:
            for category, info in self.improvement_categories.items():
                category_opportunities = []
                
                for source_path in info["source_paths"]:
                    full_path = self.godot_source_dir / source_path
                    if full_path.exists():
                        if full_path.is_dir():
                            opportunities = await self._analyze_directory(full_path, category)
                        else:
                            opportunities = await self._analyze_file(full_path, category)
                        category_opportunities.extend(opportunities)
                        
                analysis_result["category_analysis"][category] = {
                    "opportunities": len(category_opportunities),
                    "files": category_opportunities
                }
                analysis_result["improvement_opportunities"].extend(category_opportunities)
                
            analysis_result["files_analyzed"] = len(analysis_result["improvement_opportunities"])
            self.logger.info(f"총 {analysis_result['files_analyzed']}개 파일 분석 완료")
            
        except Exception as e:
            self.logger.error(f"소스 코드 분석 실패: {e}")
            
        return analysis_result
    
    async def _analyze_directory(self, directory: Path, category: str) -> List[Dict]:
        """디렉토리 내 파일 분석"""
        opportunities = []
        
        for file_path in directory.rglob("*.cpp"):
            opps = await self._analyze_file(file_path, category)
            opportunities.extend(opps)
            
        for file_path in directory.rglob("*.h"):
            opps = await self._analyze_file(file_path, category)
            opportunities.extend(opps)
            
        return opportunities
    
    async def _analyze_file(self, file_path: Path, category: str) -> List[Dict]:
        """개별 파일 분석"""
        opportunities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 카테고리별 분석
            if category == "csharp_binding":
                opportunities.extend(self._analyze_csharp_binding(file_path, content))
            elif category == "korean_support":
                opportunities.extend(self._analyze_korean_support(file_path, content))
            elif category == "networking":
                opportunities.extend(self._analyze_networking(file_path, content))
            elif category == "nakama_integration":
                opportunities.extend(self._analyze_nakama_integration(file_path, content))
            elif category == "ai_control":
                opportunities.extend(self._analyze_ai_control(file_path, content))
                
        except Exception as e:
            self.logger.debug(f"파일 분석 오류 {file_path}: {e}")
            
        return opportunities
    
    def _analyze_csharp_binding(self, file_path: Path, content: str) -> List[Dict]:
        """C# 바인딩 개선 기회 분석"""
        opportunities = []
        
        # async/await 패턴 지원 확인
        if "mono" in str(file_path) and "async" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "async_support",
                "description": "async/await 패턴 지원 추가 필요",
                "priority": 9
            })
            
        # Task 관련 최적화
        if "Task" in content and "optimization" not in content:
            opportunities.append({
                "file": str(file_path),
                "type": "task_optimization",
                "description": "Task 처리 최적화 가능",
                "priority": 7
            })
            
        # GC 최적화
        if "garbage" in content.lower() or "gc" in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "gc_optimization",
                "description": "가비지 컬렉션 최적화 가능",
                "priority": 8
            })
            
        return opportunities
    
    def _analyze_korean_support(self, file_path: Path, content: str) -> List[Dict]:
        """한국어 지원 개선 기회 분석"""
        opportunities = []
        
        # UTF-8 인코딩 지원
        if "string" in str(file_path).lower() and "utf8" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "utf8_support",
                "description": "UTF-8 한국어 처리 개선 필요",
                "priority": 8
            })
            
        # IME 지원
        if "input" in str(file_path).lower() and "ime" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "ime_support",
                "description": "한국어 IME 지원 개선 필요",
                "priority": 9
            })
            
        # 폰트 렌더링
        if "font" in content.lower() and "korean" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "font_rendering",
                "description": "한국어 폰트 렌더링 최적화 필요",
                "priority": 7
            })
            
        return opportunities
    
    def _analyze_networking(self, file_path: Path, content: str) -> List[Dict]:
        """네트워킹 성능 개선 기회 분석"""
        opportunities = []
        
        # 비동기 네트워킹
        if "socket" in content.lower() and "async" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "async_networking",
                "description": "비동기 네트워킹 지원 추가",
                "priority": 8
            })
            
        # 버퍼 최적화
        if "buffer" in content.lower() and "pool" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "buffer_pooling",
                "description": "버퍼 풀링 최적화 가능",
                "priority": 7
            })
            
        # 프로토콜 최적화
        if "protocol" in content.lower() or "packet" in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "protocol_optimization",
                "description": "네트워크 프로토콜 최적화 가능",
                "priority": 6
            })
            
        return opportunities
    
    def _analyze_nakama_integration(self, file_path: Path, content: str) -> List[Dict]:
        """Nakama 통합 개선 기회 분석"""
        opportunities = []
        
        # Nakama 프로토콜 지원
        if "multiplayer" in str(file_path).lower():
            opportunities.append({
                "file": str(file_path),
                "type": "nakama_protocol",
                "description": "Nakama 프로토콜 네이티브 지원 추가",
                "priority": 8
            })
            
        # 실시간 멀티플레이어
        if "real" in content.lower() and "time" in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "realtime_optimization",
                "description": "실시간 멀티플레이어 최적화",
                "priority": 7
            })
            
        return opportunities
    
    def _analyze_ai_control(self, file_path: Path, content: str) -> List[Dict]:
        """AI 제어 API 개선 기회 분석"""
        opportunities = []
        
        # AI 제어 인터페이스
        if "object" in str(file_path).lower() and "ai" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "ai_interface",
                "description": "AI 제어 인터페이스 추가 필요",
                "priority": 9
            })
            
        # 자동화 API
        if "editor" in str(file_path).lower() and "automation" not in content.lower():
            opportunities.append({
                "file": str(file_path),
                "type": "automation_api",
                "description": "에디터 자동화 API 확장 필요",
                "priority": 8
            })
            
        # 머신러닝 통합
        if "node" in str(file_path).lower():
            opportunities.append({
                "file": str(file_path),
                "type": "ml_integration",
                "description": "머신러닝 노드 타입 추가 가능",
                "priority": 7
            })
            
        return opportunities
    
    async def generate_improvements(self, learning_data: Dict[str, Any], 
                                  analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI 기반 개선 사항 생성"""
        self.logger.info("AI 기반 개선 사항 생성 중...")
        improvements = []
        
        # 학습 데이터 기반 개선 사항
        learned_improvements = self._generate_from_learning_data(learning_data)
        improvements.extend(learned_improvements)
        
        # 분석 결과 기반 개선 사항
        analyzed_improvements = self._generate_from_analysis(analysis_result)
        improvements.extend(analyzed_improvements)
        
        # 우선순위 정렬
        improvements.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        # 상위 개선 사항 선택
        top_improvements = improvements[:10]  # 상위 10개 개선 사항
        
        self.logger.info(f"총 {len(top_improvements)}개 개선 사항 생성")
        return top_improvements
    
    def _generate_from_learning_data(self, learning_data: Dict[str, Any]) -> List[Dict]:
        """학습 데이터 기반 개선 사항 생성"""
        improvements = []
        
        # 높은 숙련도 주제 기반 개선
        for topic, data in learning_data["topics"].items():
            if isinstance(data, dict) and data.get("mastery_score", 0) >= 85:
                if topic in self.learning_topics_mapping:
                    for category in self.learning_topics_mapping[topic]:
                        improvements.append({
                            "id": f"learn_{topic}_{category}",
                            "category": category,
                            "type": "learned_optimization",
                            "description": f"{topic} 학습 기반 {self.improvement_categories[category]['description']}",
                            "priority": self.improvement_categories[category]["priority"],
                            "confidence": data.get("mastery_score", 0) / 100.0,
                            "source": "learning_data"
                        })
        
        return improvements
    
    def _generate_from_analysis(self, analysis_result: Dict[str, Any]) -> List[Dict]:
        """분석 결과 기반 개선 사항 생성"""
        improvements = []
        
        for opportunity in analysis_result.get("improvement_opportunities", []):
            improvements.append({
                "id": f"analyze_{opportunity['type']}_{hash(opportunity['file'])}",
                "category": self._get_category_from_file(opportunity['file']),
                "type": opportunity['type'],
                "description": opportunity['description'],
                "priority": opportunity.get('priority', 5),
                "confidence": 0.8,
                "source": "code_analysis",
                "target_file": opportunity['file']
            })
            
        return improvements
    
    def _get_category_from_file(self, file_path: str) -> str:
        """파일 경로에서 카테고리 추론"""
        file_path_lower = file_path.lower()
        
        if "mono" in file_path_lower:
            return "csharp_binding"
        elif "string" in file_path_lower or "translation" in file_path_lower:
            return "korean_support"
        elif "socket" in file_path_lower or "network" in file_path_lower:
            return "networking"
        elif "multiplayer" in file_path_lower:
            return "nakama_integration"
        else:
            return "ai_control"
    
    async def create_patches(self, improvements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """개선 사항을 실제 코드 패치로 변환"""
        self.logger.info(f"{len(improvements)}개 개선 사항에 대한 패치 생성 중...")
        patches = []
        
        for improvement in improvements:
            try:
                patch = await self._create_patch_for_improvement(improvement)
                if patch:
                    patches.append(patch)
                    
            except Exception as e:
                self.logger.error(f"패치 생성 실패 {improvement['id']}: {e}")
                
        self.logger.info(f"총 {len(patches)}개 패치 생성 완료")
        return patches
    
    async def _create_patch_for_improvement(self, improvement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 개선 사항에 대한 패치 생성"""
        patch = {
            "id": improvement["id"],
            "category": improvement["category"],
            "description": improvement["description"],
            "files": []
        }
        
        # 카테고리별 패치 생성
        if improvement["category"] == "csharp_binding":
            patch["files"] = self._create_csharp_patch(improvement)
        elif improvement["category"] == "korean_support":
            patch["files"] = self._create_korean_patch(improvement)
        elif improvement["category"] == "networking":
            patch["files"] = self._create_networking_patch(improvement)
        elif improvement["category"] == "nakama_integration":
            patch["files"] = self._create_nakama_patch(improvement)
        elif improvement["category"] == "ai_control":
            patch["files"] = self._create_ai_control_patch(improvement)
            
        return patch if patch["files"] else None
    
    def _create_csharp_patch(self, improvement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """C# 바인딩 개선 패치 생성"""
        patches = []
        
        if improvement["type"] == "async_support":
            patches.append({
                "file": "modules/mono/glue/GodotSharp/GodotSharp/Core/GodotTaskScheduler.cs",
                "content": self._generate_csharp_async_support()
            })
        elif improvement["type"] == "task_optimization":
            patches.append({
                "file": "modules/mono/glue/runtime/mono_task_utils.cpp",
                "content": self._generate_task_optimization()
            })
            
        return patches
    
    def _generate_csharp_async_support(self) -> str:
        """C# async/await 지원 코드 생성"""
        return """
using System;
using System.Threading;
using System.Threading.Tasks;
using Godot;

namespace Godot
{
    public class GodotTaskScheduler : TaskScheduler
    {
        private static readonly GodotTaskScheduler _instance = new GodotTaskScheduler();
        public static GodotTaskScheduler Instance => _instance;
        
        protected override void QueueTask(Task task)
        {
            // Godot 메인 스레드에서 Task 실행
            GD.CallDeferred(() => TryExecuteTask(task));
        }
        
        protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
        {
            // 현재 스레드가 Godot 메인 스레드인 경우에만 인라인 실행
            if (OS.IsMainThread())
            {
                return TryExecuteTask(task);
            }
            return false;
        }
        
        protected override IEnumerable<Task> GetScheduledTasks()
        {
            return Enumerable.Empty<Task>();
        }
        
        public static ConfiguredTaskAwaitable RunOnGodotThread(Func<Task> asyncAction)
        {
            return Task.Factory.StartNew(asyncAction, 
                CancellationToken.None, 
                TaskCreationOptions.None, 
                Instance).Unwrap().ConfigureAwait(false);
        }
    }
}
"""
    
    def _generate_task_optimization(self) -> str:
        """Task 최적화 C++ 코드 생성"""
        return """
#include "mono_task_utils.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"

namespace GodotMonoTaskUtils {
    
    static Mutex task_pool_mutex;
    static LocalVector<MonoObject*> task_pool;
    
    void optimize_task_execution(MonoObject* p_task) {
        MutexLock lock(task_pool_mutex);
        
        // Task 풀링으로 메모리 할당 최소화
        if (task_pool.size() < 100) {
            task_pool.push_back(p_task);
        }
        
        // 병렬 실행 최적화
        if (Thread::get_hardware_concurrency() > 2) {
            // 멀티코어 환경에서 병렬 실행
            execute_task_parallel(p_task);
        } else {
            // 단일 스레드 환경에서 순차 실행
            execute_task_sequential(p_task);
        }
    }
    
    void execute_task_parallel(MonoObject* p_task) {
        // 병렬 Task 실행 로직
        // WorkerThreadPool 활용
    }
    
    void execute_task_sequential(MonoObject* p_task) {
        // 순차 Task 실행 로직
        // 메인 스레드에서 실행
    }
}
"""
    
    def _create_korean_patch(self, improvement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """한국어 지원 개선 패치 생성"""
        patches = []
        
        if improvement["type"] == "utf8_support":
            patches.append({
                "file": "core/string/ustring.cpp",
                "content": self._generate_korean_utf8_support()
            })
        elif improvement["type"] == "ime_support":
            patches.append({
                "file": "platform/windows/os_windows.cpp",
                "content": self._generate_korean_ime_support()
            })
            
        return patches
    
    def _generate_korean_utf8_support(self) -> str:
        """한국어 UTF-8 지원 코드 생성"""
        return """
// 한국어 UTF-8 처리 개선
bool String::is_valid_korean_character(char32_t c) const {
    // 한글 음절 범위: U+AC00 ~ U+D7A3
    if (c >= 0xAC00 && c <= 0xD7A3) {
        return true;
    }
    // 한글 자모 범위: U+1100 ~ U+11FF
    if (c >= 0x1100 && c <= 0x11FF) {
        return true;
    }
    // 한글 호환 자모: U+3130 ~ U+318F
    if (c >= 0x3130 && c <= 0x318F) {
        return true;
    }
    return false;
}

String String::normalize_korean() const {
    String normalized;
    for (int i = 0; i < length(); i++) {
        char32_t c = operator[](i);
        if (is_valid_korean_character(c)) {
            // NFC 정규화 적용
            normalized += normalize_hangul_character(c);
        } else {
            normalized += c;
        }
    }
    return normalized;
}

char32_t String::normalize_hangul_character(char32_t c) const {
    // 한글 조합형을 완성형으로 변환
    if (c >= 0x1100 && c <= 0x11FF) {
        // 초성, 중성, 종성 조합 로직
        return compose_hangul_syllable(c);
    }
    return c;
}
"""
    
    def _generate_korean_ime_support(self) -> str:
        """한국어 IME 지원 코드 생성"""
        return """
// Windows 한국어 IME 지원 개선
void OS_Windows::_process_ime_event() {
    if (!ime_active) return;
    
    HIMC himc = ImmGetContext(hWnd);
    if (!himc) return;
    
    // 한국어 IME 컴포지션 처리
    if (ImmGetCompositionStringW(himc, GCS_COMPSTR, nullptr, 0) > 0) {
        int len = ImmGetCompositionStringW(himc, GCS_COMPSTR, nullptr, 0);
        LocalVector<WCHAR> buffer;
        buffer.resize(len / sizeof(WCHAR) + 1);
        
        ImmGetCompositionStringW(himc, GCS_COMPSTR, buffer.ptr(), len);
        
        String composition_string;
        composition_string.parse_utf16((const char16_t*)buffer.ptr());
        
        // 한글 조합 중인 텍스트 처리
        DisplayServer::get_singleton()->ime_update_composition(
            composition_string,
            Point2i(ime_cursor_pos_x, ime_cursor_pos_y)
        );
    }
    
    // 한국어 입력 완료 처리
    if (ImmGetCompositionStringW(himc, GCS_RESULTSTR, nullptr, 0) > 0) {
        int len = ImmGetCompositionStringW(himc, GCS_RESULTSTR, nullptr, 0);
        LocalVector<WCHAR> buffer;
        buffer.resize(len / sizeof(WCHAR) + 1);
        
        ImmGetCompositionStringW(himc, GCS_RESULTSTR, buffer.ptr(), len);
        
        String result_string;
        result_string.parse_utf16((const char16_t*)buffer.ptr());
        
        // 완성된 한글 텍스트 입력
        DisplayServer::get_singleton()->ime_commit_text(result_string);
    }
    
    ImmReleaseContext(hWnd, himc);
}
"""
    
    def _create_networking_patch(self, improvement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """네트워킹 성능 개선 패치 생성"""
        patches = []
        
        if improvement["type"] == "async_networking":
            patches.append({
                "file": "core/io/stream_peer_tcp.cpp",
                "content": self._generate_async_networking()
            })
        elif improvement["type"] == "buffer_pooling":
            patches.append({
                "file": "core/io/packet_peer.cpp",
                "content": self._generate_buffer_pooling()
            })
            
        return patches
    
    def _generate_async_networking(self) -> str:
        """비동기 네트워킹 코드 생성"""
        return """
// 비동기 네트워킹 지원
Error StreamPeerTCP::connect_to_host_async(const IPAddress &p_host, int p_port) {
    ERR_FAIL_COND_V(status != STATUS_NONE, ERR_ALREADY_IN_USE);
    
    peer = Ref<NetSocket>(NetSocket::create());
    ERR_FAIL_COND_V(peer.is_null(), ERR_CANT_CREATE);
    
    // 비동기 연결 설정
    peer->set_blocking(false);
    
    Error err = peer->connect_to_host(p_host, p_port);
    if (err == ERR_BUSY) {
        // 비동기 연결 진행 중
        status = STATUS_CONNECTING;
        connection_thread.start(_async_connect_thread, this);
        return OK;
    }
    
    return err;
}

void StreamPeerTCP::_async_connect_thread(void *p_userdata) {
    StreamPeerTCP *tcp = (StreamPeerTCP *)p_userdata;
    
    while (tcp->status == STATUS_CONNECTING) {
        NetSocket::PollType poll_type;
        Error err = tcp->peer->poll(poll_type, 100000); // 100ms timeout
        
        if (err == OK && poll_type == NetSocket::POLL_TYPE_OUT) {
            // 연결 성공
            tcp->status = STATUS_CONNECTED;
            tcp->_on_async_connect_complete(OK);
            break;
        } else if (err != OK && err != ERR_BUSY) {
            // 연결 실패
            tcp->status = STATUS_ERROR;
            tcp->_on_async_connect_complete(err);
            break;
        }
        
        // CPU 사용률 감소를 위한 짧은 대기
        OS::get_singleton()->delay_usec(1000); // 1ms
    }
}

void StreamPeerTCP::_on_async_connect_complete(Error p_error) {
    // 연결 완료 시그널 emit
    call_deferred(SNAME("emit_signal"), "async_connection_completed", p_error);
}
"""
    
    def _generate_buffer_pooling(self) -> str:
        """버퍼 풀링 최적화 코드 생성"""
        return """
// 버퍼 풀링 최적화
class BufferPool {
private:
    static constexpr int POOL_SIZE = 64;
    static constexpr int BUFFER_SIZES[] = {256, 1024, 4096, 16384, 65536};
    
    struct BufferQueue {
        List<Vector<uint8_t>> available;
        Mutex mutex;
        int buffer_size;
        int max_count;
    };
    
    static BufferQueue buffer_pools[5];
    
public:
    static Vector<uint8_t> acquire(int p_size) {
        // 적절한 크기의 버퍼 풀 선택
        int pool_index = -1;
        for (int i = 0; i < 5; i++) {
            if (p_size <= BUFFER_SIZES[i]) {
                pool_index = i;
                break;
            }
        }
        
        if (pool_index == -1) {
            // 풀 크기보다 큰 경우 직접 할당
            Vector<uint8_t> buffer;
            buffer.resize(p_size);
            return buffer;
        }
        
        BufferQueue &pool = buffer_pools[pool_index];
        MutexLock lock(pool.mutex);
        
        if (pool.available.is_empty()) {
            // 풀이 비어있으면 새로 할당
            Vector<uint8_t> buffer;
            buffer.resize(BUFFER_SIZES[pool_index]);
            return buffer;
        } else {
            // 풀에서 버퍼 가져오기
            Vector<uint8_t> buffer = pool.available.front()->get();
            pool.available.pop_front();
            return buffer;
        }
    }
    
    static void release(Vector<uint8_t> &p_buffer) {
        int size = p_buffer.size();
        int pool_index = -1;
        
        for (int i = 0; i < 5; i++) {
            if (size == BUFFER_SIZES[i]) {
                pool_index = i;
                break;
            }
        }
        
        if (pool_index == -1) {
            // 풀 크기가 아닌 경우 그냥 해제
            return;
        }
        
        BufferQueue &pool = buffer_pools[pool_index];
        MutexLock lock(pool.mutex);
        
        if (pool.available.size() < POOL_SIZE) {
            // 풀에 반환
            pool.available.push_back(p_buffer);
        }
        // 풀이 가득 찬 경우 그냥 해제
    }
};

// PacketPeer에서 버퍼 풀 사용
Error PacketPeer::get_packet(const uint8_t **r_buffer, int &r_len) {
    // 버퍼 풀에서 버퍼 획득
    last_packet_buffer = BufferPool::acquire(incoming_packet_size);
    
    // 패킷 데이터 읽기
    Error err = _get_packet(last_packet_buffer.ptrw(), incoming_packet_size);
    
    if (err == OK) {
        *r_buffer = last_packet_buffer.ptr();
        r_len = incoming_packet_size;
    }
    
    return err;
}

PacketPeer::~PacketPeer() {
    // 버퍼 풀에 반환
    if (last_packet_buffer.size() > 0) {
        BufferPool::release(last_packet_buffer);
    }
}
"""
    
    def _create_nakama_patch(self, improvement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Nakama 통합 개선 패치 생성"""
        patches = []
        
        if improvement["type"] == "nakama_protocol":
            patches.append({
                "file": "modules/multiplayer/nakama_multiplayer_peer.cpp",
                "content": self._generate_nakama_protocol()
            })
            
        return patches
    
    def _generate_nakama_protocol(self) -> str:
        """Nakama 프로토콜 지원 코드 생성"""
        return """
// Nakama 멀티플레이어 피어 구현
#include "nakama_multiplayer_peer.h"
#include "core/io/marshalls.h"

void NakamaMultiplayerPeer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("create_client", "url", "port", "server_key"), &NakamaMultiplayerPeer::create_client);
    ClassDB::bind_method(D_METHOD("authenticate_device", "device_id"), &NakamaMultiplayerPeer::authenticate_device);
    ClassDB::bind_method(D_METHOD("join_match", "match_id"), &NakamaMultiplayerPeer::join_match);
    ClassDB::bind_method(D_METHOD("create_match"), &NakamaMultiplayerPeer::create_match);
    
    ADD_SIGNAL(MethodInfo("match_joined", PropertyInfo(Variant::STRING, "match_id")));
    ADD_SIGNAL(MethodInfo("match_presence", PropertyInfo(Variant::DICTIONARY, "presences")));
}

Error NakamaMultiplayerPeer::create_client(const String &p_url, int p_port, const String &p_server_key) {
    ERR_FAIL_COND_V(connection_status != CONNECTION_DISCONNECTED, ERR_ALREADY_IN_USE);
    
    // WebSocket 연결 생성
    ws_peer = Ref<WebSocketPeer>(WebSocketPeer::create());
    ERR_FAIL_COND_V(ws_peer.is_null(), ERR_CANT_CREATE);
    
    // Nakama 서버 연결
    String ws_url = "ws://" + p_url + ":" + itos(p_port) + "/ws";
    Vector<String> protocols;
    protocols.push_back("nakama");
    
    Error err = ws_peer->connect_to_url(ws_url, protocols);
    if (err != OK) {
        return err;
    }
    
    server_key = p_server_key;
    connection_status = CONNECTION_CONNECTING;
    
    return OK;
}

Error NakamaMultiplayerPeer::authenticate_device(const String &p_device_id) {
    ERR_FAIL_COND_V(connection_status != CONNECTION_CONNECTED, ERR_UNAVAILABLE);
    
    // Nakama 인증 메시지 생성
    Dictionary auth_msg;
    auth_msg["cid"] = generate_cid();
    auth_msg["auth"] = Dictionary();
    auth_msg["auth"]["device"] = Dictionary();
    auth_msg["auth"]["device"]["id"] = p_device_id;
    
    return _send_nakama_message(auth_msg);
}

Error NakamaMultiplayerPeer::join_match(const String &p_match_id) {
    ERR_FAIL_COND_V(!authenticated, ERR_UNAUTHORIZED);
    
    Dictionary join_msg;
    join_msg["cid"] = generate_cid();
    join_msg["match_join"] = Dictionary();
    join_msg["match_join"]["match_id"] = p_match_id;
    
    current_match_id = p_match_id;
    
    return _send_nakama_message(join_msg);
}

Error NakamaMultiplayerPeer::_send_nakama_message(const Dictionary &p_message) {
    String json = JSON::stringify(p_message);
    CharString utf8 = json.utf8();
    
    return ws_peer->send((const uint8_t *)utf8.ptr(), utf8.length(), WebSocketPeer::WRITE_MODE_TEXT);
}

void NakamaMultiplayerPeer::_process_nakama_message(const Dictionary &p_message) {
    // 메시지 타입별 처리
    if (p_message.has("auth")) {
        // 인증 응답 처리
        Dictionary auth = p_message["auth"];
        if (auth.has("token")) {
            session_token = auth["token"];
            authenticated = true;
            user_id = auth["user_id"];
        }
    } else if (p_message.has("match")) {
        // 매치 조인 응답
        Dictionary match = p_message["match"];
        current_match_id = match["match_id"];
        
        // 매치에 있는 유저들 처리
        if (match.has("presences")) {
            _update_match_presences(match["presences"]);
        }
        
        emit_signal(SNAME("match_joined"), current_match_id);
    } else if (p_message.has("match_data")) {
        // 매치 데이터 수신
        Dictionary match_data = p_message["match_data"];
        _handle_match_data(match_data);
    }
}

Error NakamaMultiplayerPeer::put_packet(const uint8_t *p_buffer, int p_buffer_size) {
    ERR_FAIL_COND_V(!authenticated || current_match_id.is_empty(), ERR_UNAVAILABLE);
    
    // Nakama 매치 데이터 전송
    Dictionary data_msg;
    data_msg["cid"] = generate_cid();
    data_msg["match_data_send"] = Dictionary();
    data_msg["match_data_send"]["match_id"] = current_match_id;
    data_msg["match_data_send"]["op_code"] = 1; // 게임 데이터
    
    // 바이너리 데이터를 base64로 인코딩
    String base64_data = Marshalls::raw_to_base64(p_buffer, p_buffer_size);
    data_msg["match_data_send"]["data"] = base64_data;
    
    // 타겟 피어 설정
    if (target_peer > 0) {
        Array presences;
        presences.append(get_peer_session_id(target_peer));
        data_msg["match_data_send"]["presences"] = presences;
    }
    // target_peer == 0이면 모든 피어에게 전송 (기본값)
    
    return _send_nakama_message(data_msg);
}
"""
    
    def _create_ai_control_patch(self, improvement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """AI 제어 API 개선 패치 생성"""
        patches = []
        
        if improvement["type"] == "ai_interface":
            patches.append({
                "file": "core/object/ai_controller.cpp",
                "content": self._generate_ai_interface()
            })
        elif improvement["type"] == "automation_api":
            patches.append({
                "file": "editor/editor_automation.cpp",
                "content": self._generate_automation_api()
            })
            
        return patches
    
    def _generate_ai_interface(self) -> str:
        """AI 제어 인터페이스 코드 생성"""
        return """
// AI 제어 인터페이스
#include "ai_controller.h"
#include "core/object/class_db.h"
#include "core/variant/variant.h"

void AIController::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_ai_enabled", "enabled"), &AIController::set_ai_enabled);
    ClassDB::bind_method(D_METHOD("is_ai_enabled"), &AIController::is_ai_enabled);
    
    ClassDB::bind_method(D_METHOD("execute_ai_action", "action_name", "parameters"), &AIController::execute_ai_action);
    ClassDB::bind_method(D_METHOD("get_ai_state"), &AIController::get_ai_state);
    ClassDB::bind_method(D_METHOD("set_ai_behavior", "behavior_tree"), &AIController::set_ai_behavior);
    
    ClassDB::bind_method(D_METHOD("train_model", "training_data"), &AIController::train_model);
    ClassDB::bind_method(D_METHOD("predict", "input_data"), &AIController::predict);
    
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ai_enabled"), "set_ai_enabled", "is_ai_enabled");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "ai_state"), "", "get_ai_state");
    
    ADD_SIGNAL(MethodInfo("ai_action_executed", PropertyInfo(Variant::STRING, "action_name")));
    ADD_SIGNAL(MethodInfo("ai_state_changed", PropertyInfo(Variant::DICTIONARY, "new_state")));
}

void AIController::set_ai_enabled(bool p_enabled) {
    if (ai_enabled == p_enabled) {
        return;
    }
    
    ai_enabled = p_enabled;
    
    if (ai_enabled) {
        _start_ai_processing();
    } else {
        _stop_ai_processing();
    }
}

Variant AIController::execute_ai_action(const String &p_action_name, const Dictionary &p_parameters) {
    ERR_FAIL_COND_V(!ai_enabled, Variant());
    
    // AI 액션 실행
    AIAction action;
    action.name = p_action_name;
    action.parameters = p_parameters;
    action.timestamp = OS::get_singleton()->get_ticks_msec();
    
    // 액션 큐에 추가
    action_queue_mutex.lock();
    action_queue.push_back(action);
    action_queue_mutex.unlock();
    
    // 액션 실행 시그널
    call_deferred(SNAME("emit_signal"), "ai_action_executed", p_action_name);
    
    return _process_ai_action(action);
}

Dictionary AIController::get_ai_state() const {
    Dictionary state;
    
    state["enabled"] = ai_enabled;
    state["current_behavior"] = current_behavior_name;
    state["action_count"] = action_history.size();
    state["last_action_time"] = last_action_time;
    
    // AI 모델 상태
    if (ai_model.is_valid()) {
        state["model_loaded"] = true;
        state["model_accuracy"] = ai_model->get_accuracy();
    } else {
        state["model_loaded"] = false;
    }
    
    return state;
}

Error AIController::train_model(const Dictionary &p_training_data) {
    ERR_FAIL_COND_V(!ai_enabled, ERR_UNAVAILABLE);
    
    // AI 모델 학습
    if (!ai_model.is_valid()) {
        ai_model = Ref<AIModel>(memnew(AIModel));
    }
    
    Array inputs = p_training_data["inputs"];
    Array outputs = p_training_data["outputs"];
    
    return ai_model->train(inputs, outputs);
}

Variant AIController::predict(const Array &p_input_data) {
    ERR_FAIL_COND_V(!ai_enabled || !ai_model.is_valid(), Variant());
    
    return ai_model->predict(p_input_data);
}

void AIController::_process_ai_queue() {
    while (ai_enabled) {
        action_queue_mutex.lock();
        if (!action_queue.is_empty()) {
            AIAction action = action_queue.front()->get();
            action_queue.pop_front();
            action_queue_mutex.unlock();
            
            // 액션 처리
            _execute_action_internal(action);
        } else {
            action_queue_mutex.unlock();
            OS::get_singleton()->delay_usec(1000); // 1ms 대기
        }
    }
}
"""
    
    def _generate_automation_api(self) -> str:
        """에디터 자동화 API 코드 생성"""
        return """
// 에디터 자동화 API
#include "editor_automation.h"
#include "editor/editor_node.h"
#include "scene/main/node.h"

void EditorAutomation::_bind_methods() {
    ClassDB::bind_method(D_METHOD("create_scene", "root_node_type"), &EditorAutomation::create_scene);
    ClassDB::bind_method(D_METHOD("add_node", "parent_path", "node_type", "node_name"), &EditorAutomation::add_node);
    ClassDB::bind_method(D_METHOD("set_node_property", "node_path", "property", "value"), &EditorAutomation::set_node_property);
    
    ClassDB::bind_method(D_METHOD("save_scene", "path"), &EditorAutomation::save_scene);
    ClassDB::bind_method(D_METHOD("load_scene", "path"), &EditorAutomation::load_scene);
    
    ClassDB::bind_method(D_METHOD("run_scene"), &EditorAutomation::run_scene);
    ClassDB::bind_method(D_METHOD("stop_scene"), &EditorAutomation::stop_scene);
    
    ClassDB::bind_method(D_METHOD("build_project", "platform", "export_path"), &EditorAutomation::build_project);
    
    ClassDB::bind_method(D_METHOD("execute_script", "script_content"), &EditorAutomation::execute_script);
    ClassDB::bind_method(D_METHOD("batch_process", "commands"), &EditorAutomation::batch_process);
}

Error EditorAutomation::create_scene(const String &p_root_node_type) {
    EditorNode *editor = EditorNode::get_singleton();
    ERR_FAIL_COND_V(!editor, ERR_UNAVAILABLE);
    
    // 새 씬 생성
    Node *root = Object::cast_to<Node>(ClassDB::instantiate(p_root_node_type));
    ERR_FAIL_COND_V(!root, ERR_CANT_CREATE);
    
    editor->set_edited_scene(root);
    current_scene_root = root;
    
    return OK;
}

NodePath EditorAutomation::add_node(const NodePath &p_parent_path, const String &p_node_type, const String &p_node_name) {
    ERR_FAIL_COND_V(!current_scene_root, NodePath());
    
    Node *parent = nullptr;
    if (p_parent_path.is_empty()) {
        parent = current_scene_root;
    } else {
        parent = current_scene_root->get_node(p_parent_path);
    }
    ERR_FAIL_COND_V(!parent, NodePath());
    
    // 노드 생성 및 추가
    Node *new_node = Object::cast_to<Node>(ClassDB::instantiate(p_node_type));
    ERR_FAIL_COND_V(!new_node, NodePath());
    
    new_node->set_name(p_node_name);
    parent->add_child(new_node);
    
    // 에디터에서 선택
    EditorNode::get_singleton()->get_selection()->clear();
    EditorNode::get_singleton()->get_selection()->add_node(new_node);
    
    return new_node->get_path();
}

Error EditorAutomation::set_node_property(const NodePath &p_node_path, const String &p_property, const Variant &p_value) {
    ERR_FAIL_COND_V(!current_scene_root, ERR_UNAVAILABLE);
    
    Node *node = current_scene_root->get_node(p_node_path);
    ERR_FAIL_COND_V(!node, ERR_DOES_NOT_EXIST);
    
    node->set(p_property, p_value);
    
    // 속성 변경 알림
    node->notify_property_list_changed();
    
    return OK;
}

Error EditorAutomation::execute_script(const String &p_script_content) {
    // GDScript 실행
    Ref<GDScript> script;
    script.instantiate();
    
    Error err = script->reload(false);
    if (err != OK) {
        return err;
    }
    
    script->set_source_code(p_script_content);
    err = script->reload();
    if (err != OK) {
        return err;
    }
    
    // 스크립트 인스턴스 생성 및 실행
    Ref<RefCounted> instance = script->new_instance();
    if (instance.is_valid() && instance->has_method("_run")) {
        instance->call("_run");
    }
    
    return OK;
}

Error EditorAutomation::batch_process(const Array &p_commands) {
    // 배치 명령 처리
    for (int i = 0; i < p_commands.size(); i++) {
        Dictionary cmd = p_commands[i];
        String command = cmd["command"];
        
        if (command == "create_scene") {
            create_scene(cmd["root_type"]);
        } else if (command == "add_node") {
            add_node(cmd["parent"], cmd["type"], cmd["name"]);
        } else if (command == "set_property") {
            set_node_property(cmd["node"], cmd["property"], cmd["value"]);
        } else if (command == "save_scene") {
            save_scene(cmd["path"]);
        } else if (command == "wait") {
            OS::get_singleton()->delay_msec(cmd["duration"]);
        }
    }
    
    return OK;
}
"""
    
    async def apply_patches(self, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """패치를 소스 코드에 적용"""
        self.logger.info(f"{len(patches)}개 패치 적용 중...")
        
        result = {
            "applied": [],
            "failed": [],
            "backup_created": False
        }
        
        # 백업 생성
        backup_path = await self._create_source_backup()
        if backup_path:
            result["backup_created"] = True
            result["backup_path"] = str(backup_path)
        
        # 패치 적용
        for patch in patches:
            try:
                applied_files = await self._apply_patch(patch)
                result["applied"].extend(applied_files)
                
            except Exception as e:
                self.logger.error(f"패치 적용 실패 {patch['id']}: {e}")
                result["failed"].append({
                    "patch_id": patch["id"],
                    "error": str(e)
                })
        
        self.logger.info(f"패치 적용 완료: {len(result['applied'])}개 성공, {len(result['failed'])}개 실패")
        return result
    
    async def _create_source_backup(self) -> Optional[Path]:
        """소스 코드 백업 생성"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.project_root / "godot_source_backups" / timestamp
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 중요 파일만 백업 (전체 백업은 너무 큼)
            important_dirs = ["core", "modules/mono", "editor", "scene"]
            
            for dir_name in important_dirs:
                src_dir = self.godot_source_dir / dir_name
                if src_dir.exists():
                    dst_dir = backup_dir / dir_name
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                    
            self.logger.info(f"소스 백업 생성 완료: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            self.logger.error(f"백업 생성 실패: {e}")
            return None
    
    async def _apply_patch(self, patch: Dict[str, Any]) -> List[str]:
        """개별 패치 적용"""
        applied_files = []
        
        for file_info in patch.get("files", []):
            file_path = self.godot_source_dir / file_info["file"]
            
            # 디렉토리 생성
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 기존 파일이 있으면 백업
            if file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                shutil.copy2(file_path, backup_path)
            
            # 패치 내용 쓰기
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_info["content"])
                
            applied_files.append(str(file_path))
            self.logger.debug(f"패치 적용: {file_path}")
            
        # 패치 정보 저장
        patch_info_file = self.patches_dir / f"{patch['id']}.json"
        with open(patch_info_file, 'w', encoding='utf-8') as f:
            json.dump({
                "id": patch["id"],
                "category": patch["category"],
                "description": patch["description"],
                "applied_files": applied_files,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
            
        return applied_files
    
    async def build_improved_engine(self) -> Dict[str, Any]:
        """개선된 Godot 엔진 빌드"""
        self.logger.info("개선된 Godot 엔진 빌드 시작...")
        
        result = {
            "build_started": False,
            "build_completed": False,
            "output_path": None,
            "errors": []
        }
        
        try:
            # 빌드 스크립트 경로
            build_script = self.project_root / "build-godot"
            
            if not build_script.exists():
                raise FileNotFoundError(f"빌드 스크립트를 찾을 수 없습니다: {build_script}")
            
            # 빌드 명령 준비
            build_cmd = [str(build_script)]
            
            # 빌드 환경 변수 설정
            env = os.environ.copy()
            env["GODOT_BUILD_TYPE"] = "ai_improved"
            env["GODOT_ENABLE_CSHARP"] = "yes"
            env["GODOT_ENABLE_AI_APIS"] = "yes"
            
            # 빌드 실행
            self.logger.info(f"빌드 명령 실행: {' '.join(build_cmd)}")
            result["build_started"] = True
            
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.project_root)
            )
            
            # 빌드 출력 실시간 로깅
            async def log_output(stream, log_func):
                async for line in stream:
                    log_func(line.decode('utf-8', errors='ignore').strip())
            
            await asyncio.gather(
                log_output(process.stdout, self.logger.info),
                log_output(process.stderr, self.logger.warning)
            )
            
            # 빌드 완료 대기
            return_code = await process.wait()
            
            if return_code == 0:
                result["build_completed"] = True
                result["output_path"] = str(self.build_dir / "godot_ai_improved")
                self.logger.info("빌드 성공!")
            else:
                result["errors"].append(f"빌드 실패: 반환 코드 {return_code}")
                self.logger.error(f"빌드 실패: 반환 코드 {return_code}")
                
        except Exception as e:
            error_msg = f"빌드 중 오류 발생: {e}"
            result["errors"].append(error_msg)
            self.logger.error(error_msg)
            
        return result


async def main():
    """메인 실행 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    improver = GodotEngineImprover()
    
    try:
        # 1. 학습 데이터 로드
        print("1. 학습 데이터 로드 중...")
        learning_data = await improver.load_learning_data()
        
        # 2. 엔진 소스 분석
        print("2. Godot 엔진 소스 코드 분석 중...")
        analysis_result = await improver.analyze_engine_source()
        
        # 3. 개선 사항 생성
        print("3. AI 기반 개선 사항 생성 중...")
        improvements = await improver.generate_improvements(learning_data, analysis_result)
        
        # 4. 패치 생성
        print("4. 코드 패치 생성 중...")
        patches = await improver.create_patches(improvements)
        
        # 5. 패치 적용
        print("5. 패치 적용 중...")
        apply_result = await improver.apply_patches(patches)
        
        # 6. 엔진 빌드
        print("6. 개선된 Godot 엔진 빌드 중...")
        build_result = await improver.build_improved_engine()
        
        # 결과 출력
        print("\n=== Godot 엔진 개선 완료 ===")
        print(f"학습 데이터: {len(learning_data['topics'])}개 주제")
        print(f"분석된 파일: {analysis_result['files_analyzed']}개")
        print(f"생성된 개선 사항: {len(improvements)}개")
        print(f"적용된 패치: {len(apply_result['applied'])}개")
        print(f"빌드 상태: {'성공' if build_result['build_completed'] else '실패'}")
        
        if build_result['output_path']:
            print(f"출력 경로: {build_result['output_path']}")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())