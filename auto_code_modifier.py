#!/usr/bin/env python3
"""
24시간 자동 코드 수정 시스템
사용자 명령에 따라 다른 폴더의 코드를 자동으로 수정해주는 AI 코딩 공장
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import subprocess
from dataclasses import dataclass
import requests
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_code_modifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CodeModificationTask:
    """코드 수정 작업"""
    id: str
    timestamp: str
    target_folder: str
    file_pattern: str
    modification_type: str  # 'create', 'modify', 'improve', 'fix'
    description: str
    code_content: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None

class AutoCodeModifier:
    """자동 코드 수정 시스템"""
    
    def __init__(self):
        self.base_dir = Path(".")
        self.tasks_file = "code_modification_tasks.json"
        self.completed_tasks_file = "completed_tasks.json"
        self.ai_server_url = "http://localhost:8000"
        self.tasks = []
        self.completed_tasks = []
        self.load_tasks()
        
    def load_tasks(self):
        """작업 목록 로드"""
        try:
            if Path(self.tasks_file).exists():
                with open(self.tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tasks = [CodeModificationTask(**task) for task in data]
            
            if Path(self.completed_tasks_file).exists():
                with open(self.completed_tasks_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_tasks = [CodeModificationTask(**task) for task in data]
        except Exception as e:
            logger.error(f"작업 로드 실패: {e}")
    
    def save_tasks(self):
        """작업 목록 저장"""
        try:
            with open(self.tasks_file, 'w', encoding='utf-8') as f:
                json.dump([task.__dict__ for task in self.tasks], f, indent=2, ensure_ascii=False)
            
            with open(self.completed_tasks_file, 'w', encoding='utf-8') as f:
                json.dump([task.__dict__ for task in self.completed_tasks], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"작업 저장 실패: {e}")
    
    def add_task(self, target_folder: str, file_pattern: str, modification_type: str, description: str, code_content: str = None) -> str:
        """새 작업 추가"""
        task_id = f"task_{int(time.time())}_{len(self.tasks)}"
        task = CodeModificationTask(
            id=task_id,
            timestamp=datetime.now().isoformat(),
            target_folder=target_folder,
            file_pattern=file_pattern,
            modification_type=modification_type,
            description=description,
            code_content=code_content
        )
        
        self.tasks.append(task)
        self.save_tasks()
        logger.info(f"✅ 새 작업 추가: {task_id} - {description}")
        return task_id
    
    def find_files(self, target_folder: str, file_pattern: str) -> List[Path]:
        """패턴에 맞는 파일 찾기"""
        try:
            folder_path = Path(target_folder)
            if not folder_path.exists():
                logger.warning(f"폴더가 존재하지 않음: {target_folder}")
                return []
            
            if "*" in file_pattern:
                # 와일드카드 패턴
                files = list(folder_path.glob(file_pattern))
            else:
                # 특정 파일
                file_path = folder_path / file_pattern
                files = [file_path] if file_path.exists() else []
            
            return files
        except Exception as e:
            logger.error(f"파일 검색 실패: {e}")
            return []
    
    def generate_code_fallback(self, prompt: str) -> str:
        """AI 서버 없이 간단한 코드 생성"""
        if "PlayerController" in prompt or "player" in prompt.lower():
            return '''using UnityEngine;

public class PlayerController : MonoBehaviour
{
    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float jumpForce = 10f;
    
    [Header("Ground Check")]
    public Transform groundCheck;
    public float groundDistance = 0.4f;
    public LayerMask groundMask;
    
    private Vector3 velocity;
    private bool isGrounded;
    private CharacterController controller;
    
    void Start()
    {
        controller = GetComponent<CharacterController>();
    }
    
    void Update()
    {
        // Ground check
        isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);
        
        if (isGrounded && velocity.y < 0)
        {
            velocity.y = -2f;
        }
        
        // Input
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        
        Vector3 move = transform.right * x + transform.forward * z;
        controller.Move(move * moveSpeed * Time.deltaTime);
        
        // Jump
        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            velocity.y = Mathf.Sqrt(jumpForce * -2f * Physics.gravity.y);
        }
        
        // Gravity
        velocity.y += Physics.gravity.y * Time.deltaTime;
        controller.Move(velocity * Time.deltaTime);
    }
}
'''
        elif "GameManager" in prompt or "manager" in prompt.lower():
            return '''using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }
    
    [Header("Game State")]
    public bool isGamePaused = false;
    public int score = 0;
    public float gameTime = 0f;
    
    [Header("UI References")]
    public GameObject pauseMenu;
    public UnityEngine.UI.Text scoreText;
    public UnityEngine.UI.Text timeText;
    
    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    void Update()
    {
        {
            gameTime += Time.deltaTime;
            UpdateUI();
        }
        
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            TogglePause();
        }
    }
    
    public void AddScore(int points)
    {
        score += points;
    }
    
    public void TogglePause()
    {
        Time.timeScale = isGamePaused ? 0f : 1f;
        pauseMenu.SetActive(isGamePaused);
    }
    
    void UpdateUI()
    {
        if (scoreText) scoreText.text = "Score: " + score;
        if (timeText) timeText.text = "Time: " + gameTime.ToString("F1");
    }
    
    public void RestartGame()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }
}
'''
        else:
            return f'''using UnityEngine;

// Generated code for: {prompt}
public class GeneratedScript : MonoBehaviour
{{
    void Start()
    {{
        Debug.Log("Generated script started");
    }}
    
    void Update()
    {{
        // Add your update logic here
    }}
}}
'''

    def modify_file_sync(self, file_path: Path, modification_type: str, description: str, code_content: str = None) -> str:
        """파일 수정 (동기)"""
        try:
            # 백업 생성
            backup_path = file_path.with_suffix(f".backup_{int(time.time())}")
            if file_path.exists():
                shutil.copy2(file_path, backup_path)
                logger.info(f"백업 생성: {backup_path}")
            
            if modification_type == "create":
                # 새 파일 생성
                if code_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(code_content)
                    return f"새 파일 생성: {file_path}"
                else:
                    # AI에게 코드 생성 요청 (폴백)
                    generated_code = self.generate_code_fallback(description)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(generated_code)
                    return f"AI가 새 파일 생성: {file_path}"
            
            elif modification_type == "modify":
                # 기존 파일 수정
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    # 간단한 수정 (주석 추가)
                    modified_content = f"// Modified: {description}\n" + original_content
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    return f"파일 수정 완료: {file_path}"
                else:
                    return f"파일이 존재하지 않음: {file_path}"
            
            elif modification_type == "improve":
                # 코드 개선 (간단한 포맷팅)
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    # 간단한 개선 (주석 추가)
                    improved_content = f"// Improved: {description}\n" + original_content
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(improved_content)
                    return f"코드 개선 완료: {file_path}"
                else:
                    return f"파일이 존재하지 않음: {file_path}"
            
            return f"알 수 없는 수정 타입: {modification_type}"
            
        except Exception as e:
            logger.error(f"파일 수정 실패 {file_path}: {e}")
            return f"오류: {str(e)}"
    
    def process_task_sync(self, task: CodeModificationTask):
        """작업 처리 (동기)"""
        try:
            logger.info(f"🔧 작업 처리 시작: {task.id} - {task.description}")
            task.status = "in_progress"
            self.save_tasks()
            
            # 대상 파일 찾기
            files = self.find_files(task.target_folder, task.file_pattern)
            
            if not files and task.modification_type != "create":
                task.status = "failed"
                task.result = f"파일을 찾을 수 없음: {task.target_folder}/{task.file_pattern}"
                logger.warning(task.result)
                return
            
            results = []
            
            if task.modification_type == "create" and not files:
                # 새 파일 생성
                file_path = Path(task.target_folder) / task.file_pattern
                # 디렉토리 생성
                file_path.parent.mkdir(parents=True, exist_ok=True)
                result = self.modify_file_sync(file_path, task.modification_type, task.description, task.code_content)
                results.append(result)
            else:
                # 기존 파일들 수정
                for file_path in files:
                    result = self.modify_file_sync(file_path, task.modification_type, task.description, task.code_content)
                    results.append(result)
            
            task.status = "completed"
            task.result = "\n".join(results)
            
            # 완료된 작업 목록으로 이동
            self.tasks.remove(task)
            self.completed_tasks.append(task)
            self.save_tasks()
            
            logger.info(f"✅ 작업 완료: {task.id}")
            
        except Exception as e:
            task.status = "failed"
            task.result = f"작업 처리 실패: {str(e)}"
            logger.error(f"❌ 작업 실패 {task.id}: {e}")
    
    def run_continuously_sync(self):
        """24시간 연속 실행 (동기)"""
        logger.info("🤖 24시간 자동 코드 수정 시스템 시작")
        
        while True:
            try:
                # 대기 중인 작업 확인
                pending_tasks = [task for task in self.tasks if task.status == "pending"]
                
                if pending_tasks:
                    logger.info(f"📋 처리할 작업 {len(pending_tasks)}개 발견")
                    
                    # 작업 처리 (순차적으로)
                    for task in pending_tasks[:3]:  # 한 번에 최대 3개 작업
                        self.process_task_sync(task)
                        time.sleep(1)  # 작업 간 1초 대기
                else:
                    logger.info("⏸️  대기 중인 작업이 없습니다. 30초 후 다시 확인...")
                
                # 30초 대기
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("🛑 사용자에 의해 중단됨")
                break
            except Exception as e:
                logger.error(f"시스템 오류: {e}")
                time.sleep(10)  # 오류 발생 시 10초 대기
    
    def print_status(self):
        """현재 상태 출력"""
        print("\n" + "="*60)
        print("🤖 자동 코드 수정 시스템 상태")
        print("="*60)
        print(f"📋 대기 중인 작업: {len([t for t in self.tasks if t.status == 'pending'])}개")
        print(f"🔧 진행 중인 작업: {len([t for t in self.tasks if t.status == 'in_progress'])}개")
        print(f"✅ 완료된 작업: {len(self.completed_tasks)}개")
        print(f"❌ 실패한 작업: {len([t for t in self.tasks if t.status == 'failed'])}개")
        
        if self.tasks:
            print("\n📋 최근 작업:")
            for task in self.tasks[-3:]:
                print(f"  • {task.id}: {task.description} ({task.status})")
        
        print("="*60)

def main():
    """메인 함수"""
    modifier = AutoCodeModifier()
    
    print("🤖 24시간 자동 코드 수정 시스템")
    print("=" * 50)
    print("1. 연속 실행 모드")
    print("2. 작업 추가")
    print("3. 상태 확인")
    print("4. 완료된 작업 보기")
    print("5. 예시 작업 추가")
    print("=" * 50)
    
    choice = input("선택하세요 (1-5): ").strip()
    
    if choice == "1":
        print("🚀 연속 실행 모드를 시작합니다...")
        modifier.run_continuously_sync()
    
    elif choice == "2":
        print("\n📝 새 작업 추가")
        target_folder = input("대상 폴더 경로: ").strip()
        file_pattern = input("파일 패턴 (예: *.cs, PlayerController.cs): ").strip()
        modification_type = input("수정 타입 (create/modify/improve/fix): ").strip()
        description = input("작업 설명: ").strip()
        
        task_id = modifier.add_task(target_folder, file_pattern, modification_type, description)
        print(f"✅ 작업 추가됨: {task_id}")
    
    elif choice == "3":
        modifier.print_status()
    
    elif choice == "4":
        print("\n✅ 완료된 작업:")
        for task in modifier.completed_tasks[-10:]:  # 최근 10개
            print(f"  • {task.id}: {task.description}")
            print(f"    결과: {task.result}")
            print()
    
    elif choice == "5":
        print("\n🎯 예시 작업들을 추가합니다...")
        
        # Unity 스크립트 생성 예시
        modifier.add_task(
            target_folder="../Assets/Scripts",
            file_pattern="PlayerController.cs", 
            modification_type="create",
            description="Unity 플레이어 컨트롤러 스크립트 생성"
        )
        
        modifier.add_task(
            target_folder="../Assets/Scripts",
            file_pattern="GameManager.cs",
            modification_type="create", 
            description="Unity 게임 매니저 스크립트 생성"
        )
        
        # 기존 스크립트 개선 예시
        modifier.add_task(
            target_folder="../Assets/Scripts",
            file_pattern="*.cs",
            modification_type="improve",
            description="모든 스크립트에 성능 최적화 주석 추가"
        )
        

if __name__ == "__main__":
    main()
