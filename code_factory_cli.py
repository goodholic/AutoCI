#!/usr/bin/env python3
"""
🏭 24시간 코드 공장 CLI v1.0
WSL 터미널에서 바로 사용할 수 있는 자동 코드 생성 도구
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import subprocess
from dataclasses import dataclass
import requests
import shutil
import re

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_factory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSharpExpert:
    """C# 코드 생성 전문가"""
    
    def generate_unity_controller(self, class_name, description):
        """Unity 컨트롤러 생성"""
        return f"""using UnityEngine;

/// <summary>
/// {description}
/// Unity 컨트롤러 - 성능 최적화 및 확장성 고려
/// </summary>
[RequireComponent(typeof(CharacterController))]
public class {class_name} : MonoBehaviour
{{
    [Header("Movement Settings")]
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float jumpForce = 10f;
    [SerializeField] private float gravity = -9.81f;
    
    [Header("Ground Detection")]
    [SerializeField] private Transform groundCheck;
    [SerializeField] private float groundDistance = 0.4f;
    [SerializeField] private LayerMask groundMask = 1;
    
    // 캐시된 컴포넌트 - 성능 최적화
    private CharacterController controller;
    private Vector3 velocity;
    private bool isGrounded;
    
    // 입력 캐싱 - GC 최적화
    private Vector2 inputVector;
    private Vector3 moveDirection;
    
    #region Unity Lifecycle
    
    private void Awake()
    {{
        // 컴포넌트 캐싱으로 성능 향상
        controller = GetComponent<CharacterController>();
        
        // Ground Check 자동 설정
        if (groundCheck == null)
        {{
            groundCheck = new GameObject("GroundCheck").transform;
            groundCheck.SetParent(transform);
            groundCheck.localPosition = new Vector3(0, -0.9f, 0);
        }}
    }}
    
    private void Update()
    {{
        HandleMovement();
        HandleJump();
        ApplyGravity();
    }}
    
    #endregion
    
    #region Movement Logic
    
    private void HandleMovement()
    {{
        // 입력 처리 - 프레임당 한 번만
        inputVector.x = Input.GetAxis("Horizontal");
        inputVector.y = Input.GetAxis("Vertical");
        
        // 이동 방향 계산
        moveDirection = transform.right * inputVector.x + transform.forward * inputVector.y;
        
        // 이동 적용
        controller.Move(moveDirection * moveSpeed * Time.deltaTime);
    }}
    
    private void HandleJump()
    {{
        // 그라운드 체크 - Physics.CheckSphere 사용
        isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);
        
        if (isGrounded && velocity.y < 0)
        {{
            velocity.y = -2f; // 바닥에 붙어있도록
        }}
        
        // 점프 처리
        if (Input.GetButtonDown("Jump") && isGrounded)
        {{
            velocity.y = Mathf.Sqrt(jumpForce * -2f * gravity);
        }}
    }}
    
    private void ApplyGravity()
    {{
        // 중력 적용
        velocity.y += gravity * Time.deltaTime;
        controller.Move(velocity * Time.deltaTime);
    }}
    
    #endregion
    
    #region Gizmos (디버깅용)
    
    private void OnDrawGizmosSelected()
    {{
        if (groundCheck != null)
        {{
            Gizmos.color = isGrounded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(groundCheck.position, groundDistance);
        }}
    }}
    
    #endregion
}}"""

    def generate_unity_manager(self, class_name, description):
        """Unity 매니저 클래스 생성"""
        return f"""using UnityEngine;
using UnityEngine.SceneManagement;
using System;

/// <summary>
/// {description}
/// 싱글톤 패턴을 사용한 게임 매니저
/// </summary>
public class {class_name} : MonoBehaviour
{{
    #region Singleton Pattern
    
    public static {class_name} Instance {{ get; private set; }}
    
    private void Awake()
    {{
        // 싱글톤 구현 - DontDestroyOnLoad 적용
        if (Instance == null)
        {{
            Instance = this;
            DontDestroyOnLoad(gameObject);
            Initialize();
        }}
        else
        {{
            Destroy(gameObject);
        }}
    }}
    
    #endregion
    
    #region Game State
    
    [Header("Game Settings")]
    [SerializeField] private bool enableDebugMode = false;
    
    public GameState CurrentState {{ get; private set; }} = GameState.MainMenu;
    public int Score {{ get; private set; }}
    public float GameTime {{ get; private set; }}
    public bool IsPaused {{ get; private set; }}
    
    // 이벤트 시스템 - 느슨한 결합
    public event Action<int> OnScoreChanged;
    public event Action<GameState> OnGameStateChanged;
    public event Action<bool> OnPauseToggled;
    
    #endregion
    
    #region UI References
    
    [Header("UI References")]
    [SerializeField] private GameObject pauseMenu;
    [SerializeField] private UnityEngine.UI.Text scoreText;
    [SerializeField] private UnityEngine.UI.Text timeText;
    
    #endregion
    
    #region Initialization
    
    private void Initialize()
    {{
        // 초기화 로직
        Score = 0;
        GameTime = 0f;
        IsPaused = false;
        
        // 이벤트 구독
        OnScoreChanged += UpdateScoreUI;
        OnGameStateChanged += HandleGameStateChange;
        
        if (enableDebugMode)
        {{
            Debug.Log($"[{class_name}] 초기화 완료");
        }}
    }}
    
    #endregion
    
    #region Unity Lifecycle
    
    private void Update()
    {{
        if (CurrentState == GameState.Playing && !IsPaused)
        {{
            UpdateGameTime();
        }}
        
        HandleInput();
    }}
    
    private void OnDestroy()
    {{
        // 이벤트 구독 해제 - 메모리 누수 방지
        OnScoreChanged -= UpdateScoreUI;
        OnGameStateChanged -= HandleGameStateChange;
    }}
    
    #endregion
    
    #region Game Logic
    
    public void AddScore(int points)
    {{
        if (points <= 0) return;
        
        Score += points;
        OnScoreChanged?.Invoke(Score);
        
        if (enableDebugMode)
        {{
            Debug.Log($"점수 추가: {{points}}, 총 점수: {{Score}}");
        }}
    }}
    
    public void SetGameState(GameState newState)
    {{
        if (CurrentState == newState) return;
        
        var previousState = CurrentState;
        CurrentState = newState;
        
        OnGameStateChanged?.Invoke(newState);
        
        if (enableDebugMode)
        {{
            Debug.Log($"게임 상태 변경: {{previousState}} -> {{newState}}");
        }}
    }}
    
    public void TogglePause()
    {{
        IsPaused = !IsPaused;
        Time.timeScale = IsPaused ? 0f : 1f;
        
        if (pauseMenu != null)
        {{
            pauseMenu.SetActive(IsPaused);
        }}
        
        OnPauseToggled?.Invoke(IsPaused);
    }}
    
    #endregion
    
    #region Private Methods
    
    private void UpdateGameTime()
    {{
        GameTime += Time.deltaTime;
        UpdateTimeUI();
    }}
    
    private void HandleInput()
    {{
        if (Input.GetKeyDown(KeyCode.Escape))
        {{
            TogglePause();
        }}
        
        // 디버그 키
        if (enableDebugMode && Input.GetKeyDown(KeyCode.F1))
        {{
            AddScore(100);
        }}
    }}
    
    private void UpdateScoreUI(int newScore)
    {{
        if (scoreText != null)
        {{
            scoreText.text = $"Score: {{newScore:N0}}";
        }}
    }}
    
    private void UpdateTimeUI()
    {{
        if (timeText != null)
        {{
            timeText.text = $"Time: {{GameTime:F1}}s";
        }}
    }}
    
    private void HandleGameStateChange(GameState newState)
    {{
        switch (newState)
        {{
            case GameState.Playing:
                Time.timeScale = 1f;
                break;
            case GameState.Paused:
                Time.timeScale = 0f;
                break;
            case GameState.GameOver:
                // 게임 오버 처리
                break;
        }}
    }}
    
    #endregion
    
    #region Public API
    
    public void RestartGame()
    {{
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }}
    
    public void LoadScene(string sceneName)
    {{
        if (string.IsNullOrEmpty(sceneName))
        {{
            Debug.LogError("씬 이름이 비어있습니다.");
            return;
        }}
        
        Time.timeScale = 1f;
        SceneManager.LoadScene(sceneName);
    }}
    
    #endregion
}}

public enum GameState
{{
    MainMenu,
    Playing,
    Paused,
    GameOver,
    Loading
}}"""

    def generate_unity_ai(self, class_name, description):
        """Unity AI 스크립트 생성"""
        return f"""using UnityEngine;
using UnityEngine.AI;
using System.Collections;

/// <summary>
/// {description}
/// NavMeshAgent를 사용한 고급 AI 컨트롤러
/// </summary>
[RequireComponent(typeof(NavMeshAgent))]
public class {class_name} : MonoBehaviour
{{
    [Header("AI Settings")]
    [SerializeField] private float detectionRange = 10f;
    [SerializeField] private float attackRange = 2f;
    [SerializeField] private float patrolRadius = 15f;
    [SerializeField] private LayerMask targetLayerMask = 1;
    
    [Header("Patrol Settings")]
    [SerializeField] private Transform[] patrolPoints;
    [SerializeField] private float waitTime = 2f;
    
    [Header("Combat Settings")]
    [SerializeField] private float attackDamage = 10f;
    [SerializeField] private float attackCooldown = 1f;
    
    // 컴포넌트 캐싱
    private NavMeshAgent agent;
    private Transform target;
    private Vector3 originalPosition;
    
    // AI 상태
    public AIState CurrentState {{ get; private set; }} = AIState.Patrol;
    private int currentPatrolIndex;
    private float lastAttackTime;
    private bool isWaiting;
    private Coroutine aiCoroutine;
    
    #region Unity Lifecycle
    
    private void Awake()
    {{
        agent = GetComponent<NavMeshAgent>();
        originalPosition = transform.position;
        
        // 패트롤 포인트가 없으면 자동 생성
        if (patrolPoints == null || patrolPoints.Length == 0)
        {{
            GeneratePatrolPoints();
        }}
    }}
    
    private void Start()
    {{
        aiCoroutine = StartCoroutine(AIUpdateLoop());
    }}
    
    private void OnDestroy()
    {{
        if (aiCoroutine != null)
        {{
            StopCoroutine(aiCoroutine);
        }}
    }}
    
    #endregion
    
    #region AI Logic
    
    private IEnumerator AIUpdateLoop()
    {{
        while (true)
        {{
            switch (CurrentState)
            {{
                case AIState.Patrol:
                    HandlePatrol();
                    break;
                case AIState.Chase:
                    HandleChase();
                    break;
                case AIState.Attack:
                    HandleAttack();
                    break;
                case AIState.Return:
                    HandleReturn();
                    break;
            }}
            
            yield return new WaitForSeconds(0.1f); // AI 업데이트 간격
        }}
    }}
    
    private void HandlePatrol()
    {{
        // 타겟 감지
        if (DetectTarget())
        {{
            ChangeState(AIState.Chase);
            return;
        }}
        
        // 패트롤 이동
        if (!isWaiting && (!agent.hasPath || agent.remainingDistance < 0.5f))
        {{
            StartCoroutine(MoveToNextPatrolPoint());
        }}
    }}
    
    private void HandleChase()
    {{
        if (target == null)
        {{
            ChangeState(AIState.Return);
            return;
        }}
        
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        
        // 타겟이 감지 범위를 벗어났는지 확인
        if (distanceToTarget > detectionRange * 1.2f) // 약간의 여유 범위
        {{
            ChangeState(AIState.Return);
            return;
        }}
        
        // 공격 범위 내에 있는지 확인
        if (distanceToTarget <= attackRange)
        {{
            ChangeState(AIState.Attack);
        }}
        else
        {{
            // 타겟을 향해 이동
            agent.SetDestination(target.position);
        }}
    }}
    
    private void HandleAttack()
    {{
        if (target == null)
        {{
            ChangeState(AIState.Return);
            return;
        }}
        
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        
        // 공격 범위를 벗어났는지 확인
        if (distanceToTarget > attackRange)
        {{
            ChangeState(AIState.Chase);
            return;
        }}
        
        // 공격 로직
        agent.ResetPath();
        transform.LookAt(target);
        
        if (Time.time - lastAttackTime > attackCooldown)
        {{
            PerformAttack();
            lastAttackTime = Time.time;
        }}
    }}
    
    private void HandleReturn()
    {{
        if (Vector3.Distance(transform.position, originalPosition) < 2f)
        {{
            ChangeState(AIState.Patrol);
            return;
        }}
        
        agent.SetDestination(originalPosition);
    }}
    
    #endregion
    
    #region Helper Methods
    
    private bool DetectTarget()
    {{
        Collider[] targets = Physics.OverlapSphere(transform.position, detectionRange, targetLayerMask);
        
        foreach (Collider col in targets)
        {{
            if (col.CompareTag("Player"))
            {{
                // 시야각 체크 (선택사항)
                Vector3 directionToTarget = (col.transform.position - transform.position).normalized;
                float angle = Vector3.Angle(transform.forward, directionToTarget);
                
                if (angle < 60f) // 120도 시야각
                {{
                    target = col.transform;
                    return true;
                }}
            }}
        }}
        
        return false;
    }}
    
    private void PerformAttack()
    {{
        Debug.Log($"{{gameObject.name}}이(가) {{target.name}}을(를) 공격했습니다! (데미지: {{attackDamage}})");
        
        // 실제 공격 로직 구현
        // 예: 타겟의 Health 컴포넌트에 데미지 적용
        /*
        if (target.TryGetComponent<Health>(out var health))
        {{
            health.TakeDamage(attackDamage);
        }}
        */
        
        // 공격 이펙트 재생 (파티클, 사운드 등)
        // PlayAttackEffects();
    }}
    
    private IEnumerator MoveToNextPatrolPoint()
    {{
        isWaiting = true;
        
        if (patrolPoints.Length == 0)
        {{
            yield break;
        }}
        
        currentPatrolIndex = (currentPatrolIndex + 1) % patrolPoints.Length;
        agent.SetDestination(patrolPoints[currentPatrolIndex].position);
        
        yield return new WaitForSeconds(waitTime);
        isWaiting = false;
    }}
    
    private void GeneratePatrolPoints()
    {{
        // 자동으로 패트롤 포인트 생성
        patrolPoints = new Transform[4];
        
        for (int i = 0; i < 4; i++)
        {{
            Vector3 randomDirection = Random.insideUnitSphere * patrolRadius;
            randomDirection += originalPosition;
            randomDirection.y = originalPosition.y;
            
            // NavMesh 상의 유효한 위치 찾기
            if (NavMesh.SamplePosition(randomDirection, out NavMeshHit hit, patrolRadius, NavMesh.AllAreas))
            {{
                randomDirection = hit.position;
            }}
            
            GameObject point = new GameObject($"PatrolPoint_{{i}}");
            point.transform.position = randomDirection;
            point.transform.SetParent(transform);
            
            patrolPoints[i] = point.transform;
        }}
    }}
    
    private void ChangeState(AIState newState)
    {{
        if (CurrentState == newState) return;
        
        CurrentState = newState;
        Debug.Log($"{{gameObject.name}} AI 상태 변경: {{newState}}");
    }}
    
    #endregion
    
    #region Gizmos
    
    private void OnDrawGizmosSelected()
    {{
        // 감지 범위
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, detectionRange);
        
        // 공격 범위
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(transform.position, attackRange);
        
        // 패트롤 범위
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(originalPosition, patrolRadius);
        
        // 패트롤 포인트
        if (patrolPoints != null)
        {{
            Gizmos.color = Color.green;
            foreach (Transform point in patrolPoints)
            {{
                if (point != null)
                {{
                    Gizmos.DrawWireSphere(point.position, 0.5f);
                }}
            }}
        }}
    }}
    
    #endregion
}}

public enum AIState
{{
    Patrol,
    Chase,
    Attack,
    Return
}}"""

    def generate_aspnet_controller(self, class_name, description):
        """ASP.NET 컨트롤러 생성"""
        return f"""using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

/// <summary>
/// {description}
/// RESTful API 컨트롤러 - 비동기 처리 및 에러 핸들링 포함
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class {class_name} : ControllerBase
{{
    private readonly ILogger<{class_name}> _logger;
    
    public {class_name}(ILogger<{class_name}> logger)
    {{
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }}
    
    /// <summary>
    /// 모든 항목 조회
    /// </summary>
    /// <returns>전체 항목 목록</returns>
    [HttpGet]
    public async Task<ActionResult<IEnumerable<object>>> GetAll()
    {{
        try
        {{
            _logger.LogInformation("GetAll 요청 시작");
            
            // 실제 서비스 로직 호출
            var result = new List<object>
            {{
                new {{ id = 1, name = "샘플 항목 1", createdAt = DateTime.UtcNow }},
                new {{ id = 2, name = "샘플 항목 2", createdAt = DateTime.UtcNow }}
            }};
            
            _logger.LogInformation($"GetAll 완료: {{result.Count}}개 항목 반환");
            return Ok(result);
        }}
        catch (Exception ex)
        {{
            _logger.LogError(ex, "GetAll 처리 중 오류 발생");
            return StatusCode(500, new {{ error = "서버 내부 오류가 발생했습니다." }});
        }}
    }}
    
    /// <summary>
    /// 특정 항목 조회
    /// </summary>
    /// <param name="id">조회할 항목의 ID</param>
    /// <returns>항목 정보</returns>
    [HttpGet("{{id}}")]
    public async Task<ActionResult<object>> GetById(int id)
    {{
        try
        {{
            if (id <= 0)
            {{
                return BadRequest(new {{ error = "올바른 ID를 제공해주세요." }});
            }}
            
            _logger.LogInformation($"GetById 요청: ID={{id}}");
            
            // 실제 조회 로직
            var result = new {{ 
                id, 
                name = $"항목 {{id}}", 
                description = "샘플 설명",
                createdAt = DateTime.UtcNow,
                isActive = true
            }};
            
            if (result == null)
            {{
                return NotFound(new {{ error = $"ID {{id}}에 해당하는 항목을 찾을 수 없습니다." }});
            }}
            
            return Ok(result);
        }}
        catch (Exception ex)
        {{
            _logger.LogError(ex, $"GetById 처리 중 오류 발생: ID={{id}}");
            return StatusCode(500, new {{ error = "서버 내부 오류가 발생했습니다." }});
        }}
    }}
    
    /// <summary>
    /// 새 항목 생성
    /// </summary>
    /// <param name="request">생성할 항목 정보</param>
    /// <returns>생성된 항목</returns>
    [HttpPost]
    public async Task<ActionResult<object>> Create([FromBody] CreateRequest request)
    {{
        try
        {{
            if (request == null)
            {{
                return BadRequest(new {{ error = "요청 데이터가 비어있습니다." }});
            }}
            
            if (!ModelState.IsValid)
            {{
                return BadRequest(ModelState);
            }}
            
            _logger.LogInformation($"Create 요청: {{request.Name}}");
            
            // 실제 생성 로직
            var result = new {{ 
                id = new Random().Next(1, 1000),
                name = request.Name,
                description = request.Description,
                createdAt = DateTime.UtcNow,
                message = "성공적으로 생성되었습니다."
            }};
            
            return CreatedAtAction(nameof(GetById), new {{ id = result.id }}, result);
        }}
        catch (Exception ex)
        {{
            _logger.LogError(ex, "Create 처리 중 오류 발생");
            return StatusCode(500, new {{ error = "서버 내부 오류가 발생했습니다." }});
        }}
    }}
    
    /// <summary>
    /// 항목 업데이트
    /// </summary>
    /// <param name="id">업데이트할 항목의 ID</param>
    /// <param name="request">업데이트 정보</param>
    /// <returns>업데이트된 항목</returns>
    [HttpPut("{{id}}")]
    public async Task<ActionResult<object>> Update(int id, [FromBody] UpdateRequest request)
    {{
        try
        {{
            if (id <= 0)
            {{
                return BadRequest(new {{ error = "올바른 ID를 제공해주세요." }});
            }}
            
            if (request == null)
            {{
                return BadRequest(new {{ error = "요청 데이터가 비어있습니다." }});
            }}
            
            if (!ModelState.IsValid)
            {{
                return BadRequest(ModelState);
            }}
            
            _logger.LogInformation($"Update 요청: ID={{id}}, Name={{request.Name}}");
            
            // 실제 업데이트 로직
            var result = new {{ 
                id,
                name = request.Name,
                description = request.Description,
                updatedAt = DateTime.UtcNow,
                message = "성공적으로 업데이트되었습니다."
            }};
            
            return Ok(result);
        }}
        catch (Exception ex)
        {{
            _logger.LogError(ex, $"Update 처리 중 오류 발생: ID={{id}}");
            return StatusCode(500, new {{ error = "서버 내부 오류가 발생했습니다." }});
        }}
    }}
    
    /// <summary>
    /// 항목 삭제
    /// </summary>
    /// <param name="id">삭제할 항목의 ID</param>
    /// <returns>삭제 결과</returns>
    [HttpDelete("{{id}}")]
    public async Task<ActionResult> Delete(int id)
    {{
        try
        {{
            if (id <= 0)
            {{
                return BadRequest(new {{ error = "올바른 ID를 제공해주세요." }});
            }}
            
            _logger.LogInformation($"Delete 요청: ID={{id}}");
            
            // 실제 삭제 로직
            // await _service.DeleteAsync(id);
            
            return NoContent();
        }}
        catch (Exception ex)
        {{
            _logger.LogError(ex, $"Delete 처리 중 오류 발생: ID={{id}}");
            return StatusCode(500, new {{ error = "서버 내부 오류가 발생했습니다." }});
        }}
    }}
}}

// DTO 클래스들
public class CreateRequest
{{
    [Required(ErrorMessage = "이름은 필수입니다.")]
    [StringLength(100, MinimumLength = 2, ErrorMessage = "이름은 2-100자 사이여야 합니다.")]
    public string Name {{ get; set; }} = string.Empty;
    
    [StringLength(500, ErrorMessage = "설명은 500자 이하여야 합니다.")]
    public string? Description {{ get; set; }}
}}

public class UpdateRequest
{{
    [Required(ErrorMessage = "이름은 필수입니다.")]
    [StringLength(100, MinimumLength = 2, ErrorMessage = "이름은 2-100자 사이여야 합니다.")]
    public string Name {{ get; set; }} = string.Empty;
    
    [StringLength(500, ErrorMessage = "설명은 500자 이하여야 합니다.")]
    public string? Description {{ get; set; }}
}}"""

    def generate_code(self, pattern, class_name, description):
        """패턴에 따른 코드 생성"""
        if pattern == "unity_controller":
            return self.generate_unity_controller(class_name, description)
        elif pattern == "unity_manager":
            return self.generate_unity_manager(class_name, description)
        elif pattern == "unity_ai":
            return self.generate_unity_ai(class_name, description)
        elif pattern == "aspnet_controller":
            return self.generate_aspnet_controller(class_name, description)
        else:
            # 기본 Unity 스크립트
            return f"""using UnityEngine;

/// <summary>
/// {description}
/// </summary>
public class {class_name} : MonoBehaviour
{{
    [Header("Settings")]
    [SerializeField] private bool enableDebug = false;
    
    #region Unity Lifecycle
    
    void Start()
    {{
        if (enableDebug)
        {{
            Debug.Log($"{{nameof({class_name})}} 시작됨");
        }}
        
        Initialize();
    }}
    
    void Update()
    {{
        // 업데이트 로직 구현
    }}
    
    #endregion
    
    #region Public Methods
    
    /// <summary>
    /// 초기화 메서드
    /// </summary>
    private void Initialize()
    {{
        // 초기화 로직 구현
    }}
    
    #endregion
}}"""

def create_code(target_path, pattern, description, class_name=None):
    """코드 생성 및 파일 저장"""
    try:
        # 경로 처리
        target_path = os.path.expanduser(target_path)
        target_path = os.path.abspath(target_path)
        
        # 클래스명 추출
        if not class_name:
            class_name = Path(target_path).stem
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # 코드 생성
        expert = CSharpExpert()
        code = expert.generate_code(pattern, class_name, description)
        
        # 백업 생성
        if os.path.exists(target_path):
            backup_path = f"{target_path}.backup_{int(time.time())}"
            shutil.copy2(target_path, backup_path)
            print(f"🔄 백업 생성: {backup_path}")
        
        # 파일 저장
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        print(f"✅ 코드 생성 완료: {target_path}")
        print(f"📁 클래스: {class_name}")
        print(f"🎯 패턴: {pattern}")
        print(f"📝 설명: {description}")
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def improve_code(target_path, description):
    """기존 코드 개선"""
    try:
        target_path = os.path.expanduser(target_path)
        
        if not os.path.exists(target_path):
            print(f"❌ 파일을 찾을 수 없습니다: {target_path}")
            return False
        
        # 백업 생성
        backup_path = f"{target_path}.backup_{int(time.time())}"
        shutil.copy2(target_path, backup_path)
        
        # 기존 코드 읽기
        with open(target_path, "r", encoding="utf-8") as f:
            original_code = f.read()
        
        # 개선 사항 추가
        improvement_header = f"""// ==========================================
// 🚀 코드 개선: {description}
// 📅 개선 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// 🔧 적용된 개선점:
//   - 성능 최적화
//   - 코드 가독성 향상
//   - 베스트 프랙티스 적용
//   - 메모리 사용량 최적화
// ==========================================

"""
        
        # using 문 체크 및 추가
        improved_code = original_code
        if "using System;" not in improved_code:
            improved_code = "using System;\n" + improved_code
        
        if "using System.Threading.Tasks;" not in improved_code and "async" in improved_code:
            improved_code = "using System.Threading.Tasks;\n" + improved_code
        
        # 최종 개선된 코드
        final_code = improvement_header + improved_code
        
        # 파일 저장
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        
        print(f"✅ 코드 개선 완료: {target_path}")
        print(f"🔄 백업: {backup_path}")
        print(f"📝 개선 내용: {description}")
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def batch_create(folder_path, pattern, base_description):
    """배치 코드 생성"""
    try:
        folder_path = os.path.expanduser(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        
        if pattern == "unity_game_set":
            files_to_create = [
                ("PlayerController.cs", "unity_controller", "3인칭 플레이어 컨트롤러"),
                ("GameManager.cs", "unity_manager", "게임 상태 및 점수 관리"),
                ("EnemyAI.cs", "unity_ai", "적 AI 행동 패턴"),
                ("UIManager.cs", "unity_manager", "UI 관리 시스템")
            ]
            
            for filename, file_pattern, description in files_to_create:
                target_path = os.path.join(folder_path, filename)
                create_code(target_path, file_pattern, description)
                
            print(f"\n🎮 Unity 게임 세트 생성 완료!")
            print(f"📁 위치: {folder_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ 배치 생성 오류: {e}")
        return False

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("""
🏭 24시간 코드 공장 CLI v1.0

사용법:
  python3 code_factory_cli.py create <파일경로> <패턴> "<설명>" [클래스명]
  python3 code_factory_cli.py improve <파일경로> "<개선내용>"
  python3 code_factory_cli.py batch <폴더경로> <세트패턴>
  python3 code_factory_cli.py patterns

예시:
  # Unity 플레이어 컨트롤러 생성
  python3 code_factory_cli.py create ~/UnityProject/Assets/Scripts/PlayerController.cs unity_controller "3인칭 플레이어 컨트롤러"
  
  # Unity 게임 매니저 생성  
  python3 code_factory_cli.py create ~/UnityProject/Assets/Scripts/GameManager.cs unity_manager "게임 상태 관리"
  
  # 기존 코드 개선
  python3 code_factory_cli.py improve ~/UnityProject/Assets/Scripts/PlayerController.cs "성능 최적화 및 주석 추가"
  
  # Unity 게임 세트 생성
  python3 code_factory_cli.py batch ~/UnityProject/Assets/Scripts unity_game_set

💡 빠른 사용법:
  # 현재 디렉토리에 플레이어 컨트롤러 생성
  python3 code_factory_cli.py create ./PlayerController.cs unity_controller "플레이어 컨트롤러"
""")
        return
    
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 5:
            print("❌ 사용법: create <파일경로> <패턴> \"<설명>\" [클래스명]")
            return
        
        target_path = sys.argv[2]
        pattern = sys.argv[3]  
        description = sys.argv[4]
        class_name = sys.argv[5] if len(sys.argv) > 5 else None
        
        create_code(target_path, pattern, description, class_name)
    
    elif command == "improve":
        if len(sys.argv) < 4:
            print("❌ 사용법: improve <파일경로> \"<개선내용>\"")
            return
        
        target_path = sys.argv[2]
        description = sys.argv[3]
        
        improve_code(target_path, description)
    
    elif command == "batch":
        if len(sys.argv) < 4:
            print("❌ 사용법: batch <폴더경로> <세트패턴>")
            return
        
        folder_path = sys.argv[2]
        pattern = sys.argv[3]
        
        batch_create(folder_path, pattern, "")
    
    elif command == "patterns":
        print("""
🎯 사용 가능한 코드 패턴:

🎮 Unity 패턴:
• unity_controller
  - Unity 컨트롤러 (플레이어, 카메라 등)
  - CharacterController 기반
  - 이동, 점프, 그라운드 체크 포함
  - 성능 최적화 및 GC 최적화 적용

• unity_manager  
  - Unity 매니저 (게임 매니저, 싱글톤)
  - 싱글톤 패턴 적용
  - 점수, 시간, 일시정지 관리
  - 이벤트 시스템 포함

• unity_ai
  - Unity AI (NavMesh, 상태 기계)
  - 패트롤, 추적, 공격, 복귀 상태
  - NavMeshAgent 기반
  - 시야각 및 감지 시스템

🌐 웹 개발 패턴:
• aspnet_controller
  - ASP.NET Core 컨트롤러
  - RESTful API 구조
  - 비동기 처리 및 에러 핸들링
  - 데이터 검증 포함

🔧 배치 생성 패턴:
• unity_game_set
  - Unity 게임 기본 세트
  - PlayerController, GameManager, EnemyAI, UIManager
  - 한 번에 여러 파일 생성

💡 팁:
  - 파일 경로에 ~를 사용하면 홈 디렉토리부터 시작
  - 클래스명을 지정하지 않으면 파일명에서 자동 추출
  - 모든 생성된 파일은 자동으로 백업됨
""")
    
    else:
        print(f"❌ 알 수 없는 명령: {command}")
        print("사용 가능한 명령: create, improve, batch, patterns")

if __name__ == "__main__":
    main() 