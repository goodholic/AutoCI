using UnityEngine;
using UnityEngine.AI;
using System.Collections;

/// <summary>
/// 적 AI 행동 패턴
/// NavMeshAgent를 사용한 고급 AI 컨트롤러
/// </summary>
[RequireComponent(typeof(NavMeshAgent))]
public class EnemyAI : MonoBehaviour
{
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
    public AIState CurrentState { get; private set; } = AIState.Patrol;
    private int currentPatrolIndex;
    private float lastAttackTime;
    private bool isWaiting;
    private Coroutine aiCoroutine;
    
    #region Unity Lifecycle
    
    private void Awake()
    {
        agent = GetComponent<NavMeshAgent>();
        originalPosition = transform.position;
        
        // 패트롤 포인트가 없으면 자동 생성
        if (patrolPoints == null || patrolPoints.Length == 0)
        {
            GeneratePatrolPoints();
        }
    }
    
    private void Start()
    {
        aiCoroutine = StartCoroutine(AIUpdateLoop());
    }
    
    private void OnDestroy()
    {
        if (aiCoroutine != null)
        {
            StopCoroutine(aiCoroutine);
        }
    }
    
    #endregion
    
    #region AI Logic
    
    private IEnumerator AIUpdateLoop()
    {
        while (true)
        {
            switch (CurrentState)
            {
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
            }
            
            yield return new WaitForSeconds(0.1f); // AI 업데이트 간격
        }
    }
    
    private void HandlePatrol()
    {
        // 타겟 감지
        if (DetectTarget())
        {
            ChangeState(AIState.Chase);
            return;
        }
        
        // 패트롤 이동
        if (!isWaiting && (!agent.hasPath || agent.remainingDistance < 0.5f))
        {
            StartCoroutine(MoveToNextPatrolPoint());
        }
    }
    
    private void HandleChase()
    {
        if (target == null)
        {
            ChangeState(AIState.Return);
            return;
        }
        
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        
        // 타겟이 감지 범위를 벗어났는지 확인
        if (distanceToTarget > detectionRange * 1.2f) // 약간의 여유 범위
        {
            ChangeState(AIState.Return);
            return;
        }
        
        // 공격 범위 내에 있는지 확인
        if (distanceToTarget <= attackRange)
        {
            ChangeState(AIState.Attack);
        }
        else
        {
            // 타겟을 향해 이동
            agent.SetDestination(target.position);
        }
    }
    
    private void HandleAttack()
    {
        if (target == null)
        {
            ChangeState(AIState.Return);
            return;
        }
        
        float distanceToTarget = Vector3.Distance(transform.position, target.position);
        
        // 공격 범위를 벗어났는지 확인
        if (distanceToTarget > attackRange)
        {
            ChangeState(AIState.Chase);
            return;
        }
        
        // 공격 로직
        agent.ResetPath();
        transform.LookAt(target);
        
        if (Time.time - lastAttackTime > attackCooldown)
        {
            PerformAttack();
            lastAttackTime = Time.time;
        }
    }
    
    private void HandleReturn()
    {
        if (Vector3.Distance(transform.position, originalPosition) < 2f)
        {
            ChangeState(AIState.Patrol);
            return;
        }
        
        agent.SetDestination(originalPosition);
    }
    
    #endregion
    
    #region Helper Methods
    
    private bool DetectTarget()
    {
        Collider[] targets = Physics.OverlapSphere(transform.position, detectionRange, targetLayerMask);
        
        foreach (Collider col in targets)
        {
            if (col.CompareTag("Player"))
            {
                // 시야각 체크 (선택사항)
                Vector3 directionToTarget = (col.transform.position - transform.position).normalized;
                float angle = Vector3.Angle(transform.forward, directionToTarget);
                
                if (angle < 60f) // 120도 시야각
                {
                    target = col.transform;
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private void PerformAttack()
    {
        Debug.Log($"{gameObject.name}이(가) {target.name}을(를) 공격했습니다! (데미지: {attackDamage})");
        
        // 실제 공격 로직 구현
        // 예: 타겟의 Health 컴포넌트에 데미지 적용
        /*
        if (target.TryGetComponent<Health>(out var health))
        {
            health.TakeDamage(attackDamage);
        }
        */
        
        // 공격 이펙트 재생 (파티클, 사운드 등)
        // PlayAttackEffects();
    }
    
    private IEnumerator MoveToNextPatrolPoint()
    {
        isWaiting = true;
        
        if (patrolPoints.Length == 0)
        {
            yield break;
        }
        
        currentPatrolIndex = (currentPatrolIndex + 1) % patrolPoints.Length;
        agent.SetDestination(patrolPoints[currentPatrolIndex].position);
        
        yield return new WaitForSeconds(waitTime);
        isWaiting = false;
    }
    
    private void GeneratePatrolPoints()
    {
        // 자동으로 패트롤 포인트 생성
        patrolPoints = new Transform[4];
        
        for (int i = 0; i < 4; i++)
        {
            Vector3 randomDirection = Random.insideUnitSphere * patrolRadius;
            randomDirection += originalPosition;
            randomDirection.y = originalPosition.y;
            
            // NavMesh 상의 유효한 위치 찾기
            if (NavMesh.SamplePosition(randomDirection, out NavMeshHit hit, patrolRadius, NavMesh.AllAreas))
            {
                randomDirection = hit.position;
            }
            
            GameObject point = new GameObject($"PatrolPoint_{i}");
            point.transform.position = randomDirection;
            point.transform.SetParent(transform);
            
            patrolPoints[i] = point.transform;
        }
    }
    
    private void ChangeState(AIState newState)
    {
        if (CurrentState == newState) return;
        
        CurrentState = newState;
        Debug.Log($"{gameObject.name} AI 상태 변경: {newState}");
    }
    
    #endregion
    
    #region Gizmos
    
    private void OnDrawGizmosSelected()
    {
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
        {
            Gizmos.color = Color.green;
            foreach (Transform point in patrolPoints)
            {
                if (point != null)
                {
                    Gizmos.DrawWireSphere(point.position, 0.5f);
                }
            }
        }
    }
    
    #endregion
}

public enum AIState
{
    Patrol,
    Chase,
    Attack,
    Return
}