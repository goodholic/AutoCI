using UnityEngine;

/// <summary>
/// 3인칭 플레이어 컨트롤러
/// Unity 컨트롤러 - 성능 최적화 및 확장성 고려
/// </summary>
[RequireComponent(typeof(CharacterController))]
public class PlayerController : MonoBehaviour
{
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
    {
        // 컴포넌트 캐싱으로 성능 향상
        controller = GetComponent<CharacterController>();
        
        // Ground Check 자동 설정
        if (groundCheck == null)
        {
            groundCheck = new GameObject("GroundCheck").transform;
            groundCheck.SetParent(transform);
            groundCheck.localPosition = new Vector3(0, -0.9f, 0);
        }
    }
    
    private void Update()
    {
        HandleMovement();
        HandleJump();
        ApplyGravity();
    }
    
    #endregion
    
    #region Movement Logic
    
    private void HandleMovement()
    {
        // 입력 처리 - 프레임당 한 번만
        inputVector.x = Input.GetAxis("Horizontal");
        inputVector.y = Input.GetAxis("Vertical");
        
        // 이동 방향 계산
        moveDirection = transform.right * inputVector.x + transform.forward * inputVector.y;
        
        // 이동 적용
        controller.Move(moveDirection * moveSpeed * Time.deltaTime);
    }
    
    private void HandleJump()
    {
        // 그라운드 체크 - Physics.CheckSphere 사용
        isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);
        
        if (isGrounded && velocity.y < 0)
        {
            velocity.y = -2f; // 바닥에 붙어있도록
        }
        
        // 점프 처리
        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            velocity.y = Mathf.Sqrt(jumpForce * -2f * gravity);
        }
    }
    
    private void ApplyGravity()
    {
        // 중력 적용
        velocity.y += gravity * Time.deltaTime;
        controller.Move(velocity * Time.deltaTime);
    }
    
    #endregion
    
    #region Gizmos (디버깅용)
    
    private void OnDrawGizmosSelected()
    {
        if (groundCheck != null)
        {
            Gizmos.color = isGrounded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(groundCheck.position, groundDistance);
        }
    }
    
    #endregion
}