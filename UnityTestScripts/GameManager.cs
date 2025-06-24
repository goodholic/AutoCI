using UnityEngine;
using UnityEngine.SceneManagement;
using System;

/// <summary>
/// 게임 상태 및 점수 관리
/// 싱글톤 패턴을 사용한 게임 매니저
/// </summary>
public class GameManager : MonoBehaviour
{
    #region Singleton Pattern
    
    public static GameManager Instance { get; private set; }
    
    private void Awake()
    {
        // 싱글톤 구현 - DontDestroyOnLoad 적용
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
            Initialize();
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    #endregion
    
    #region Game State
    
    [Header("Game Settings")]
    [SerializeField] private bool enableDebugMode = false;
    
    public GameState CurrentState { get; private set; } = GameState.MainMenu;
    public int Score { get; private set; }
    public float GameTime { get; private set; }
    public bool IsPaused { get; private set; }
    
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
    {
        // 초기화 로직
        Score = 0;
        GameTime = 0f;
        IsPaused = false;
        
        // 이벤트 구독
        OnScoreChanged += UpdateScoreUI;
        OnGameStateChanged += HandleGameStateChange;
        
        if (enableDebugMode)
        {
            Debug.Log($"[GameManager] 초기화 완료");
        }
    }
    
    #endregion
    
    #region Unity Lifecycle
    
    private void Update()
    {
        if (CurrentState == GameState.Playing && !IsPaused)
        {
            UpdateGameTime();
        }
        
        HandleInput();
    }
    
    private void OnDestroy()
    {
        // 이벤트 구독 해제 - 메모리 누수 방지
        OnScoreChanged -= UpdateScoreUI;
        OnGameStateChanged -= HandleGameStateChange;
    }
    
    #endregion
    
    #region Game Logic
    
    public void AddScore(int points)
    {
        if (points <= 0) return;
        
        Score += points;
        OnScoreChanged?.Invoke(Score);
        
        if (enableDebugMode)
        {
            Debug.Log($"점수 추가: {points}, 총 점수: {Score}");
        }
    }
    
    public void SetGameState(GameState newState)
    {
        if (CurrentState == newState) return;
        
        var previousState = CurrentState;
        CurrentState = newState;
        
        OnGameStateChanged?.Invoke(newState);
        
        if (enableDebugMode)
        {
            Debug.Log($"게임 상태 변경: {previousState} -> {newState}");
        }
    }
    
    public void TogglePause()
    {
        IsPaused = !IsPaused;
        Time.timeScale = IsPaused ? 0f : 1f;
        
        if (pauseMenu != null)
        {
            pauseMenu.SetActive(IsPaused);
        }
        
        OnPauseToggled?.Invoke(IsPaused);
    }
    
    #endregion
    
    #region Private Methods
    
    private void UpdateGameTime()
    {
        GameTime += Time.deltaTime;
        UpdateTimeUI();
    }
    
    private void HandleInput()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            TogglePause();
        }
        
        // 디버그 키
        if (enableDebugMode && Input.GetKeyDown(KeyCode.F1))
        {
            AddScore(100);
        }
    }
    
    private void UpdateScoreUI(int newScore)
    {
        if (scoreText != null)
        {
            scoreText.text = $"Score: {newScore:N0}";
        }
    }
    
    private void UpdateTimeUI()
    {
        if (timeText != null)
        {
            timeText.text = $"Time: {GameTime:F1}s";
        }
    }
    
    private void HandleGameStateChange(GameState newState)
    {
        switch (newState)
        {
            case GameState.Playing:
                Time.timeScale = 1f;
                break;
            case GameState.Paused:
                Time.timeScale = 0f;
                break;
            case GameState.GameOver:
                // 게임 오버 처리
                break;
        }
    }
    
    #endregion
    
    #region Public API
    
    public void RestartGame()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }
    
    public void LoadScene(string sceneName)
    {
        if (string.IsNullOrEmpty(sceneName))
        {
            Debug.LogError("씬 이름이 비어있습니다.");
            return;
        }
        
        Time.timeScale = 1f;
        SceneManager.LoadScene(sceneName);
    }
    
    #endregion
}

public enum GameState
{
    MainMenu,
    Playing,
    Paused,
    GameOver,
    Loading
}