using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using Godot;

namespace AutoCI.Modules
{
    /// <summary>
    /// 변형된 Godot 엔진 자동화 컨트롤러
    /// AI가 Godot 엔진을 직접 조작하여 게임을 개발
    /// </summary>
    public class GodotAutomationController : Node
    {
        private string projectPath;
        private Process godotProcess;
        private SceneTree sceneTree;
        private Dictionary<string, Node> gameNodes;
        private bool isEngineRunning;
        
        // 게임 개발 액션 매핑
        private readonly Dictionary<string, Action<Dictionary<string, object>>> actionHandlers;
        
        public GodotAutomationController()
        {
            gameNodes = new Dictionary<string, Node>();
            actionHandlers = new Dictionary<string, Action<Dictionary<string, object>>>
            {
                ["create_scene"] = CreateScene,
                ["add_player"] = AddPlayer,
                ["add_enemy"] = AddEnemy,
                ["add_platform"] = AddPlatform,
                ["add_collision"] = AddCollision,
                ["add_physics"] = AddPhysics,
                ["add_ui"] = AddUI,
                ["add_sound"] = AddSound,
                ["optimize_performance"] = OptimizePerformance,
                ["debug_error"] = DebugError,
                ["generate_csharp_code"] = GenerateCSharpCode,
                ["test_gameplay"] = TestGameplay
            };
        }
        
        public override void _Ready()
        {
            GD.Print("🎮 Godot 자동화 컨트롤러 초기화 완료");
            sceneTree = GetTree();
        }
        
        /// <summary>
        /// Godot 엔진 시작
        /// </summary>
        public async Task<bool> StartEngine(string projectName)
        {
            try
            {
                projectPath = Path.Combine(OS.GetUserDataDir(), "game_projects", projectName);
                
                // 프로젝트 디렉토리 생성
                if (!Directory.Exists(projectPath))
                {
                    Directory.CreateDirectory(projectPath);
                    await CreateProjectFiles(projectPath, projectName);
                }
                
                // Godot 프로세스 시작
                var godotPath = FindGodotExecutable();
                if (string.IsNullOrEmpty(godotPath))
                {
                    GD.PrintErr("❌ Godot 실행 파일을 찾을 수 없습니다.");
                    return false;
                }
                
                var startInfo = new ProcessStartInfo
                {
                    FileName = godotPath,
                    Arguments = $"--path \"{projectPath}\" --editor",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                };
                
                godotProcess = Process.Start(startInfo);
                isEngineRunning = true;
                
                GD.Print($"✅ Godot 엔진 시작됨: {projectName}");
                return true;
            }
            catch (Exception e)
            {
                GD.PrintErr($"❌ Godot 엔진 시작 실패: {e.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// AI 액션 실행
        /// </summary>
        public async Task<bool> ExecuteAction(string action, Dictionary<string, object> parameters)
        {
            if (!isEngineRunning)
            {
                GD.PrintErr("❌ Godot 엔진이 실행되지 않았습니다.");
                return false;
            }
            
            if (actionHandlers.ContainsKey(action))
            {
                try
                {
                    actionHandlers[action](parameters);
                    GD.Print($"✅ 액션 실행 완료: {action}");
                    return true;
                }
                catch (Exception e)
                {
                    GD.PrintErr($"❌ 액션 실행 실패: {action} - {e.Message}");
                    return false;
                }
            }
            
            GD.PrintErr($"❌ 알 수 없는 액션: {action}");
            return false;
        }
        
        /// <summary>
        /// 씬 생성
        /// </summary>
        private void CreateScene(Dictionary<string, object> parameters)
        {
            var sceneName = parameters.GetValueOrDefault("name", "MainScene").ToString();
            var sceneType = parameters.GetValueOrDefault("type", "2D").ToString();
            
            PackedScene newScene = new PackedScene();
            Node rootNode;
            
            if (sceneType == "3D")
            {
                rootNode = new Node3D { Name = sceneName };
            }
            else
            {
                rootNode = new Node2D { Name = sceneName };
            }
            
            // 씬 저장
            var scenePath = $"{projectPath}/scenes/{sceneName}.tscn";
            newScene.Pack(rootNode);
            ResourceSaver.Save(scenePath, newScene);
            
            gameNodes[sceneName] = rootNode;
            GD.Print($"📄 씬 생성됨: {sceneName} ({sceneType})");
        }
        
        /// <summary>
        /// 플레이어 추가
        /// </summary>
        private void AddPlayer(Dictionary<string, object> parameters)
        {
            var sceneName = parameters.GetValueOrDefault("scene", "MainScene").ToString();
            var playerType = parameters.GetValueOrDefault("type", "CharacterBody2D").ToString();
            
            if (!gameNodes.ContainsKey(sceneName))
            {
                GD.PrintErr($"❌ 씬을 찾을 수 없습니다: {sceneName}");
                return;
            }
            
            CharacterBody2D player = new CharacterBody2D { Name = "Player" };
            
            // 스프라이트 추가
            var sprite = new Sprite2D { Name = "Sprite" };
            player.AddChild(sprite);
            
            // 충돌 형태 추가
            var collision = new CollisionShape2D { Name = "CollisionShape" };
            var shape = new RectangleShape2D { Size = new Vector2(32, 32) };
            collision.Shape = shape;
            player.AddChild(collision);
            
            // 플레이어 스크립트 생성
            var scriptPath = $"{projectPath}/scripts/Player.cs";
            File.WriteAllText(scriptPath, GeneratePlayerScript());
            
            gameNodes[sceneName].AddChild(player);
            gameNodes["Player"] = player;
            
            GD.Print("🏃 플레이어 추가됨");
        }
        
        /// <summary>
        /// 적 추가
        /// </summary>
        private void AddEnemy(Dictionary<string, object> parameters)
        {
            var sceneName = parameters.GetValueOrDefault("scene", "MainScene").ToString();
            var enemyCount = Convert.ToInt32(parameters.GetValueOrDefault("count", 1));
            
            for (int i = 0; i < enemyCount; i++)
            {
                var enemy = new CharacterBody2D { Name = $"Enemy{i}" };
                
                // 적 스프라이트
                var sprite = new Sprite2D { Name = "Sprite" };
                enemy.AddChild(sprite);
                
                // 충돌 형태
                var collision = new CollisionShape2D { Name = "CollisionShape" };
                var shape = new CircleShape2D { Radius = 16 };
                collision.Shape = shape;
                enemy.AddChild(collision);
                
                // 위치 설정
                enemy.Position = new Vector2(
                    GD.RandRange(100, 500),
                    GD.RandRange(100, 300)
                );
                
                if (gameNodes.ContainsKey(sceneName))
                {
                    gameNodes[sceneName].AddChild(enemy);
                }
                
                gameNodes[$"Enemy{i}"] = enemy;
            }
            
            GD.Print($"👾 {enemyCount}개의 적 추가됨");
        }
        
        /// <summary>
        /// 플랫폼 추가
        /// </summary>
        private void AddPlatform(Dictionary<string, object> parameters)
        {
            var sceneName = parameters.GetValueOrDefault("scene", "MainScene").ToString();
            var platformCount = Convert.ToInt32(parameters.GetValueOrDefault("count", 5));
            
            for (int i = 0; i < platformCount; i++)
            {
                var platform = new StaticBody2D { Name = $"Platform{i}" };
                
                // 플랫폼 형태
                var collision = new CollisionShape2D();
                var shape = new RectangleShape2D { Size = new Vector2(200, 20) };
                collision.Shape = shape;
                platform.AddChild(collision);
                
                // 시각적 표현
                var rect = new ColorRect
                {
                    Size = new Vector2(200, 20),
                    Position = new Vector2(-100, -10),
                    Color = Colors.Brown
                };
                platform.AddChild(rect);
                
                // 위치 설정
                platform.Position = new Vector2(
                    i * 250 + 100,
                    GD.RandRange(200, 400)
                );
                
                if (gameNodes.ContainsKey(sceneName))
                {
                    gameNodes[sceneName].AddChild(platform);
                }
                
                gameNodes[$"Platform{i}"] = platform;
            }
            
            GD.Print($"🟫 {platformCount}개의 플랫폼 추가됨");
        }
        
        /// <summary>
        /// 충돌 처리 추가
        /// </summary>
        private void AddCollision(Dictionary<string, object> parameters)
        {
            // Area2D를 사용한 충돌 감지 영역 추가
            var area = new Area2D { Name = "CollisionArea" };
            
            area.BodyEntered += (body) =>
            {
                if (body.Name == "Player" && body.HasMethod("OnCollision"))
                {
                    body.Call("OnCollision", area);
                }
            };
            
            var collision = new CollisionShape2D();
            var shape = new RectangleShape2D { Size = new Vector2(50, 50) };
            collision.Shape = shape;
            area.AddChild(collision);
            
            GD.Print("💥 충돌 처리 시스템 추가됨");
        }
        
        /// <summary>
        /// 물리 엔진 설정
        /// </summary>
        private void AddPhysics(Dictionary<string, object> parameters)
        {
            var gravity = Convert.ToSingle(parameters.GetValueOrDefault("gravity", 980.0f));
            
            // 프로젝트 설정에서 물리 설정
            ProjectSettings.SetSetting("physics/2d/default_gravity", gravity);
            ProjectSettings.SetSetting("physics/2d/default_gravity_vector", Vector2.Down);
            
            GD.Print($"⚙️ 물리 엔진 설정됨 (중력: {gravity})");
        }
        
        /// <summary>
        /// UI 추가
        /// </summary>
        private void AddUI(Dictionary<string, object> parameters)
        {
            var uiType = parameters.GetValueOrDefault("type", "HUD").ToString();
            
            var canvasLayer = new CanvasLayer { Name = "UI" };
            
            // 점수 표시
            var scoreLabel = new Label
            {
                Name = "ScoreLabel",
                Text = "Score: 0",
                Position = new Vector2(10, 10),
                AddThemeStyleboxOverride("normal", new StyleBoxFlat())
            };
            canvasLayer.AddChild(scoreLabel);
            
            // 체력바
            var healthBar = new ProgressBar
            {
                Name = "HealthBar",
                Value = 100,
                Position = new Vector2(10, 40),
                Size = new Vector2(200, 20)
            };
            canvasLayer.AddChild(healthBar);
            
            if (sceneTree != null)
            {
                sceneTree.Root.AddChild(canvasLayer);
            }
            
            gameNodes["UI"] = canvasLayer;
            GD.Print("🎨 UI 시스템 추가됨");
        }
        
        /// <summary>
        /// 사운드 추가
        /// </summary>
        private void AddSound(Dictionary<string, object> parameters)
        {
            var soundType = parameters.GetValueOrDefault("type", "background").ToString();
            
            var audioPlayer = new AudioStreamPlayer { Name = $"Audio_{soundType}" };
            
            // 오디오 버스 설정
            audioPlayer.Bus = soundType == "background" ? "Music" : "SFX";
            audioPlayer.VolumeDb = -10.0f;
            
            if (sceneTree != null)
            {
                sceneTree.Root.AddChild(audioPlayer);
            }
            
            gameNodes[$"Audio_{soundType}"] = audioPlayer;
            GD.Print($"🔊 사운드 시스템 추가됨: {soundType}");
        }
        
        /// <summary>
        /// 성능 최적화
        /// </summary>
        private void OptimizePerformance(Dictionary<string, object> parameters)
        {
            // 렌더링 최적화
            RenderingServer.SetDefaultClearColor(Colors.Black);
            
            // 물리 최적화
            ProjectSettings.SetSetting("physics/2d/thread_model", 2); // Multi-threaded
            
            // 메모리 최적화
            GC.Collect();
            GC.WaitForPendingFinalizers();
            
            GD.Print("⚡ 성능 최적화 적용됨");
        }
        
        /// <summary>
        /// 오류 디버깅
        /// </summary>
        private void DebugError(Dictionary<string, object> parameters)
        {
            var errorType = parameters.GetValueOrDefault("type", "unknown").ToString();
            var errorMessage = parameters.GetValueOrDefault("message", "").ToString();
            
            GD.PrintErr($"🐛 디버깅: {errorType} - {errorMessage}");
            
            // 자동 수정 시도
            switch (errorType)
            {
                case "null_reference":
                    CheckAndFixNullReferences();
                    break;
                case "physics_error":
                    ResetPhysicsSettings();
                    break;
                case "rendering_error":
                    ResetRenderingSettings();
                    break;
            }
        }
        
        /// <summary>
        /// C# 코드 생성
        /// </summary>
        private void GenerateCSharpCode(Dictionary<string, object> parameters)
        {
            var className = parameters.GetValueOrDefault("class", "GameController").ToString();
            var code = GenerateGameControllerScript(className);
            
            var scriptPath = $"{projectPath}/scripts/{className}.cs";
            Directory.CreateDirectory(Path.GetDirectoryName(scriptPath));
            File.WriteAllText(scriptPath, code);
            
            GD.Print($"💻 C# 코드 생성됨: {className}.cs");
        }
        
        /// <summary>
        /// 게임플레이 테스트
        /// </summary>
        private void TestGameplay(Dictionary<string, object> parameters)
        {
            var testDuration = Convert.ToInt32(parameters.GetValueOrDefault("duration", 30));
            
            GD.Print($"🎮 게임플레이 테스트 시작 ({testDuration}초)");
            
            // 게임 실행
            if (sceneTree != null)
            {
                sceneTree.ChangeSceneToFile($"{projectPath}/scenes/MainScene.tscn");
            }
        }
        
        /// <summary>
        /// 프로젝트 파일 생성
        /// </summary>
        private async Task CreateProjectFiles(string path, string projectName)
        {
            // project.godot 파일 생성
            var projectConfig = $@"
config_version=5

[application]
config/name=""{projectName}""
config/features=PackedStringArray(""4.2"", ""C#"", ""Forward Plus"")
config/icon=""res://icon.svg""

[dotnet]
project/assembly_name=""{projectName}""

[rendering]
renderer/rendering_method=""forward_plus""
";
            File.WriteAllText(Path.Combine(path, "project.godot"), projectConfig);
            
            // 디렉토리 구조 생성
            Directory.CreateDirectory(Path.Combine(path, "scenes"));
            Directory.CreateDirectory(Path.Combine(path, "scripts"));
            Directory.CreateDirectory(Path.Combine(path, "assets"));
            
            await Task.CompletedTask;
        }
        
        /// <summary>
        /// Godot 실행 파일 찾기
        /// </summary>
        private string FindGodotExecutable()
        {
            var possiblePaths = new[]
            {
                @"C:\Program Files\Godot\Godot.exe",
                @"C:\Program Files (x86)\Godot\Godot.exe",
                @"/usr/local/bin/godot",
                @"/usr/bin/godot",
                @"~/Applications/Godot.app/Contents/MacOS/Godot"
            };
            
            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                    return path;
            }
            
            // 환경 변수에서 찾기
            var envPath = Environment.GetEnvironmentVariable("GODOT_PATH");
            if (!string.IsNullOrEmpty(envPath) && File.Exists(envPath))
                return envPath;
            
            return null;
        }
        
        /// <summary>
        /// 플레이어 스크립트 생성
        /// </summary>
        private string GeneratePlayerScript()
        {
            return @"using Godot;

public partial class Player : CharacterBody2D
{
    private float speed = 300.0f;
    private float jumpVelocity = -400.0f;
    private float gravity = ProjectSettings.GetSetting(""physics/2d/default_gravity"").AsSingle();
    
    public override void _PhysicsProcess(double delta)
    {
        Vector2 velocity = Velocity;
        
        // 중력 적용
        if (!IsOnFloor())
            velocity.Y += gravity * (float)delta;
        
        // 점프 처리
        if (Input.IsActionJustPressed(""jump"") && IsOnFloor())
            velocity.Y = jumpVelocity;
        
        // 좌우 이동
        Vector2 direction = Input.GetVector(""move_left"", ""move_right"", ""move_up"", ""move_down"");
        if (direction != Vector2.Zero)
        {
            velocity.X = direction.X * speed;
        }
        else
        {
            velocity.X = Mathf.MoveToward(Velocity.X, 0, speed * (float)delta);
        }
        
        Velocity = velocity;
        MoveAndSlide();
    }
    
    public void OnCollision(Area2D area)
    {
        GD.Print($""Player collided with {area.Name}"");
    }
}";
        }
        
        /// <summary>
        /// 게임 컨트롤러 스크립트 생성
        /// </summary>
        private string GenerateGameControllerScript(string className)
        {
            return $@"using Godot;
using System;

public partial class {className} : Node
{{
    private int score = 0;
    private float timeElapsed = 0.0f;
    private Label scoreLabel;
    private ProgressBar healthBar;
    
    public override void _Ready()
    {{
        GD.Print(""Game Started!"");
        
        // UI 요소 찾기
        scoreLabel = GetNode<Label>(""/root/UI/ScoreLabel"");
        healthBar = GetNode<ProgressBar>(""/root/UI/HealthBar"");
    }}
    
    public override void _Process(double delta)
    {{
        timeElapsed += (float)delta;
        
        // 게임 로직 업데이트
        UpdateGameState();
    }}
    
    private void UpdateGameState()
    {{
        // 점수 업데이트
        if (scoreLabel != null)
            scoreLabel.Text = $""Score: {{score}}"";
    }}
    
    public void AddScore(int points)
    {{
        score += points;
        EmitSignal(SignalName.ScoreChanged, score);
    }}
    
    public void TakeDamage(float damage)
    {{
        if (healthBar != null)
        {{
            healthBar.Value -= damage;
            if (healthBar.Value <= 0)
            {{
                GameOver();
            }}
        }}
    }}
    
    private void GameOver()
    {{
        GD.Print(""Game Over!"");
        GetTree().Paused = true;
    }}
    
    [Signal]
    public delegate void ScoreChangedEventHandler(int newScore);
}}";
        }
        
        /// <summary>
        /// Null 참조 확인 및 수정
        /// </summary>
        private void CheckAndFixNullReferences()
        {
            foreach (var kvp in gameNodes)
            {
                if (kvp.Value == null || !IsInstanceValid(kvp.Value))
                {
                    GD.PrintErr($"Null reference found: {kvp.Key}");
                    gameNodes.Remove(kvp.Key);
                }
            }
        }
        
        /// <summary>
        /// 물리 설정 초기화
        /// </summary>
        private void ResetPhysicsSettings()
        {
            ProjectSettings.SetSetting("physics/2d/default_gravity", 980.0);
            ProjectSettings.SetSetting("physics/2d/default_linear_damp", 0.1);
            ProjectSettings.SetSetting("physics/2d/default_angular_damp", 1.0);
        }
        
        /// <summary>
        /// 렌더링 설정 초기화
        /// </summary>
        private void ResetRenderingSettings()
        {
            RenderingServer.SetDefaultClearColor(Colors.Black);
            ProjectSettings.SetSetting("rendering/anti_aliasing/quality/msaa_2d", 0);
            ProjectSettings.SetSetting("rendering/anti_aliasing/quality/msaa_3d", 0);
        }
        
        /// <summary>
        /// 엔진 종료
        /// </summary>
        public void StopEngine()
        {
            if (godotProcess != null && !godotProcess.HasExited)
            {
                godotProcess.Kill();
                godotProcess.Dispose();
            }
            
            isEngineRunning = false;
            gameNodes.Clear();
            
            GD.Print("🛑 Godot 엔진 종료됨");
        }
        
        /// <summary>
        /// 리소스 정리
        /// </summary>
        public override void _ExitTree()
        {
            StopEngine();
        }
    }
}