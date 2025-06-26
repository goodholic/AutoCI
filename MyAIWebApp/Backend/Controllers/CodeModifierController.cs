using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Text.Json;

[ApiController]
[Route("api/[controller]")]
public class CodeModifierController : ControllerBase
{
    private readonly ILogger<CodeModifierController> _logger;
    private readonly string _autoModifierPath;
    
    public CodeModifierController(ILogger<CodeModifierController> logger)
    {
        _logger = logger;
        _autoModifierPath = Path.Combine(Directory.GetCurrentDirectory(), "../auto_code_modifier.py");
    }
    
    [HttpPost("add-task")]
    public async Task<IActionResult> AddTask([FromBody] AddTaskRequest request)
    {
        try
        {
            // Python 스크립트 실행하여 작업 추가
            var processInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"-c \"" +
                    $"import sys; sys.path.append('.'); " +
                    $"from auto_code_modifier import AutoCodeModifier; " +
                    $"modifier = AutoCodeModifier(); " +
                    $"task_id = modifier.add_task('{request.TargetFolder}', '{request.FilePattern}', '{request.ModificationType}', '{request.Description}', {(request.CodeContent != null ? $"'{request.CodeContent}'" : "None")}); " +
                    $"print(task_id)\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = "../"
            };
            
            using var process = Process.Start(processInfo);
            if (process == null)
                return BadRequest(new { error = "프로세스 시작에 실패했습니다." });
                
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            
            await process.WaitForExitAsync();
            
            if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
            {
                var taskId = output.Trim();
                _logger.LogInformation($"✅ 작업 추가 성공: {taskId}");
                return Ok(new { taskId, message = "작업이 성공적으로 추가되었습니다." });
            }
            else
            {
                _logger.LogError($"작업 추가 실패: {error}");
                return BadRequest(new { error = "작업 추가에 실패했습니다.", details = error });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "작업 추가 중 오류 발생");
            return StatusCode(500, new { error = "서버 오류가 발생했습니다.", details = ex.Message });
        }
    }
    
    [HttpGet("status")]
    public async Task<IActionResult> GetStatus()
    {
        try
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"-c \"" +
                    $"import sys; sys.path.append('.'); " +
                    $"from auto_code_modifier import AutoCodeModifier; " +
                    $"import json; " +
                    $"modifier = AutoCodeModifier(); " +
                    $"pending = [t for t in modifier.tasks if t.status == 'pending']; " +
                    $"in_progress = [t for t in modifier.tasks if t.status == 'in_progress']; " +
                    $"completed = modifier.completed_tasks; " +
                    $"failed = [t for t in modifier.tasks if t.status == 'failed']; " +
                    $"status = {{'pending': len(pending), 'in_progress': len(in_progress), 'completed': len(completed), 'failed': len(failed), 'recent_tasks': [{{'id': t.id, 'description': t.description, 'status': t.status}} for t in modifier.tasks[-5:]]}}; " +
                    $"print(json.dumps(status))\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = "../"
            };
            
            using var process = Process.Start(processInfo);
            if (process == null)
                return BadRequest(new { error = "프로세스 시작에 실패했습니다." });
                
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            
            if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
            {
                var status = JsonSerializer.Deserialize<object>(output.Trim());
                return Ok(status);
            }
            else
            {
                return BadRequest(new { error = "상태 조회에 실패했습니다." });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "상태 조회 중 오류 발생");
            return StatusCode(500, new { error = "서버 오류가 발생했습니다.", details = ex.Message });
        }
    }
    
    [HttpGet("completed-tasks")]
    public async Task<IActionResult> GetCompletedTasks([FromQuery] int count = 10)
    {
        try
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"-c \"" +
                    $"import sys; sys.path.append('.'); " +
                    $"from auto_code_modifier import AutoCodeModifier; " +
                    $"import json; " +
                    $"modifier = AutoCodeModifier(); " +
                    $"tasks = [{{'id': t.id, 'description': t.description, 'result': t.result, 'timestamp': t.timestamp}} for t in modifier.completed_tasks[-{count}:]]; " +
                    $"print(json.dumps(tasks))\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = "../"
            };
            
            using var process = Process.Start(processInfo);
            if (process == null)
                return BadRequest(new { error = "프로세스 시작에 실패했습니다." });
                
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            
            if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
            {
                var tasks = JsonSerializer.Deserialize<object>(output.Trim());
                return Ok(tasks);
            }
            else
            {
                return BadRequest(new { error = "완료된 작업 조회에 실패했습니다." });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "완료된 작업 조회 중 오류 발생");
            return StatusCode(500, new { error = "서버 오류가 발생했습니다.", details = ex.Message });
        }
    }
    
    [HttpPost("start-continuous")]
    public IActionResult StartContinuous()
    {
        try
        {
            // 백그라운드에서 연속 실행 모드 시작
            var processInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"auto_code_modifier.py",
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = "../"
            };
            
            // 백그라운드 프로세스로 시작
            Process.Start(processInfo);
            
            _logger.LogInformation("🚀 24시간 자동 코드 수정 시스템이 시작되었습니다.");
            return Ok(new { message = "24시간 자동 코드 수정 시스템이 백그라운드에서 시작되었습니다." });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "연속 실행 모드 시작 중 오류 발생");
            return StatusCode(500, new { error = "서버 오류가 발생했습니다.", details = ex.Message });
        }
    }
    
    [HttpPost("add-examples")]
    public async Task<IActionResult> AddExamples()
    {
        try
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"-c \"" +
                    $"import sys; sys.path.append('.'); " +
                    $"from auto_code_modifier import AutoCodeModifier; " +
                    $"modifier = AutoCodeModifier(); " +
                    $"modifier.add_task('../Assets/Scripts', 'PlayerController.cs', 'create', 'Unity 플레이어 컨트롤러 스크립트 생성'); " +
                    $"modifier.add_task('../Assets/Scripts', 'GameManager.cs', 'create', 'Unity 게임 매니저 스크립트 생성'); " +
                    $"modifier.add_task('../Assets/Scripts', '*.cs', 'improve', '모든 스크립트에 성능 최적화 주석 추가'); " +
                    $"print('✅ 3개의 예시 작업이 추가되었습니다!')\"",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                WorkingDirectory = "../"
            };
            
            using var process = Process.Start(processInfo);
            if (process == null)
                return BadRequest(new { error = "프로세스 시작에 실패했습니다." });
                
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();
            
            if (process.ExitCode == 0)
            {
                return Ok(new { message = "3개의 예시 작업이 성공적으로 추가되었습니다." });
            }
            else
            {
                return BadRequest(new { error = "예시 작업 추가에 실패했습니다." });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "예시 작업 추가 중 오류 발생");
            return StatusCode(500, new { error = "서버 오류가 발생했습니다.", details = ex.Message });
        }
    }
}

public class AddTaskRequest
{
    public string TargetFolder { get; set; } = "";
    public string FilePattern { get; set; } = "";
    public string ModificationType { get; set; } = ""; // create, modify, improve, fix
    public string Description { get; set; } = "";
    public string? CodeContent { get; set; }
} 