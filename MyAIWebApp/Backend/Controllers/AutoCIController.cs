using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Text;

namespace Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AutoCIController : ControllerBase
    {
        private readonly ILogger<AutoCIController> _logger;
        private readonly string _autoциPath;

        public AutoCIController(ILogger<AutoCIController> logger)
        {
            _logger = logger;
            _autoциPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "autoci");
        }

        [HttpPost("execute")]
        public async Task<IActionResult> ExecuteCommand([FromBody] AutoCICommandRequest request)
        {
            try
            {
                _logger.LogInformation($"🚀 AutoCI 명령 실행: {request.Command}");

                // autoci 스크립트 경로 확인
                if (!System.IO.File.Exists(_autoциPath))
                {
                    _logger.LogError($"❌ autoci 스크립트를 찾을 수 없습니다: {_autoциPath}");
                    return BadRequest(new { error = "autoci 스크립트를 찾을 수 없습니다." });
                }

                var processInfo = new ProcessStartInfo
                {
                    FileName = "bash",
                    Arguments = $"{_autoциPath} {request.Command}",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(_autoциPath)
                };

                var output = new StringBuilder();
                var error = new StringBuilder();

                using var process = Process.Start(processInfo);
                if (process == null)
                {
                    return BadRequest(new { error = "프로세스를 시작할 수 없습니다." });
                }

                // 비동기적으로 출력 읽기
                var outputTask = process.StandardOutput.ReadToEndAsync();
                var errorTask = process.StandardError.ReadToEndAsync();

                // 타임아웃 설정 (30초)
                var completed = await Task.Run(() => process.WaitForExit(30000));

                if (!completed)
                {
                    process.Kill();
                    return StatusCode(408, new { error = "명령 실행 시간이 초과되었습니다." });
                }

                output.Append(await outputTask);
                error.Append(await errorTask);

                if (process.ExitCode == 0)
                {
                    _logger.LogInformation($"✅ AutoCI 명령 실행 성공: {request.Command}");
                    return Ok(new 
                    { 
                        success = true,
                        output = output.ToString(),
                        command = request.Command,
                        exitCode = process.ExitCode
                    });
                }
                else
                {
                    _logger.LogError($"❌ AutoCI 명령 실행 실패: {error}");
                    return BadRequest(new 
                    { 
                        success = false,
                        error = error.ToString(),
                        output = output.ToString(),
                        command = request.Command,
                        exitCode = process.ExitCode
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "AutoCI 명령 실행 중 오류 발생");
                return StatusCode(500, new { error = "서버 오류가 발생했습니다.", details = ex.Message });
            }
        }

        [HttpGet("status")]
        public async Task<IActionResult> GetSystemStatus()
        {
            try
            {
                var status = new
                {
                    autociAvailable = System.IO.File.Exists(_autoциPath),
                    workingDirectory = Path.GetDirectoryName(_autoциPath),
                    pythonProcesses = await GetPythonProcesses(),
                    reportFiles = GetRecentReports()
                };

                return Ok(status);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "시스템 상태 확인 중 오류");
                return StatusCode(500, new { error = "상태 확인 실패", details = ex.Message });
            }
        }

        private async Task<List<object>> GetPythonProcesses()
        {
            var processes = new List<object>();
            
            try
            {
                var processInfo = new ProcessStartInfo
                {
                    FileName = "bash",
                    Arguments = "-c \"ps aux | grep python | grep -E '(autoci|rag|dual|enhance)' | grep -v grep\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(processInfo);
                if (process != null)
                {
                    var output = await process.StandardOutput.ReadToEndAsync();
                    await process.WaitForExitAsync();

                    var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length > 10)
                        {
                            processes.Add(new
                            {
                                pid = parts[1],
                                cpu = parts[2],
                                mem = parts[3],
                                command = string.Join(" ", parts.Skip(10))
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Python 프로세스 조회 실패");
            }

            return processes;
        }

        private List<object> GetRecentReports()
        {
            var reports = new List<object>();
            var reportDir = Path.Combine(Path.GetDirectoryName(_autoциPath), "autoci_reports");

            if (Directory.Exists(reportDir))
            {
                var files = Directory.GetFiles(reportDir, "*.md")
                    .OrderByDescending(f => new FileInfo(f).LastWriteTime)
                    .Take(5);

                foreach (var file in files)
                {
                    var info = new FileInfo(file);
                    reports.Add(new
                    {
                        name = info.Name,
                        size = info.Length,
                        modified = info.LastWriteTime,
                        path = file
                    });
                }
            }

            return reports;
        }
    }

    public class AutoCICommandRequest
    {
        public string Command { get; set; } = "";
        public Dictionary<string, string>? Options { get; set; }
    }
}