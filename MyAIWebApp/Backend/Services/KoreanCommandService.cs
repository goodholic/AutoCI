using System.Diagnostics;
using System.Text.Json;

namespace Backend.Services
{
    public class KoreanCommandService
    {
        private readonly ILogger<KoreanCommandService> _logger;
        private readonly string _pythonScriptPath;

        public KoreanCommandService(ILogger<KoreanCommandService> logger)
        {
            _logger = logger;
            _pythonScriptPath = Path.Combine(Directory.GetCurrentDirectory(), "..", "korean_command_trainer.py");
        }

        public async Task<KoreanCommandResult> AnalyzeCommand(string text)
        {
            try
            {
                _logger.LogInformation($"🤖 한글 명령 분석: {text}");

                var processInfo = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = $"-c \"import sys; sys.path.append('{Path.GetDirectoryName(_pythonScriptPath)}'); " +
                               $"from korean_command_trainer import analyze_korean_command; " +
                               $"import json; " +
                               $"result = analyze_korean_command('{text.Replace("'", "\\'")}'); " +
                               $"print(json.dumps(result, ensure_ascii=False))\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(_pythonScriptPath)
                };

                using var process = Process.Start(processInfo);
                if (process == null)
                {
                    return new KoreanCommandResult { Success = false, Message = "프로세스 시작 실패" };
                }

                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (!string.IsNullOrEmpty(error))
                {
                    _logger.LogError($"Python 오류: {error}");
                }

                if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
                {
                    try
                    {
                        var result = JsonSerializer.Deserialize<KoreanCommandResult>(output);
                        if (result?.Success == true)
                        {
                            _logger.LogInformation($"✅ 명령 인식 성공: {result.CommandType} (신뢰도: {result.Confidence:F2})");
                        }
                        return result ?? new KoreanCommandResult { Success = false };
                    }
                    catch (JsonException ex)
                    {
                        _logger.LogError($"JSON 파싱 오류: {ex.Message}");
                        return new KoreanCommandResult { Success = false, Message = "결과 파싱 오류" };
                    }
                }

                return new KoreanCommandResult { Success = false, Message = "명령 분석 실패" };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "한글 명령 분석 중 오류");
                return new KoreanCommandResult { Success = false, Message = ex.Message };
            }
        }

        public async Task<bool> TrainCommand(string text, string commandType)
        {
            try
            {
                _logger.LogInformation($"📚 한글 명령 학습: '{text}' → {commandType}");

                var processInfo = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = $"-c \"import sys; sys.path.append('{Path.GetDirectoryName(_pythonScriptPath)}'); " +
                               $"from korean_command_trainer import train_korean_command; " +
                               $"import json; " +
                               $"result = train_korean_command('{text.Replace("'", "\\'")}', '{commandType}'); " +
                               $"print(json.dumps(result))\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(_pythonScriptPath)
                };

                using var process = Process.Start(processInfo);
                if (process == null) return false;

                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (process.ExitCode == 0)
                {
                    _logger.LogInformation("✅ 한글 명령 학습 완료");
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "한글 명령 학습 중 오류");
                return false;
            }
        }

        public async Task<TrainingReport> GetTrainingReport()
        {
            try
            {
                var processInfo = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = $"-c \"import sys; sys.path.append('{Path.GetDirectoryName(_pythonScriptPath)}'); " +
                               $"from korean_command_trainer import KoreanCommandTrainer; " +
                               $"import json; " +
                               $"trainer = KoreanCommandTrainer(); " +
                               $"report = trainer.generate_training_report(); " +
                               $"print(json.dumps(report, ensure_ascii=False))\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = Path.GetDirectoryName(_pythonScriptPath)
                };

                using var process = Process.Start(processInfo);
                if (process == null)
                {
                    return new TrainingReport();
                }

                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
                {
                    var report = JsonSerializer.Deserialize<TrainingReport>(output);
                    return report ?? new TrainingReport();
                }

                return new TrainingReport();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "학습 리포트 조회 중 오류");
                return new TrainingReport();
            }
        }
    }

    public class KoreanCommandResult
    {
        public bool Success { get; set; }
        public string? CommandType { get; set; }
        public string? Intent { get; set; }
        public double Confidence { get; set; }
        public List<string> Entities { get; set; } = new();
        public string? OriginalText { get; set; }
        public string? Message { get; set; }
        public string? Suggestion { get; set; }
    }

    public class TrainingReport
    {
        public int TotalExamples { get; set; }
        public Dictionary<string, int> Intents { get; set; } = new();
        public List<string> Keywords { get; set; } = new();
        public List<object> RecentExamples { get; set; } = new();
    }
}