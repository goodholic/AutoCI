using System.Diagnostics;
using System.Text;
using Microsoft.AspNetCore.SignalR;

namespace MyAIWebApp.Backend.Services
{
    public class AutoCIMonitorService : BackgroundService
    {
        private readonly ILogger<AutoCIMonitorService> _logger;
        private readonly IHubContext<AutoCIHub> _hubContext;
        private readonly IServiceProvider _serviceProvider;

        public AutoCIMonitorService(
            ILogger<AutoCIMonitorService> logger,
            IHubContext<AutoCIHub> hubContext,
            IServiceProvider serviceProvider)
        {
            _logger = logger;
            _hubContext = hubContext;
            _serviceProvider = serviceProvider;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Check AutoCI system status
                    var statusInfo = await GetAutoCIStatus();
                    
                    // Broadcast status to all connected clients
                    await _hubContext.Clients.All.SendAsync("SystemStatus", statusInfo, stoppingToken);
                    
                    // Check for any Python processes
                    var pythonProcesses = await GetPythonProcesses();
                    if (pythonProcesses.Any())
                    {
                        await _hubContext.Clients.All.SendAsync("PythonProcesses", pythonProcesses, stoppingToken);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in AutoCI monitor service");
                }

                // Check every 5 seconds
                await Task.Delay(5000, stoppingToken);
            }
        }

        private async Task<AutoCIStatus> GetAutoCIStatus()
        {
            var status = new AutoCIStatus();

            try
            {
                // Check if enhance system is running
                var enhanceCheck = await ExecuteCommand("ps aux | grep 'advanced_autoci_system.py' | grep -v grep");
                status.EnhanceSystemRunning = !string.IsNullOrWhiteSpace(enhanceCheck);

                // Check if dual phase is running
                var dualCheck = await ExecuteCommand("ps aux | grep 'robust_dual_phase.py' | grep -v grep");
                status.DualPhaseRunning = !string.IsNullOrWhiteSpace(dualCheck);

                // Check if RAG server is running
                var ragCheck = await ExecuteCommand("ps aux | grep 'enhanced_rag_system' | grep -v grep");
                status.RAGServerRunning = !string.IsNullOrWhiteSpace(ragCheck);

                // Check for autoci reports
                var reportsPath = "/mnt/c/Users/super/Desktop/Unity Project(25년도 제작)/26.AutoCI/AutoCI/autoci_reports";
                if (Directory.Exists(reportsPath))
                {
                    var reports = Directory.GetFiles(reportsPath, "*.md")
                        .OrderByDescending(f => File.GetLastWriteTime(f))
                        .Take(5)
                        .Select(f => new FileInfo(f))
                        .Select(fi => new ReportInfo
                        {
                            FileName = fi.Name,
                            CreatedAt = fi.LastWriteTime,
                            Size = fi.Length
                        })
                        .ToList();
                    
                    status.RecentReports = reports;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting AutoCI status");
            }

            return status;
        }

        private async Task<List<PythonProcessInfo>> GetPythonProcesses()
        {
            var processes = new List<PythonProcessInfo>();

            try
            {
                var output = await ExecuteCommand("ps aux | grep python | grep -E '(autoci|rag|dual)' | grep -v grep");
                if (!string.IsNullOrWhiteSpace(output))
                {
                    var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                    foreach (var line in lines)
                    {
                        var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 11)
                        {
                            processes.Add(new PythonProcessInfo
                            {
                                PID = parts[1],
                                CPU = parts[2],
                                Memory = parts[3],
                                Command = string.Join(" ", parts.Skip(10))
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting Python processes");
            }

            return processes;
        }

        private async Task<string> ExecuteCommand(string command)
        {
            try
            {
                var startInfo = new ProcessStartInfo
                {
                    FileName = "bash",
                    Arguments = $"-c \"{command}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(startInfo);
                if (process == null) return "";

                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();
                
                return output;
            }
            catch
            {
                return "";
            }
        }
    }

    public class AutoCIStatus
    {
        public bool EnhanceSystemRunning { get; set; }
        public bool DualPhaseRunning { get; set; }
        public bool RAGServerRunning { get; set; }
        public List<ReportInfo> RecentReports { get; set; } = new();
    }

    public class ReportInfo
    {
        public string FileName { get; set; } = "";
        public DateTime CreatedAt { get; set; }
        public long Size { get; set; }
    }

    public class PythonProcessInfo
    {
        public string PID { get; set; } = "";
        public string CPU { get; set; } = "";
        public string Memory { get; set; } = "";
        public string Command { get; set; } = "";
    }
}