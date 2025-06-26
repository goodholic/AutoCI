using Microsoft.AspNetCore.Mvc;
using Backend.Services;

namespace Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class KoreanCommandController : ControllerBase
    {
        private readonly KoreanCommandService _koreanCommandService;
        private readonly ILogger<KoreanCommandController> _logger;

        public KoreanCommandController(KoreanCommandService koreanCommandService, ILogger<KoreanCommandController> logger)
        {
            _koreanCommandService = koreanCommandService;
            _logger = logger;
        }

        [HttpPost("analyze")]
        public async Task<IActionResult> AnalyzeCommand([FromBody] AnalyzeRequest request)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Text))
                {
                    return BadRequest(new { success = false, message = "텍스트가 비어있습니다." });
                }

                _logger.LogInformation($"🤖 한글 명령 분석 요청: {request.Text}");

                var result = await _koreanCommandService.AnalyzeCommand(request.Text);
                
                // 분석 성공시 학습도 진행 (피드백 루프)
                if (result.Success && !string.IsNullOrEmpty(result.CommandType))
                {
                    _ = Task.Run(async () => 
                    {
                        await _koreanCommandService.TrainCommand(request.Text, result.CommandType);
                    });
                }

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "한글 명령 분석 중 오류");
                return StatusCode(500, new { success = false, message = "서버 오류가 발생했습니다." });
            }
        }

        [HttpPost("train")]
        public async Task<IActionResult> TrainCommand([FromBody] TrainRequest request)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request.Text) || string.IsNullOrWhiteSpace(request.CommandType))
                {
                    return BadRequest(new { success = false, message = "필수 파라미터가 누락되었습니다." });
                }

                _logger.LogInformation($"📚 한글 명령 학습: '{request.Text}' → {request.CommandType}");

                var success = await _koreanCommandService.TrainCommand(request.Text, request.CommandType);

                if (success)
                {
                    return Ok(new { success = true, message = "학습이 완료되었습니다." });
                }
                else
                {
                    return BadRequest(new { success = false, message = "학습에 실패했습니다." });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "한글 명령 학습 중 오류");
                return StatusCode(500, new { success = false, message = "서버 오류가 발생했습니다." });
            }
        }

        [HttpGet("report")]
        public async Task<IActionResult> GetTrainingReport()
        {
            try
            {
                var report = await _koreanCommandService.GetTrainingReport();
                
                return Ok(new
                {
                    success = true,
                    data = report,
                    timestamp = DateTime.UtcNow
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "학습 리포트 조회 중 오류");
                return StatusCode(500, new { success = false, message = "서버 오류가 발생했습니다." });
            }
        }

        [HttpGet("examples")]
        public IActionResult GetExamples()
        {
            var examples = new[]
            {
                new { text = "Unity에서 플레이어 캐릭터 컨트롤러를 만들어주세요", commandType = "unity_player" },
                new { text = "적 AI 스크립트를 구현하고 싶습니다", commandType = "unity_enemy" },
                new { text = "게임 전체를 관리하는 매니저를 작성해줘", commandType = "game_manager" },
                new { text = "모든 코드를 최적화하고 개선해주세요", commandType = "improve_all" },
                new { text = "몬스터 추적 시스템 만들어줘", commandType = "unity_enemy" },
                new { text = "플레이어 움직임 스크립트 생성", commandType = "unity_player" },
                new { text = "씬 전환 매니저 구현", commandType = "game_manager" },
                new { text = "코드 리팩토링 진행해줘", commandType = "improve_all" }
            };

            return Ok(new
            {
                success = true,
                examples = examples,
                totalCount = examples.Length
            });
        }
    }

    public class AnalyzeRequest
    {
        public string Text { get; set; } = "";
    }

    public class TrainRequest
    {
        public string Text { get; set; } = "";
        public string CommandType { get; set; } = "";
    }
}