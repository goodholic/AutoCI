using Microsoft.AspNetCore.Mvc;
using Backend.Services;

namespace Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class RAGStatusController : ControllerBase
    {
        private readonly EnhancedRAGService _ragService;
        private readonly ILogger<RAGStatusController> _logger;

        public RAGStatusController(EnhancedRAGService ragService, ILogger<RAGStatusController> logger)
        {
            _ragService = ragService;
            _logger = logger;
        }

        [HttpGet("status")]
        public IActionResult GetRAGStatus()
        {
            try
            {
                var status = _ragService.GetSystemStatus();
                _logger.LogInformation("📊 RAG 상태 조회 성공");
                
                return Ok(new 
                { 
                    success = true, 
                    message = "Enhanced RAG 시스템 활성화됨",
                    data = status,
                    timestamp = DateTime.UtcNow
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"❌ RAG 상태 조회 실패: {ex.Message}");
                return BadRequest(new { success = false, error = ex.Message });
            }
        }

        [HttpPost("search")]
        public IActionResult SearchKnowledge([FromBody] SearchRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.Query))
                    return BadRequest(new { success = false, error = "검색 쿼리가 필요합니다." });

                var results = _ragService.SearchRelevantCode(request.Query, request.MaxResults ?? 5);
                
                _logger.LogInformation($"🔍 RAG 검색 완료: '{request.Query}' -> {results.Count}개 결과");
                
                return Ok(new 
                { 
                    success = true, 
                    query = request.Query,
                    results = results.Select(r => new {
                        id = r.Id,
                        description = r.Description,
                        category = r.Category,
                        templateName = r.TemplateName,
                        qualityScore = r.QualityScore,
                        relevanceScore = r.RelevanceScore,
                        codePreview = r.Code.Length > 200 ? r.Code.Substring(0, 200) + "..." : r.Code,
                        keywords = r.Keywords.Take(5)
                    })
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"❌ RAG 검색 실패: {ex.Message}");
                return BadRequest(new { success = false, error = ex.Message });
            }
        }

        [HttpPost("enhance")]
        public IActionResult EnhancePrompt([FromBody] EnhanceRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.Query))
                    return BadRequest(new { success = false, error = "쿼리가 필요합니다." });

                var enhancedPrompt = _ragService.EnhancePrompt(request.Query);
                
                _logger.LogInformation($"✨ RAG 프롬프트 향상 완료: {request.Query.Length}자 -> {enhancedPrompt.Length}자");
                
                return Ok(new 
                { 
                    success = true, 
                    originalQuery = request.Query,
                    enhancedPrompt = enhancedPrompt,
                    enhancementRatio = Math.Round((double)enhancedPrompt.Length / request.Query.Length, 2)
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"❌ RAG 프롬프트 향상 실패: {ex.Message}");
                return BadRequest(new { success = false, error = ex.Message });
            }
        }
        
        [HttpGet("learning-status")]
        public IActionResult GetLearningStatus()
        {
            try
            {
                var learningStatus = _ragService.GetLearningStatus();
                
                return Ok(new 
                { 
                    success = true,
                    data = new
                    {
                        isLearning = learningStatus.IsLearning,
                        message = learningStatus.Message,
                        progress = learningStatus.Progress,
                        currentFile = learningStatus.CurrentFile
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError($"❌ 학습 상태 조회 실패: {ex.Message}");
                return Ok(new 
                { 
                    success = true,
                    data = new
                    {
                        isLearning = false,
                        message = "시스템이 유휴 상태입니다.",
                        progress = 0,
                        currentFile = ""
                    }
                });
            }
        }
    }

    public class SearchRequest
    {
        public string Query { get; set; } = "";
        public int? MaxResults { get; set; }
    }

    public class EnhanceRequest
    {
        public string Query { get; set; } = "";
    }
} 