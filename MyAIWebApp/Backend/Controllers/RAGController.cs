using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

namespace MyAIWebApp.Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class RAGController : ControllerBase
    {
        private readonly RAGService _ragService;
        
        public RAGController(RAGService ragService)
        {
            _ragService = ragService;
        }
        
        [HttpPost("index-readme")]
        public async Task<ActionResult> IndexReadme([FromBody] IndexReadmeRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.ReadmePath))
            {
                return BadRequest(new { error = "README 파일 경로가 필요합니다." });
            }
            
            var success = await _ragService.IndexReadme(request.ReadmePath);
            
            if (success)
            {
                return Ok(new { message = "README 파일이 성공적으로 인덱싱되었습니다." });
            }
            
            return StatusCode(500, new { error = "README 파일 인덱싱에 실패했습니다." });
        }
        
        [HttpPost("index-document")]
        public async Task<ActionResult> IndexDocument([FromBody] Document document)
        {
            if (string.IsNullOrWhiteSpace(document.Content))
            {
                return BadRequest(new { error = "문서 내용이 필요합니다." });
            }
            
            await _ragService.IndexDocument(document);
            return Ok(new { message = "문서가 성공적으로 인덱싱되었습니다." });
        }
        
        [HttpPost("query")]
        public async Task<ActionResult<QueryResponse>> Query([FromBody] QueryRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.Question))
            {
                return BadRequest(new { error = "질문이 필요합니다." });
            }
            
            var answer = await _ragService.Query(request.Question);
            
            return Ok(new QueryResponse
            {
                Question = request.Question,
                Answer = answer
            });
        }
        
        [HttpGet("documents")]
        public ActionResult GetDocuments()
        {
            var documents = _ragService.GetAllDocuments();
            return Ok(documents);
        }
    }
    
    public class IndexReadmeRequest
    {
        public string ReadmePath { get; set; } = string.Empty;
    }
    
    public class QueryRequest
    {
        public string Question { get; set; } = string.Empty;
    }
    
    public class QueryResponse
    {
        public string Question { get; set; } = string.Empty;
        public string Answer { get; set; } = string.Empty;
    }
}