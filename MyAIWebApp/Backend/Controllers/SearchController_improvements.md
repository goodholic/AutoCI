# SearchController.cs 개선 제안

품질 점수: 0.35 → 0.85

## 개선 사항

- null 체크 추가 필요
- 비동기 메서드로 변환 권장
- SOLID 원칙 적용 필요
- 에러 처리 강화 필요

## 개선된 코드

```csharp
using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MyAIWebApp.Backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SearchController : ControllerBase
    {
        private readonly SearchService _searchService;
        
        public SearchController(SearchService searchService)
        {
            _searchService = searchService;
        }
        
        [HttpGet("code")]
        public async Task<ActionResult<List<SearchResult>>> SearchCode([FromQuery] string query, [FromQuery] int maxResults = 10)
        {
            if (string.IsNullOrWhiteSpace(query))
            {
                return BadRequest(new { error = "Query parameter is required" });
            }
            
            var results = await _searchService.Search(query, maxResults);
            return Ok(results);
        }
        
        [HttpPost("index")]
        public async Task<ActionResult> IndexCode([FromBody] IndexCodeRequest request)
        {
            if (string.IsNullOrWhiteSpace(request.FileName) || string.IsNullOrWhiteSpace(request.Content))
            {
                return BadRequest(new { error = "FileName and Content are required" });
            }
            
            var success = await _searchService.IndexCode(request.FileName, request.Content, request.Language ?? "csharp");
            
            if (success)
            {
                return Ok(new { message = "Code indexed successfully" });
            }
            
            return StatusCode(500, new { error = "Failed to index code" });
        }
    }
    
    public class IndexCodeRequest
    {
        public string FileName { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
        public string? Language { get; set; }
    }
}
// AI가 개선한 코드
```
