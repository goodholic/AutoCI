using Microsoft.AspNetCore.Mvc;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Services;

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