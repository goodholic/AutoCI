// Backend/Controllers/AIController.cs
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using Backend.Services;

[ApiController]
[Route("api/[controller]")]
public class AIController : ControllerBase
{
    private readonly AIService _aiService;
    private readonly SearchService _searchService;
    private readonly RAGService _ragService;
    
    public AIController(
        AIService aiService, 
        SearchService searchService, 
        RAGService ragService)
    {
        _aiService = aiService;
        _searchService = searchService;
        _ragService = ragService;
    }
    
    [HttpPost("generate")]
    public async Task<IActionResult> Generate([FromBody] GenerateRequest request)
    {
        var result = await _aiService.GenerateText(request.Prompt);
        return Ok(new { result });
    }
    
    [HttpPost("search")]
    public async Task<IActionResult> Search([FromBody] SearchRequest request)
    {
        var results = await _searchService.Search(request.Query);
        return Ok(results);
    }
    
    [HttpPost("rag/index")]
    public async Task<IActionResult> IndexDocument([FromBody] Document document)
    {
        await _ragService.IndexDocument(document);
        return Ok(new { message = "문서가 인덱싱되었습니다." });
    }
    
    [HttpPost("rag/query")]
    public async Task<IActionResult> RAGQuery([FromBody] RAGQueryRequest request)
    {
        var answer = await _ragService.Query(request.Question);
        return Ok(new { answer });
    }
}

public class GenerateRequest
{
    public string Prompt { get; set; } = string.Empty;
}

public class SearchRequest
{
    public string Query { get; set; } = string.Empty;
}

public class RAGQueryRequest
{
    public string Question { get; set; } = string.Empty;
}