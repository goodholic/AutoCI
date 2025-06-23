using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;

[ApiController]
[Route("api/[controller]")]
public class LlamaController : ControllerBase
{
    private readonly LlamaService _llamaService;
    
    public LlamaController(LlamaService llamaService)
    {
        _llamaService = llamaService;
    }
    
    [HttpPost("generate")]
    public async Task<IActionResult> GenerateCode([FromBody] GenerateCodeRequest request)
    {
        var code = await _llamaService.GenerateCode(request.Prompt);
        return Ok(new { generatedCode = code });
    }
}

public class GenerateCodeRequest
{
    public string Prompt { get; set; } = string.Empty;
}