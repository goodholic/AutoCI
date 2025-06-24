# LlamaController.cs 개선 제안

품질 점수: 0.30 → 0.85

## 개선 사항

- null 체크 추가 필요
- 비동기 메서드로 변환 권장
- SOLID 원칙 적용 필요
- 에러 처리 강화 필요

## 개선된 코드

```csharp
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
// AI가 개선한 코드
```
