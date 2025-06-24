using System;
using System.Threading.Tasks;

public class AIService
{
    private readonly LlamaService _llamaService;
    
    public AIService(LlamaService llamaService)
    {
        _llamaService = llamaService;
    }
    
    public async Task<string> GenerateText(string prompt)
    {
        // LlamaService를 통해 텍스트 생성
        return await _llamaService.GenerateCode(prompt);
    }
    
    public async Task<string> GenerateCode(string prompt, string language = "csharp")
    {
        // 언어별 프롬프트 조정
        var enhancedPrompt = language.ToLower() switch
        {
            "csharp" => $"Generate C# code for the following request:\n{prompt}",
            "javascript" => $"Generate JavaScript code for the following request:\n{prompt}",
            "python" => $"Generate Python code for the following request:\n{prompt}",
            _ => prompt
        };
        
        return await _llamaService.GenerateCode(enhancedPrompt);
    }
    
    public async Task<string> ExplainCode(string code)
    {
        var prompt = $@"Explain the following code in detail:

```
{code}
```

Please explain what this code does, how it works, and any important concepts it uses.";
        
        return await _llamaService.GenerateCode(prompt);
    }
    
    public async Task<string> ImproveCode(string code)
    {
        var prompt = $@"Improve the following code by making it more efficient, readable, and following best practices:

```
{code}
```

Please provide the improved version with explanations of the changes made.";
        
        return await _llamaService.GenerateCode(prompt);
    }
}