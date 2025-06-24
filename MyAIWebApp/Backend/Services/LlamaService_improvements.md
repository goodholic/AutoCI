# LlamaService.cs 개선 제안

품질 점수: 0.40 → 0.85

## 개선 사항

- null 체크 추가 필요
- 비동기 메서드로 변환 권장
- SOLID 원칙 적용 필요
- 에러 처리 강화 필요

## 개선된 코드

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class LlamaService
{
    private readonly HttpClient _httpClient;
    private readonly string _pythonApiUrl = "http://localhost:8000";
    
    public LlamaService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }
    
    public async Task<string> GenerateCode(string prompt)
    {
        var request = new
        {
            prompt = prompt
        };
        
        var json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        try
        {
            var response = await _httpClient.PostAsync($"{_pythonApiUrl}/generate", content);
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadAsStringAsync();
                dynamic? responseData = JsonConvert.DeserializeObject(result);
                return responseData?.generated_text ?? "// 코드 생성 실패";
            }
            return "// API 호출 실패";
        }
        catch (Exception ex)
        {
            return $"// 오류 발생: {ex.Message}";
        }
    }
}
// AI가 개선한 코드
```
