// Backend/Services/AIService.cs
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class AIService
{
    private readonly HttpClient _httpClient;
    private readonly string _huggingFaceApiKey = "YOUR_API_KEY";
    
    public AIService(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }
    
    public async Task<string> GenerateText(string prompt)
    {
        var apiUrl = "https://api-inference.huggingface.co/models/gpt2";
        
        var request = new
        {
            inputs = prompt,
            parameters = new
            {
                max_length = 100,
                temperature = 0.7
            }
        };
        
        var json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        _httpClient.DefaultRequestHeaders.Authorization = 
            new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _huggingFaceApiKey);
        
        var response = await _httpClient.PostAsync(apiUrl, content);
        var result = await response.Content.ReadAsStringAsync();
        
        return result;
    }
}