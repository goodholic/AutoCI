using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

public class SearchService
{
    private readonly List<CodeDocument> _codeDocuments;
    private readonly string _indexPath = "code_index.json";
    
    public SearchService()
    {
        _codeDocuments = new List<CodeDocument>();
        LoadCodeIndex();
    }
    
    private void LoadCodeIndex()
    {
        if (File.Exists(_indexPath))
        {
            try
            {
                var json = File.ReadAllText(_indexPath);
                var documents = JsonSerializer.Deserialize<List<CodeDocument>>(json);
                if (documents != null)
                {
                    _codeDocuments.AddRange(documents);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"인덱스 로드 실패: {ex.Message}");
            }
        }
    }
    
    public async Task<List<SearchResult>> Search(string query, int maxResults = 10)
    {
        if (string.IsNullOrWhiteSpace(query))
            return new List<SearchResult>();
        
        var results = new List<SearchResult>();
        var queryLower = query.ToLower();
        var queryTokens = TokenizeCode(queryLower);
        
        foreach (var doc in _codeDocuments)
        {
            var contentLower = doc.Content.ToLower();
            var docTokens = TokenizeCode(contentLower);
            
            // 점수 계산
            var keywordScore = CalculateKeywordScore(queryTokens, docTokens, contentLower, queryLower);
            var tagScore = CalculateTagScore(queryLower, doc);
            var titleScore = doc.FileName.ToLower().Contains(queryLower) ? 0.3f : 0f;
            
            var totalScore = (keywordScore * 0.5f) + (tagScore * 0.3f) + (titleScore * 0.2f);
            
            if (totalScore > 0.1f) // 임계값
            {
                results.Add(new SearchResult
                {
                    Id = doc.Id,
                    Title = doc.FileName,
                    Content = GetCodeSnippet(doc.Content, query),
                    Score = totalScore,
                    Language = doc.Language,
                    Tags = doc.Tags
                });
            }
        }
        
        return await Task.FromResult(results
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList());
    }
    
    public async Task<bool> IndexCode(string fileName, string content, string language = "csharp")
    {
        try
        {
            var doc = new CodeDocument
            {
                Id = Guid.NewGuid().ToString(),
                FileName = fileName,
                Content = content,
                Language = language,
                Tags = ExtractTags(content)
            };
            
            _codeDocuments.Add(doc);
            
            // 인덱스 저장
            var json = JsonSerializer.Serialize(_codeDocuments, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_indexPath, json);
            
            return true;
        }
        catch
        {
            return false;
        }
    }
    
    private float CalculateKeywordScore(List<string> queryTokens, List<string> docTokens, string content, string query)
    {
        // 정확한 매칭
        if (content.Contains(query))
            return 1.0f;
        
        // 토큰 매칭
        var matches = queryTokens.Intersect(docTokens).Count();
        return (float)matches / Math.Max(queryTokens.Count, 1);
    }
    
    private float CalculateTagScore(string query, CodeDocument doc)
    {
        if (doc.Tags == null || doc.Tags.Length == 0)
            return 0;
        
        var matchingTags = doc.Tags.Count(tag => query.Contains(tag.ToLower()));
        return (float)matchingTags / doc.Tags.Length;
    }
    
    private string GetCodeSnippet(string content, string query, int contextLines = 3)
    {
        var lines = content.Split('\n');
        var queryLower = query.ToLower();
        
        // 일치하는 첫 번째 줄 찾기
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i].ToLower().Contains(queryLower))
            {
                var start = Math.Max(0, i - contextLines);
                var end = Math.Min(lines.Length - 1, i + contextLines);
                
                var snippet = string.Join("\n", lines.Skip(start).Take(end - start + 1));
                return snippet.Length > 500 ? snippet.Substring(0, 500) + "..." : snippet;
            }
        }
        
        // 일치하는 줄이 없으면 처음 몇 줄 반환
        var firstLines = string.Join("\n", lines.Take(Math.Min(5, lines.Length)));
        return firstLines.Length > 500 ? firstLines.Substring(0, 500) + "..." : firstLines;
    }
    
    private string[] ExtractTags(string content)
    {
        var tags = new List<string>();
        
        // 클래스명 추출
        var classMatches = Regex.Matches(content, @"class\s+(\w+)", RegexOptions.IgnoreCase);
        tags.AddRange(classMatches.Cast<Match>().Select(m => m.Groups[1].Value.ToLower()));
        
        // 메서드명 추출
        var methodMatches = Regex.Matches(content, @"(public|private|protected|internal)\s+\w+\s+(\w+)\s*\(", RegexOptions.IgnoreCase);
        tags.AddRange(methodMatches.Cast<Match>().Select(m => m.Groups[2].Value.ToLower()));
        
        // 공통 패턴 추출
        if (content.Contains("async")) tags.Add("async");
        if (content.Contains("Task")) tags.Add("task");
        if (content.Contains("Controller")) tags.Add("controller");
        if (content.Contains("Service")) tags.Add("service");
        if (content.Contains("Repository")) tags.Add("repository");
        if (content.Contains("using UnityEngine")) tags.Add("unity");
        
        return tags.Distinct().ToArray();
    }
    
    private List<string> TokenizeCode(string text)
    {
        // 코드 토큰화: 카멜케이스 분리, 특수문자 제거 등
        var tokens = new List<string>();
        
        // 카멜케이스 분리
        var camelCasePattern = @"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])";
        var words = Regex.Split(text, camelCasePattern);
        
        foreach (var word in words)
        {
            var cleanWord = Regex.Replace(word, @"[^a-zA-Z0-9]", " ");
            tokens.AddRange(cleanWord.Split(' ', StringSplitOptions.RemoveEmptyEntries));
        }
        
        return tokens.Where(t => t.Length > 2).Distinct().ToList();
    }
}

public class CodeDocument
{
    public string Id { get; set; } = string.Empty;
    public string FileName { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public string Language { get; set; } = "csharp";
    public string[]? Tags { get; set; }
}

public class SearchResult
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public float Score { get; set; }
    public string Language { get; set; } = string.Empty;
    public string[]? Tags { get; set; }
}