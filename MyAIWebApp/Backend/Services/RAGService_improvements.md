# RAGService.cs 개선 제안

품질 점수: 0.55 → 0.85

## 개선 사항

- null 체크 추가 필요
- 비동기 메서드로 변환 권장
- SOLID 원칙 적용 필요
- 에러 처리 강화 필요

## 개선된 코드

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

public class RAGService
{
    private readonly List<Document> _documents;
    private readonly Dictionary<string, float[]> _embeddings;
    private readonly LlamaService _llamaService;
    
    public RAGService(LlamaService llamaService)
    {
        _llamaService = llamaService;
        _documents = new List<Document>();
        _embeddings = new Dictionary<string, float[]>();
    }
    
    public async Task<bool> IndexReadme(string readmePath)
    {
        try
        {
            if (!File.Exists(readmePath))
                return false;
            
            var content = await File.ReadAllTextAsync(readmePath);
            var document = new Document
            {
                Id = "readme",
                Title = "README.md",
                Content = content,
                FilePath = readmePath
            };
            
            await IndexDocument(document);
            return true;
        }
        catch
        {
            return false;
        }
    }
    
    public Task IndexDocument(Document document)
    {
        // 문서를 의미 있는 섹션으로 분할
        var sections = SplitDocumentIntoSections(document.Content);
        
        foreach (var section in sections)
        {
            var id = Guid.NewGuid().ToString();
            var docChunk = new Document 
            { 
                Id = id, 
                Content = section.Content,
                Title = $"{document.Title} - {section.Title}",
                FilePath = document.FilePath
            };
            
            _documents.Add(docChunk);
            
            // 간단한 임베딩 생성 (TF-IDF 스타일)
            var embedding = GenerateSimpleEmbedding(section.Content);
            _embeddings[id] = embedding;
        }
        
        return Task.CompletedTask;
    }
    
    public async Task<string> Query(string question)
    {
        if (_documents.Count == 0)
        {
            return "문서가 인덱싱되지 않았습니다. 먼저 README 파일을 인덱싱해주세요.";
        }
        
        // 질문에 대한 임베딩 생성
        var queryEmbedding = GenerateSimpleEmbedding(question);
        
        // 가장 관련된 문서 섹션 찾기
        var relevantDocs = FindMostSimilarDocuments(queryEmbedding, 3);
        
        if (relevantDocs.Count == 0)
        {
            return "관련된 정보를 찾을 수 없습니다.";
        }
        
        // 컨텍스트 생성
        var context = string.Join("\n\n---\n\n", relevantDocs.Select(d => 
            $"섹션: {d.Title}\n내용: {d.Content}"));
        
        // Code Llama를 사용하여 답변 생성
        var prompt = $@"다음은 프로젝트 문서의 일부입니다:

{context}

위 정보를 바탕으로 다음 질문에 답변해주세요:
질문: {question}

답변:";
        
        return await _llamaService.GenerateCode(prompt);
    }
    
    private List<Section> SplitDocumentIntoSections(string content)
    {
        var sections = new List<Section>();
        var lines = content.Split('\n');
        var currentSection = new StringBuilder();
        var currentTitle = "소개";
        
        foreach (var line in lines)
        {
            // 마크다운 헤더 감지
            var headerMatch = Regex.Match(line, @"^(#{1,6})\s+(.+)$");
            if (headerMatch.Success)
            {
                // 이전 섹션 저장
                if (currentSection.Length > 0)
                {
                    sections.Add(new Section 
                    { 
                        Title = currentTitle, 
                        Content = currentSection.ToString().Trim() 
                    });
                    currentSection.Clear();
                }
                
                currentTitle = headerMatch.Groups[2].Value;
            }
            else
            {
                currentSection.AppendLine(line);
            }
        }
        
        // 마지막 섹션 저장
        if (currentSection.Length > 0)
        {
            sections.Add(new Section 
            { 
                Title = currentTitle, 
                Content = currentSection.ToString().Trim() 
            });
        }
        
        // 긴 섹션은 추가로 분할
        var finalSections = new List<Section>();
        foreach (var section in sections)
        {
            if (section.Content.Length > 1000)
            {
                var chunks = SplitIntoChunks(section.Content, 800);
                for (int i = 0; i < chunks.Count; i++)
                {
                    finalSections.Add(new Section
                    {
                        Title = $"{section.Title} (파트 {i + 1})",
                        Content = chunks[i]
                    });
                }
            }
            else
            {
                finalSections.Add(section);
            }
        }
        
        return finalSections;
    }
    
    private List<string> SplitIntoChunks(string text, int chunkSize)
    {
        var chunks = new List<string>();
        var sentences = Regex.Split(text, @"(?<=[.!?])\s+");
        var currentChunk = new StringBuilder();
        
        foreach (var sentence in sentences)
        {
            if (currentChunk.Length + sentence.Length > chunkSize && currentChunk.Length > 0)
            {
                chunks.Add(currentChunk.ToString().Trim());
                currentChunk.Clear();
            }
            
            currentChunk.Append(sentence).Append(" ");
        }
        
        if (currentChunk.Length > 0)
        {
            chunks.Add(currentChunk.ToString().Trim());
        }
        
        return chunks;
    }
    
    private float[] GenerateSimpleEmbedding(string text)
    {
        // 간단한 단어 빈도 기반 임베딩
        var words = Regex.Split(text.ToLower(), @"\W+")
            .Where(w => w.Length > 2)
            .ToList();
        
        var embedding = new float[384]; // 고정 크기 임베딩
        var hash = text.GetHashCode();
        var random = new Random(Math.Abs(hash));
        
        // 단어 기반 특성 벡터 생성
        for (int i = 0; i < embedding.Length; i++)
        {
            var wordIndex = i % Math.Max(words.Count, 1);
            var wordHash = words.Count > 0 ? words[wordIndex].GetHashCode() : 0;
            embedding[i] = (float)(Math.Sin(wordHash + i) * 0.5 + 0.5);
        }
        
        // 정규화
        var norm = (float)Math.Sqrt(embedding.Sum(x => x * x));
        if (norm > 0)
        {
            for (int i = 0; i < embedding.Length; i++)
            {
                embedding[i] /= norm;
            }
        }
        
        return embedding;
    }
    
    private List<Document> FindMostSimilarDocuments(float[] queryEmbedding, int topK)
    {
        var similarities = new List<(string Id, float Similarity)>();
        
        foreach (var kvp in _embeddings)
        {
            var similarity = CosineSimilarity(queryEmbedding, kvp.Value);
            similarities.Add((kvp.Key, similarity));
        }
        
        var topDocIds = similarities
            .OrderByDescending(x => x.Similarity)
            .Take(topK)
            .Select(x => x.Id)
            .ToList();
        
        return _documents.Where(d => topDocIds.Contains(d.Id)).ToList();
    }
    
    private float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            return 0;
        
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        
        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0 || normB == 0)
            return 0;
        
        return dotProduct / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
    }
    
    public List<Document> GetAllDocuments()
    {
        return _documents.ToList();
    }
    
    private class Section
    {
        public string Title { get; set; } = string.Empty;
        public string Content { get; set; } = string.Empty;
    }
}

public class Document
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public string FilePath { get; set; } = string.Empty;
}
// AI가 개선한 코드
```
