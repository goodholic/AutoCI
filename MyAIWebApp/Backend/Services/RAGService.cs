// Backend/Services/RAGService.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

public class RAGService
{
    private readonly List<Document> _documents;
    private readonly Dictionary<string, Vector<float>> _embeddings;
    private readonly AIService _aiService;
    
    public RAGService(AIService aiService)
    {
        _aiService = aiService;
        _documents = new List<Document>();
        _embeddings = new Dictionary<string, Vector<float>>();
    }
    
    public async Task IndexDocument(Document document)
    {
        // 문서를 청크로 분할
        var chunks = SplitIntoChunks(document.Content, 500);
        
        foreach (var chunk in chunks)
        {
            var embedding = await GenerateEmbedding(chunk);
            var id = Guid.NewGuid().ToString();
            
            _documents.Add(new Document 
            { 
                Id = id, 
                Content = chunk,
                Title = document.Title
            });
            
            _embeddings[id] = embedding;
        }
    }
    
    public async Task<string> Query(string question)
    {
        // 질문에 대한 임베딩 생성
        var queryEmbedding = await GenerateEmbedding(question);
        
        // 가장 유사한 문서 찾기
        var relevantDocs = FindMostSimilarDocuments(queryEmbedding, 3);
        
        // 컨텍스트 생성
        var context = string.Join("\n\n", relevantDocs.Select(d => d.Content));
        
        // LLM에 컨텍스트와 함께 질문
        var prompt = $"다음 정보를 바탕으로 질문에 답하세요.\n\n컨텍스트:\n{context}\n\n질문: {question}\n\n답변:";
        
        return await _aiService.GenerateText(prompt);
    }
    
    private List<string> SplitIntoChunks(string text, int chunkSize)
    {
        var chunks = new List<string>();
        var words = text.Split(' ');
        var currentChunk = new List<string>();
        var currentSize = 0;
        
        foreach (var word in words)
        {
            if (currentSize + word.Length > chunkSize && currentChunk.Count > 0)
            {
                chunks.Add(string.Join(" ", currentChunk));
                currentChunk.Clear();
                currentSize = 0;
            }
            
            currentChunk.Add(word);
            currentSize += word.Length + 1;
        }
        
        if (currentChunk.Count > 0)
        {
            chunks.Add(string.Join(" ", currentChunk));
        }
        
        return chunks;
    }
    
    private async Task<Vector<float>> GenerateEmbedding(string text)
    {
        // 실제로는 임베딩 API를 사용해야 함
        // 여기서는 간단한 예시
        var random = new Random();
        var values = new float[384];
        
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)random.NextDouble();
        }
        
        return new Vector<float>(values);
    }
    
    private List<Document> FindMostSimilarDocuments(Vector<float> queryEmbedding, int topK)
    {
        var similarities = new Dictionary<string, float>();
        
        foreach (var kvp in _embeddings)
        {
            // Vector<float>의 Length() 대신 정규화된 코사인 유사도 계산
            var dotProduct = Vector.Dot(queryEmbedding, kvp.Value);
            var magnitude1 = Math.Sqrt(Vector.Dot(queryEmbedding, queryEmbedding));
            var magnitude2 = Math.Sqrt(Vector.Dot(kvp.Value, kvp.Value));
            var similarity = dotProduct / (float)(magnitude1 * magnitude2);
            
            similarities[kvp.Key] = similarity;
        }
        
        var topDocIds = similarities
            .OrderByDescending(x => x.Value)
            .Take(topK)
            .Select(x => x.Key);
        
        return _documents.Where(d => topDocIds.Contains(d.Id)).ToList();
    }
}

public class Document
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
}