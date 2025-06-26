using System.Text.Json;
using System.Text.RegularExpressions;

namespace Backend.Services
{
    public class KnowledgeEntry
    {
        public string Id { get; set; } = "";
        public string Description { get; set; } = "";
        public string Code { get; set; } = "";
        public string Category { get; set; } = "";
        public string TemplateName { get; set; } = "";
        public List<string> Keywords { get; set; } = new();
        public int QualityScore { get; set; } = 80;
        public double RelevanceScore { get; set; } = 0;
    }

    public class LearningStatus
    {
        public bool IsLearning { get; set; }
        public string Message { get; set; } = "";
        public int Progress { get; set; }
        public string CurrentFile { get; set; } = "";
    }

    public class EnhancedRAGService
    {
        private readonly List<KnowledgeEntry> _knowledgeBase = new();
        private readonly Dictionary<string, int> _categories = new();
        private readonly Dictionary<string, int> _templates = new();
        private readonly ILogger<EnhancedRAGService> _logger;
        private LearningStatus _learningStatus = new();

        public EnhancedRAGService(ILogger<EnhancedRAGService> logger)
        {
            _logger = logger;
            LoadKnowledgeBase();
        }

        private void LoadKnowledgeBase()
        {
            _logger.LogInformation("🔍 Enhanced RAG 지식 베이스 로딩 중...");
            
            _learningStatus.IsLearning = true;
            _learningStatus.Message = "C# 코드 학습 중...";
            _learningStatus.Progress = 0;
            _learningStatus.CurrentFile = "";
            
            var dataDir = "../../expert_learning_data";
            if (!Directory.Exists(dataDir))
            {
                _logger.LogError($"❌ 데이터 디렉토리가 없습니다: {dataDir}");
                _learningStatus.IsLearning = false;
                _learningStatus.Message = "학습 데이터가 없습니다";
                return;
            }

            var jsonFiles = Directory.GetFiles(dataDir, "*.json");
            var validCount = 0;

            for (int fileIndex = 0; fileIndex < jsonFiles.Length; fileIndex++)
            {
                var filePath = jsonFiles[fileIndex];
                _learningStatus.CurrentFile = Path.GetFileName(filePath);
                _learningStatus.Progress = (int)((fileIndex + 1) * 100.0 / jsonFiles.Length);
                _learningStatus.Message = $"파일 분석 중 ({fileIndex + 1}/{jsonFiles.Length})";
                
                try
                {
                    var jsonContent = File.ReadAllText(filePath);
                    var data = JsonSerializer.Deserialize<JsonElement>(jsonContent);

                    var code = data.GetProperty("code").GetString()?.Trim() ?? "";
                    var category = data.GetProperty("category").GetString() ?? "general";
                    var templateName = data.TryGetProperty("template_name", out var template) ? template.GetString() ?? "" : "";
                    var qualityScore = data.TryGetProperty("quality_score", out var quality) ? quality.GetInt32() : 80;

                    if (code.Length > 100) // 유효한 코드만
                    {
                        var description = GetTemplateDescription(templateName);
                        var keywords = ExtractKeywords(code, templateName, category);

                        var entry = new KnowledgeEntry
                        {
                            Id = Path.GetFileName(filePath),
                            Description = description,
                            Code = code,
                            Category = category,
                            TemplateName = templateName,
                            Keywords = keywords,
                            QualityScore = qualityScore
                        };

                        _knowledgeBase.Add(entry);
                        validCount++;

                        // 카테고리별 통계
                        if (!_categories.ContainsKey(category))
                            _categories[category] = 0;
                        _categories[category]++;

                        // 템플릿별 통계
                        if (!string.IsNullOrEmpty(templateName))
                        {
                            if (!_templates.ContainsKey(templateName))
                                _templates[templateName] = 0;
                            _templates[templateName]++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError($"❌ 파일 로드 실패 {filePath}: {ex.Message}");
                }
            }

            _logger.LogInformation($"✅ Enhanced RAG 로드 완료: {validCount}/{jsonFiles.Length}개 유효한 항목");
            _logger.LogInformation($"📊 카테고리: {_categories.Count}개, 템플릿: {_templates.Count}개");
            
            _learningStatus.IsLearning = false;
            _learningStatus.Message = $"학습 완료: {validCount}개 패턴 학습";
            _learningStatus.Progress = 100;
        }

        private string GetTemplateDescription(string templateName)
        {
            var descriptions = new Dictionary<string, string>
            {
                ["async_command_pattern"] = "C# 비동기 Command 패턴 구현",
                ["repository_pattern"] = "Repository 패턴을 활용한 데이터 액세스",
                ["unity_object_pool"] = "Unity Object Pool 패턴으로 성능 최적화",
                ["memory_optimization"] = "C# 메모리 최적화 기법",
                ["advanced_async"] = "C# 고급 비동기 프로그래밍",
                ["performance_optimization"] = "C# 성능 최적화 패턴",
                ["unity_optimization"] = "Unity 게임 엔진 최적화",
                ["architecture_patterns"] = "C# 아키텍처 디자인 패턴",
                ["unity_advanced"] = "Unity 고급 개발 기법",
                ["implementation_patterns"] = "C# 구현 패턴 및 모범 사례"
            };

            return descriptions.GetValueOrDefault(templateName, $"{templateName} 패턴");
        }

        private List<string> ExtractKeywords(string code, string templateName, string category)
        {
            var keywords = new List<string>();

            // 템플릿명에서 키워드
            if (!string.IsNullOrEmpty(templateName))
                keywords.AddRange(templateName.Split('_'));

            // 카테고리에서 키워드
            keywords.AddRange(category.Split('_'));

            // 코드에서 C# 키워드 추출
            var patterns = new[]
            {
                @"\b(async|await|Task|public|private|protected|class|interface|struct)\b",
                @"\b(Unity|GameObject|MonoBehaviour|Component)\b",
                @"\b(List|Dictionary|Array|IEnumerable)\b",
                @"\b(Command|Handler|Repository|Service|Manager)\b"
            };

            foreach (var pattern in patterns)
            {
                var matches = Regex.Matches(code, pattern, RegexOptions.IgnoreCase);
                keywords.AddRange(matches.Select(m => m.Value.ToLower()));
            }

            // 중복 제거 및 정리
            return keywords
                .Where(k => !string.IsNullOrWhiteSpace(k) && k.Length > 2)
                .Distinct()
                .Take(10)
                .ToList();
        }

        public List<KnowledgeEntry> SearchRelevantCode(string query, int maxResults = 3)
        {
            if (!_knowledgeBase.Any()) return new List<KnowledgeEntry>();

            var queryLower = query.ToLower();
            var queryWords = queryLower.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var relevantCodes = new List<KnowledgeEntry>();

            foreach (var entry in _knowledgeBase)
            {
                double score = 0;

                // 1. 템플릿명 매칭 (높은 가중치)
                var templateName = entry.TemplateName.ToLower();
                if (queryWords.Any(word => templateName.Contains(word)))
                    score += 10;

                // 2. 카테고리 매칭
                var category = entry.Category.ToLower();
                if (queryWords.Any(word => category.Contains(word)))
                    score += 8;

                // 3. 설명 매칭
                var description = entry.Description.ToLower();
                if (queryWords.Any(word => description.Contains(word)))
                    score += 6;

                // 4. 키워드 매칭
                var keywords = entry.Keywords.Select(k => k.ToLower()).ToList();
                foreach (var word in queryWords)
                {
                    if (keywords.Any(keyword => keyword.Contains(word)))
                        score += 5;
                }

                // 5. 코드 내용 매칭
                var code = entry.Code.ToLower();
                foreach (var word in queryWords)
                {
                    if (code.Contains(word))
                        score += 3;
                }

                // 6. Unity 관련 특별 처리
                if (queryLower.Contains("unity"))
                {
                    if (templateName.Contains("unity") || category.Contains("unity") || code.Contains("unity"))
                        score += 15;
                }

                // 7. 비동기 관련 특별 처리
                if (queryWords.Any(word => new[] { "async", "await", "task" }.Contains(word)))
                {
                    if (templateName.Contains("async") || templateName.Contains("command"))
                        score += 12;
                }

                // 8. 패턴 관련 특별 처리
                if (queryLower.Contains("pattern") || queryLower.Contains("pool"))
                {
                    if (templateName.Contains("pattern") || templateName.Contains("pool"))
                        score += 10;
                }

                // 9. 품질 점수 반영
                var qualityBonus = entry.QualityScore / 20.0; // 80점 기준으로 정규화
                score += qualityBonus;

                if (score > 5) // 최소 임계값
                {
                    var entryCopy = new KnowledgeEntry
                    {
                        Id = entry.Id,
                        Description = entry.Description,
                        Code = entry.Code,
                        Category = entry.Category,
                        TemplateName = entry.TemplateName,
                        Keywords = entry.Keywords,
                        QualityScore = entry.QualityScore,
                        RelevanceScore = Math.Round(score, 2)
                    };
                    relevantCodes.Add(entryCopy);
                }
            }

            // 점수순으로 정렬하고 상위 결과만 반환
            return relevantCodes
                .OrderByDescending(x => x.RelevanceScore)
                .Take(maxResults)
                .ToList();
        }

        public string EnhancePrompt(string userQuery)
        {
            var relevantCodes = SearchRelevantCode(userQuery, 3);

            if (!relevantCodes.Any())
            {
                return $"{userQuery}\n\n(수집된 C# 지식 베이스에서 관련 예제를 찾지 못했습니다. 일반적인 C# 모범 사례를 사용해주세요.)";
            }

            var enhancedPrompt = $@"사용자 요청: {userQuery}

🔍 관련 C# 코드 예제들 (수집된 {_knowledgeBase.Count}개 지식 베이스에서 검색):

";

            for (int i = 0; i < relevantCodes.Count; i++)
            {
                var codeEntry = relevantCodes[i];
                var codePreview = codeEntry.Code.Length > 600 
                    ? codeEntry.Code.Substring(0, 600) + "..." 
                    : codeEntry.Code;

                enhancedPrompt += $@"
━━━ 예제 {i + 1} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂 카테고리: {codeEntry.Category}
🏷️ 패턴: {codeEntry.TemplateName}
📊 품질점수: {codeEntry.QualityScore}/100
🎯 관련도: {codeEntry.RelevanceScore}점
💡 설명: {codeEntry.Description}

🔧 코드:
{codePreview}

";
            }

            enhancedPrompt += $@"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 요청사항:
위의 관련 예제들을 참고하여 사용자 요청에 맞는 고품질 C# 코드를 생성해주세요.

🎯 중점사항:
• Unity 최적화 및 성능 패턴 적극 활용
• 모던 C# 기법 (async/await, LINQ, Pattern Matching 등) 사용  
• 메모리 효율성과 가독성 고려
• 실제 프로덕션에서 사용 가능한 수준의 코드
• 관련 예제의 구조와 패턴을 참고하되, 사용자 요청에 맞게 customization

🚀 활용 가능한 전체 지식: {_knowledgeBase.Count}개 C# 전문 예제";

            return enhancedPrompt;
        }

        public object GetSystemStatus()
        {
            return new
            {
                TotalKnowledge = _knowledgeBase.Count,
                Categories = _categories.Count,
                Templates = _templates.Count,
                TopCategories = _categories.OrderByDescending(x => x.Value).Take(5).Select(x => new { x.Key, x.Value }),
                TopTemplates = _templates.OrderByDescending(x => x.Value).Take(5).Select(x => new { x.Key, x.Value }),
                RAGEnabled = _knowledgeBase.Any()
            };
        }
        
        public LearningStatus GetLearningStatus()
        {
            return _learningStatus;
        }
    }
} 