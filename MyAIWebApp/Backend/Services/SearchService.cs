// Backend/Services/SearchService.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

public class SearchService
{
    private MLContext _mlContext;
    private ITransformer? _model;
    
    public SearchService()
    {
        _mlContext = new MLContext();
        LoadOrTrainModel();
    }
    
    private void LoadOrTrainModel()
    {
        // 텍스트 분류 모델 학습
        var data = _mlContext.Data.LoadFromTextFile<SearchData>(
            "search_data.csv", 
            hasHeader: true, 
            separatorChar: ','
        );
        
        var pipeline = _mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features", 
                inputColumnName: nameof(SearchData.Query))
            .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label"))
            .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        
        _model = pipeline.Fit(data);
    }
    
    public Task<List<SearchResult>> Search(string query)
    {
        if (_model == null)
        {
            return Task.FromResult(new List<SearchResult>());
        }
        
        var predictionEngine = _mlContext.Model.CreatePredictionEngine<SearchData, SearchPrediction>(_model);
        
        var input = new SearchData { Query = query };
        var prediction = predictionEngine.Predict(input);
        
        // 예측 결과를 기반으로 검색 수행
        var results = new List<SearchResult>();
        
        // 실제 검색 로직 구현
        results.Add(new SearchResult 
        { 
            Title = "검색 결과 1",
            Content = "관련 내용...",
            Score = prediction.Score?.Max() ?? 0
        });
        
        return Task.FromResult(results);
    }
}

public class SearchData
{
    [LoadColumn(0)]
    public string Query { get; set; } = string.Empty;
    
    [LoadColumn(1)]
    public string Category { get; set; } = string.Empty;
}

public class SearchPrediction
{
    public string PredictedLabel { get; set; } = string.Empty;
    public float[]? Score { get; set; }
}

public class SearchResult
{
    public string Title { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public float Score { get; set; }
}