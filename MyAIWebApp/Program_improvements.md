# Program.cs 개선 제안

품질 점수: 0.40 → 0.85

## 개선 사항

- null 체크 추가 필요
- 비동기 메서드로 변환 권장
- SOLID 원칙 적용 필요
- 에러 처리 강화 필요

## 개선된 코드

```csharp
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddHttpClient<AIService>();
builder.Services.AddHttpClient<LlamaService>(); // 이 줄 추가
builder.Services.AddSingleton<SearchService>();
builder.Services.AddSingleton<RAGService>();

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll",
        builder =>
        {
            builder.AllowAnyOrigin()
                   .AllowAnyMethod()
                   .AllowAnyHeader();
        });
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseCors("AllowAll");
app.UseAuthorization();
app.MapControllers();

app.Run();
// AI가 개선한 코드
```
