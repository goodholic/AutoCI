using Backend.Services;
using Microsoft.OpenApi.Models;
using MyAIWebApp.Backend;
using MyAIWebApp.Backend.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Add SignalR
builder.Services.AddSignalR();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowBlazorClient",
        builder => builder
            .WithOrigins("https://localhost:7100", "http://localhost:5100") // Blazor client URLs
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials());
});

// Register services
builder.Services.AddHttpClient<LlamaService>();
builder.Services.AddScoped<RAGService>();
builder.Services.AddSingleton<EnhancedRAGService>();
builder.Services.AddScoped<AIService>();
builder.Services.AddScoped<SearchService>();
builder.Services.AddSingleton<KoreanCommandService>();

// Register AutoCI monitoring service
builder.Services.AddHostedService<AutoCIMonitorService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseCors("AllowBlazorClient");
app.MapControllers();

// Map SignalR hub
app.MapHub<AutoCIHub>("/hubs/autoci");

app.Run();
