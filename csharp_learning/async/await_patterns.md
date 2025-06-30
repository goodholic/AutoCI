
# Async/Await Patterns in Godot C#

## 기본 개념
Godot에서 C# async/await를 사용하여 비동기 작업을 처리합니다.

```csharp
public partial class Player : CharacterBody2D
{
    public async Task LoadResourcesAsync()
    {
        var texture = await LoadTextureAsync("res://player.png");
        GetNode<Sprite2D>("Sprite2D").Texture = texture;
    }
    
    private async Task<Texture2D> LoadTextureAsync(string path)
    {
        await Task.Delay(100); // 시뮬레이션
        return GD.Load<Texture2D>(path);
    }
}
```

## Godot 특화 패턴
- ToSignal() 사용하여 시그널 대기
- SceneTreeTimer로 비동기 타이머
- HTTP 요청의 비동기 처리
