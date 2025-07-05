
# LINQ in Godot C#

## 기본 LINQ 사용법
Godot에서 LINQ를 사용하여 노드 및 데이터 처리:

```csharp
public partial class GameManager : Node
{
    public void ProcessEnemies()
    {
        var enemies = GetTree().GetNodesInGroup("enemies")
            .Cast<Enemy>()
            .Where(e => e.Health > 0)
            .OrderByDescending(e => e.Threat)
            .Take(5);
            
        foreach (var enemy in enemies)
        {
            enemy.UpdateAI();
        }
    }
}
```

## Godot 특화 패턴
- GetNodesInGroup()과 LINQ 결합
- 씨스템 컬렉션 처리
- 성능 최적화 고려사항
