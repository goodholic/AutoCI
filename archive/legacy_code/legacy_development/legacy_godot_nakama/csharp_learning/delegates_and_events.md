
# Delegates and Events in Godot C#

## Godot 시그널 vs C# 이벤트

```csharp
public partial class Player : CharacterBody2D
{
    // C# 이벤트
    public delegate void HealthChangedEventHandler(int newHealth);
    public event HealthChangedEventHandler HealthChanged;
    
    // Godot 시그널
    [Signal]
    public delegate void DamagedEventHandler(int damage);
    
    private int _health = 100;
    
    public void TakeDamage(int damage)
    {
        _health -= damage;
        HealthChanged?.Invoke(_health);
        EmitSignal(SignalName.Damaged, damage);
    }
}
```

## 베스트 프랙티스
- Godot 시그널 선호 (에디터 통합)
- C# 이벤트는 내부 로직에 사용
