using Godot;

public partial class Player : Node2D
{
    private float _speed = 300.0f;
    
    public override void _Ready()
    {
        GD.Print("Player ready!");
    }
    
    public override void _Process(double delta)
    {
        Vector2 velocity = Vector2.Zero;
        
        if (Input.IsActionPressed("ui_right"))
            velocity.X += 1;
        if (Input.IsActionPressed("ui_left"))
            velocity.X -= 1;
        if (Input.IsActionPressed("ui_down"))
            velocity.Y += 1;
        if (Input.IsActionPressed("ui_up"))
            velocity.Y -= 1;
        
        if (velocity.Length() > 0)
        {
            velocity = velocity.Normalized() * _speed;
            Position += velocity * (float)delta;
        }
    }
}