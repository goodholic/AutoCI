using System;
using System.Collections.Generic;

namespace GameCharacterSystem
{
    // 캐릭터 인터페이스
    public interface ICharacter
    {
        string Name { get; }
        int Level { get; }
        void Attack(ICharacter target);
        void TakeDamage(int damage);
        bool IsAlive { get; }
    }
    
    // 스킬 인터페이스
    public interface ISkill
    {
        string Name { get; }
        int ManaCost { get; }
        void Use(Character caster, ICharacter target);
    }
    
    // 기본 캐릭터 클래스
    public abstract class Character : ICharacter
    {
        public string Name { get; protected set; }
        public int Level { get; protected set; }
        public int Health { get; protected set; }
        public int MaxHealth { get; protected set; }
        public int Mana { get; protected set; }
        public int MaxMana { get; protected set; }
        public int AttackPower { get; protected set; }
        public int Defense { get; protected set; }
        
        public bool IsAlive => Health > 0;
        
        protected List<ISkill> skills = new List<ISkill>();
        
        public Character(string name, int level)
        {
            Name = name;
            Level = level;
            InitializeStats();
        }
        
        protected abstract void InitializeStats();
        
        public virtual void Attack(ICharacter target)
        {
            Console.WriteLine($"{Name}이(가) {target.Name}을(를) 공격합니다!");
            int damage = AttackPower;
            target.TakeDamage(damage);
        }
        
        public virtual void TakeDamage(int damage)
        {
            int actualDamage = Math.Max(damage - Defense, 0);
            Health -= actualDamage;
            Console.WriteLine($"{Name}이(가) {actualDamage}의 피해를 입었습니다! (남은 HP: {Health}/{MaxHealth})");
            
            if (!IsAlive)
            {
                Console.WriteLine($"{Name}이(가) 쓰러졌습니다!");
            }
        }
        
        public void UseSkill(int skillIndex, ICharacter target)
        {
            if (skillIndex < 0 || skillIndex >= skills.Count)
            {
                Console.WriteLine("잘못된 스킬 번호입니다.");
                return;
            }
            
            var skill = skills[skillIndex];
            if (Mana >= skill.ManaCost)
            {
                skill.Use(this, target);
                Mana -= skill.ManaCost;
            }
            else
            {
                Console.WriteLine($"마나가 부족합니다! (필요: {skill.ManaCost}, 현재: {Mana})");
            }
        }
    }
    
    // 전사 클래스
    public class Warrior : Character
    {
        public Warrior(string name, int level) : base(name, level)
        {
            skills.Add(new PowerStrike());
            skills.Add(new ShieldBash());
        }
        
        protected override void InitializeStats()
        {
            MaxHealth = 100 + (Level * 20);
            Health = MaxHealth;
            MaxMana = 50 + (Level * 5);
            Mana = MaxMana;
            AttackPower = 15 + (Level * 3);
            Defense = 10 + (Level * 2);
        }
    }
    
    // 마법사 클래스
    public class Mage : Character
    {
        public Mage(string name, int level) : base(name, level)
        {
            skills.Add(new Fireball());
            skills.Add(new FrostBolt());
        }
        
        protected override void InitializeStats()
        {
            MaxHealth = 60 + (Level * 10);
            Health = MaxHealth;
            MaxMana = 100 + (Level * 15);
            Mana = MaxMana;
            AttackPower = 10 + (Level * 2);
            Defense = 5 + Level;
        }
    }
    
    // 스킬 구현
    public class PowerStrike : ISkill
    {
        public string Name => "파워 스트라이크";
        public int ManaCost => 10;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.AttackPower * 2;
            target.TakeDamage(damage);
        }
    }
    
    public class ShieldBash : ISkill
    {
        public string Name => "방패 강타";
        public int ManaCost => 15;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.Defense + 10;
            target.TakeDamage(damage);
        }
    }
    
    public class Fireball : ISkill
    {
        public string Name => "화염구";
        public int ManaCost => 20;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.Level * 10 + 20;
            target.TakeDamage(damage);
        }
    }
    
    public class FrostBolt : ISkill
    {
        public string Name => "서리 화살";
        public int ManaCost => 15;
        
        public void Use(Character caster, ICharacter target)
        {
            Console.WriteLine($"{caster.Name}이(가) {Name}를 사용합니다!");
            int damage = caster.Level * 8 + 15;
            target.TakeDamage(damage);
        }
    }
    
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== 게임 캐릭터 전투 시스템 ===");
            
            var warrior = new Warrior("전사", 5);
            var mage = new Mage("마법사", 5);
            
            Console.WriteLine($"
{warrior.Name} (레벨 {warrior.Level}) vs {mage.Name} (레벨 {mage.Level})");
            Console.WriteLine($"{warrior.Name}: HP {warrior.Health}/{warrior.MaxHealth}, MP {warrior.Mana}/{warrior.MaxMana}");
            Console.WriteLine($"{mage.Name}: HP {mage.Health}/{mage.MaxHealth}, MP {mage.Mana}/{mage.MaxMana}");
            
            // 전투 시뮬레이션
            Console.WriteLine("
=== 전투 시작! ===");
            
            // 전사의 턴
            Console.WriteLine("
[전사의 턴]");
            warrior.Attack(mage);
            warrior.UseSkill(0, mage); // 파워 스트라이크
            
            if (mage.IsAlive)
            {
                // 마법사의 턴
                Console.WriteLine("
[마법사의 턴]");
                mage.Attack(warrior);
                mage.UseSkill(0, warrior); // 화염구
            }
            
            // 결과
            Console.WriteLine("
=== 전투 결과 ===");
            Console.WriteLine($"{warrior.Name}: HP {warrior.Health}/{warrior.MaxHealth}");
            Console.WriteLine($"{mage.Name}: HP {mage.Health}/{mage.MaxHealth}");
        }
    }
}