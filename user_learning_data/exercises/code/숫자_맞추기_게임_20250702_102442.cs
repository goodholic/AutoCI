using System;

namespace NumberGuessingGame
{
    class Program
    {
        static void Main(string[] args)
        {
            Random random = new Random();
            int targetNumber = random.Next(1, 101);
            int attempts = 0;
            int maxAttempts = 10;
            
            Console.WriteLine("숫자 맞추기 게임!");
            Console.WriteLine("1부터 100 사이의 숫자를 맞춰보세요.");
            Console.WriteLine($"기회는 {maxAttempts}번입니다.
");
            
            while (attempts < maxAttempts)
            {
                attempts++;
                Console.Write($"시도 {attempts}/{maxAttempts}: ");
                
                if (!int.TryParse(Console.ReadLine(), out int guess))
                {
                    Console.WriteLine("올바른 숫자를 입력하세요.");
                    attempts--;
                    continue;
                }
                
                if (guess < 1 || guess > 100)
                {
                    Console.WriteLine("1부터 100 사이의 숫자를 입력하세요.");
                    attempts--;
                    continue;
                }
                
                if (guess == targetNumber)
                {
                    Console.WriteLine($"
🎉 정답입니다! {attempts}번 만에 맞추셨습니다!");
                    break;
                }
                else if (guess < targetNumber)
                {
                    Console.WriteLine("더 큰 숫자입니다.");
                }
                else
                {
                    Console.WriteLine("더 작은 숫자입니다.");
                }
                
                if (attempts == maxAttempts)
                {
                    Console.WriteLine($"
😢 게임 오버! 정답은 {targetNumber}였습니다.");
                }
            }
            
            Console.WriteLine("
게임을 종료합니다.");
        }
    }
}