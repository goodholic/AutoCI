using System;

namespace Calculator
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("간단한 계산기 프로그램");
            
            while (true)
            {
                Console.WriteLine("
1. 덧셈  2. 뺄셈  3. 곱셈  4. 나눗셈  5. 종료");
                Console.Write("선택: ");
                
                if (!int.TryParse(Console.ReadLine(), out int choice))
                {
                    Console.WriteLine("잘못된 입력입니다.");
                    continue;
                }
                
                if (choice == 5) break;
                
                Console.Write("첫 번째 숫자: ");
                if (!double.TryParse(Console.ReadLine(), out double num1))
                {
                    Console.WriteLine("잘못된 숫자입니다.");
                    continue;
                }
                
                Console.Write("두 번째 숫자: ");
                if (!double.TryParse(Console.ReadLine(), out double num2))
                {
                    Console.WriteLine("잘못된 숫자입니다.");
                    continue;
                }
                
                double result = 0;
                switch (choice)
                {
                    case 1:
                        result = num1 + num2;
                        Console.WriteLine($"{num1} + {num2} = {result}");
                        break;
                    case 2:
                        result = num1 - num2;
                        Console.WriteLine($"{num1} - {num2} = {result}");
                        break;
                    case 3:
                        result = num1 * num2;
                        Console.WriteLine($"{num1} * {num2} = {result}");
                        break;
                    case 4:
                        if (num2 != 0)
                        {
                            result = num1 / num2;
                            Console.WriteLine($"{num1} / {num2} = {result}");
                        }
                        else
                        {
                            Console.WriteLine("0으로 나눌 수 없습니다.");
                        }
                        break;
                    default:
                        Console.WriteLine("잘못된 선택입니다.");
                        break;
                }
            }
            
            Console.WriteLine("계산기를 종료합니다.");
        }
    }
}