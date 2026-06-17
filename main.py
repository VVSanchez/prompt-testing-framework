"""
Точка входа для тестирования промптов.

Команды:
    python main.py                      # Mock режим
    python main.py --deepseek           # Тестирование с DeepSeek (v1)
    python main.py --deepseek --v2      # Тестирование с DeepSeek (v2)
    python main.py --compare v1 v2      # Сравнение двух версий
"""

import sys
from src.tester import PromptTester
from src.reporters import ReportGenerator
from src.comparator import PromptComparator


def print_usage():
    """Выводит справку по использованию."""
    print("""
🧪 Universal Prompt Testing Framework

Использование:
  python main.py                          Mock режим (без API)
  python main.py --deepseek               Тестирование v1 через DeepSeek
  python main.py --deepseek --v2          Тестирование v2 через DeepSeek
  python main.py --compare v1 v2          Сравнение двух версий

Примеры:
  python main.py --compare v1 v2
    → Запустит тесты для v1 и v2, сравнит результаты, покажет победителя
""")


def run_single_version(version: str):
    """Запускает тесты для одной версии промпта."""
    print("\n" + "="*60)
    print(f"🧪 Тестирование версии: {version}")
    print("="*60)
    
    # Загружаем промпт
    try:
        with open(f"examples/prompts/{version}/system_prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"📄 Загружен промпт: {len(system_prompt)} символов")
    except FileNotFoundError:
        print(f"❌ Файл не найден: examples/prompts/{version}/system_prompt.txt")
        return
    
    # Создаем тестер
    tester = PromptTester(use_mock=False)
    
    # Запускаем тесты
    results_df = tester.run_all_tests(
        "examples/test_cases.csv",
        system_prompt=system_prompt,
        use_deepseek=True
    )
    
    # Статистика
    stats = tester.get_statistics(results_df)
    
    print("\n" + "="*60)
    print(f"📊 СТАТИСТИКА ДЛЯ {version.upper()}")
    print("="*60)
    print(f"✅ Пройдено: {stats['passed']} из {stats['total']} ({stats['pass_rate']}%)")
    print(f"📈 Среднее сходство: {stats['avg_similarity']}")
    print(f"🎯 Среднее покрытие: {stats['avg_coverage']}")
    print(f"⚡ Средняя задержка: {stats['avg_latency']}s")
    
    # Отчеты
    reporter = ReportGenerator(output_dir="reports")
    paths = reporter.generate_all(results_df, stats)
    
    print(f"\n🎉 Отчеты сохранены:")
    print(f"   {paths['html']}")


def run_comparison(version1: str, version2: str):
    """Запускает сравнение двух версий."""
    print("\n" + "="*60)
    print(f"🔬 СРАВНЕНИЕ: {version1} vs {version2}")
    print("="*60)
    
    # Создаем компаратор
    comparator = PromptComparator()
    
    # Запускаем сравнение
    comparison = comparator.compare(
        version1=version1,
        version2=version2,
        test_cases_path="examples/test_cases.csv"
    )
    
    # Выводим резюме
    comparator.print_summary(comparison)
    
    # Генерируем HTML отчет
    reporter = ReportGenerator(output_dir="reports")
    html_path = reporter.save_comparison_html(comparison)
    
    print("\n" + "="*60)
    print("🎉 Готово! Открой сравнительный отчет:")
    print(f"   {html_path}")
    print("="*60 + "\n")


def run_mock():
    """Запускает тесты в mock режиме."""
    print("\n" + "="*60)
    print("🧪 Universal Prompt Testing Framework (MOCK)")
    print("="*60)
    
    tester = PromptTester(use_mock=True)
    results_df = tester.run_all_tests("examples/test_cases.csv")
    
    stats = tester.get_statistics(results_df)
    
    print("\n" + "="*60)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)
    print(f"✅ Пройдено: {stats['passed']} из {stats['total']} ({stats['pass_rate']}%)")
    print(f"📈 Среднее сходство: {stats['avg_similarity']}")
    print(f"🎯 Среднее покрытие: {stats['avg_coverage']}")
    
    reporter = ReportGenerator(output_dir="reports")
    paths = reporter.generate_all(results_df, stats)
    
    print(f"\n🎉 Отчеты: {paths['html']}")


def main():
    """Главная функция — роутер команд."""
    args = sys.argv[1:]
    
    # Если нет аргументов — показываем справку
    if not args:
        run_mock()
        return
    
    # Команда: сравнение версий
    if "--compare" in args:
        idx = args.index("--compare")
        if len(args) >= idx + 3:
            version1 = args[idx + 1]
            version2 = args[idx + 2]
            run_comparison(version1, version2)
        else:
            print("❌ Укажи две версии: python main.py --compare v1 v2")
        return
    
    # Команда: тестирование с DeepSeek
    if "--deepseek" in args:
        version = "v2" if "--v2" in args else "v1"
        run_single_version(version)
        return
    
    # Неизвестная команда
    print(f"❌ Неизвестная команда: {' '.join(args)}")
    print_usage()


if __name__ == "__main__":
    main()