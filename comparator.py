"""
Модуль для сравнения разных версий промптов.

Это ключевая фича фреймворка — она позволяет:
1. Сравнить две версии промпта (v1 vs v2)
2. Понять, какая версия лучше
3. Найти конкретные улучшения/ухудшения

Аналогия с JS:
    Это как если бы ты написал функцию, которая запускает
    два разных алгоритма на одних и тех же данных и сравнивает результаты.
"""

import pandas as pd
from typing import Dict, List, Tuple
from .tester import PromptTester


class PromptComparator:
    """
    Класс для сравнения версий промптов.
    
    Пример использования:
        comparator = PromptComparator()
        comparison = comparator.compare(
            prompt_v1="Ты консультант...",
            prompt_v2="Ты эксперт-консультант...",
            test_cases_path="examples/test_cases.csv"
        )
    """
    
    def __init__(self):
        """Инициализация компаратора."""
        self.test_cases_path = None
        self.results_v1 = None
        self.results_v2 = None
    
    def _load_prompt(self, version: str) -> str:
        """
        Загружает промпт из файла.
        
        Args:
            version: Версия промпта (например, "v1" или "v2")
        
        Returns:
            Текст промпта
        """
        file_path = f"examples/prompts/{version}/system_prompt.txt"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Файл не найден: {file_path}\n"
                f"Создай папку examples/prompts/{version}/ и добавь system_prompt.txt"
            )
    
    def _run_tests_for_version(self, prompt: str, version_name: str) -> pd.DataFrame:
        """
        Запускает тесты для конкретной версии промпта.
        
        Args:
            prompt: Текст промпта
            version_name: Название версии (для логов)
        
        Returns:
            DataFrame с результатами тестов
        """
        print(f"\n{'='*60}")
        print(f"🧪 Тестирование версии: {version_name}")
        print(f"{'='*60}")
        
        # Создаем тестер для DeepSeek
        tester = PromptTester(use_mock=False)
        
        # Запускаем тесты с DeepSeek
        results = tester.run_all_tests(
            self.test_cases_path,
            system_prompt=prompt,
            use_deepseek=True
        )
        
        return results
    
    def compare(self, version1: str, version2: str, 
                test_cases_path: str = "examples/test_cases.csv") -> Dict:
        """
        Сравнивает две версии промпта.
        
        Args:
            version1: Первая версия (например, "v1")
            version2: Вторая версия (например, "v2")
            test_cases_path: Путь к CSV с тестами
        
        Returns:
            Словарь с результатами сравнения
        """
        self.test_cases_path = test_cases_path
        
        # Загружаем промпты
        print("\n📄 Загрузка промптов...")
        prompt_v1 = self._load_prompt(version1)
        prompt_v2 = self._load_prompt(version2)
        print(f"✅ v1: {len(prompt_v1)} символов")
        print(f"✅ v2: {len(prompt_v2)} символов")
        
        # Запускаем тесты для v1
        self.results_v1 = self._run_tests_for_version(prompt_v1, version1)
        
        # Запускаем тесты для v2
        self.results_v2 = self._run_tests_for_version(prompt_v2, version2)
        
        # Считаем статистику для каждой версии
        stats_v1 = self._calculate_stats(self.results_v1)
        stats_v2 = self._calculate_stats(self.results_v2)
        
        # Сравниваем по каждому тесту
        comparison_by_test = self._compare_by_test(self.results_v1, self.results_v2)
        
        # Определяем победителя
        winner = self._determine_winner(stats_v1, stats_v2)
        
        return {
            "version1": version1,
            "version2": version2,
            "stats_v1": stats_v1,
            "stats_v2": stats_v2,
            "comparison_by_test": comparison_by_test,
            "winner": winner,
            "results_v1": self.results_v1,
            "results_v2": self.results_v2
        }
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """
        Считает статистику по результатам тестов.
        
        Аналогия с JS:
            function calculateStats(results) {
                const total = results.length;
                const passed = results.filter(r => r.passed).length;
                const avgSimilarity = results.reduce((sum, r) => sum + r.similarity, 0) / total;
                return { total, passed, avgSimilarity, ... };
            }
        """
        total = len(df)
        passed = int(df["passed"].sum())
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "avg_similarity": round(df["similarity"].mean(), 3),
            "avg_coverage": round(df["keyword_coverage"].mean(), 3),
            "avg_latency": round(df["latency_sec"].mean(), 3),
            "total_latency": round(df["latency_sec"].sum(), 3)
        }
    
    def _compare_by_test(self, df_v1: pd.DataFrame, df_v2: pd.DataFrame) -> List[Dict]:
        """
        Сравнивает результаты по каждому тесту.
        
        Returns:
            Список словарей с сравнением по каждому тесту
        """
        comparison = []
        
        # Проходим по всем тестам (предполагаем, что они в одном порядке)
        for i in range(len(df_v1)):
            row_v1 = df_v1.iloc[i]
            row_v2 = df_v2.iloc[i]
            
            # Считаем разницу в метриках
            similarity_diff = row_v2["similarity"] - row_v1["similarity"]
            coverage_diff = row_v2["keyword_coverage"] - row_v1["keyword_coverage"]
            latency_diff = row_v2["latency_sec"] - row_v1["latency_sec"]
            
            # Определяем, улучшился ли тест
            improved = similarity_diff > 0.05 or coverage_diff > 0.1
            degraded = similarity_diff < -0.05 or coverage_diff < -0.1
            
            if improved:
                status = "📈 improved"
            elif degraded:
                status = "📉 degraded"
            else:
                status = "➡️ same"
            
            comparison.append({
                "id": row_v1["id"],
                "question": row_v1["question"],
                "v1_similarity": row_v1["similarity"],
                "v2_similarity": row_v2["similarity"],
                "similarity_diff": round(similarity_diff, 3),
                "v1_coverage": row_v1["keyword_coverage"],
                "v2_coverage": row_v2["keyword_coverage"],
                "coverage_diff": round(coverage_diff, 3),
                "v1_latency": row_v1["latency_sec"],
                "v2_latency": row_v2["latency_sec"],
                "latency_diff": round(latency_diff, 3),
                "status": status
            })
        
        return comparison
    
    def _determine_winner(self, stats_v1: Dict, stats_v2: Dict) -> Dict:
        """
        Определяет победителя на основе метрик.
        
        Логика:
        - Главная метрика: pass_rate (процент пройденных тестов)
        - Вторичная: avg_similarity (качество ответов)
        - Третичная: avg_coverage (полнота ответов)
        """
        # Считаем "очки" для каждой версии
        score_v1 = (
            stats_v1["pass_rate"] * 2 +           # x2 вес
            stats_v1["avg_similarity"] * 100 +    # переводим в проценты
            stats_v1["avg_coverage"] * 100
        )
        
        score_v2 = (
            stats_v2["pass_rate"] * 2 +
            stats_v2["avg_similarity"] * 100 +
            stats_v2["avg_coverage"] * 100
        )
        
        if score_v2 > score_v1:
            winner = "v2"
            improvement = round(score_v2 - score_v1, 2)
        elif score_v1 > score_v2:
            winner = "v1"
            improvement = round(score_v1 - score_v2, 2)
        else:
            winner = "tie"
            improvement = 0
        
        return {
            "winner": winner,
            "score_v1": round(score_v1, 2),
            "score_v2": round(score_v2, 2),
            "improvement": improvement
        }
    
    def print_summary(self, comparison: Dict):
        """
        Выводит красивое резюме сравнения в консоль.
        """
        stats_v1 = comparison["stats_v1"]
        stats_v2 = comparison["stats_v2"]
        winner = comparison["winner"]
        
        print("\n" + "="*60)
        print("📊 СРАВНЕНИЕ ВЕРСИЙ")
        print("="*60)
        
        # Заголовки колонок
        print(f"\n{'Метрика':<25} {comparison['version1']:<15} {comparison['version2']:<15} {'Δ':<10}")
        print("-"*65)
        
        # Pass Rate
        diff = stats_v2["pass_rate"] - stats_v1["pass_rate"]
        emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"{'Pass Rate':<25} {stats_v1['pass_rate']:<15.1f} {stats_v2['pass_rate']:<15.1f} {emoji} {diff:+.1f}")
        
        # Avg Similarity
        diff = stats_v2["avg_similarity"] - stats_v1["avg_similarity"]
        emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"{'Avg Similarity':<25} {stats_v1['avg_similarity']:<15.3f} {stats_v2['avg_similarity']:<15.3f} {emoji} {diff:+.3f}")
        
        # Avg Coverage
        diff = stats_v2["avg_coverage"] - stats_v1["avg_coverage"]
        emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"{'Avg Coverage':<25} {stats_v1['avg_coverage']:<15.3f} {stats_v2['avg_coverage']:<15.3f} {emoji} {diff:+.3f}")
        
        # Avg Latency
        diff = stats_v2["avg_latency"] - stats_v1["avg_latency"]
        emoji = "🐢" if diff > 0 else "🐇" if diff < 0 else "➡️"
        print(f"{'Avg Latency':<25} {stats_v1['avg_latency']:<15.3f}s {stats_v2['avg_latency']:<14.3f}s {emoji} {diff:+.3f}s")
        
        # Победитель
        print("\n" + "="*60)
        if winner["winner"] == "v2":
            print(f"🏆 ПОБЕДИТЕЛЬ: {comparison['version2']} (улучшение на {winner['improvement']} очков)")
        elif winner["winner"] == "v1":
            print(f"🏆 ПОБЕДИТЕЛЬ: {comparison['version1']} (улучшение на {winner['improvement']} очков)")
        else:
            print(f"🤝 НИЧЬЯ: версии показывают одинаковый результат")
        print("="*60)
        
        # Детали по тестам
        print("\n📋 ДЕТАЛИ ПО ТЕСТАМ:")
        print("-"*60)
        
        improved = [t for t in comparison["comparison_by_test"] if "improved" in t["status"]]
        degraded = [t for t in comparison["comparison_by_test"] if "degraded" in t["status"]]
        same = [t for t in comparison["comparison_by_test"] if "same" in t["status"]]
        
        if improved:
            print(f"\n📈 Улучшилось ({len(improved)}):")
            for t in improved:
                print(f"   #{t['id']}: similarity {t['v1_similarity']:.2f} → {t['v2_similarity']:.2f} (+{t['similarity_diff']:.2f})")
        
        if degraded:
            print(f"\n📉 Ухудшилось ({len(degraded)}):")
            for t in degraded:
                print(f"   #{t['id']}: similarity {t['v1_similarity']:.2f} → {t['v2_similarity']:.2f} ({t['similarity_diff']:.2f})")
        
        if same:
            print(f"\n➡️ Без изменений ({len(same)})")