"""
Модуль для генерации отчетов о тестировании.

Создает красивые HTML и CSV отчеты с результатами.
"""

import pandas as pd
from datetime import datetime
import os
from typing import Dict


class ReportGenerator:
    """Генератор отчетов."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Args:
            output_dir: Папка для сохранения отчетов
        """
        self.output_dir = output_dir
        # Создаем папку, если её нет
        os.makedirs(output_dir, exist_ok=True)
    
    def save_csv(self, df: pd.DataFrame, stats: Dict) -> str:
        """
        Сохраняет результаты в CSV.
        
        Returns:
            Путь к сохраненному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.output_dir, f"test_report_{timestamp}.csv")
        
        # Сохраняем в CSV с BOM для корректного открытия в Excel
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"💾 CSV сохранен: {csv_path}")
        return csv_path
    
    def save_html(self, df: pd.DataFrame, stats: Dict) -> str:
        """
        Создает красивый HTML отчет.
        
        Returns:
            Путь к сохраненному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.output_dir, f"test_report_{timestamp}.html")
        
        # Генерируем строки таблицы
        table_rows = ""
        for _, row in df.iterrows():
            status_class = "passed" if row["passed"] else "failed"
            status_emoji = "✅" if row["passed"] else "❌"
            
            # Обрезаем длинные тексты для таблицы
            question_preview = str(row['question'])[:80]
            expected_preview = str(row['expected'])[:100]
            response_preview = str(row['response'])[:100]
            
            table_rows += f"""
                <tr>
                    <td>{row['id']}</td>
                    <td>{question_preview}</td>
                    <td class="expected">{expected_preview}</td>
                    <td class="response">{response_preview}</td>
                    <td>{row['similarity']:.2f}</td>
                    <td>{row['keyword_coverage']:.2f}</td>
                    <td>{row['latency_sec']}s</td>
                    <td class="{status_class}">{status_emoji}</td>
                </tr>
            """
        
        # Полный HTML документ
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt Test Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c5aa0;
            margin-bottom: 10px;
        }}
        .timestamp {{
            color: #666;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
        }}
        .metric {{
            flex: 1;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c5aa0;
        }}
        .metric-value.success {{ color: #27ae60; }}
        .metric-value.danger {{ color: #e74c3c; }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #2c5aa0;
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .expected, .response {{
            font-size: 13px;
            color: #555;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .passed {{
            color: #27ae60;
            font-weight: bold;
            font-size: 18px;
        }}
        .failed {{
            color: #e74c3c;
            font-weight: bold;
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Prompt Test Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-value">{stats['total']}</div>
                <div class="metric-label">Всего тестов</div>
            </div>
            <div class="metric">
                <div class="metric-value success">{stats['passed']}</div>
                <div class="metric-label">Пройдено</div>
            </div>
            <div class="metric">
                <div class="metric-value danger">{stats['failed']}</div>
                <div class="metric-label">Провалено</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats['pass_rate']}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats['avg_similarity']}</div>
                <div class="metric-label">Avg Similarity</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats['avg_coverage']}</div>
                <div class="metric-label">Avg Coverage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats['avg_latency']}s</div>
                <div class="metric-label">Avg Latency</div>
            </div>
        </div>
        
        <h2>Результаты тестов</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Вопрос</th>
                    <th>Ожидаемый ответ</th>
                    <th>Ответ LLM</th>
                    <th>Similarity</th>
                    <th>Coverage</th>
                    <th>Latency</th>
                    <th>Статус</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        # Сохраняем файл
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🎨 HTML сохранен: {html_path}")
        return html_path
    
    def generate_all(self, df: pd.DataFrame, stats: Dict) -> Dict:
        """
        Генерирует все форматы отчетов.
        
        Returns:
            Словарь с путями к файлам
        """
        csv_path = self.save_csv(df, stats)
        html_path = self.save_html(df, stats)
        
        return {
            "csv": csv_path,
            "html": html_path
        }
    
    def save_comparison_html(self, comparison: Dict) -> str:
        """
        Создает HTML отчет для сравнения версий.
        
        Args:
            comparison: Результат сравнения из PromptComparator
        
        Returns:
            Путь к сохраненному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.output_dir, f"comparison_{timestamp}.html")
        
        stats_v1 = comparison["stats_v1"]
        stats_v2 = comparison["stats_v2"]
        winner = comparison["winner"]
        
        # Генерируем строки таблицы сравнения
        comparison_rows = ""
        for t in comparison["comparison_by_test"]:
            status_class = "improved" if "improved" in t["status"] else \
                          "degraded" if "degraded" in t["status"] else "same"
            
            comparison_rows += f"""
                <tr class="{status_class}">
                    <td>{t['id']}</td>
                    <td>{t['question'][:60]}</td>
                    <td>{t['v1_similarity']:.2f}</td>
                    <td>{t['v2_similarity']:.2f}</td>
                    <td class="diff {'positive' if t['similarity_diff'] > 0 else 'negative' if t['similarity_diff'] < 0 else ''}">
                        {t['similarity_diff']:+.3f}
                    </td>
                    <td>{t['v1_coverage']:.2f}</td>
                    <td>{t['v2_coverage']:.2f}</td>
                    <td class="diff {'positive' if t['coverage_diff'] > 0 else 'negative' if t['coverage_diff'] < 0 else ''}">
                        {t['coverage_diff']:+.3f}
                    </td>
                    <td>{t['status']}</td>
                </tr>
            """
        
        # Определяем текст победителя
        if winner["winner"] == "v2":
            winner_text = f"🏆 Победитель: {comparison['version2']}"
            winner_class = "success"
        elif winner["winner"] == "v1":
            winner_text = f"🏆 Победитель: {comparison['version1']}"
            winner_class = "success"
        else:
            winner_text = "🤝 Ничья"
            winner_class = "neutral"
        
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Prompt Comparison Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #2c5aa0; margin-bottom: 10px; }}
        .timestamp {{ color: #666; margin-bottom: 30px; }}
        
        .winner-banner {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .winner-banner h2 {{ margin: 0 0 10px 0; font-size: 32px; }}
        .winner-banner .score {{ font-size: 18px; opacity: 0.9; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 15px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-values {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-value.v1 {{ color: #3498db; }}
        .stat-value.v2 {{ color: #e74c3c; }}
        .stat-diff {{
            font-size: 14px;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .stat-diff.positive {{ background: #d4edda; color: #155724; }}
        .stat-diff.negative {{ background: #f8d7da; color: #721c24; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{
            background: #2c5aa0;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{ background: #f9f9f9; }}
        tr.improved {{ background: #f0fff4; }}
        tr.degraded {{ background: #fff5f5; }}
        
        .diff {{ font-weight: bold; font-family: monospace; }}
        .diff.positive {{ color: #27ae60; }}
        .diff.negative {{ color: #e74c3c; }}
        
        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-box h3 {{ color: #2c5aa0; margin-top: 0; }}
        .summary-list {{ list-style: none; padding: 0; }}
        .summary-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .summary-list li:last-child {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Prompt Comparison Report</h1>
        <p class="timestamp">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
            Сравнение: {comparison['version1']} vs {comparison['version2']}
        </p>
        
        <div class="winner-banner">
            <h2>{winner_text}</h2>
            <div class="score">
                {comparison['version1']}: {winner['score_v1']} очков |
                {comparison['version2']}: {winner['score_v2']} очков
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Pass Rate</h3>
                <div class="stat-values">
                    <span class="stat-value v1">{stats_v1['pass_rate']}%</span>
                    <span>→</span>
                    <span class="stat-value v2">{stats_v2['pass_rate']}%</span>
                    <span class="stat-diff {'positive' if stats_v2['pass_rate'] > stats_v1['pass_rate'] else 'negative'}">
                        {stats_v2['pass_rate'] - stats_v1['pass_rate']:+.1f}%
                    </span>
                </div>
            </div>
            
            <div class="stat-card">
                <h3>Avg Similarity</h3>
                <div class="stat-values">
                    <span class="stat-value v1">{stats_v1['avg_similarity']}</span>
                    <span>→</span>
                    <span class="stat-value v2">{stats_v2['avg_similarity']}</span>
                    <span class="stat-diff {'positive' if stats_v2['avg_similarity'] > stats_v1['avg_similarity'] else 'negative'}">
                        {stats_v2['avg_similarity'] - stats_v1['avg_similarity']:+.3f}
                    </span>
                </div>
            </div>
            
            <div class="stat-card">
                <h3>Avg Coverage</h3>
                <div class="stat-values">
                    <span class="stat-value v1">{stats_v1['avg_coverage']}</span>
                    <span>→</span>
                    <span class="stat-value v2">{stats_v2['avg_coverage']}</span>
                    <span class="stat-diff {'positive' if stats_v2['avg_coverage'] > stats_v1['avg_coverage'] else 'negative'}">
                        {stats_v2['avg_coverage'] - stats_v1['avg_coverage']:+.3f}
                    </span>
                </div>
            </div>
            
            <div class="stat-card">
                <h3>Avg Latency</h3>
                <div class="stat-values">
                    <span class="stat-value v1">{stats_v1['avg_latency']}s</span>
                    <span>→</span>
                    <span class="stat-value v2">{stats_v2['avg_latency']}s</span>
                    <span class="stat-diff {'positive' if stats_v2['avg_latency'] < stats_v1['avg_latency'] else 'negative'}">
                        {stats_v2['avg_latency'] - stats_v1['avg_latency']:+.3f}s
                    </span>
                </div>
            </div>
        </div>
        
        <div class="summary-box">
            <h3>📋 Анализ по тестам</h3>
            <ul class="summary-list">
                <li>📈 Улучшилось: {len([t for t in comparison['comparison_by_test'] if 'improved' in t['status']])} тестов</li>
                <li>📉 Ухудшилось: {len([t for t in comparison['comparison_by_test'] if 'degraded' in t['status']])} тестов</li>
                <li>➡️ Без изменений: {len([t for t in comparison['comparison_by_test'] if 'same' in t['status']])} тестов</li>
            </ul>
        </div>
        
        <h2>🔍 Детальное сравнение</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Вопрос</th>
                    <th>v1 Similarity</th>
                    <th>v2 Similarity</th>
                    <th>Δ Similarity</th>
                    <th>v1 Coverage</th>
                    <th>v2 Coverage</th>
                    <th>Δ Coverage</th>
                    <th>Статус</th>
                </tr>
            </thead>
            <tbody>
                {comparison_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🎨 Сравнительный отчет: {html_path}")
        return html_path