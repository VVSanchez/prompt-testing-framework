"""
Модуль с метриками для оценки качества ответов LLM.

Аналогия с JS:
- Это как отдельный файл utils.js с вспомогательными функциями
- Экспортируем функции через return в конце (в Python — просто определяем)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Вычисляет семантическое сходство между двумя текстами.
    
    Как работает:
    1. Превращает тексты в числовые векторы (TF-IDF)
    2. Считает косинус угла между векторами
    3. Возвращает число от 0 (совсем разные) до 1 (идентичные)
    
    Аналогия с JS:
        function calculateSimilarity(text1, text2) {
            // Превращаем тексты в векторы чисел
            const vec1 = tfidf(text1);
            const vec2 = tfidf(text2);
            // Считаем косинус между ними
            return cosine(vec1, vec2);
        }
    
    Args:
        text1: Первый текст
        text2: Второй текст
    
    Returns:
        float от 0 до 1 (1 = идентичны)
    """
    # Если один из текстов пустой — сходство 0
    if not text1.strip() or not text2.strip():
        return 0.0
    
    try:
        # Создаем векторизатор (превращает текст в числа)
        vectorizer = TfidfVectorizer()
        
        # Превращаем оба текста в векторы одновременно
        # fit_transform учится на данных и сразу их трансформирует
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Считаем косинусное сходство между первым и вторым вектором
        # Результат — матрица 2x2, нам нужен элемент [0][1]
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    
    except Exception as e:
        # Если что-то пошло не так (например, тексты слишком короткие)
        print(f"⚠️ Ошибка при расчете сходства: {e}")
        return 0.0


def calculate_keyword_coverage(response: str, expected: str) -> dict:
    """
    Проверяет, какие ключевые слова из эталона есть в ответе.
    
    Как работает:
    1. Разбивает эталонный ответ на слова
    2. Убирает стоп-слова (и, в, на, с...)
    3. Проверяет, какие слова есть в ответе LLM
    4. Считает процент покрытия
    
    Аналогия с JS:
        function checkKeywords(response, expected) {
            const keywords = expected.split(' ').filter(w => !stopWords.includes(w));
            const found = keywords.filter(w => response.includes(w));
            return {
                coverage: found.length / keywords.length,
                found: found,
                missing: keywords.filter(w => !found.includes(w))
            };
        }
    
    Args:
        response: Ответ от LLM
        expected: Эталонный ответ
    
    Returns:
        dict с coverage (0-1), found (список), missing (список)
    """
    # Стоп-слова — частые слова, которые не несут смысла
    stop_words = {
        # Русские
        'и', 'в', 'на', 'с', 'по', 'для', 'что', 'как', 'это', 'который',
        'а', 'но', 'да', 'не', 'из', 'от', 'до', 'за', 'о', 'об', 'когда',
        'уже', 'или', 'же', 'бы', 'ли', 'при', 'про', 'над', 'под', 'через',
        # Английские
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either'
    }
    
    # Функция для очистки текста: убираем пунктуацию, приводим к нижнему регистру
    def clean_text(text):
        # Приводим к нижнему регистру (как .toLowerCase() в JS)
        text = text.lower()
        # Убираем пунктуацию через регулярное выражение
        # [^\w\s] означает "всё, что не буква/цифра и не пробел"
        text = re.sub(r'[^\w\s]', ' ', text)
        # Разбиваем на слова (как .split(' ') в JS)
        words = text.split()
        # Убираем стоп-слова (как .filter() в JS)
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return words
    
    # Извлекаем ключевые слова из эталона
    expected_words = clean_text(expected)
    response_words = set(clean_text(response))  # set — как Set в JS (уникальные значения)
    
    # Если ключевых слов нет — возвращаем 100% (нечего проверять)
    if not expected_words:
        return {
            "coverage": 1.0,
            "found": [],
            "missing": []
        }
    
    # Проверяем, какие ключевые слова есть в ответе
    found = [w for w in expected_words if w in response_words]
    missing = [w for w in expected_words if w not in response_words]
    
    # Считаем процент покрытия
    coverage = len(found) / len(expected_words)
    
    return {
        "coverage": coverage,
        "found": found,
        "missing": missing
    }


def is_test_passed(similarity: float, coverage: float, 
                   similarity_threshold: float = 0.3,
                   coverage_threshold: float = 0.5) -> bool:
    """
    Определяет, пройден ли тест.
    
    Тест считается пройденным, если:
    - Семантическое сходство > порога ИЛИ
    - Покрытие ключевых слов > порога
    
    Args:
        similarity: Семантическое сходство (0-1)
        coverage: Покрытие ключевых слов (0-1)
        similarity_threshold: Порог для сходства (по умолчанию 0.3)
        coverage_threshold: Порог для покрытия (по умолчанию 0.5)
    
    Returns:
        bool: True если тест пройден
    """
    return similarity >= similarity_threshold or coverage >= coverage_threshold