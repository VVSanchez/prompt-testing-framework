"""
Основной модуль для тестирования промптов и AI-агентов.

Здесь находится класс PromptTester, который:
1. Читает тестовые кейсы из CSV
2. Отправляет запросы к LLM / агенту на Agent Platform (Just AI)
3. Считает метрики качества
4. Возвращает результаты

Режимы отправки запросов (--mode):
  * jaicp      — агент на Just AI Agent Platform через Chat API (основной режим)
  * deepseek   — прямой вызов DeepSeek API (OpenAI-compatible), legacy
  * mock       — локальная заглушка (проверка фреймворка без ключей)
  * custom-api — свой HTTP endpoint (api_endpoint/api_key)
"""

import pandas as pd
import requests
import time
import uuid
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

# Импортируем наши метрики из metrics.py
from .metrics import (
    calculate_semantic_similarity,
    calculate_keyword_coverage,
    is_test_passed
)

# Клиент Agent Platform (Just AI)
from .agent_platform import AgentPlatformClient, AgentPlatformError


class PromptTester:
    """
    Основной класс для тестирования промптов и агентов.

    Пример использования (Agent Platform):
        tester = PromptTester(mode="jaicp")
        results = tester.run_all_tests("examples/test_cases.csv")

    Пример использования (legacy DeepSeek):
        tester = PromptTester(mode="deepseek")
        results = tester.run_all_tests("examples/test_cases.csv",
                                       system_prompt=open("system_prompt.txt").read())
    """

    VALID_MODES = ("mock", "jaicp", "deepseek", "custom-api")

    def __init__(self, api_endpoint: str = None, api_key: Optional[str] = None,
                 use_mock: bool = False,
                 mode: Optional[str] = None,
                 jaicp_token: Optional[str] = None,
                 jaicp_base_url: Optional[str] = None,
                 reuse_session: bool = False):
        """
        Конструктор класса.

        Args:
            api_endpoint: URL API для тестирования (режим custom-api)
            api_key: API ключ для авторизации (режим custom-api)
            use_mock: Совместимость со старым кодом: True => mode="mock"
            mode: Режим тестирования: "jaicp" | "deepseek" | "mock" | "custom-api".
                  По умолчанию берется из env PROMPT_TEST_MODE, иначе "mock".
            jaicp_token: Токен канала Chat API Agent Platform
                         (по умолчанию env JAICP_CHAT_API_TOKEN)
            jaicp_base_url: Адрес сервера (по умолчанию https://bot.jaicp.com)
            reuse_session: Если True — все тесты идут в один диалог (один clientId).
                           По умолчанию False: уникальный clientId на тест,
                           чтобы контексты/сессии агента не смешивались.
        """
        # Обратная совместимость: use_mock=True => mock-режим
        if mode is None:
            mode = "mock" if use_mock else os.getenv("PROMPT_TEST_MODE", "mock")
        if mode not in self.VALID_MODES:
            raise ValueError(f"Неизвестный режим '{mode}'. "
                             f"Доступны: {', '.join(self.VALID_MODES)}")

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.mode = mode
        self.use_mock = (mode == "mock")  # для старого кода/отчетов
        self.reuse_session = reuse_session
        self.results: List[Dict] = []  # Список для хранения результатов

        # Загружаем переменные окружения из .env файла
        load_dotenv()

        # Клиент Agent Platform создаём только в режиме jaicp
        # (ошибка про недостающий токен прилетит сразу, а не в середине прогона)
        self._jaicp_client: Optional[AgentPlatformClient] = None
        if mode == "jaicp":
            self._jaicp_client = AgentPlatformClient(
                token=jaicp_token, base_url=jaicp_base_url)

    @property
    def jaicp_client(self) -> AgentPlatformClient:
        """Ленивый доступ к клиенту Agent Platform."""
        if self._jaicp_client is None:
            self._jaicp_client = AgentPlatformClient()
        return self._jaicp_client
    
    def _mock_response(self, question: str) -> str:
        """
        Заглушка для тестирования без реального API.
        Возвращает осмысленный ответ на основе вопроса.
        
        Это нужно, чтобы ты мог сразу увидеть работу фреймворка,
        не получая API ключей.
        """
        # Простая логика: если в вопросе есть ключевые слова — возвращаем ответ
        question_lower = question.lower()
        
        if 'документ' in question_lower:
            return "Для подачи заявки необходимы устав, учредительные документы и финансовая отчетность за последний год."
        elif 'сумм' in question_lower or 'максим' in question_lower:
            return "Максимальная сумма кредита составляет 500 млн рублей для промышленных предприятий и 200 млн для МСП."
        elif 'требован' in question_lower or 'заемщик' in question_lower:
            return "Основные требования к заемщику: выручка от 100 млн рублей, отсутствие просрочек, положительная кредитная история."
        elif 'ставк' in question_lower or 'процент' in question_lower:
            return "Ставка по программе составляет от 8% годовых для МСП и от 6% для приоритетных отраслей."
        elif 'срок' in question_lower:
            return "Максимальный срок кредитования — до 10 лет с возможностью пролонгации."
        else:
            return "Для получения подробной информации обратитесь к менеджеру или ознакомьтесь с условиями на сайте."
        
    def send_to_deepseek(self, question: str, system_prompt: str = None, 
                        context: Dict = None) -> str:
        """
        Отправка запроса напрямую в DeepSeek API.
        
        Args:
            question: Вопрос пользователя
            system_prompt: Системный промпт (из файла prompts/v1/system_prompt.txt)
            context: Дополнительный контекст
        
        Returns:
            Ответ от DeepSeek
        """
        try:
            # Загружаем API ключ из .env
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                return "ERROR: DEEPSEEK_API_KEY not found in .env"
            
            # Формируем системный промпт (если не передан)
            if not system_prompt:
                system_prompt = """Ты консультант по льготному кредитованию юридических лиц.
    Отвечай подробно, используя информацию о программах.
    Учитывай контекст: ИНН клиента и выбранную программу."""
            
            # Формируем контекст для промпта
            context_text = ""
            if context:
                if context.get("inn"):
                    context_text += f"ИНН клиента: {context['inn']}\n"
                if context.get("program"):
                    context_text += f"Выбранная программа: {context['program']}\n"
            
            # Формируем полный промпт
            full_prompt = f"{context_text}\nВопрос клиента: {question}"
            
            # Отправляем запрос через OpenAI-compatible API
            # DeepSeek использует тот же формат, что и OpenAI
            import requests
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",  # или deepseek-coder
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.3,  # Низкая температура для стабильности
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"ERROR: HTTP {response.status_code} - {response.text[:200]}"
        
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def send_to_agent_platform(self, question: str, context: Dict = None,
                               client_id: Optional[str] = None) -> str:
        """
        Отправка запроса к агенту на Just AI Agent Platform через Chat API.

        Args:
            question: Вопрос пользователя
            context: Дополнительный контекст (inn, program и т.п.)
            client_id: Идентификатор клиента/сессии Chat API

        Returns:
            Текст ответа агента
        """
        return self.jaicp_client.send_query(question, client_id=client_id,
                                             context=context)

    def send_query(self, question: str, context: Dict = None,
                   system_prompt: str = None, use_deepseek: bool = False,
                   client_id: Optional[str] = None) -> str:
        """
        Отправка запроса к целевой LLM/агенту в зависимости от режима.

        Args:
            question: Вопрос для тестирования
            context: Дополнительный контекст
            system_prompt: Системный промпт (для DeepSeek; в Agent Platform
                           промпт задан на стороне процесса/блока «Агент»)
            use_deepseek: Совместимость со старым кодом — True форсит deepseek
            client_id: clientId для Chat API (Agent Platform)

        Returns:
            Ответ от LLM/агента (строка)
        """
        mode = "deepseek" if use_deepseek else self.mode

        # Агент на Agent Platform (Just AI) — основной новый режим
        if mode == "jaicp":
            return self.send_to_agent_platform(question, context, client_id)

        # DeepSeek напрямую (legacy)
        if mode == "deepseek":
            return self.send_to_deepseek(question, system_prompt, context)

        # Mock-режим (без внешних вызовов)
        if mode == "mock":
            time.sleep(0.5)
            return self._mock_response(question)

        # Кастомный HTTP endpoint
        try:
            payload = {
                "query": question,
                "context": context or {}
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response") or data.get("answer") or data.get("text", "")
            else:
                return f"ERROR: HTTP {response.status_code}"

        except Exception as e:
            return f"ERROR: {str(e)}"

    def run_single_test(self, test_case: Dict, system_prompt: str = None,
                        use_deepseek: bool = False,
                        client_id: Optional[str] = None) -> Dict:
        """
        Запуск одного тестового кейса.
        """
        question = str(test_case.get("question", ""))
        expected = str(test_case.get("expected_answer", ""))
        context = {
            "inn": str(test_case.get("inn", "")),
            "program": str(test_case.get("program", ""))
        }

        # Для Agent Platform: уникальный clientId на тест, чтобы сессии
        # и контексты разных тестов не смешивались (см. Chat API docs)
        if client_id is None and self.mode == "jaicp" and not self.reuse_session:
            client_id = f"prompt-test-{test_case.get('id', '?')}-{uuid.uuid4().hex[:8]}"

        # Измеряем время
        start_time = time.time()
        response = self.send_query(question, context, system_prompt,
                                   use_deepseek, client_id)
        latency = time.time() - start_time

        # Считаем метрики
        similarity = calculate_semantic_similarity(response, expected)
        keyword_data = calculate_keyword_coverage(response, expected)
        coverage = keyword_data["coverage"]

        # Определяем, пройден ли тест
        passed = is_test_passed(similarity, coverage)

        result = {
            "id": test_case.get("id", ""),
            "question": question,
            "expected": expected,
            "response": response,
            "latency_sec": round(latency, 3),
            "similarity": round(similarity, 3),
            "keyword_coverage": round(coverage, 3),
            "missing_keywords": keyword_data["missing"],
            "passed": passed
        }
        if self.mode == "jaicp":
            result["client_id"] = client_id

        return result

    def run_all_tests(self, csv_path: str, system_prompt: str = None,
                      use_deepseek: bool = False) -> pd.DataFrame:
        """
        Запуск всех тестов из CSV файла.
        """
        df = pd.read_csv(csv_path, encoding='utf-8')

        # Определяем режим для лога
        mode = "deepseek" if use_deepseek else self.mode
        mode_labels = {
            "jaicp": "🤖 AGENT PLATFORM (Just AI)",
            "deepseek": "🤖 DEEPSEEK",
            "mock": "🎭 MOCK",
            "custom-api": "🌐 API",
        }
        mode_label = mode_labels.get(mode, mode.upper())

        print(f"\n{'='*60}")
        print(f"🚀 Запуск {len(df)} тестов в режиме {mode_label}")
        print(f"{'='*60}\n")

        self.results = []

        for index, row in df.iterrows():
            test_case = row.to_dict()
            result = self.run_single_test(test_case, system_prompt, use_deepseek)
            self.results.append(result)

            status = "✅" if result["passed"] else "❌"
            print(f"{status} Тест #{result['id']}")
            print(f"   Вопрос: {result['question'][:60]}...")
            print(f"   Сходство: {result['similarity']:.2f} | Покрытие: {result['keyword_coverage']:.2f} | Время: {result['latency_sec']}s")
            if result['missing_keywords']:
                print(f"   Пропущены: {', '.join(result['missing_keywords'][:5])}")
            print()

        return pd.DataFrame(self.results)

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Подсчет общей статистики по результатам.
        
        Returns:
            Словарь со статистикой
        """
        total = len(df)
        passed = int(df["passed"].sum())
        failed = total - passed
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "avg_similarity": round(df["similarity"].mean(), 3),
            "avg_coverage": round(df["keyword_coverage"].mean(), 3),
            "avg_latency": round(df["latency_sec"].mean(), 3),
            "max_latency": round(df["latency_sec"].max(), 3),
            "min_latency": round(df["latency_sec"].min(), 3)
        }