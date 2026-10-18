"""
Клиент для Agent Platform от Just AI (https://docs.agentplatform.just-ai.com).

Платформа: единая среда для создания AI-агентов (LLM + промпт + инструменты)
с визуальным конструктором процессов (Flow Builder) и каналми подключения.

Для тестирования промптов/агентов снаружи платформы используется Chat API:
    https://docs.agentplatform.just-ai.com/channels/chat-api

Основные методы Chat API:
    POST /chatapi/{token}                 — синхронная отправка запроса клиента
    GET  /chatapi/{token}                 — упрощенный метод (clientId, query, event)
    POST /chatapi/{token}/async           — асинхронная отправка (вернется id сообщения)
    GET  /chatapi/{token}/events          — long polling асинхронных событий
    GET  /chatapi/{token}/client/{clientId}/history — история переписки

Токен канала — часть адреса после https://bot.jaicp.com/chatapi/
(получается в разделе «Интеграции» → «Получить вебхук» у канала Chat API).

Пример ответа (упрощенно):
    {
      "token": "...",
      "clientId": "tester-1",
      "questionId": "12345",
      "data": {
        "replies": [{"type": "text", "text": "Ответ агента", "state": "/CatchAll"}],
        "answer": "Ответ агента",
        "newSessionStarted": false,
        "endSession": false
      },
      "timestamp": "2026-01-01T12:00:00"
    }

Важные особенности платформы (влияют на тестирование):
  * Промпт агента задается НЕ одним system_prompt-текстом, а полями блока «Агент»:
    Роль / Цель / Инструкции (+ Расширенный режим для полного контроля
    системного промпта). См. https://docs.agentplatform.just-ai.com/blocks/agent/agent-block
  * Контекст передается через Context.set(key, value) и подстановки {{ ... }}
    (движок v2 — Pebble Templates): {{ request.query }}, {{ previousResult }} и др.
  * Диалог привязан к clientId/sessionId: чтобы каждый тест начинался «с чистого
    листа», нужно использовать уникальный clientId либо сброс сессии (/start).
"""

import os
import time
import uuid
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


class AgentPlatformError(RuntimeError):
    """Ошибка взаимодействия с Agent Platform."""


class AgentPlatformClient:
    """
    Клиент Agent Platform (Just AI) через Chat API.

    Пример:
        client = AgentPlatformClient(token="ZaBcdEfG:123...")
        answer = client.send_query("Какие документы нужны для кредита?",
                                   context={"inn": "1234567890", "program": "МСП"})
    """

    DEFAULT_BASE_URL = "https://bot.jaicp.com"

    def __init__(self,
                 token: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 60,
                 max_retries: int = 3,
                 retry_backoff: float = 2.0):
        """
        Args:
            token: Токен канала Chat API (`ЗаглавнаяЧасть:хеш`).
                   Берется из env JAICP_CHAT_API_TOKEN, если не передан.
            base_url: Адрес сервера (по умолчанию https://bot.jaicp.com).
            timeout: Таймаут HTTP-запроса, сек.
            max_retries: Количество повторов при сетевых ошибках/5xx
                         (см. «Повторные попытки при ошибке» в движке v2).
            retry_backoff: Множитель экспоненциальной задержки между попытками.
        """
        load_dotenv()

        self.token = token or os.getenv("JAICP_CHAT_API_TOKEN")
        self.base_url = (base_url or os.getenv("JAICP_BASE_URL")
                         or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        if not self.token:
            raise AgentPlatformError(
                "Не задан токен Chat API. Передайте token=... или добавьте "
                "JAICP_CHAT_API_TOKEN в .env. Токен получается в проекте Agent "
                "Platform: Интеграции → Chat API → Получить вебхук "
                "(часть адреса после https://bot.jaicp.com/chatapi/)."
            )

    # ------------------------------------------------------------------ #
    # Низкоуровневые методы Chat API
    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = requests.request(method, url, timeout=self.timeout, **kwargs)
                # 5xx — временная ошибка сервера, повторяем
                if resp.status_code >= 500:
                    last_exc = AgentPlatformError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(self.retry_backoff * (2 ** attempt))

        raise AgentPlatformError(f"Chat API недоступна: {last_exc}")

    @property
    def _chatapi_path(self) -> str:
        return f"/chatapi/{self.token}"

    def send_sync(self,
                  query: str,
                  client_id: str,
                  extra_fields: Optional[Dict] = None) -> Dict:
        """
        Синхронная отправка запроса клиента: POST /chatapi/{token}.

        Args:
            query: Текст запроса клиента.
            client_id: Идентификатор клиента (определяет сессию/контекст диалога).
            extra_fields: Дополнительные поля тела запроса.

        Returns:
            Разобранный JSON-ответ платформы.
        """
        payload = {
            "clientId": client_id,
            "query": query,
            "questionId": uuid.uuid4().hex,
        }
        if extra_fields:
            payload.update(extra_fields)

        resp = self._request("POST", self._chatapi_path, json=payload)
        if resp.status_code != 200:
            raise AgentPlatformError(
                f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def send_get(self, query: str, client_id: str) -> Dict:
        """Упрощенный метод: GET /chatapi/{token}?clientId=...&query=..."""
        resp = self._request("GET", self._chatapi_path,
                             params={"clientId": client_id, "query": query})
        if resp.status_code != 200:
            raise AgentPlatformError(
                f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def send_async(self, query: str, client_id: str) -> str:
        """
        Асинхронная отправка: POST /chatapi/{token}/async.
        Возвращает идентификатор сообщения; ответ ждите через poll_events().
        """
        resp = self._request("POST", f"{self._chatapi_path}/async",
                             json={"clientId": client_id, "query": query})
        if resp.status_code != 200:
            raise AgentPlatformError(
                f"HTTP {resp.status_code}: {resp.text[:300]}")
        return str(resp.json().get("id", ""))

    def poll_events(self, client_id: str, last_event_id: Optional[str] = None,
                    wait_sec: int = 30) -> List[Dict]:
        """
        Long polling событий: GET /chatapi/{token}/events.
        Используется вместе с send_async().
        """
        params = {"clientId": client_id, "wait": wait_sec}
        if last_event_id:
            params["lastEventId"] = last_event_id
        resp = self._request("GET", f"{self._chatapi_path}/events",
                             params=params)
        if resp.status_code != 200:
            raise AgentPlatformError(
                f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        events = data.get("data") or data.get("events") or []
        return events if isinstance(events, list) else [events]

    def get_history(self, client_id: str) -> List[Dict]:
        """История переписки: GET /chatapi/{token}/client/{clientId}/history."""
        resp = self._request("GET",
                             f"{self._chatapi_path}/client/{client_id}/history")
        if resp.status_code != 200:
            raise AgentPlatformError(
                f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data.get("data") or data.get("messages") or []

    # ------------------------------------------------------------------ #
    # Высокоуровневый API для фреймворка тестирования
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_answer(response_json: Dict) -> str:
        """
        Достаёт текстовый ответ агента из ответа Chat API.

        Порядок поиска: data.answer → text-реплики в data.replies → raw text.
        """
        data = response_json.get("data") or {}

        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        replies = data.get("replies") or []
        texts = [r.get("text", "") for r in replies
                 if isinstance(r, dict) and r.get("type") in ("text", "TEXT")]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined

        # fallback: попробуем верхнеуровневые поля
        return (response_json.get("text") or "").strip()

    def send_query(self, question: str,
                   client_id: Optional[str] = None,
                   context: Optional[Dict] = None) -> str:
        """
        Отправить вопрос агенту и вернуть текст ответа.

        Каждому тесту соответствует свой client_id (по умолчанию уникальный),
        чтобы диалоги/контексты тестов не смешивались.

        Args:
            question: Вопрос пользователя.
            client_id: Идентификатор клиента/сессии. Если None — генерируется.
            context: Дополнительный контекст (inn, program и т.п.) —
                     добавляется к тексту запроса; на стороне процесса
                     Agent Platform может разбираться блоками кода/подстановками.

        Returns:
            Текст ответа агента (или строка "ERROR: ...").
        """
        cid = client_id or f"prompt-test-{uuid.uuid4().hex[:12]}"

        full_question = question
        if context:
            ctx_lines = [f"{k}: {v}" for k, v in context.items() if v]
            if ctx_lines:
                full_question = f"{question}\n(контекст: {'; '.join(ctx_lines)})"

        try:
            result = self.send_sync(full_question, cid)
            return self.extract_answer(result) or "ERROR: пустой ответ агента"
        except AgentPlatformError as exc:
            return f"ERROR: {exc}"
