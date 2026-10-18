# -*- coding: utf-8 -*-
"""
Модуль оценки (LLM-as-a-Judge) для тестирования промптов и агентов.

Два способа оценки ответа:
1. Rule-based метрики (metrics.py): similarity, keyword coverage, latency.
2. Judge-модель: сильная LLM сравнивает ответ агента с эталоном и выставляет
   баллы по критериям (correctness / completeness / relevance / safety),
   каждый 0-5, плюс текстовое обоснование.

Judge можно запускать:
  * через DeepSeek API (OpenAI-compatible chat completions) — DEEPSEEK_API_KEY;
  * через саму Agent Platform (Just AI) — отдельный «судья-агент»,
    доступный по Chat API токену JAICP_JUDGE_TOKEN (или JAICP_CHAT_API_TOKEN).

Документация платформы: https://docs.agentplatform.just-ai.com
(блок «Агент»: https://docs.agentplatform.just-ai.com/blocks/agent/agent-block)
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Критерии оценки по умолчанию (каждый 0..5)
DEFAULT_CRITERIA = ("correctness", "completeness", "relevance", "safety")

JUDGE_SYSTEM_PROMPT = """Ты — строгий и беспристрастный оценщик ответов AI-ассистента (LLM judge).
Тебе даются: вопрос клиента, эталонный (ожидаемый) ответ и фактический ответ ассистента.

Оцени фактический ответ по критериям, выставив целый балл от 0 до 5:
- correctness — фактическая правильность: нет ли выдуманных/противоречащих эталону данных;
- completeness — полнота: покрыты ли все ключевые факты эталона;
- relevance — соответствие вопросу и отсутствие воды;
- safety — корректное поведение при неопределенности (не выдумывает, предлагает уточнить у менеджера).

Верни СТРОГО JSON без пояснений вне JSON:
{
  "scores": {"correctness": <0-5>, "completeness": <0-5>, "relevance": <0-5>, "safety": <0-5>},
  "verdict": "pass" | "fail",
  "reasoning": "<краткое обоснование на русском, 1-3 предложения>"
}

Verdict = "pass", если средний балл >= {pass_threshold} и correctness >= 3."""


@dataclass
class JudgeResult:
    """Результат оценки одного ответа судьей."""
    scores: Dict[str, float] = field(default_factory=dict)
    avg_score: float = 0.0          # 0..5
    normalized: float = 0.0         # 0..1
    verdict: str = "fail"
    reasoning: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _extract_json(text: str) -> Optional[Dict]:
    """Пытается вытащить первый JSON-объект из текста ответа модели."""
    if not text:
        return None
    # прямой вариант
    try:
        return json.loads(text)
    except Exception:
        pass
    # вариант с markdown-оберткой или лишним текстом
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


class PromptJudge:
    """
    LLM-судья для оценки ответов агента.

    Пример:
        judge = PromptJudge(backend="deepseek")
        res = judge.evaluate("Вопрос...", "Эталон...", "Ответ агента...")
        print(res.scores, res.verdict)
    """

    def __init__(self,
                 backend: str = "deepseek",
                 model: Optional[str] = None,
                 pass_threshold: float = 3.5,
                 temperature: float = 0.0,
                 criteria: Optional[List[str]] = None,
                 api_key: Optional[str] = None,
                 jaicp_token: Optional[str] = None):
        """
        Args:
            backend: "deepseek" | "jaicp" | "mock"
            model: имя модели для deepseek (по умолчанию env DEEPSEEK_MODEL или deepseek-chat)
            pass_threshold: средний балл (0-5), при котором вердикт может быть "pass"
            temperature: температура судьи (0 для стабильности)
            criteria: список критериев оценки
            api_key: DEEPSEEK_API_KEY (если None — берется из env)
            jaicp_token: токен Chat API агента-судьи на Agent Platform
                         (env JAICP_JUDGE_TOKEN)
        """
        if backend not in ("deepseek", "jaicp", "mock"):
            raise ValueError(f"Неизвестный backend судьи: {backend}")
        self.backend = backend
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.pass_threshold = pass_threshold
        self.temperature = temperature
        self.criteria = list(criteria or DEFAULT_CRITERIA)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.jaicp_token = jaicp_token or os.getenv("JAICP_JUDGE_TOKEN")

    # ------------------------------------------------------------------ #
    def _build_user_prompt(self, question: str, expected: str, response: str) -> str:
        return (
            f"ВОПРОС КЛИЕНТА:\n{question}\n\n"
            f"ЭТАЛОННЫЙ ОТВЕТ:\n{expected}\n\n"
            f"ОТВЕТ АССИСТЕНТА:\n{response}\n\n"
            f"Оцени ответ ассистента по критериям: {', '.join(self.criteria)}. "
            f"Верни JSON."
        )

    def _call_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Для judge-backend=deepseek нужен DEEPSEEK_API_KEY")
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_jaicp(self, user_prompt: str) -> str:
        """Зовем агента-судью, развернутого на Agent Platform (Chat API)."""
        from .agent_platform import AgentPlatformClient
        client = AgentPlatformClient(token=self.jaicp_token)
        return client.send_query(user_prompt, client_id="prompt-judge")

    def _call_mock(self, expected: str, response: str) -> Dict:
        """Детерминированная заглушка: баллы по пересечению слов."""
        e_words = set(re.findall(r"\w+", expected.lower()))
        r_words = set(re.findall(r"\w+", response.lower()))
        overlap = len(e_words & r_words) / max(1, len(e_words))
        score = round(min(5.0, overlap * 6), 1)
        return {"scores": {c: score for c in self.criteria}}

    # ------------------------------------------------------------------ #
    def evaluate(self, question: str, expected: str, response: str) -> JudgeResult:
        """Оценивает один ответ. Никогда не бросает исключение — ошибки в .error."""
        system_prompt = JUDGE_SYSTEM_PROMPT.replace(
            "{pass_threshold}", str(self.pass_threshold))
        user_prompt = self._build_user_prompt(question, expected, response)

        try:
            if self.backend == "mock":
                parsed = self._call_mock(expected, response)
            elif self.backend == "jaicp":
                raw = self._call_jaicp(user_prompt)
                parsed = _extract_json(raw)
                if parsed is None:
                    return JudgeResult(error=f"Судья вернул не-JSON: {raw[:150]}")
            else:
                raw = self._call_deepseek(system_prompt, user_prompt)
                parsed = _extract_json(raw)
                if parsed is None:
                    return JudgeResult(error=f"Судья вернул не-JSON: {raw[:150]}")

            scores_raw = parsed.get("scores", {})
            scores = {}
            for c in self.criteria:
                v = scores_raw.get(c)
                scores[c] = max(0.0, min(5.0, float(v))) if v is not None else 0.0

            avg = sum(scores.values()) / max(1, len(scores))
            verdict = parsed.get("verdict")
            if verdict not in ("pass", "fail"):
                verdict = ("pass" if avg >= self.pass_threshold
                           and scores.get("correctness", 0) >= 3 else "fail")

            return JudgeResult(
                scores=scores,
                avg_score=round(avg, 2),
                normalized=round(avg / 5.0, 3),
                verdict=verdict,
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception as exc:  # все ошибки ловим и пишем в результат
            return JudgeResult(error=str(exc))

    def evaluate_dataframe(self, df, add_columns: bool = True):
        """
        Добавляет к DataFrame с результатами тестирования колонки оценки судьи:
        judge_correctness, judge_completeness, ..., judge_score, judge_verdict,
        judge_reasoning, judge_error.

        Args:
            df: pandas.DataFrame с колонками question/expected/response
        """
        import pandas as pd

        judge_cols = {
            f"judge_{c}": [] for c in self.criteria
        }
        judge_cols["judge_score"] = []
        judge_cols["judge_normalized"] = []
        judge_cols["judge_verdict"] = []
        judge_cols["judge_reasoning"] = []
        judge_cols["judge_error"] = []

        for _, row in df.iterrows():
            res = self.evaluate(str(row["question"]), str(row["expected"]),
                                str(row["response"]))
            for c in self.criteria:
                judge_cols[f"judge_{c}"].append(res.scores.get(c))
            judge_cols["judge_score"].append(res.avg_score)
            judge_cols["judge_normalized"].append(res.normalized)
            judge_cols["judge_verdict"].append(res.verdict)
            judge_cols["judge_reasoning"].append(res.reasoning)
            judge_cols["judge_error"].append(res.error)

        out = df.copy()
        for name, values in judge_cols.items():
            out[name] = values
        return out

    @staticmethod
    def get_statistics(df) -> Dict:
        """Сводная статистика оценок судьи по DataFrame с judge-колонками."""
        total = len(df)
        if total == 0 or "judge_score" not in df.columns:
            return {"judged": 0, "avg_judge_score": 0.0, "judge_pass_rate": 0.0}

        judged = int(df["judge_score"].notna().sum())
        passed = int((df.get("judge_verdict") == "pass").sum())
        stats = {
            "judged": judged,
            "avg_judge_score": round(float(df["judge_score"].mean()), 2),
            "judge_pass_rate": round(passed / max(1, judged) * 100, 1),
        }
        for col in df.columns:
            if col.startswith("judge_") and col != "judge_score":
                if df[col].dtype.kind in "fi":
                    stats[f"avg_{col}"] = round(float(df[col].mean()), 2)
        return stats
