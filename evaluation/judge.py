from __future__ import annotations

import json
import logging
import re
from typing import Any

import ssl_fix  # noqa: F401 — must run before google.genai / aiohttp
from google import genai

from config import Config

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an evaluator for a legal retrieval-augmented generation (RAG) system.

Given a user query, retrieved context, and generated answer, score:
1. faithfulness (0.0-1.0): Is the answer fully supported by the retrieved context? Penalize hallucinations.
2. answer_relevancy (0.0-1.0): Does the answer directly address the user's question?

If the answer states insufficient evidence and the context is empty or irrelevant, score faithfulness high only if the answer does not invent facts.

Return valid JSON only:
{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "rationale": "one short sentence"
}

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""


class GeminiJudge:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._client = genai.Client(api_key=api_key or Config.GEMINI_API_KEY)
        self._model = model or Config.GEMINI_JUDGE_MODEL

    def score(
        self,
        *,
        query: str,
        context: str,
        answer: str,
    ) -> dict[str, Any]:
        prompt = JUDGE_PROMPT.format(
            query=(query or "").strip(),
            context=(context or "(empty)")[:12000],
            answer=(answer or "").strip()[:8000],
        )
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "response_mime_type": "application/json",
                },
            )
            return self._parse_json(response.text or "")
        except Exception as exc:
            log.warning("Judge call failed: %s", exc)
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "rationale": f"judge_error: {exc}",
            }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ValueError("Empty judge response")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        raise ValueError("Judge response is not valid JSON")
