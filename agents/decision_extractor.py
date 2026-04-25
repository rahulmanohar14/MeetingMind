import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
    CONFIDENCE_THRESHOLD,
)

_SYSTEM_PROMPT = """You are an expert meeting analyst. Classify this meeting utterance into exactly one category:
DECISION: A binding final choice. Look for: that's final, we're going with, decided, agreed upon.
ACTION_ITEM: A specific personal commitment. Must have a doer. Extract owner (named person only, not team), verb (specific action), deadline (specific date or next meeting).
OPEN_QUESTION: A question raised with no answer given.
DEFERRAL: A polite postponement. Look for: circle back, revisit, maybe next week, let's think about it.
DISCUSSION: Everything else including opinions, exploration, small talk.
Return ONLY valid JSON with no markdown:
{"category": "DECISION|ACTION_ITEM|OPEN_QUESTION|DEFERRAL|DISCUSSION", "confidence": 0.85, "owner": "name or null", "verb": "action or null", "deadline": "date or null", "reasoning": "one sentence explanation"}


JSON rules: use null (not the string "null") for missing values; category must be one of the five uppercase tokens above."""


def _parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class DecisionExtractor:
    def __init__(self, mode: str = "api") -> None:
        self._mode = (mode or "api").strip().lower()
        self._local_classifier = None
        self._client = None

        if self._mode == "local":
            try:
                from models.local_classifier import LocalClassifier

                if LocalClassifier.is_available():
                    self._local_classifier = LocalClassifier()
                else:
                    print(
                        "[WARN] Local mode selected but model not available. Falling back to API mode."
                    )
                    self._mode = "api"
            except Exception as exc:
                print(
                    f"[WARN] Local mode initialization failed ({exc}). Falling back to API mode."
                )
                self._mode = "api"

        if self._mode != "local":
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )

    def _classify_utterance(self, user_text: str) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Classify this utterance:\n\n{user_text}",
                },
            ],
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()
        return _parse_json_object(content)

    @staticmethod
    def _str_or_null(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() in ("null", "none", "n/a"):
            return None
        return s

    @staticmethod
    def _resolve_owner_pronoun(owner: Optional[str], speaker: str) -> Optional[str]:
        if owner is None:
            return None
        if owner.strip().lower() in {"i", "me", "my"}:
            return speaker
        return owner

    def extract(self, segments: list) -> list:
        if self._mode == "local" and self._local_classifier is not None:
            return self._local_classifier.classify_batch(segments)

        results: List[Dict[str, Any]] = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            out = self._classify_utterance(text)
            category = (out.get("category") or "DISCUSSION").strip().upper()
            try:
                conf = float(out.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            owner = self._str_or_null(out.get("owner"))
            verb = self._str_or_null(out.get("verb"))
            deadline = self._str_or_null(out.get("deadline"))
            reasoning = str(out.get("reasoning") or "No explanation provided.").strip()

            turn_index = int(seg.get("turn_index", 0))
            speaker = str(seg.get("speaker") or "Unknown")
            owner = self._resolve_owner_pronoun(owner, speaker)
            line_number = seg.get("line_number", 0)
            source_line = f"Line {line_number}: {speaker}: {text}"

            low_conf = conf < CONFIDENCE_THRESHOLD
            completion_risk = "high"
            if category == "ACTION_ITEM" and (owner is None or deadline is None):
                completion_risk = "high"
            else:
                completion_risk = "low"

            results.append(
                {
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "text": text,
                    "category": category,
                    "confidence": conf,
                    "owner": owner,
                    "verb": verb,
                    "deadline": deadline,
                    "reasoning": reasoning,
                    "source_line": source_line,
                    "low_confidence": low_conf,
                    "completion_risk": completion_risk,
                    "line_number": line_number,
                }
            )
        return results
