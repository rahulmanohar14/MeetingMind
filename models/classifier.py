import json
import re
from typing import List

from openai import OpenAI
from sklearn.metrics import classification_report

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME

_CATEGORIES = (
    "DECISION",
    "ACTION_ITEM",
    "OPEN_QUESTION",
    "DEFERRAL",
    "DISCUSSION",
)

_CLASSIFY_SYSTEM = f"""Classify a single meeting utterance into exactly one of these categories: {', '.join(_CATEGORIES)}.
Return ONLY a JSON object with a single key "label" and value one of: {', '.join(_CATEGORIES)}. No markdown."""


def _parse_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


class MeetingClassifier:
    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    def classify_one(self, utterance: str) -> str:
        u = (utterance or "").strip()
        if not u:
            return "DISCUSSION"
        try:
            resp = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Utterance:\n{u}",
                    },
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            data = _parse_json_object(content)
            label = (data.get("label") or "DISCUSSION").strip().upper()
            if label not in _CATEGORIES:
                return "DISCUSSION"
            return label
        except Exception:
            return "DISCUSSION"

    def classify_batch(self, utterances: list) -> list:
        return [self.classify_one(u) for u in utterances]

    @staticmethod
    def _sample_gold() -> List[tuple]:
        return [
            (
                "We're going with the Q3 launch date, that's final.",
                "DECISION",
            ),
            ("I'll send the contract to legal by Thursday.", "ACTION_ITEM"),
            (
                "Sarah will update the mockups before Friday's meeting.",
                "ACTION_ITEM",
            ),
            (
                "Let's circle back on the budget question next week.",
                "DEFERRAL",
            ),
            ("Yeah that sounds right to me.", "DISCUSSION"),
            (
                "Has anyone checked the compliance requirements?",
                "OPEN_QUESTION",
            ),
            ("We've decided to use React for the frontend.", "DECISION"),
            ("Someone should look into the vendor situation.", "ACTION_ITEM"),
            (
                "I wonder if Q3 even makes sense given the timeline.",
                "DISCUSSION",
            ),
            (
                "Marcus will review the legal terms by end of month.",
                "ACTION_ITEM",
            ),
            ("Maybe we revisit the pricing model later.", "DEFERRAL"),
            (
                "What's our fallback if the API integration fails?",
                "OPEN_QUESTION",
            ),
            (
                "Bob will set up the staging environment by Wednesday.",
                "ACTION_ITEM",
            ),
            ("That's agreed then — we launch in October.", "DECISION"),
            (
                "Let's think about the onboarding flow separately.",
                "DEFERRAL",
            ),
            (
                "The design looks great, nice work Sarah.",
                "DISCUSSION",
            ),
            ("Who owns the user testing plan?", "OPEN_QUESTION"),
            (
                "We need to finalize the pricing — let's decide now: $49/month.",
                "DECISION",
            ),
            (
                "Let's table the internationalization discussion for now.",
                "DEFERRAL",
            ),
            (
                "Alice will share the updated roadmap by Monday.",
                "ACTION_ITEM",
            ),
        ]

    def evaluate_sample(self) -> dict:
        gold = self._sample_gold()
        texts = [g[0] for g in gold]
        y_true = [g[1] for g in gold]
        y_pred = self.classify_batch(texts)
        return classification_report(
            y_true, y_pred, labels=list(_CATEGORIES), output_dict=True, zero_division=0
        )
