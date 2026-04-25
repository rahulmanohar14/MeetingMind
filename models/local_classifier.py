from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIDENCE_THRESHOLD, LOCAL_MODEL_PATH


class LocalClassifier:
    def __init__(self) -> None:
        self._model_path = Path(LOCAL_MODEL_PATH)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Local model folder not found at '{self._model_path}'. "
                "Offline mode requires a fine-tuned DistilBERT model in models/meetingmind_model/."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self._model_path))
        self._model.eval()

    @staticmethod
    def is_available() -> bool:
        return Path(LOCAL_MODEL_PATH, "config.json").exists()

    def classify(self, text: str) -> Dict[str, Any]:
        utterance = (text or "").strip()
        if not utterance:
            return {
                "category": "DISCUSSION",
                "confidence": 0.0,
                "method": "distilbert_local",
            }

        encoded = self._tokenizer(
            utterance,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=-1)

        label = self._model.config.id2label.get(int(pred_idx.item()), "DISCUSSION")
        return {
            "category": str(label).upper(),
            "confidence": float(conf.item()),
            "method": "distilbert_local",
        }

    def classify_batch(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue

            out = self.classify(text)
            category = str(out.get("category") or "DISCUSSION").upper()
            confidence = float(out.get("confidence", 0.0) or 0.0)

            turn_index = int(seg.get("turn_index", 0))
            speaker = str(seg.get("speaker") or "Unknown")
            line_number = seg.get("line_number", 0)
            source_line = f"Line {line_number}: {speaker}: {text}"

            owner = None
            verb = None
            deadline = None

            low_confidence = confidence < CONFIDENCE_THRESHOLD
            completion_risk = "high" if (category == "ACTION_ITEM" and (owner is None or deadline is None)) else "low"

            results.append(
                {
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "text": text,
                    "category": category,
                    "confidence": confidence,
                    "owner": owner,
                    "verb": verb,
                    "deadline": deadline,
                    "reasoning": "Classified by local DistilBERT model",
                    "source_line": source_line,
                    "low_confidence": low_confidence,
                    "completion_risk": completion_risk,
                    "line_number": line_number,
                }
            )
        return results
