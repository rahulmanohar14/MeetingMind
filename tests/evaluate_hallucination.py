from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.decision_extractor import DecisionExtractor
from agents.transcript_parser import TranscriptParser


CATEGORIES = ["DECISION", "ACTION_ITEM", "OPEN_QUESTION", "DEFERRAL", "DISCUSSION"]
TARGET_MAX_RATE = 5.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _source_candidate(source_line: str) -> str:
    """
    DecisionExtractor source_line format:
      "Line N: Speaker: utterance"
    We remove "Line N:" and compare the rest against transcript text.
    """
    src = (source_line or "").strip()
    return re.sub(r"^Line\s+\d+\s*:\s*", "", src, flags=re.IGNORECASE).strip()


def _is_hallucinated(source_line: str, transcript_text: str, item_text: str) -> bool:
    normalized_transcript = _normalize(transcript_text)
    candidate = _normalize(_source_candidate(source_line))
    fallback = _normalize(item_text or "")

    if candidate and candidate in normalized_transcript:
        return False
    if fallback and fallback in normalized_transcript:
        return False
    return True


def _load_transcript_paths() -> List[Path]:
    sample_path = PROJECT_ROOT / "data" / "sample_transcript.txt"
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic_transcripts"

    if not sample_path.exists():
        raise FileNotFoundError(f"Missing file: {sample_path}")
    if not synthetic_dir.exists():
        raise FileNotFoundError(f"Missing directory: {synthetic_dir}")

    base_synthetic = sorted(
        p
        for p in synthetic_dir.glob("*.txt")
        if not p.name.endswith("_augmented.txt")
    )
    return [sample_path, *base_synthetic]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _evaluate_transcript(
    path: Path,
    parser: TranscriptParser,
    extractor: DecisionExtractor,
) -> Tuple[List[Dict[str, Any]], int]:
    transcript_text = path.read_text(encoding="utf-8")
    segments = parser.parse(transcript_text)
    if not segments:
        # Synthetic files may be unnumbered ("Speaker: text").
        numbered_lines = []
        for idx, line in enumerate(transcript_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^\d+\s*\.\s*[^:\n]+:\s*.+$", stripped):
                numbered_lines.append(stripped)
            else:
                numbered_lines.append(f"{idx}. {stripped}")
        normalized = "\n".join(numbered_lines)
        segments = parser.parse(normalized)
    extracted_items = extractor.extract(segments)

    hallucinated = 0
    for item in extracted_items:
        source_line = str(item.get("source_line") or "")
        item_text = str(item.get("text") or "")
        if _is_hallucinated(source_line, transcript_text, item_text):
            hallucinated += 1

    return extracted_items, hallucinated


def main() -> None:
    transcript_paths = _load_transcript_paths()
    parser = TranscriptParser()
    extractor = DecisionExtractor()

    total_items = 0
    total_hallucinated = 0
    tested_transcripts: List[str] = []

    per_category_totals: Dict[str, int] = defaultdict(int)
    per_category_hallucinated: Dict[str, int] = defaultdict(int)
    per_category_conf_sum: Dict[str, float] = defaultdict(float)
    per_category_conf_count: Dict[str, int] = defaultdict(int)

    table_rows: List[Tuple[str, int, int, float]] = []

    for path in transcript_paths:
        extracted_items, hallucinated_for_transcript = _evaluate_transcript(path, parser, extractor)
        transcript_total = len(extracted_items)
        transcript_rate = _safe_rate(hallucinated_for_transcript, transcript_total)

        tested_transcripts.append(str(path))
        table_rows.append((path.name, transcript_total, hallucinated_for_transcript, transcript_rate))

        total_items += transcript_total
        total_hallucinated += hallucinated_for_transcript

        transcript_text = path.read_text(encoding="utf-8")
        for item in extracted_items:
            category = (item.get("category") or "DISCUSSION").upper()
            if category not in CATEGORIES:
                category = "DISCUSSION"

            per_category_totals[category] += 1
            conf = float(item.get("confidence", 0.0) or 0.0)
            per_category_conf_sum[category] += conf
            per_category_conf_count[category] += 1

            if _is_hallucinated(
                str(item.get("source_line") or ""),
                transcript_text,
                str(item.get("text") or ""),
            ):
                per_category_hallucinated[category] += 1

    overall_rate = _safe_rate(total_hallucinated, total_items)
    verdict = "PASS" if overall_rate <= TARGET_MAX_RATE else "FAIL"

    per_category_rates: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORIES:
        cat_total = int(per_category_totals.get(category, 0))
        cat_hallucinated = int(per_category_hallucinated.get(category, 0))
        cat_conf_count = int(per_category_conf_count.get(category, 0))
        avg_conf = (
            round(per_category_conf_sum[category] / cat_conf_count, 4)
            if cat_conf_count > 0
            else 0.0
        )
        per_category_rates[category] = {
            "total_items": cat_total,
            "hallucinated_items": cat_hallucinated,
            "hallucination_rate_percent": _safe_rate(cat_hallucinated, cat_total),
            "average_confidence": avg_conf,
        }

    report = {
        "total_items_extracted": total_items,
        "hallucinated_items": total_hallucinated,
        "hallucination_rate_percent": overall_rate,
        "per_category_rates": per_category_rates,
        "verdict": verdict,
        "tested_transcripts": tested_transcripts,
    }

    out_dir = PROJECT_ROOT / "data" / "example_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hallucination_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    name_w = max(len("Transcript"), max(len(row[0]) for row in table_rows))
    print("Hallucination Evaluation Summary")
    print("-" * (name_w + 36))
    print(f"{'Transcript'.ljust(name_w)} | {'Items':>5} | {'Hallucinated':>12} | {'Rate%':>7}")
    print("-" * (name_w + 36))
    for name, items, hallucinated, rate in table_rows:
        print(f"{name.ljust(name_w)} | {items:>5} | {hallucinated:>12} | {rate:>7.2f}")
    print("-" * (name_w + 36))
    print(f"{'TOTAL'.ljust(name_w)} | {total_items:>5} | {total_hallucinated:>12} | {overall_rate:>7.2f}")
    print()
    print(f"Saved report: {out_path}")
    print(f"Final verdict: {verdict} (target <= {TARGET_MAX_RATE:.2f}%)")


if __name__ == "__main__":
    main()
