from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.accountability_tracker import AccountabilityTracker
from agents.cross_meeting_memory import CrossMeetingMemory
from agents.decision_extractor import DecisionExtractor
from agents.report_generator import ReportGenerator
from agents.transcript_parser import TranscriptParser


def _count_by_category(extracted_items: List[Dict[str, Any]], category: str) -> int:
    cat = category.upper()
    return sum(1 for item in extracted_items if (item.get("category") or "").upper() == cat)


def _build_stats(
    meeting_id: str,
    segments: List[Dict[str, Any]],
    extracted_items: List[Dict[str, Any]],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "meeting_id": meeting_id,
        "total_turns": len(segments),
        "decisions_count": _count_by_category(extracted_items, "DECISION"),
        "action_items_count": _count_by_category(extracted_items, "ACTION_ITEM"),
        "open_questions_count": _count_by_category(extracted_items, "OPEN_QUESTION"),
        "deferrals_count": _count_by_category(extracted_items, "DEFERRAL"),
        "discussion_count": _count_by_category(extracted_items, "DISCUSSION"),
        "high_risk_items_count": len(report.get("high_risk_items", []) or []),
        "low_confidence_items_count": len(report.get("low_confidence_items", []) or []),
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    input_path = PROJECT_ROOT / "data" / "sample_transcript.txt"
    output_dir = PROJECT_ROOT / "data" / "example_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input transcript: {input_path}")

    meeting_id = "sample_meeting_001"
    transcript_text = input_path.read_text(encoding="utf-8")

    parser = TranscriptParser()
    extractor = DecisionExtractor()
    memory = CrossMeetingMemory()
    tracker = AccountabilityTracker()
    reporter = ReportGenerator()

    segments = parser.parse(transcript_text)
    extracted_items = extractor.extract(segments)

    # Keep parity with app pipeline for recurring issue detection.
    memory.add_meeting(meeting_id, extracted_items)
    memory.save_graph()
    recurring_issues = memory.check_recurring()

    tracker.track(extracted_items, meeting_id)
    tracker.save()
    accountability_data = tracker.get_dashboard_data()

    structured_output = reporter.generate(
        extracted_items=extracted_items,
        recurring_issues=recurring_issues,
        accountability_data=accountability_data,
        meeting_id=meeting_id,
    )
    markdown_report = reporter.generate_markdown(structured_output)
    stats = _build_stats(meeting_id, segments, extracted_items, structured_output)

    structured_json_path = output_dir / "sample_output.json"
    markdown_path = output_dir / "sample_report.md"
    stats_path = output_dir / "sample_stats.json"

    structured_json_path.write_text(
        json.dumps(structured_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(markdown_report, encoding="utf-8")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Example output generation complete.")
    print(f"Saved structured JSON: {structured_json_path}")
    print(f"Saved markdown report: {markdown_path}")
    print(f"Saved summary stats JSON: {stats_path}")


if __name__ == "__main__":
    main()
