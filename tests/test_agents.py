from unittest.mock import patch

from agents.accountability_tracker import AccountabilityTracker
from agents.report_generator import ReportGenerator
from agents.transcript_parser import TranscriptParser

VALID = frozenset(
    {
        "DECISION",
        "ACTION_ITEM",
        "OPEN_QUESTION",
        "DEFERRAL",
        "DISCUSSION",
    }
)


def test_transcript_parser_basic():
    p = TranscriptParser()
    out = p.parse("1. Alice: Hello\n2. Bob: How are you")
    assert len(out) == 2
    assert out[0]["speaker"] == "Alice"
    assert out[0]["text"] == "Hello"
    assert out[1]["speaker"] == "Bob"


def test_transcript_parser_blank_lines():
    p = TranscriptParser()
    text = "1. Alice: Hi\n\n2. Bob: Hey there\n\n"
    out = p.parse(text)
    assert not any(
        (not s.get("text", "").strip()) for s in out
    ), "no empty text segments"
    assert len(out) == 2


def test_accountability_ratio():
    t = AccountabilityTracker()
    t._stats = {"Alice": {"assigned": 0, "completed": 0, "items": []}}
    base = {
        "category": "ACTION_ITEM",
        "text": "x",
        "owner": "Alice",
        "verb": "v",
        "deadline": "d",
        "source_line": "Line 1: Alice: x",
    }
    t.track(
        [base, dict(base, source_line="Line 2"), dict(base, source_line="Line 3")],
        "m1",
    )
    t._stats["Alice"]["completed"] = 1
    AccountabilityTracker._recompute_ratio(t._stats["Alice"])
    assert round(t._stats["Alice"]["follow_through_ratio"], 2) == 0.33


@patch.object(ReportGenerator, "_meeting_summary", return_value="Test summary.")
def test_report_has_required_keys(_mock):
    g = ReportGenerator()
    items = [
        {
            "category": "DISCUSSION",
            "text": "hello",
            "speaker": "A",
            "source_line": "Line 1: A: hello",
            "low_confidence": False,
            "reasoning": "r",
        },
        {
            "category": "DECISION",
            "text": "d",
            "speaker": "B",
            "source_line": "Line 2: B: d",
            "low_confidence": True,
            "reasoning": "r2",
        },
    ]
    r = g.generate(items, [], {}, "m1")
    for k in (
        "meeting_id",
        "summary",
        "decisions",
        "action_items",
        "open_questions",
        "deferrals",
        "recurring_issues",
        "accountability",
        "low_confidence_items",
        "high_risk_items",
    ):
        assert k in r


def test_valid_categories():
    assert VALID == frozenset(
        {
            "DECISION",
            "ACTION_ITEM",
            "OPEN_QUESTION",
            "DEFERRAL",
            "DISCUSSION",
        }
    )
