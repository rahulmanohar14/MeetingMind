from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agents.cross_meeting_memory as cmm
from agents.cross_meeting_memory import CrossMeetingMemory
from agents.decision_extractor import DecisionExtractor
from agents.transcript_parser import TranscriptParser


TARGET_RECALL = 0.70
OUTPUT_PATH = PROJECT_ROOT / "data" / "example_outputs" / "cross_meeting_recall_report.json"

SEQUENCE_1 = [
    """1. Alice: Let's discuss the vendor contract - we still haven't signed with DataCorp.
2. Bob: I'll follow up with DataCorp legal team by Friday.
3. Alice: Also the API integration is blocked until contract is signed.
4. Carol: What's our fallback if DataCorp falls through?
5. Alice: Let's circle back on the vendor question next week.""",
    """1. Alice: Update on DataCorp - still waiting on their legal review.
2. Bob: I emailed them but no response yet on the vendor contract.
3. Carol: This vendor contract issue is blocking two other workstreams.
4. Alice: Has anyone escalated the DataCorp vendor situation?
5. Bob: Let's revisit the vendor contract status in next meeting.""",
    """1. Alice: DataCorp vendor contract is still unresolved - third meeting in a row.
2. Bob: I'll escalate to our legal counsel today regarding the vendor contract.
3. Carol: We need to decide on the vendor contract by end of week.
4. Alice: Agreed - vendor contract resolution is our top priority.
5. Bob: What happens if the vendor contract falls through entirely?""",
]

SEQUENCE_2 = [
    """1. David: The Q3 budget approval is still pending from finance.
2. Emma: We can't start hiring until the budget is approved.
3. David: Finance said budget review takes two weeks.
4. Emma: Has anyone escalated the budget approval timeline?
5. David: Let's table the budget discussion until we hear back.""",
    """1. David: Budget approval is still stuck in finance review.
2. Emma: The budget delay is now affecting our hiring plan.
3. Frank: Can we get interim budget approval for at least the critical hires?
4. David: What's the status on the Q3 budget approval?
5. Emma: Let's revisit the budget situation next meeting.""",
    """1. David: Three weeks now and budget approval still pending.
2. Emma: The budget is blocking everything - we need a decision.
3. Frank: I'll meet with CFO directly to resolve the budget approval.
4. David: Budget approval needs to happen this week, no more delays.
5. Emma: Who is the final decision maker on the budget approval?""",
]

SEQUENCE_3 = [
    """1. Grace: The launch timeline keeps slipping - we were supposed to launch last month.
2. Henry: Engineering needs two more weeks for the authentication fix.
3. Grace: That pushes the launch timeline to end of quarter.
4. Iris: Has the launch timeline been communicated to customers?
5. Grace: Let's revisit the launch timeline once engineering is done.""",
    """1. Grace: Launch timeline update - engineering needs one more week.
2. Henry: The authentication issue is almost resolved, launch timeline should hold.
3. Iris: Marketing is ready but waiting on the launch timeline confirmation.
4. Grace: Can we commit to a launch timeline today?
5. Henry: Let's circle back on the launch timeline Thursday.""",
    """1. Grace: Launch timeline is our critical path item for this quarter.
2. Henry: Authentication fix is done - we can now lock the launch timeline.
3. Iris: We've been discussing the launch timeline for three meetings.
4. Grace: We're going with October 15th as the launch date - that's final.
5. Henry: Launch timeline confirmed - everyone needs to align their workstreams.""",
]

SEQUENCE_4 = [
    """1. Jack: Compliance review for the new data pipeline hasn't started yet.
2. Karen: Legal said compliance review could take up to three weeks.
3. Jack: We can't go to production without completing the compliance review.
4. Karen: Who owns the compliance review process?
5. Jack: Let's defer the compliance discussion to next week.""",
    """1. Jack: Compliance review update - legal team is reviewing now.
2. Karen: The compliance review is taking longer than expected.
3. Liam: Can we get a partial compliance sign-off to unblock production?
4. Jack: Has anyone checked the compliance review status with legal?
5. Karen: Let's circle back on compliance review once legal responds.""",
    """1. Jack: Compliance review is still the blocker for production release.
2. Karen: Legal finished the compliance review - there are three issues to fix.
3. Liam: I'll address the compliance review findings by Friday.
4. Jack: Compliance review has been discussed three meetings in a row now.
5. Karen: We need to close out the compliance review this week.""",
]

SEQUENCE_5 = [
    """1. Mike: The API performance issues are affecting our enterprise customers.
2. Nina: API response times are averaging 4 seconds - way too slow.
3. Mike: Has anyone identified the root cause of the API performance problem?
4. Omar: I'll investigate the API performance and report back by Wednesday.
5. Mike: Let's revisit the API performance issue next sprint.""",
    """1. Mike: API performance is still degraded - customers are complaining.
2. Nina: We've identified one bottleneck but API performance is still poor.
3. Omar: The API performance fix requires a database schema change.
4. Mike: What's the timeline for resolving the API performance issue?
5. Nina: Let's table the API performance fix until Omar's analysis is done.""",
    """1. Mike: API performance has been discussed for three meetings - we need resolution.
2. Nina: The API performance degradation is now a P1 issue.
3. Omar: I've completed the API performance analysis - here are the findings.
4. Mike: We're implementing the database optimization to fix API performance today.
5. Nina: API performance fix will be deployed by end of week - that's our commitment.""",
]

SEQUENCES = [
    ("Sequence 1", SEQUENCE_1, ["datacorp", "vendor"]),
    ("Sequence 2", SEQUENCE_2, ["budget approval", "budget"]),
    ("Sequence 3", SEQUENCE_3, ["launch timeline", "launch"]),
    ("Sequence 4", SEQUENCE_4, ["compliance review", "compliance"]),
    ("Sequence 5", SEQUENCE_5, ["api performance", "api"]),
]


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _topic_matches_text(topic: str, text: str) -> bool:
    topic_words = [w.strip().lower() for w in str(topic).split() if w.strip()]
    text_l = str(text).lower()
    return any(word in text_l for word in topic_words)


def _cleanup_paths(chroma_path: str, graph_path: str) -> None:
    cpath = PROJECT_ROOT / chroma_path
    gpath = PROJECT_ROOT / graph_path
    if cpath.exists():
        shutil.rmtree(cpath, ignore_errors=True)
    if gpath.exists():
        gpath.unlink()


def _evaluate_sequence(
    seq_num: int,
    seq_name: str,
    meetings: List[str],
    ground_truth_topics: List[str],
    parser: TranscriptParser,
    extractor: DecisionExtractor,
) -> Dict[str, Any]:
    temp_chroma = f"./data/chroma_db_test_{seq_num}"
    temp_graph = f"./data/graph_test_{seq_num}.json"
    _cleanup_paths(temp_chroma, temp_graph)

    original_chroma = cmm.CHROMA_PATH
    original_graph = cmm.GRAPH_PATH
    cmm.CHROMA_PATH = temp_chroma
    cmm.GRAPH_PATH = temp_graph
    try:
        memory = CrossMeetingMemory()
        for meeting_idx, meeting_text in enumerate(meetings, start=1):
            segments = parser.parse(meeting_text)
            extracted_items = extractor.extract(segments)
            memory.add_meeting(
                meeting_id=f"{seq_name.lower().replace(' ', '_')}_meeting_{meeting_idx}",
                extracted_items=extracted_items,
            )
        recurring = memory.check_recurring()
    finally:
        cmm.CHROMA_PATH = original_chroma
        cmm.GRAPH_PATH = original_graph
        _cleanup_paths(temp_chroma, temp_graph)

    recurring_texts = [str(item.get("text") or "").lower() for item in recurring]
    matched = []
    missed = []
    for topic in ground_truth_topics:
        if any(_topic_matches_text(topic, txt) for txt in recurring_texts):
            matched.append(topic)
        else:
            missed.append(topic)

    false_positives = 0
    for txt in recurring_texts:
        if not any(_topic_matches_text(topic, txt) for topic in ground_truth_topics):
            false_positives += 1

    return {
        "sequence_name": seq_name,
        "ground_truth_topics": ground_truth_topics,
        "flagged_count": len(recurring),
        "flagged_items": recurring,
        "correctly_flagged_topics": matched,
        "missed_topics": missed,
        "correctly_flagged_count": len(matched),
        "false_positives_count": false_positives,
    }


def main() -> None:
    parser = TranscriptParser()
    extractor = DecisionExtractor(mode="api")

    sequence_results = []
    for idx, (name, meetings, gt_topics) in enumerate(SEQUENCES, start=1):
        sequence_results.append(
            _evaluate_sequence(
                seq_num=idx,
                seq_name=name,
                meetings=meetings,
                ground_truth_topics=gt_topics,
                parser=parser,
                extractor=extractor,
            )
        )

    total_known = sum(len(res["ground_truth_topics"]) for res in sequence_results)
    correctly_flagged = sum(res["correctly_flagged_count"] for res in sequence_results)
    total_flagged = sum(res["flagged_count"] for res in sequence_results)
    false_positives = sum(res["false_positives_count"] for res in sequence_results)
    missed = total_known - correctly_flagged

    recall = _safe_div(correctly_flagged, total_known)
    precision = _safe_div(correctly_flagged, total_flagged)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    verdict = "PASS" if recall >= TARGET_RECALL else "FAIL"

    report = {
        "total_known_recurring_issues": total_known,
        "correctly_flagged": correctly_flagged,
        "missed": missed,
        "false_positives": false_positives,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "verdict": verdict,
        "sequence_results": sequence_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Cross-Meeting Recall Evaluation (Hardcoded Recurrence Sequences)")
    print("=" * 88)
    for res in sequence_results:
        print(f"\n{res['sequence_name']}")
        print(f"  Ground truth topics: {', '.join(res['ground_truth_topics'])}")
        print(f"  Flagged by system: {res['flagged_count']}")
        print(f"  Correctly flagged: {res['correctly_flagged_count']} -> {res['correctly_flagged_topics']}")
        print(f"  Missed: {len(res['missed_topics'])} -> {res['missed_topics']}")
        print(f"  False positives: {res['false_positives_count']}")
        if res["flagged_items"]:
            print("  Flagged issue texts:")
            for item in res["flagged_items"]:
                text = str(item.get("text") or "")
                appearances = item.get("appearance_count")
                print(f"    - ({appearances} appearances) {text}")
        else:
            print("  Flagged issue texts: []")

    print("\n" + "=" * 88)
    print("Final Metrics")
    print("=" * 88)
    print(f"Total known recurring issues: {total_known}")
    print(f"Correctly flagged: {correctly_flagged}")
    print(f"Missed: {missed}")
    print(f"False positives: {false_positives}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Verdict: {verdict} (target recall >= {TARGET_RECALL:.2f})")
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
