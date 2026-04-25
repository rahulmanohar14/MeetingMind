import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_NAME

_LOW_CONF = "⚠️"
_HIGH_RISK = "🔴"


class ReportGenerator:
    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

    def _meeting_summary(
        self, extracted_items: list, meeting_id: str
    ) -> str:
        try:
            brief = [
                f"{(x.get('category') or '')}: {x.get('text', '')[:500]}"
                for x in extracted_items[:30]
            ]
            resp = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write 2-4 sentence executive meeting summaries. "
                            "Be concrete; name participants when relevant. No JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Meeting {meeting_id}.\n" + "\n".join(brief),
                    },
                ],
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return f"Report for {meeting_id} based on {len(extracted_items)} classified utterance(s)."

    @staticmethod
    def _group_items(
        items: list, category: str
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        c = category.upper()
        for it in items:
            if (it.get("category") or "").upper() != c:
                continue
            row: Dict[str, Any] = {
                "text": it.get("text", ""),
                "source_line": it.get("source_line", ""),
                "speaker": it.get("speaker", ""),
            }
            if c == "ACTION_ITEM":
                row["owner"] = it.get("owner")
                row["verb"] = it.get("verb")
                row["deadline"] = it.get("deadline")
                row["low_confidence"] = it.get("low_confidence", False)
                row["completion_risk"] = it.get("completion_risk", "low")
            out.append(row)
        return out

    def generate(
        self,
        extracted_items: list,
        recurring_issues: list,
        accountability_data: dict,
        meeting_id: str,
    ) -> dict:
        low_confidence_items: List[Dict[str, Any]] = []
        high_risk_items: List[Dict[str, Any]] = []
        for it in extracted_items:
            if it.get("low_confidence"):
                low_confidence_items.append(
                    {
                        "text": it.get("text", ""),
                        "source_line": it.get("source_line", ""),
                        "reason": (it.get("reasoning", "") or "Low model confidence")
                        if isinstance(it, dict)
                        else "",
                    }
                )
            if (
                (it.get("category") or "").upper() == "ACTION_ITEM"
                and it.get("completion_risk") == "high"
            ):
                high_risk_items.append(
                    {
                        "text": it.get("text", ""),
                        "source_line": it.get("source_line", ""),
                        "reason": "Missing owner or deadline on action item",
                    }
                )
        return {
            "meeting_id": meeting_id,
            "summary": self._meeting_summary(extracted_items, meeting_id),
            "decisions": self._group_items(extracted_items, "DECISION"),
            "action_items": self._group_items(extracted_items, "ACTION_ITEM"),
            "open_questions": self._group_items(
                extracted_items, "OPEN_QUESTION"
            ),
            "deferrals": self._group_items(extracted_items, "DEFERRAL"),
            "recurring_issues": list(recurring_issues or []),
            "accountability": accountability_data
            if isinstance(accountability_data, dict)
            else {},
            "low_confidence_items": low_confidence_items,
            "high_risk_items": high_risk_items,
        }

    @staticmethod
    def _fmt_item_body(it: Dict[str, Any]) -> str:
        parts: List[str] = []
        t = (it.get("text") or "").strip()
        if t:
            parts.append(t)
        if (it.get("owner") is not None) or (it.get("verb") is not None):
            o = it.get("owner")
            v = it.get("verb")
            d = it.get("deadline")
            extra = f" (owner: {o}; verb: {v}; due: {d})"
            if extra.strip() != " (owner: None; verb: None; due: None)":
                parts[0:1] = [parts[0] + " " + extra.strip()] if parts else [extra]
        return "\n\n".join(parts) if parts else "—"

    def generate_markdown(self, report: dict) -> str:
        lines: List[str] = []
        lines.append(
            f"# Meeting report: {report.get('meeting_id', 'unknown')}\n"
        )
        lines.append(f"## Summary\n{report.get('summary', '')}\n")

        def add_section(
            title: str,
            key: str,
            mark_low: bool = False,
            mark_high: bool = False,
        ) -> None:
            lines.append(f"## {title}\n")
            for it in report.get(key) or []:
                if isinstance(it, str):
                    lines.append(f"- {it}\n\n")
                    continue
                if not isinstance(it, dict):
                    continue
                body = self._fmt_item_body(it) if "text" in it else str(it)
                pre = []
                if mark_low and it.get("low_confidence"):
                    pre.append(f"{_LOW_CONF} Low confidence extraction. ")
                if mark_high and it.get("completion_risk") == "high":
                    pre.append(
                        f"{_HIGH_RISK} High completion risk — missing owner or deadline. "
                    )
                if pre:
                    body = ("".join(pre) + body).strip()
                lines.append(f"- {body}\n")
                sl = (it.get("source_line") or "").strip()
                if sl:
                    lines.append(f"  *{sl}*\n")
                lines.append("")

        add_section("Decisions", "decisions")
        add_section("Action items", "action_items", mark_high=True)
        add_section("Open questions", "open_questions")
        add_section("Deferrals", "deferrals", mark_low=True)

        lines.append("## Recurring issues\n")
        for it in report.get("recurring_issues", []) or []:
            if isinstance(it, str):
                lines.append(f"- {it}\n")
            elif isinstance(it, dict) and (it.get("text") or it.get("source_line")):
                t = it.get("text", "")
                sl = (it.get("source_line", "") or "").strip()
                lines.append(f"- {t}\n")
                if sl:
                    lines.append(f"  *{sl}*\n")
        lines.append("")

        acc = report.get("accountability", {})
        lines.append("## Accountability\n")
        lines.append(
            "```\n" + json.dumps(acc, ensure_ascii=False, indent=2) + "\n```\n"
        )
        lines.append("")

        lines.append("## Low confidence\n")
        for it in report.get("low_confidence_items", []) or []:
            if isinstance(it, dict):
                lines.append(
                    f"- {_LOW_CONF} {it.get('text', '') or it.get('reason', '')}\n"
                )
                sl = (it.get("source_line", "") or "").strip()
                if sl:
                    lines.append(f"  *{sl}*\n")
        lines.append("## High risk\n")
        for it in report.get("high_risk_items", []) or []:
            if isinstance(it, dict):
                lines.append(
                    f"- {_HIGH_RISK} {it.get('text', '') or it.get('reason', '')}\n"
                )
                sl = (it.get("source_line", "") or "").strip()
                if sl:
                    lines.append(f"  *{sl}*\n")
        return "\n".join(lines).strip() + "\n"
