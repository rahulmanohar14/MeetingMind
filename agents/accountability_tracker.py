import json
import os
from typing import Any, Dict, List

ACCOUNTABILITY_PATH = "./data/accountability.json"


class AccountabilityTracker:
    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, Any]] = {}
        if os.path.isfile(ACCOUNTABILITY_PATH):
            with open(ACCOUNTABILITY_PATH, "r", encoding="utf-8") as f:
                self._stats = json.load(f)

    @staticmethod
    def _recompute_ratio(owner_data: Dict[str, Any]) -> None:
        assigned = int(owner_data.get("assigned", 0) or 0)
        completed = int(owner_data.get("completed", 0) or 0)
        if assigned <= 0:
            owner_data["follow_through_ratio"] = 0.0
        else:
            owner_data["follow_through_ratio"] = round(completed / assigned, 4)

    def _ensure_owner(self, owner: str) -> Dict[str, Any]:
        if owner not in self._stats:
            self._stats[owner] = {
                "assigned": 0,
                "completed": 0,
                "follow_through_ratio": 0.0,
                "items": [],
            }
        return self._stats[owner]

    def track(self, extracted_items: list, meeting_id: str) -> dict:
        for item in extracted_items:
            if (item.get("category") or "").upper() != "ACTION_ITEM":
                continue
            owner = item.get("owner")
            if not owner:
                # Still count in meeting-level if needed; per spec stats are per owner
                continue
            o = self._ensure_owner(str(owner).strip())
            o["assigned"] = int(o.get("assigned", 0)) + 1
            o["items"] = o.get("items") or []
            o["items"].append(
                {
                    "meeting_id": meeting_id,
                    "text": item.get("text"),
                    "verb": item.get("verb"),
                    "deadline": item.get("deadline"),
                    "source_line": item.get("source_line"),
                }
            )
            self._recompute_ratio(o)
        return dict(self.get_dashboard_data())

    def save(self) -> None:
        os.makedirs("./data", exist_ok=True)
        with open(ACCOUNTABILITY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._stats, f, ensure_ascii=False, indent=2)

    def get_dashboard_data(self) -> dict:
        for odata in self._stats.values():
            self._recompute_ratio(odata)
        return dict(self._stats)
