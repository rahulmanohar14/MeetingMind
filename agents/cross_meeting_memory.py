import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import chromadb
import networkx as nx

CHROMA_PATH = "./data/chroma_db"
GRAPH_PATH = "./data/graph.json"
COLLECTION = "meeting_issues"
CATEGORIES = frozenset({"DECISION", "ACTION_ITEM", "OPEN_QUESTION"})
SIMILARITY_THRESHOLD = 1.0


def _node_id_for_text(text: str) -> str:
    h = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
    return f"issue_{h}"


class CrossMeetingMemory:
    def __init__(self) -> None:
        os.makedirs("./data", exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION,
        )
        self.graph: nx.DiGraph = nx.DiGraph()
        if os.path.isfile(GRAPH_PATH):
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, edges="links")
            # Backward-compatible normalization of node schema.
            for node_id, attrs in list(self.graph.nodes(data=True)):
                if attrs.get("type") == "meeting":
                    continue
                meetings = attrs.get("meetings")
                if meetings is None:
                    meetings = attrs.get("meeting_ids", [])
                self.graph.nodes[node_id]["meetings"] = list(meetings or [])
                if self.graph.nodes[node_id].get("resolution_status") is None:
                    self.graph.nodes[node_id]["resolution_status"] = "unresolved"

    def add_meeting(self, meeting_id: str, extracted_items: list) -> None:
        meeting_node_id = f"meeting::{meeting_id}"
        if meeting_node_id not in self.graph:
            self.graph.add_node(meeting_node_id, type="meeting", meeting_id=meeting_id)

        for item in extracted_items:
            category = (item.get("category") or "").upper()
            if category not in CATEGORIES:
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue

            source = item.get("source_line") or f"{item.get('speaker')}: {text}"

            # Look for an existing similar issue in Chroma first.
            similar = self.query_similar(text)
            best = similar[0] if similar else None
            best_id = (best or {}).get("id")
            best_distance = (best or {}).get("distance")
            if isinstance(best_distance, (float, int)):
                print(
                    f"[CrossMeetingMemory] item='{text[:80]}' top_match_id={best_id} distance={float(best_distance):.4f}"
                )
            else:
                print(
                    f"[CrossMeetingMemory] item='{text[:80]}' top_match_id={best_id} distance=None"
                )
            is_match = (
                isinstance(best_id, str)
                and best_id in self.graph
                and isinstance(best_distance, (float, int))
                and float(best_distance) < SIMILARITY_THRESHOLD
            )

            if is_match:
                nid = str(best_id)
                meetings: List[str] = list(self.graph.nodes[nid].get("meetings", []))
                if meeting_id not in meetings:
                    meetings.append(meeting_id)
                self.graph.nodes[nid]["meetings"] = meetings
                self.graph.nodes[nid]["text"] = self.graph.nodes[nid].get("text") or text
                self.graph.nodes[nid]["resolution_status"] = (
                    self.graph.nodes[nid].get("resolution_status") or "unresolved"
                )
                print(
                    f"[CrossMeetingMemory] MATCHED existing node={nid} meetings_count={len(meetings)}"
                )
            else:
                nid = _node_id_for_text(text)
                # Ensure unique id if hashing collides with an existing different issue node.
                if nid in self.graph and self.graph.nodes[nid].get("text") != text:
                    suffix = 1
                    while f"{nid}_{suffix}" in self.graph:
                        suffix += 1
                    nid = f"{nid}_{suffix}"
                self.graph.add_node(
                    nid,
                    node_id=nid,
                    text=text,
                    meetings=[meeting_id],
                    resolution_status="unresolved",
                )
                print(
                    f"[CrossMeetingMemory] CREATED new node={nid} meetings_count=1"
                )

            # Represent each mention as a meeting -> issue edge.
            self.graph.add_edge(meeting_node_id, nid, category=category, source=(source or "")[:2000])

            # Keep the Chroma document mapped to the canonical issue node id.
            self._collection.upsert(
                ids=[nid],
                documents=[text],
                metadatas=[
                    {
                        "meeting_id": meeting_id,
                        "category": category,
                        "source": (source or "")[:2000],
                    }
                ],
            )

    def check_recurring(self) -> list:
        out: List[Dict[str, Any]] = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "meeting":
                continue
            meetings = data.get("meetings") or []
            if len(meetings) >= 3 and (data.get("resolution_status") != "resolved"):
                out.append(
                    {
                        "node_id": node_id,
                        "text": data.get("text"),
                        "meetings": meetings,
                        # Compatibility key for existing UI/readers.
                        "meeting_ids": meetings,
                        "resolution_status": data.get("resolution_status"),
                        "appearance_count": len(meetings),
                    }
                )
        return out

    def query_similar(self, issue_text: str) -> list:
        if not issue_text.strip():
            return []
        res = self._collection.query(
            query_texts=[issue_text.strip()],
            n_results=3,
        )
        out: List[Dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])
        distances = dists[0] if dists and dists[0] else [None] * len(ids)
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "text": documents[i] if i < len(documents) else None,
                    "metadata": metas[i] if i < len(metas) else None,
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return out

    def save_graph(self) -> None:
        os.makedirs("./data", exist_ok=True)
        data = nx.node_link_data(self.graph, edges="links")
        with open(GRAPH_PATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))

    def mark_resolved(self, node_id: str) -> None:
        if node_id in self.graph:
            self.graph.nodes[node_id]["resolution_status"] = "resolved"
