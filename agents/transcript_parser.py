import re
from typing import List, Dict, Any


# Matches "1. Alice: text" (number, speaker, text after first colon on that line)
_LINE_WITH_SPEAKER = re.compile(
    r"^\s*(\d+)\s*\.\s*([^:\n]+?)\s*:\s*(.*)$", re.DOTALL
)


class TranscriptParser:
    """Parses numbered meeting transcripts with optional continuation lines."""

    def parse(self, text: str) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue
            m = _LINE_WITH_SPEAKER.match(line)
            if m:
                line_number = int(m.group(1))
                speaker = m.group(2).strip()
                rest = m.group(3).strip()
                turn_index = len(segments)
                segments.append(
                    {
                        "turn_index": turn_index,
                        "speaker": speaker,
                        "text": rest,
                        "line_number": line_number,
                    }
                )
            elif segments:
                # Continuation: append to previous segment
                prev = segments[-1]
                prev["text"] = (prev["text"] + " " + line.lstrip()).strip()
            # Else: no previous segment to attach to; skip orphan lines
        return segments
