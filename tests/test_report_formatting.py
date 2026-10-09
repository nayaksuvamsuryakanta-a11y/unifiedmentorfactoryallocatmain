import os
import re
from pathlib import Path


REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", Path(__file__).parents[1]))


def _outside_fenced_code(text):
    lines = text.splitlines()
    visible = []
    fence_char = None
    fence_length = 0

    for line in lines:
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence_char is None:
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = None
            continue
        if fence_char is None:
            visible.append(line)

    return visible


def test_generated_reports_have_valid_tables_and_singular_counts():
    for report_name in ("research_paper.md", "executive_summary.md"):
        report_path = REPORTS_DIR / report_name
        lines = _outside_fenced_code(report_path.read_text(encoding="utf-8"))
        previous_line = ""

        for line_number, line in enumerate(lines, start=1):
            if re.match(r"^\s*\|", line):
                assert line.startswith("|"), f"{report_name}:{line_number}: table row is indented"
                if not previous_line.strip().startswith("|"):
                    assert previous_line == "", f"{report_name}:{line_number}: table lacks a preceding blank line"
            previous_line = line

        text = "\n".join(lines)
        assert not re.search(r"\b1\s+(?:products|days|orders)\b", text, re.IGNORECASE), report_name