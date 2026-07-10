import uuid

import pytest

import extract
import main
from main import extract_stage1_nodes


@pytest.fixture(autouse=True)
def force_heuristic_extraction(monkeypatch):
    # These tests target the deterministic heuristic classifier; without this,
    # a real OPENAI_API_KEY/ANTHROPIC_API_KEY in .env would route through the
    # live LLM instead, making tests flaky, network-dependent, and billable.
    monkeypatch.setattr(extract, "_llm_configured", lambda: False)


def test_extract_stage1_nodes_labels_barriers_facilitators_and_touch_points():
    transcript = (
        "I had a hard time getting to the clinic because the bus was late. "
        "The nurse was very supportive and helped me schedule an appointment. "
        "After that, I was able to talk to my doctor."
    )

    nodes, _ = extract_stage1_nodes(transcript)

    categories = [node.category for node in nodes]

    assert len(nodes) >= 3
    assert "barrier" in categories
    assert "facilitator" in categories
    assert any(category in {"state", "touch_point", "event"} for category in categories)

    barrier_node = next(node for node in nodes if node.category == "barrier")
    assert len(barrier_node.text.split()) <= 5
    assert "late" in barrier_node.text.lower() or "transport" in barrier_node.text.lower()


def test_extract_stage1_nodes_catches_afford_waitlist_and_awareness_barriers():
    transcript = (
        "I couldn't afford a bus pass so getting to interviews was tough. "
        "The waitlist was 6 months long which was so frustrating. "
        "I didn't know I qualified for Medicaid until a social worker told me."
    )

    nodes, _ = extract_stage1_nodes(transcript)
    categories = [node.category for node in nodes]

    assert categories.count("barrier") == 3


def test_extract_stage1_nodes_ignores_irrelevant_sentences():
    transcript = "The weather was nice. I ate lunch at noon."

    nodes, _ = extract_stage1_nodes(transcript)

    assert nodes == []


def test_extract_stage1_nodes_strips_apostrophes_cleanly():
    transcript = "I couldn't afford the copay so I skipped the visit."

    nodes, _ = extract_stage1_nodes(transcript)
    barrier_node = next(node for node in nodes if node.category == "barrier")

    assert "'" not in barrier_node.text


class _FakeExec:
    def __init__(self, row):
        self._row = row

    def execute(self):
        return type("Result", (), {"data": [self._row]})()


class _FakeTable:
    ID_COLUMNS = {
        "states": "state_id",
        "barriers": "barrier_id",
        "facilitators": "facilitator_id",
        "provenance": "provenance_id",
    }

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def insert(self, payload):
        self.calls.append((self.name, payload))
        row = dict(payload)
        row[self.ID_COLUMNS[self.name]] = f"{self.name}-{len(self.calls)}"
        return _FakeExec(row)


class _FakeSupabase:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return _FakeTable(name, self.calls)


def test_save_stage1_nodes_routes_categories_and_skips_unmapped(monkeypatch):
    fake_supabase = _FakeSupabase()
    monkeypatch.setattr(main, "supabase", fake_supabase)

    session_id = uuid.uuid4()
    media_id = uuid.uuid4()
    raw_nodes = [
        {
            "text": "clinic",
            "category": "state",
            "evidence": "went to the clinic",
            "confidence": 0.9,
            "description": None,
            "span_start": 0,
            "span_end": 19,
        },
        {
            "text": "transportation",
            "category": "barrier",
            "evidence": "the bus was late",
            "confidence": 0.8,
            "description": None,
            "span_start": None,
            "span_end": None,
        },
        {
            "text": "nurse support",
            "category": "facilitator",
            "evidence": "the nurse helped",
            "confidence": 0.85,
            "description": None,
            "span_start": None,
            "span_end": None,
        },
        {
            "text": "went for a walk",
            "category": "event",
            "evidence": "went for a walk",
            "confidence": 0.6,
            "description": None,
            "span_start": None,
            "span_end": None,
        },
    ]

    main.save_stage1_nodes(session_id, media_id, "text", raw_nodes, preferred_model=None)

    calls_by_table: dict[str, list[dict]] = {}
    for table_name, payload in fake_supabase.calls:
        calls_by_table.setdefault(table_name, []).append(payload)

    assert "events" not in calls_by_table
    assert len(calls_by_table["states"]) == 1
    assert calls_by_table["states"][0]["state_type"] == "clinical"
    assert "evidence_text" not in calls_by_table["states"][0]

    assert len(calls_by_table["barriers"]) == 1
    assert calls_by_table["barriers"][0]["barrier_type"] == "logistical"
    assert calls_by_table["barriers"][0]["evidence_text"] == "the bus was late"

    assert len(calls_by_table["facilitators"]) == 1
    assert calls_by_table["facilitators"][0]["facilitator_type"] == "person"

    # one provenance row per persisted node (state + barrier + facilitator); the
    # unmapped "event" node produces no entity row and therefore no provenance row
    assert len(calls_by_table["provenance"]) == 3
