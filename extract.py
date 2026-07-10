import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Literal, Optional

SYSTEM_PROMPT = """You are a psychologist working in clinical and behavioral health research, helping analyze interview transcripts about people navigating healthcare, justice, recovery, and social service systems.

Read the transcript carefully and identify every meaningful span of text that represents one of these entity types:

- state: A place, program, or situation a person visits or is in (for example: went to the ER, enrolled in drug court, released from prison, in a shelter, started outpatient treatment)
- barrier: A factor that blocked, delayed, or prevented access to a service or transition (for example: couldn't afford the copay, no transportation, waitlist was 6 months, didn't know the program existed, lost insurance)
- facilitator: A factor that helped or accelerated a transition or access to a service (for example: case manager helped me, church provided a ride, doctor wrote a referral, peer support worker connected me)

Rules:
- Extract all relevant spans, even if they seem minor.
- Use the exact words from the transcript for the excerpt field.
- span_start and span_end are character offsets into the original transcript (0-indexed).
- label should be a short 2-5 word phrase that names the entity.
- confidence is your certainty this is a real entity (0.0-1.0). Use less than 0.7 if you are unsure.
- Do not invent information. Only use what is in the transcript.
- Return only a valid JSON array. No explanation, no markdown, no preamble.

Each item must follow this exact structure:
{
  "entity_type": "state" | "barrier" | "facilitator",
  "label": "short label here",
  "description": "one sentence explaining what this represents",
  "excerpt": "exact quote from the transcript (max 200 characters)",
  "span_start": 0,
  "span_end": 50,
  "confidence": 0.92
}"""

PROMPT_VERSION = "v1"


def extract_stage1_nodes(
    transcript_text: str, preferred_model: Optional[str] = None
) -> tuple[list[dict[str, Any]], Literal["llm", "heuristic"]]:
    if not transcript_text or not transcript_text.strip():
        return [], "heuristic"

    if _llm_configured():
        try:
            raw_nodes = _extract_with_llm(transcript_text, preferred_model=preferred_model)
            if raw_nodes:
                return [_normalize_llm_node(node) for node in raw_nodes], "llm"
        except Exception:
            pass

    heuristic_nodes = _extract_with_rules(transcript_text)
    return heuristic_nodes, "heuristic"


def _normalize_llm_node(node: dict[str, Any]) -> dict[str, Any]:
    entity_type = (node.get("entity_type") or node.get("category") or "state").strip().lower()
    category = _map_entity_type(entity_type)
    excerpt = node.get("excerpt") or node.get("text") or ""
    normalized_text = _normalize_node_text(excerpt, category)
    return {
        "text": normalized_text or excerpt,
        "category": category,
        "evidence": excerpt,
        "confidence": float(node.get("confidence", 0.75)),
        "description": node.get("description"),
        "span_start": node.get("span_start"),
        "span_end": node.get("span_end"),
    }


def _map_entity_type(entity_type: str) -> str:
    if entity_type == "barrier":
        return "barrier"
    if entity_type == "facilitator":
        return "facilitator"
    if entity_type == "state":
        return "state"
    return "event"


def _llm_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def _extract_with_llm(
    transcript_text: str, preferred_model: Optional[str] = None
) -> list[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        payload = {
            "model": preferred_model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Please extract all entities from this transcript:\n\n{transcript_text}",
                },
            ],
            "temperature": 0.1,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return _parse_llm_json(content)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        payload = {
            "model": preferred_model or "claude-3-5-sonnet-latest",
            "max_tokens": 800,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": f"Please extract all entities from this transcript:\n\n{transcript_text}",
                }
            ],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body["content"][0]["text"]
            return _parse_llm_json(content)

    raise RuntimeError("No LLM API key configured")


def _parse_llm_json(content: str) -> list[dict[str, Any]]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if isinstance(data, dict) and "nodes" in data:
        return data["nodes"]
    if isinstance(data, list):
        return data
    return []


def _extract_with_rules(transcript_text: str) -> list[dict[str, Any]]:
    sentences = _split_sentences(transcript_text)
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()

    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())

        scores = _score_sentence(lowered)
        if max(scores.values(), default=0) == 0:
            continue

        category = _classify_sentence(lowered)
        confidence = min(0.95, 0.55 + 0.1 * max(scores.values()))
        nodes.append(
            {
                "text": _normalize_node_text(normalized, category),
                "category": category,
                "evidence": normalized,
                "confidence": confidence,
                "description": None,
                "span_start": None,
                "span_end": None,
            }
        )

    return nodes


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if segment.strip()]


BARRIER_HINTS = (
    "barrier",
    "hard",
    "difficult",
    "problem",
    "issue",
    "challenge",
    "struggle",
    "can't",
    "cannot",
    "unable",
    "late",
    "delay",
    "cost",
    "afford",
    "waitlist",
    "wait list",
    "frustrat",
    "didn't know",
    "wasn't aware",
    "lack",
    "risk",
    "worry",
    "pain",
    "stuck",
)

FACILITATOR_HINTS = (
    "help",
    "support",
    "supported",
    "helped",
    "encourage",
    "encouraged",
    "resource",
    "resources",
    "able",
    "easier",
    "improve",
    "improved",
    "good",
    "better",
    "strong",
    "trust",
    "family",
    "friend",
)

TOUCHPOINT_HINTS = (
    "clinic",
    "doctor",
    "nurse",
    "appointment",
    "care",
    "visit",
    "meeting",
    "session",
    "contact",
    "phone",
    "call",
    "talk",
    "discuss",
    "provider",
)

EVENT_HINTS = (
    "started",
    "began",
    "happened",
    "went",
    "after",
    "then",
    "when",
    "before",
    "during",
    "later",
)


def _score_sentence(lowered: str) -> dict[str, int]:
    return {
        "barrier": sum(1 for hint in BARRIER_HINTS if hint in lowered),
        "facilitator": sum(1 for hint in FACILITATOR_HINTS if hint in lowered),
        "touch_point": sum(1 for hint in TOUCHPOINT_HINTS if hint in lowered),
        "event": sum(1 for hint in EVENT_HINTS if hint in lowered),
    }


def _normalize_node_text(text: str, category: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""

    lowered = normalized.lower()
    if category == "barrier":
        if "because" in lowered:
            clause = lowered.split("because", 1)[1].strip()
            clause = re.sub(r"^(the|a|an)\s+", "", clause)
            clause = clause.rstrip(".,;:!? ")
            if "bus" in clause or "transport" in clause or "ride" in clause:
                return "transportation"
            if "late" in clause:
                return "bus was late"
            return clause[:80]
        if any(hint in lowered for hint in ("hard", "difficult", "problem", "issue", "challenge", "late", "delay", "cost", "lack")):
            return "barrier"
        return _shorten_phrase(normalized)

    if category == "facilitator":
        if "supportive" in lowered or "support" in lowered:
            for cue in ("nurse", "doctor", "counselor", "case manager", "provider", "peer", "friend", "family"):
                if cue in lowered:
                    return f"{cue} support"
        if "helped" in lowered:
            clause = re.sub(r".*?\bhelped\b", "", lowered, count=1)
            clause = re.sub(r"\bme\b", "", clause)
            clause = re.sub(r"^(the|a|an)\s+", "", clause)
            clause = clause.strip()
            if "schedule" in clause or "appointment" in clause:
                return "schedule appointment"
            if "talk" in clause or "doctor" in clause:
                return "talk to doctor"
            return _shorten_phrase(clause)
        return _shorten_phrase(normalized)

    if category == "touch_point":
        for cue in ("clinic", "doctor", "nurse", "appointment", "visit", "talk"):
            if cue in lowered:
                if cue == "appointment":
                    return "appointment"
                if cue == "talk":
                    return "doctor talk"
                return f"{cue} visit" if cue != "clinic" else "clinic visit"
        return _shorten_phrase(normalized)

    if category == "state":
        if "clinic" in lowered:
            return "clinic"
        if "treatment" in lowered:
            return "treatment"
        return _shorten_phrase(normalized)

    return _shorten_phrase(normalized)


def _shorten_phrase(text: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", " ", text.replace("'", "")).split()
    if not words:
        return ""
    if len(words) <= 4:
        return " ".join(words)
    return " ".join(words[:4])


def _classify_sentence(lowered: str) -> str:
    scores = _score_sentence(lowered)
    barrier_score = scores["barrier"]
    facilitator_score = scores["facilitator"]
    touch_point_score = scores["touch_point"]
    event_score = scores["event"]

    if barrier_score > facilitator_score and barrier_score >= touch_point_score:
        return "barrier"
    if facilitator_score > barrier_score and facilitator_score >= touch_point_score:
        return "facilitator"
    if touch_point_score > 0 and touch_point_score >= max(barrier_score, facilitator_score, event_score):
        return "touch_point"
    return "event"
