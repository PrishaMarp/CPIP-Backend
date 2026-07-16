import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Literal, Optional

SYSTEM_PROMPT = """You are coding qualitative interview transcripts describing a person's journey through systems related to opioid use, treatment, and recovery. Your job is to identify the STATES the person passed through, the TRANSITIONS between those states, and any BARRIERS or FACILITATORS that affected those states or transitions.

DEFINITIONS

A STATE is a status, location, or condition the person was in — a point in their journey, not an action. Every state must have a state_type:
- clinical: medical care, hospitals, prescribing providers, MAT clinics
- justice: courts, jail, probation, legal involvement
- recovery: peer support, sober living, recovery programs
- social_service: harm reduction orgs, patient navigators, county services, housing
- risk_event: overdose, withdrawal, relapse episode, crisis moment
- administrative: insurance status, paperwork, waitlists, eligibility processes

A TRANSITION is a move from one state to another, with a transition_type:
- referral: formally referred from one provider/system to another
- discharge: released or exited from a state (e.g. discharged from ER)
- relapse: return to substance use after a period of recovery
- reentry: returning to a system after having left it (e.g. reentry to treatment after a gap)
- dropout: disengaged or stopped participating without formal discharge
- handoff: informal transfer of care or information between people/systems

A BARRIER is something that impeded the person's progress, with a barrier_type:
- structural: systemic issues like understaffing, policy gaps, eligibility rules
- financial: cost, insurance, ability to pay
- logistical: transportation, scheduling, wait times, distance
- stigma: judgment, shame, fear of consequences from disclosure
- knowledge: not knowing a resource exists, unclear information
- social: relationships, lack of support, family/social dynamics

A FACILITATOR is something that helped the person's progress, with a facilitator_type:
- person: an individual who helped (peer, provider, family member, staff)
- program: a specific service or program that helped
- policy: a rule or policy that enabled access (e.g. walk-in policy, no insurance required)
- resource: a material resource (transportation voucher, printed materials, phone)
- relationship: an existing relationship or connection that enabled access

CRITICAL RULES

1. For every state, transition, barrier, and facilitator you extract, you MUST include an "evidence_text" field containing an EXACT VERBATIM QUOTE from the transcript that supports it. Do not paraphrase or summarize — copy the exact words. This is required for source traceability.
2. Only extract what is actually supported by the text. Do not infer states, transitions, barriers, or facilitators that aren't clearly present.
3. Assign severity (barriers) / impact (facilitators) as minor/moderate/major, and risk_level (transitions) as low/medium/high/critical, based on how the person describes the consequence — err toward "moderate"/"medium" if genuinely ambiguous.
4. A barrier or facilitator should be linked to either a state OR a transition, not both — pick whichever it most directly affected, and put that element's temp_id in the "affects" field.
5. Use temp_id values (e.g. "s1", "t1", "b1", "f1") to reference elements within your own response only — these are local identifiers, not permanent IDs. Transitions reference states via "from_state"/"to_state" temp_ids.

EXAMPLE

Transcript excerpt: "I called all three. One wasn't taking new patients. One wanted like eight hundred dollars for an intake because they didn't take my insurance. The third one, the wait was six weeks out."

Correct extraction includes a barrier:
{"temp_id": "b1", "label": "High cost of intake without insurance", "barrier_type": "financial", "severity": "major", "evidence_text": "One wanted like eight hundred dollars for an intake because they didn't take my insurance."}

and a second barrier:
{"temp_id": "b2", "label": "Long wait time for clinic intake", "barrier_type": "logistical", "severity": "major", "evidence_text": "the wait was six weeks out"}

Return a JSON object with this exact top-level shape: {"states": [...], "transitions": [...], "barriers": [...], "facilitators": [...]}. Include a "confidence" field (0.0-1.0) on every item."""

RESPONSE_JSON_SCHEMA = {
    "name": "transcript_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "states": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "temp_id": {"type": "string"},
                        "label": {"type": "string"},
                        "state_type": {
                            "type": "string",
                            "enum": ["clinical", "justice", "recovery", "social_service", "risk_event", "administrative"],
                        },
                        "evidence_text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["temp_id", "label", "state_type", "evidence_text", "confidence"],
                    "additionalProperties": False,
                },
            },
            "transitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "temp_id": {"type": "string"},
                        "from_state": {"type": "string"},
                        "to_state": {"type": "string"},
                        "transition_type": {
                            "type": "string",
                            "enum": ["referral", "discharge", "relapse", "reentry", "dropout", "handoff"],
                        },
                        "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "evidence_text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "temp_id", "from_state", "to_state", "transition_type", "risk_level",
                        "evidence_text", "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "barriers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "temp_id": {"type": "string"},
                        "label": {"type": "string"},
                        "barrier_type": {
                            "type": "string",
                            "enum": ["structural", "financial", "logistical", "stigma", "knowledge", "social"],
                        },
                        "severity": {"type": "string", "enum": ["minor", "moderate", "major"]},
                        "affects": {"type": "string"},
                        "evidence_text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "temp_id", "label", "barrier_type", "severity", "affects", "evidence_text", "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "facilitators": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "temp_id": {"type": "string"},
                        "label": {"type": "string"},
                        "facilitator_type": {
                            "type": "string",
                            "enum": ["person", "program", "policy", "resource", "relationship"],
                        },
                        "impact": {"type": "string", "enum": ["minor", "moderate", "major"]},
                        "affects": {"type": "string"},
                        "evidence_text": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "temp_id", "label", "facilitator_type", "impact", "affects", "evidence_text", "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["states", "transitions", "barriers", "facilitators"],
        "additionalProperties": False,
    },
}

PROMPT_VERSION = "v2"


def extract_stage1_nodes(
    transcript_text: str, preferred_model: Optional[str] = None
) -> tuple[list[dict[str, Any]], Literal["llm", "heuristic"]]:
    if not transcript_text or not transcript_text.strip():
        return [], "heuristic"

    if _llm_configured():
        try:
            raw_result = _extract_with_llm(transcript_text, preferred_model=preferred_model)
            nodes = _flatten_llm_result(raw_result)
            if nodes:
                return nodes, "llm"
        except Exception:
            pass

    heuristic_nodes = _extract_with_rules(transcript_text)
    return heuristic_nodes, "heuristic"


def _flatten_llm_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flattens the LLM's {states, transitions, barriers, facilitators} object
    into the unified node-dict shape main.py persists, preserving temp_id
    references so relationships (transition endpoints, affected state/transition)
    can be resolved to real DB ids after states/transitions are inserted."""
    nodes: list[dict[str, Any]] = []

    for state in result.get("states", []):
        nodes.append({
            "temp_id": state.get("temp_id"),
            "text": state.get("label", ""),
            "category": "state",
            "evidence": state.get("evidence_text", ""),
            "confidence": float(state.get("confidence", 0.75)),
            "description": None,
            "span_start": None,
            "span_end": None,
            "type": state.get("state_type"),
            "impact": None,
            "affects_temp_id": None,
            "from_temp_id": None,
            "to_temp_id": None,
        })

    for transition in result.get("transitions", []):
        nodes.append({
            "temp_id": transition.get("temp_id"),
            "text": transition.get("transition_type", "transition"),
            "category": "transition",
            "evidence": transition.get("evidence_text", ""),
            "confidence": float(transition.get("confidence", 0.75)),
            "description": None,
            "span_start": None,
            "span_end": None,
            "type": transition.get("transition_type"),
            "impact": transition.get("risk_level"),
            "affects_temp_id": None,
            "from_temp_id": transition.get("from_state"),
            "to_temp_id": transition.get("to_state"),
        })

    for barrier in result.get("barriers", []):
        nodes.append({
            "temp_id": barrier.get("temp_id"),
            "text": barrier.get("label", ""),
            "category": "barrier",
            "evidence": barrier.get("evidence_text", ""),
            "confidence": float(barrier.get("confidence", 0.75)),
            "description": None,
            "span_start": None,
            "span_end": None,
            "type": barrier.get("barrier_type"),
            "impact": barrier.get("severity"),
            "affects_temp_id": barrier.get("affects"),
            "from_temp_id": None,
            "to_temp_id": None,
        })

    for facilitator in result.get("facilitators", []):
        nodes.append({
            "temp_id": facilitator.get("temp_id"),
            "text": facilitator.get("label", ""),
            "category": "facilitator",
            "evidence": facilitator.get("evidence_text", ""),
            "confidence": float(facilitator.get("confidence", 0.75)),
            "description": None,
            "span_start": None,
            "span_end": None,
            "type": facilitator.get("facilitator_type"),
            "impact": facilitator.get("impact"),
            "affects_temp_id": facilitator.get("affects"),
            "from_temp_id": None,
            "to_temp_id": None,
        })

    return nodes


def _llm_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def _extract_with_llm(
    transcript_text: str, preferred_model: Optional[str] = None
) -> dict[str, Any]:
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
            "response_format": {"type": "json_schema", "json_schema": RESPONSE_JSON_SCHEMA},
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return _parse_llm_json(content)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        payload = {
            "model": preferred_model or "claude-3-5-sonnet-latest",
            "max_tokens": 4000,
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
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body["content"][0]["text"]
            return _parse_llm_json(content)

    raise RuntimeError("No LLM API key configured")


def _parse_llm_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    return data if isinstance(data, dict) else {}


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
