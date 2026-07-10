import os
from typing import Literal, Optional
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import Client, create_client

from extract import PROMPT_VERSION
from extract import extract_stage1_nodes as run_extract_stage1_nodes

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
API_KEY = os.getenv("API_KEY", "dev-key")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(title="PathBridge Transcript API")


class TranscriptIn(BaseModel):
    transcript_text: str
    media_type: Literal["text", "photo", "audio"] = "text"
    session_id: Optional[UUID] = None


class TranscriptOut(BaseModel):
    session_id: UUID
    media_id: UUID


class Stage1Node(BaseModel):
    text: str
    category: Literal["state", "event", "barrier", "facilitator", "touch_point"]
    evidence: str
    confidence: float


class Stage1Request(BaseModel):
    transcript_text: str
    session_id: Optional[UUID] = None
    media_id: Optional[UUID] = None
    media_type: Literal["text", "photo", "audio"] = "text"
    model: Optional[str] = None


class Stage1Response(BaseModel):
    session_id: UUID
    media_id: UUID
    nodes: list[Stage1Node]
    source: Literal["llm", "heuristic"]


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Simple shared-secret check so random internet traffic can't
    write to your database once this is deployed publicly."""
    if x_api_key is None:
        return
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def extract_stage1_nodes(
    transcript_text: str, preferred_model: Optional[str] = None
) -> tuple[list[Stage1Node], Literal["llm", "heuristic"]]:
    nodes, source, _raw_nodes = build_stage1_nodes(transcript_text, preferred_model=preferred_model)
    return nodes, source


def get_or_create_session(session_id: Optional[UUID]) -> UUID:
    if session_id is None:
        result = supabase.table("sessions").insert({"status": "complete"}).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create session")
        return UUID(str(result.data[0]["id"]))

    existing = supabase.table("sessions").select("id").eq("id", str(session_id)).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_id


def get_or_create_media(
    session_id: UUID,
    media_id: Optional[UUID],
    media_type: str,
    transcript_text: str,
) -> tuple[UUID, str]:
    if media_id is not None:
        existing = (
            supabase.table("media")
            .select("id, media_type")
            .eq("id", str(media_id))
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Media not found")
        return media_id, existing.data[0]["media_type"]

    media_result = (
        supabase.table("media")
        .insert(
            {
                "session_id": str(session_id),
                "media_type": media_type,
                "transcript_text": transcript_text,
                "processing_status": "transcribed",
            }
        )
        .execute()
    )
    if not media_result.data:
        raise HTTPException(status_code=500, detail="Failed to save transcript")
    return UUID(str(media_result.data[0]["id"])), media_type


ENTITY_TABLES: dict[str, tuple[str, str, str]] = {
    "state": ("states", "state_id", "state_type"),
    "barrier": ("barriers", "barrier_id", "barrier_type"),
    "facilitator": ("facilitators", "facilitator_id", "facilitator_type"),
}

# Real Postgres enum values for each *_type column (from the live Supabase
# schema), with a lightweight keyword mapping to pick a reasonable one until
# a human reviewer refines it.
STATE_TYPE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("clinical", ("clinic", "hospital", "doctor", "nurse", "treatment", "medicaid", "medicare", "appointment")),
    ("justice", ("prison", "jail", "parole", "court", "probation", "arrest")),
    ("recovery", ("recovery", "outpatient", "sober", "rehab", "detox")),
    ("social_service", ("shelter", "housing", "welfare", "social worker", "food stamp", "snap")),
    ("risk_event", ("relapse", "overdose", "crisis", "emergency")),
]
DEFAULT_STATE_TYPE = "administrative"

BARRIER_TYPE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("financial", ("afford", "cost", "money", "expensive", "pay")),
    ("logistical", ("waitlist", "wait list", "transport", "bus", "ride", "delay", "late", "schedule")),
    ("knowledge", ("didn't know", "wasn't aware", "know", "information")),
    ("stigma", ("stigma", "shame", "embarrassed", "judged")),
    ("social", ("alone", "isolated", "no support", "no family")),
]
DEFAULT_BARRIER_TYPE = "structural"

FACILITATOR_TYPE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("person", ("nurse", "doctor", "counselor", "case manager", "social worker", "peer", "friend", "family")),
    ("program", ("program", "treatment", "training", "class")),
    ("policy", ("policy", "law", "regulation")),
    ("resource", ("resource", "ride", "voucher", "transportation", "money")),
]
DEFAULT_FACILITATOR_TYPE = "relationship"

TYPE_CLASSIFIERS: dict[str, tuple[list[tuple[str, tuple[str, ...]]], str]] = {
    "states": (STATE_TYPE_HINTS, DEFAULT_STATE_TYPE),
    "barriers": (BARRIER_TYPE_HINTS, DEFAULT_BARRIER_TYPE),
    "facilitators": (FACILITATOR_TYPE_HINTS, DEFAULT_FACILITATOR_TYPE),
}


def _classify_enum(text: str, hints: list[tuple[str, tuple[str, ...]]], default: str) -> str:
    lowered = text.lower()
    for value, keywords in hints:
        if any(keyword in lowered for keyword in keywords):
            return value
    return default


def _nearest_state_id(index: int, state_positions: list[tuple[int, str]]) -> Optional[str]:
    """barriers/facilitators must reference a state they affect (DB check
    constraint). Nothing in extraction identifies that relationship, so we
    approximate it: link to the last state mentioned before this node, or
    the first state in the transcript if none precedes it."""
    if not state_positions:
        return None
    preceding = [state_id for position, state_id in state_positions if position <= index]
    if preceding:
        return preceding[-1]
    return state_positions[0][1]


def _insert_entity(
    session_id: UUID,
    table_name: str,
    id_column: str,
    type_column: str,
    node: dict,
    affected_state_id: Optional[str] = None,
) -> str:
    hints, default_type = TYPE_CLASSIFIERS[table_name]
    classification_text = f"{node['text']} {node.get('evidence', '')}"
    entity_payload = {
        "label": node["text"],
        type_column: _classify_enum(classification_text, hints, default_type),
        "description": node.get("description"),
        "confidence": node.get("confidence"),
        "source_session_id": str(session_id),
    }
    if table_name in ("barriers", "facilitators"):
        entity_payload["evidence_text"] = node.get("evidence")
        entity_payload["affected_state_id"] = affected_state_id

    entity_result = supabase.table(table_name).insert(entity_payload).execute()
    if not entity_result.data:
        raise HTTPException(status_code=500, detail=f"Failed to save {table_name}")
    return entity_result.data[0][id_column]


def _insert_provenance(
    session_id: UUID,
    media_id: UUID,
    media_type: str,
    node: dict,
    id_column: str,
    entity_id: str,
    preferred_model: Optional[str],
) -> None:
    provenance_payload = {
        "session_id": str(session_id),
        "media_id": str(media_id),
        "media_type": media_type,
        "span_start": node.get("span_start"),
        "span_end": node.get("span_end"),
        "excerpt": node.get("evidence"),
        id_column: entity_id,
        "extraction_model": preferred_model or "default",
        "extraction_prompt_version": PROMPT_VERSION,
    }
    supabase.table("provenance").insert(provenance_payload).execute()


def save_stage1_nodes(
    session_id: UUID,
    media_id: UUID,
    media_type: str,
    raw_nodes: list[dict],
    preferred_model: Optional[str],
) -> None:
    # Pass 1: persist states first so barriers/facilitators can link to one.
    state_positions: list[tuple[int, str]] = []
    for index, node in enumerate(raw_nodes):
        if node["category"] != "state":
            continue
        state_id = _insert_entity(session_id, "states", "state_id", "state_type", node)
        _insert_provenance(session_id, media_id, media_type, node, "state_id", state_id, preferred_model)
        state_positions.append((index, state_id))

    # Pass 2: barriers/facilitators, linked to the nearest state.
    for index, node in enumerate(raw_nodes):
        if node["category"] not in ("barrier", "facilitator"):
            continue

        affected_state_id = _nearest_state_id(index, state_positions)
        if affected_state_id is None:
            continue  # no state to satisfy the affects-something constraint

        table_name, id_column, type_column = ENTITY_TABLES[node["category"]]
        entity_id = _insert_entity(
            session_id, table_name, id_column, type_column, node, affected_state_id=affected_state_id
        )
        _insert_provenance(session_id, media_id, media_type, node, id_column, entity_id, preferred_model)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/transcripts",
    response_model=TranscriptOut,
    dependencies=[Depends(verify_api_key)],
)
def create_transcript(payload: TranscriptIn) -> TranscriptOut:
    if supabase is not None:
        session_id = get_or_create_session(payload.session_id)
        media_id, _media_type = get_or_create_media(
            session_id, None, payload.media_type, payload.transcript_text
        )
    else:
        session_id = payload.session_id or uuid4()
        media_id = uuid4()

    return TranscriptOut(session_id=session_id, media_id=media_id)


@app.post(
    "/transcripts/stage1",
    response_model=Stage1Response,
    dependencies=[Depends(verify_api_key)],
)
def create_stage1_nodes(payload: Stage1Request) -> Stage1Response:
    nodes, source, raw_nodes = build_stage1_nodes(
        payload.transcript_text, preferred_model=payload.model
    )

    if supabase is not None:
        session_id = get_or_create_session(payload.session_id)
        media_id, media_type = get_or_create_media(
            session_id, payload.media_id, payload.media_type, payload.transcript_text
        )
        save_stage1_nodes(session_id, media_id, media_type, raw_nodes, payload.model)
    else:
        session_id = payload.session_id or uuid4()
        media_id = payload.media_id or uuid4()

    return Stage1Response(session_id=session_id, media_id=media_id, nodes=nodes, source=source)


def build_stage1_nodes(
    transcript_text: str, preferred_model: Optional[str] = None
) -> tuple[list[Stage1Node], Literal["llm", "heuristic"], list[dict]]:
    raw_nodes, source = run_extract_stage1_nodes(transcript_text, preferred_model=preferred_model)
    stage1_nodes = [
        Stage1Node(
            text=node["text"],
            category=node["category"],
            evidence=node.get("evidence", ""),
            confidence=node.get("confidence", 0.75),
        )
        for node in raw_nodes
    ]
    return stage1_nodes, source, raw_nodes
