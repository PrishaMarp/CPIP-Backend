import os
from typing import Literal, Optional
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
API_KEY = os.environ["API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(title="PathBridge Transcript API")


class TranscriptIn(BaseModel):
    transcript_text: str
    media_type: Literal["text", "photo", "audio"]
    session_id: Optional[UUID] = None


class TranscriptOut(BaseModel):
    session_id: UUID
    media_id: UUID


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """Simple shared-secret check so random internet traffic can't
    write to your database once this is deployed publicly."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/transcripts",
    response_model=TranscriptOut,
    dependencies=[Depends(verify_api_key)],
)
def create_transcript(payload: TranscriptIn) -> TranscriptOut:
    session_id = payload.session_id

    if session_id is None:
        result = supabase.table("sessions").insert({"status": "complete"}).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create session")
        session_id = result.data[0]["id"]
    else:
        existing = (
            supabase.table("sessions")
            .select("id")
            .eq("id", str(session_id))
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Session not found")

    media_result = (
        supabase.table("media")
        .insert(
            {
                "session_id": str(session_id),
                "media_type": payload.media_type,
                "transcript_text": payload.transcript_text,
                "processing_status": "transcribed",
            }
        )
        .execute()
    )
    if not media_result.data:
        raise HTTPException(status_code=500, detail="Failed to save transcript")

    return TranscriptOut(session_id=session_id, media_id=media_result.data[0]["id"])
