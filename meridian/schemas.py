"""Pydantic models for all Meridian data types."""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


def gen_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── Memory Types ──

class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("mem"))
    content: str
    type: str  # decision, pattern, debug, entity, warning, note
    tags: list[str] = Field(default_factory=list)
    project_id: str = "default"
    importance: int = 3  # 1-5
    source: Optional[str] = None  # giga, webbie, chris, or None for legacy
    volatile: bool = False  # ephemeral facts (ports, PIDs, URLs, temp paths) — decay faster
    related_ids: list[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    superseded_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("chk"))
    session_id: str = ""
    project_id: str = "default"
    source: Optional[str] = None  # agent identity — giga, webbie, etc.
    task_state: str = ""
    decisions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    working_set: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Knowledge Graph ──

class Entity(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("ent"))
    name: str
    type: str  # service, file, concept, person, config
    metadata: dict = Field(default_factory=dict)
    project_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("rel"))
    source_id: str
    target_id: str
    type: str  # calls, depends_on, configured_by, implements, broke
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Decision(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("dec"))
    title: str
    rationale: str
    alternatives: list[dict] = Field(default_factory=list)
    status: str = "active"  # active, superseded, reverted
    superseded_by: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    importance: int = 3
    project_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Warning(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("wrn"))
    content: str
    severity: str = "medium"  # low, medium, high, critical
    related_entity: Optional[str] = None
    resolved: bool = False
    project_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


# ── Episodic ──

class EpisodicSession(BaseModel):
    session_id: str = Field(default_factory=lambda: gen_id("ses"))
    project_id: str = "default"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    summary: str = ""
    key_events: list[dict] = Field(default_factory=list)
    transcript_hash: str = ""


# ── Tool I/O ──

class RecallRequest(BaseModel):
    query: str
    scope: str = "all"
    max_tokens: int = 800
    time_range: Optional[dict] = None


class RecallResponse(BaseModel):
    results: str
    sources: list[dict] = Field(default_factory=list)
    token_count: int = 0


class RememberRequest(BaseModel):
    content: str
    type: str
    tags: list[str] = Field(default_factory=list)
    related_to: list[str] = Field(default_factory=list)
    importance: int = 3


class CheckpointRequest(BaseModel):
    task_state: str
    decisions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    working_set: dict = Field(default_factory=dict)


class GraphQueryRequest(BaseModel):
    entity: str
    relation: Optional[str] = None
    depth: int = 2


class HistoryRequest(BaseModel):
    query: str
    session_filter: Optional[dict] = None
