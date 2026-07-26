from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ItemSummary(BaseModel):
    id: str
    source_type: str
    title: Optional[str]
    source_ref: Optional[str]
    status: str
    created_at: datetime
    chunk_count: int

class PaginatedItems(BaseModel):
    items: list[ItemSummary]
    total_count: int
    page: int
    page_size: int
    has_more: bool

class ChunkDetail(BaseModel):
    id: str
    chunk_index: int
    content: str

class JobDetail(BaseModel):
    id: str
    job_type: str
    status: str
    attempts: int
    last_error: Optional[str]
    created_at: datetime
    updated_at: datetime

class ItemDetail(BaseModel):
    id: str
    source_type: str
    title: Optional[str]
    source_ref: Optional[str]
    content_hash: Optional[str]
    status: str
    created_at: datetime
    chunks: list[ChunkDetail]
    jobs: list[JobDetail]