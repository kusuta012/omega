from pydantic import BaseModel, Field
from typing import Optional

class IngestionRequest(BaseModel):
    source_type: str = Field(..., description="Must be url, pdf, text or code")
    source_ref: Optional[str] = None
    raw_content: Optional[str] = None
    title: Optional[str] = None