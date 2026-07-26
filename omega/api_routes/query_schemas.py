from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language search queryy")
    top_k: int = Field(5, ge=1, le=20, description="Results to retrieve")