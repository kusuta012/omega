import logging
from uuid import UUID
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from omega.agent.agent_loop import AgentLoop

logger = logging.getLogger("AgentEndpoint")
router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

class AgentRequest(BaseModel):
    message: str = Field(..., description="Natural language prompt for the agent")
    session_id: UUID | None = Field(
        default=None,
        description="Session capability returned by a previous agent response"
    )

class ToolCallLog(BaseModel):
    tool: str
    input: dict
    result_summary: str

class AgentResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[dict]
    tool_calls: list[ToolCallLog]

@router.post("", response_model=AgentResponse)
async def handle_agent_request(req: AgentRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    logger.info(f"Received agent request: '{req.message}'")
    agent = AgentLoop()
    try:
        return await agent.process(
            req.message,
            session_id=str(req.session_id) if req.session_id is not None else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

@router.post("/new")
async def start_new_session():
    agent = AgentLoop()
    session_id = await agent.new_session()
    return {"message": "New session started", "session_id": str(session_id)}