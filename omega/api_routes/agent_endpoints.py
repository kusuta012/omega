import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from omega.agent.agent_loop import AgentLoop

logger = logging.getLogger("AgentEndpoint")
router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

agent_loop = AgentLoop()

class AgentRequest(BaseModel):
    message: str = Field(..., description="Natural language prompt for the agent")

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
    result = await agent_loop.process(req.message)
    return result

@router.post("/new")
async def start_new_session():
    session_id = await agent_loop.new_session()
    return {"message": "New session started", "session_id": str(session_id)}