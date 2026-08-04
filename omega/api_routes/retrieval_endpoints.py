from fastapi import APIRouter
from omega.api_routes.query_schemas import QueryRequest
from omega.rag.synthesis import Synthesis

retrieval_router = APIRouter()
rag_engine = Synthesis()

@retrieval_router.post("/search")
async def raw_hybrid_search(request: QueryRequest):
    results = await rag_engine.search_knowledge(request.query, request.top_k)
    return {"query": request.query, "results": results}


@retrieval_router.post("/ask")
async def ask_knowledge(request: QueryRequest):
    return await rag_engine.answer_question(request.query, request.top_k)
