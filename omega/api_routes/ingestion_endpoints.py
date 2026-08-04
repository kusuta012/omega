from fastapi import APIRouter, HTTPException
from omega.api_routes.payload_schemas import IngestionRequest
from omega.knowledge_ingestion import validate_ingestion_req
from omega.storage.queue_queries import enqueue_ingestion_job

ingestion_router = APIRouter()

@ingestion_router.post("/items")
async def add_item(request: IngestionRequest):
    if request.source_type not in ["url", "pdf", "text", "code"]:
        raise HTTPException(status_code=400, detail="Invalid source type")
    if request.source_type != "pdf":
        try:
            validate_ingestion_req(request.source_type, request.source_ref, request.raw_content)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
            
    item_id, job_id, is_duplicate = await enqueue_ingestion_job(
        source_type=request.source_type,
        source_ref=request.source_ref,
        raw_content=request.raw_content,
        title=request.title
    )

    if is_duplicate:
        return {
            "message": "Item already exists in the knowledge base, Skipped duplication",
            "item_id": item_id,
            "job_id": None
        }
            
    return {
        "message": "Item successfully queued for processing",
        "item_id": item_id,
        "job_id": job_id
    }
