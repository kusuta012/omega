from fastapi import APIRouter, HTTPException, Query
from omega.storage.management_queries import (
    list_items_paginated, get_item_detail, delete_item,
    list_jobs, retry_failed_item
)

management_router = APIRouter()

@management_router.get("/items")
async def get_all_items(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: str | None = Query(None, description="Filter by status: pending, done, failed"),
    source_type: str | None = Query(None, description="Filter by type: url, pdf, text, code")
):
    items, total_count = await list_items_paginated(page, page_size, status, source_type)
    return {
        "items": items,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "has_more": (page * page_size) < total_count
    }

@management_router.get("/items/{item_id}")
async def get_single_item(item_id: str):
    result = await get_item_detail(item_id)
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")

    item = result["item"]
    return {
        "id": str(item["id"]),
        "source_type": item["source_type"],
        "title": item["title"],
        "source_ref": item["source_ref"],
        "content_hash": item["content_hash"],
        "status": item["status"],
        "created_at": str(item["created_at"]),
        "chunks": [
            {
                "id": str(c["id"]),
                "chunk_index": c["chunk_index"],
                "content": c["content"],
                "start_offset": c["start_offset"],
                "end_offset": c["end_offset"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            }
            for c in result["chunks"]
        ],
        "jobs": [
            {
                "id": str(j["id"]),
                "job_type": j["job_type"],
                "status": j["status"],
                "attempts": j["attempts"],
                "last_error": j["last_error"],
                "created_at": str(j["created_at"]),
                "updated_at": str(j["updated_at"])
            }
            for j in result["jobs"]
        ]
    }

@management_router.delete("/items/{item_id}")
async def remove_item(item_id: str):
    deleted = await delete_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")

    return {
        "message": f"Permanently deleted '{deleted['title'] or 'Untitled'}'",
        "deleted_id": str(deleted["id"]),
        "source_type": deleted["source_type"]
    }

@management_router.get("/jobs")
async def get_all_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None, description="filter by status: pending, running, done, failed")
):
    jobs, total_count = await list_jobs(page, page_size, status)
    return {
        "jobs": jobs,
        "total_count": total_count,
        "page": page,
        "page_size": page_size,
        "has_more": (page * page_size) < total_count
    }

@management_router.post("/items/{item_id}/retry")
async def retry_item(item_id: str):
    result = await retry_failed_item(item_id)
    if "error" in result:
        status_map = {
            "not_found": 404,
            "invalid_status": 400,
            "conflict": 409
        }
        raise HTTPException(
            status_code=status_map.get(result["error"], 500),
            detail=result["message"]
        )

    return result