import asyncio
import logging
from omega.storage.postgres_session import db_pool
from omega.storage.queue_queries import (
    claim_next_job, mark_job_complete, mark_job_failed, reset_stuck_jobs
)
from omega.storage.item_queries import fetch_item_by_id, update_item_parsed_content
from omega.parsing.content_extractor_router import ContentExtractorRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmegaWorker")

extractor_router = ContentExtractorRouter()

async def execute_job(job: dict):
    job_id = job['id']
    item_id = job['item_id']
    attempts = job['attempts']

    logger.info(f"Picked up Job {job_id} | Item {item_id} | Attempt {attempts}")

    try:
        item = await fetch_item_by_id(item_id)
        if not item:
            raise ValueError(f"Item record {item_id} not found in database")
        
        parsed = await extractor_router.extract_content(
            source_type=item['source_type'],
            source_ref=item['source_ref'],
            raw_content=item['raw_content'],
            title=item['title']
        )
        await update_item_parsed_content(
            item_id=item_id,
            title=parsed['title'],
            parsed_content=parsed['raw_content']
        )
        await mark_job_complete(job_id, item_id)
        logger.info(f"Job {job_id} successfully completed. Parsed '{parsed['title']} ({len(parsed['raw_content'])} chars)")
    
    except Exception as error:
        logger.error(f"Job {job_id} failed {str(error)}")
        await mark_job_failed(job_id, item_id, str(error), attempts)

async def worker_loop():
    await db_pool.connect()
    logger.info("Worker connected to Database")
    await reset_stuck_jobs()
    logger.info("checked for stuck jobs, beginning polling loop...")

    try:
        while True:
            job = await claim_next_job()
            if job:
                await execute_job(job)
            else:
                await asyncio.sleep(2)
    except asyncio.CancelledError:
        logger.info("Worker shutting down gracefully...")
    finally:
        await db_pool.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker process killed by user")