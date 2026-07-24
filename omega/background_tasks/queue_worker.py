import asyncio
import logging
from omega.storage.postgres_session import db_pool
from omega.storage.queue_queries import (
    claim_next_job, mark_job_complete, mark_job_failed, reset_stuck_jobs
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmegaWorker")

async def execute_job(job: dict):
    job_id = job['id']
    item_id = job['item_id']
    attempts = job['attempts']

    logger.info(f"Picked up Job {job_id} | Item {item_id} | Attempt {attempts}")

    try:
        # parsing will do later
        await asyncio.sleep(3)
        await mark_job_complete(job_id, item_id)
        logger.info(f"Job {job_id} successfully completed")
    
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