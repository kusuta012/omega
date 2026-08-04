import asyncio
import contextlib
import logging
import multiprocessing
from queue import Empty
from omega.storage.postgres_session import db_pool
from omega.storage.queue_queries import (
    claim_next_job, renew_job_claim, release_job_claim, mark_job_failed, reset_stuck_jobs
)
from omega.storage.item_queries import fetch_item_by_id
from omega.parsing.content_extractor_router import ContentExtractorRouter
from omega.storage.chunk_queries import replace_item_content_and_chunks
from omega.chunking.chunk_splitter import ChunkSplitter
from omega.embeddings.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmegaWorker")

extractor_router = ContentExtractorRouter()
chunk_splitter = ChunkSplitter(chunk_size=700, chunk_overlap=100)
embedding_service = get_embedding_service()

CLAIM_HEARTBEAT_SECONDS = 60

async def heartbeat_job_claim(job_id: str, claim_token: str, ownership_lost: asyncio.Event):
    while True:
        await asyncio.sleep(CLAIM_HEARTBEAT_SECONDS)
        try:
            renewed = await renew_job_claim(job_id, claim_token)
        except Exception as err:
            logger.warning(f"Claim heartbeat failed for job {job_id}: {err}")
            continue
        if not renewed:
            ownership_lost.set()
            logger.warning(f"Stopped claim heartbeat for job {job_id}: claim is no longer active")
            return

def extract_item_content(item: dict) -> dict:
    return asyncio.run(extractor_router.extract_content(
        source_type=item['source_type'],
        source_ref=item['source_ref'],
        raw_content=item['raw_content'],
        title=item['title'],
    ))

JOB_EXECUTION_TIMEOUT = 9 * 60

def parse_and_embed_item(item: dict, result_queue):
    try:
        parsed = extract_item_content(item)
        chunks = chunk_splitter.split_document(parsed['raw_content'])
        embeddings = embedding_service.generate_embeddings([chunk['content'] for chunk in chunks])
        result_queue.put(("ok", parsed, chunks, embeddings))
    except Exception as err:
        result_queue.put(("error", str(err)))

async def parse_and_embed_in_subprocess(item: dict):
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(target=parse_and_embed_item, args=(dict(item), result_queue))
    process.start()
    result = None
    try:
        while process.is_alive():
            try:
                result = result_queue.get_nowait()
                break
            except Empty:
                await asyncio.sleep(0.05)
        if result is None:
            try:
                result = result_queue.get_nowait()
            except Empty:
                raise RuntimeError(f"Ingestion subprocess exited without a result (exit code {process.exitcode})")
        if result[0] != "ok":
            raise RuntimeError(f"Document processing failed: {result[1]}")
        return result[1:]
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 2)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 2)
        result_queue.close()

async def process_job(job_id: str, item_id: str, claim_token: str, ownership_lost: asyncio.Event):
    item = await fetch_item_by_id(item_id)
    if not item:
        raise ValueError(f"Item record {item_id} not found in database")

    processing_task = asyncio.create_task(parse_and_embed_in_subprocess(item))
    ownership_task = asyncio.create_task(ownership_lost.wait())
    try:
        done, _ = await asyncio.wait(
            (processing_task, ownership_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ownership_task in done:
            raise RuntimeError(f"Ingestion job {job_id} lost its claim during document processing")
        parsed, chunks, embeddings = await processing_task
    finally:
        for task in (processing_task, ownership_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(processing_task, ownership_task, return_exceptions=True)
    await replace_item_content_and_chunks(
        item_id=item_id,
        title=parsed['title'],
        parsed_content=parsed['raw_content'],
        chunks=chunks,
        embeddings=embeddings,
        job_id=job_id,
        claim_token=claim_token,
    )
    logger.info(f"Job {job_id} successfully completed. Parsed '{parsed['title']} ({len(parsed['raw_content'])} chars)'")

async def execute_job(job: dict):
    job_id = job['id']
    item_id = job['item_id']
    attempts = job['attempts']
    claim_token = job['claim_token']

    logger.info(f"Picked up Job {job_id} | Item {item_id} | Attempt {attempts}")

    ownership_lost = asyncio.Event()
    heartbeat_task = asyncio.create_task(heartbeat_job_claim(job_id, claim_token, ownership_lost))
    try:
        await asyncio.wait_for(
            process_job(job_id, item_id, claim_token, ownership_lost),
            timeout=JOB_EXECUTION_TIMEOUT
        )
    except asyncio.CancelledError:
        await asyncio.shield(release_job_claim(job_id, claim_token, "worker shutdown"))
        raise
    except Exception as error:
        logger.error(f"Job {job_id} failed {str(error)}")
        await mark_job_failed(job_id, item_id, str(error), attempts, claim_token)
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

async def stuck_job_monitor():
    while True:
        try:
            await reset_stuck_jobs()
            logger.debug("swept for stuck jobs")
        except Exception as e:
            logger.error(f"failed to reset stuck jobs: {e}")
        await asyncio.sleep(300)

async def worker_loop():
    await db_pool.connect()
    logger.info("Worker connected to Database")
    monitor_task = asyncio.create_task(stuck_job_monitor())
    logger.info("beginning polling loop...")

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
        monitor_task.cancel()
        await db_pool.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker process killed by user")