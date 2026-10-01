"""Request-owned generation cleanup and safe public failure contracts."""
import asyncio


class GenerationFailure(Exception):
    def __init__(self, code="inference_failed", status=503):
        super().__init__(code)
        self.code = code
        self.status = status

    def payload(self):
        return {"error": {"message": {
            "inference_failed": "Inference failed; retry after the engine is ready.",
            "inference_timeout": "Inference exceeded the configured request deadline.",
            "generation_cancelled": "Generation ended before a completion was produced.",
            "image_inference_failed": "Image inference failed; no image features were added to the prompt.",
            "logprobs_unavailable": "Native token probabilities are unavailable or invalid.",
            "logprobs_alignment_failed": "Native probabilities could not be aligned to the returned text.",
        }.get(self.code, "Inference is unavailable."), "type": "server_error", "code": self.code}}


async def job_results(job, timeout=0):
    completed = False
    try:
        async with asyncio.timeout(timeout or None):
            async for result in job:
                completed = completed or bool(result.get("eos"))
                yield result
        if not completed:
            raise GenerationFailure("generation_cancelled", 499)
    except TimeoutError as exc:
        raise GenerationFailure("inference_timeout", 504) from exc
    except (asyncio.CancelledError, GenerationFailure):
        raise
    except Exception as exc:
        # Exceptions may contain prompt text or native pointers. Do not disclose
        # them to clients or record them in the default request log.
        print(f" -- generation failed: {type(exc).__name__}", flush=True)
        raise GenerationFailure() from exc
    finally:
        if not completed:
            try:
                await job.cancel()
            except Exception as exc:
                print(f" -- generation cleanup failed: {type(exc).__name__}", flush=True)


async def collect_all(jobs, collect):
    tasks = [asyncio.create_task(collect(job)) for job in jobs]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def create_and_collect(count, create, collect):
    jobs = []
    try:
        for index in range(count):
            jobs.append(create(index))
    except BaseException:
        await asyncio.gather(*(job.cancel() for job in jobs), return_exceptions=True)
        raise
    return await collect_all(jobs, collect)


def readiness(generator):
    if generator is None:
        return {"status": "unavailable", "code": "not_ready"}
    task = getattr(generator, "iteration_task", None)
    if getattr(generator, "error", None) is not None or (task is not None and task.done()):
        return {"status": "unavailable", "code": "iteration_stopped"}
    sync = getattr(generator, "generator", None)
    return {"status": "ok", "jobs": {
        "active": len(getattr(sync, "active_jobs", [])),
        "queued": len(getattr(sync, "pending_jobs", [])),
    }}
