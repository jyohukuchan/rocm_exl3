import asyncio
from types import SimpleNamespace

import pytest

from rocm_tools.exl3_server.lifecycle import GenerationFailure, job_results, create_and_collect, readiness


class Job:
    def __init__(self, mode):
        self.mode = mode
        self.cancelled = False

    async def __aiter__(self):
        if self.mode == "fail":
            raise RuntimeError("private prompt and credential")
        if self.mode == "wait":
            await asyncio.Future()
        if self.mode == "ok":
            yield {"eos": True, "text": "answer"}

    async def cancel(self):
        self.cancelled = True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,code", [("fail", "inference_failed"), ("empty", "generation_cancelled"), ("wait", "inference_timeout")])
async def test_failure_is_explicit_and_always_releases_the_owned_job(mode, code):
    job = Job(mode)
    with pytest.raises(GenerationFailure) as error:
        async for _ in job_results(job, timeout=.01):
            pass
    assert error.value.code == code
    assert "private" not in str(error.value.payload())
    assert job.cancelled


@pytest.mark.asyncio
async def test_failed_choice_cancels_remaining_choices_before_returning():
    jobs = [Job("fail"), Job("wait")]

    async def collect(job):
        return [r async for r in job_results(job)]

    with pytest.raises(GenerationFailure):
        await asyncio.wait_for(create_and_collect(2, jobs.__getitem__, collect), 1)
    assert all(job.cancelled for job in jobs)


@pytest.mark.asyncio
async def test_failed_construction_cancels_jobs_already_enqueued():
    first = Job("wait")

    def create(index):
        if index:
            raise GenerationFailure()
        return first

    with pytest.raises(GenerationFailure):
        await create_and_collect(2, create, None)
    assert first.cancelled


def test_readiness_distinguishes_queues_and_never_discloses_exceptions():
    assert readiness(None)["status"] == "unavailable"
    engine = SimpleNamespace(error=None, iteration_task=None,
                             generator=SimpleNamespace(active_jobs=[1], pending_jobs=[2, 3]))
    assert readiness(engine) == {"status": "ok", "jobs": {"active": 1, "queued": 2}}
    engine.error = RuntimeError("secret")
    assert readiness(engine) == {"status": "unavailable", "code": "iteration_stopped"}
