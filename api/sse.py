"""Server-sent events for scan progress and results."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from core.schemas import ScanReport


def _event(name: str, payload: str) -> str:
    return f"event: {name}\ndata: {payload}\n\n"


async def scan_events(scan: Callable[[], Awaitable[ScanReport]], *, heartbeat: float = 10) -> AsyncIterator[str]:
    yield _event("progress", json.dumps({"stage": "scanning"}))
    task = asyncio.ensure_future(scan())
    try:
        while not task.done():
            done, _ = await asyncio.wait((task,), timeout=heartbeat)
            if not done:
                yield ": keep-alive\n\n"
        yield _event("result", task.result().model_dump_json())
    except Exception:
        yield _event("error", json.dumps({"detail": "The scan could not be completed. Try again later."}))
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
