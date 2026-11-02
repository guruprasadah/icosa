"""A tiny asyncio fan-out hub for streaming events to SSE subscribers.

One producer (an audio task) publishes events; every connected browser tab is
its own asyncio queue. Slow or vanished subscribers are dropped rather than
allowed to block the producer.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

# Bound on per-subscriber backlog; a tab that falls this far behind is stale.
_MAX_QUEUE = 1024


class Broadcaster:
    """Fan-out hub: :meth:`publish` delivers to every active subscriber."""

    def __init__(self, name: str = "events") -> None:
        self._name = name
        self._subscribers: set[asyncio.Queue[Any]] = set()
        self._sentinel = object()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, item: Any) -> None:
        """Non-blocking broadcast. Must be called from the event loop thread."""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                # Subscriber is too far behind; drop the item for that tab.
                pass

    def close(self) -> None:
        """Wake every subscriber so its stream loop can end."""
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(self._sentinel)

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Any]]:
        """Register a subscriber queue for the duration of the context."""
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_MAX_QUEUE)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    async def stream(self) -> AsyncIterator[Any]:
        """Yield published items until :meth:`close` is called."""
        async with self.subscribe() as queue:
            while True:
                item = await queue.get()
                if item is self._sentinel:
                    return
                yield item
