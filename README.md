# icosa

Near-ultrasonic acoustic data transmission: data is sent as two-tone bursts
and decoded from microphone audio, asynchronously so it can drive a UI.

```
uv sync
uv run icosa --rx
uv run icosa --tx "hello world"
```

## Layout

| Module            | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `protocol.py`     | Wire format: encoding, burst plan, frame decoding, `Event`  |
| `tx.py`           | `async transmit(data)` + tone generation                    |
| `rx.py`           | `Receiver` async generator, `receive()`, CLI                |
| `logging.py`      | Pretty (coloured) rendering of events                       |
| `__init__.py`     | Public re-exports and the CLI                               |

## Use from an app (FastAPI, SSE, HTMX)

```python
from icosa import Receiver, transmit

# Stream events as bytes arrive
async with Receiver() as rx:
    async for event in rx:
        yield event          # END event carries the whole payload in event.data

# Transmit, muting a receiver so it doesn't hear its own speaker
await transmit(b"hello world", receiver=rx)

# Or one-shot: listen until the end marker and get the bytes back
from icosa import receive
payload = await receive()
```

`Receiver()` also supports `await rx.listen(on_event=...)` (sync or async
callback), and `await receiver(data)` is shorthand for it. Stopping/cancelling
a receiver returns whatever was decoded so far via `receiver.result()`.
