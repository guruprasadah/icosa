# icosa

Near-ultrasonic acoustic digital data transmission prototype.

```
uv sync
uv run icosa --rx
uv run icosa --tx "hello world"
```

## Web UI

```
uv run icosa --serve            # http://127.0.0.1:8000
uv run icosa --serve --port 9000
```

The landing page offers **Transmitter** and **Receiver** modes:

- **Transmitter** — type a message, see exactly how it splits into 2-bit
  symbols and tone bursts (a minimizable plan), then press **Transmit** to play
  it over the speaker with a live progress log.
- **Receiver** — the microphone opens automatically on load. Three columns
  update live: incoming bit pairs, each completed byte, and the assembled
  message. Use **Stop**/**Start** to control the mic.

Styling is Tailwind (Play CDN) with HTMX plus the SSE extension; no build step.
