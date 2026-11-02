// Receiver: consume JSON SSE events and fan them into the three columns.
//
// We open the EventSource here (rather than via HTMX's SSE extension) because
// the payloads are JSON that we route to three different columns, not HTML
// fragments to swap. Falls back cleanly to reconnect-on-error.
(function () {
  "use strict";

  var MAX_ITEMS = 400;
  var source = null;
  var retry = 0;
  var userStopped = false;
  var ended = false;

  function el(id) {
    return document.getElementById(id);
  }

  function clearEmpty(container) {
    var placeholder = container.querySelector("[data-empty]");
    if (placeholder) placeholder.remove();
  }

  function append(container, node) {
    clearEmpty(container);
    container.appendChild(node);
    while (container.children.length > MAX_ITEMS) {
      container.removeChild(container.firstChild);
    }
    container.scrollTop = container.scrollHeight;
  }

  function bitPairRow(pair, slot) {
    var bits = ["00", "01", "10", "11"][pair] || "??";
    var row = document.createElement("div");
    row.className =
      "flex items-center gap-2 rounded-md border border-slate-800/60 bg-slate-950/60 px-2 py-1";
    var badge = document.createElement("span");
    badge.className =
      "inline-flex h-4 w-4 items-center justify-center rounded text-[10px] " +
      (slot === "A"
        ? "bg-sky-500/15 text-sky-300"
        : "bg-violet-500/15 text-violet-300");
    badge.textContent = slot;
    var value = document.createElement("span");
    value.className = "text-slate-200";
    value.textContent = "pair " + pair + "  " + bits;
    row.appendChild(badge);
    row.appendChild(value);
    return row;
  }

  function byteRow(value, symbols, char) {
    var hex = value.toString(16).padStart(2, "0").toUpperCase();
    var row = document.createElement("div");
    row.className =
      "flex items-center gap-3 rounded-md border border-slate-800/60 bg-slate-950/60 px-2 py-1";

    var hexEl = document.createElement("span");
    hexEl.className = "text-slate-200";
    hexEl.textContent = "0x" + hex;

    var symEl = document.createElement("span");
    symEl.className = "text-slate-500";
    symEl.textContent = "[" + (symbols || []).join(",") + "]";

    var charEl = document.createElement("span");
    charEl.className = "ml-auto text-emerald-300";
    charEl.textContent = char || "\u00b7";

    row.appendChild(hexEl);
    row.appendChild(symEl);
    row.appendChild(charEl);
    return row;
  }

  function systemNote(container, text, tone) {
    var row = document.createElement("div");
    row.className =
      "rounded-md px-2 py-1 text-[11px] " +
      (tone === "warn" ? "text-amber-400" : "text-slate-500");
    row.textContent = text;
    append(container, row);
  }

  function setStatus(running) {
    var dot = el("rx-dot");
    var label = el("rx-label");
    if (dot) {
      dot.className =
        "h-2 w-2 rounded-full " +
        (running ? "bg-emerald-400 animate-pulse" : "bg-slate-600");
    }
    if (label) {
      label.textContent = running ? "listening" : "stopped";
      label.className =
        "text-xs font-medium " + (running ? "text-emerald-300" : "text-slate-500");
    }
  }

  function clearColumns() {
    var bits = el("rx-bits");
    var bytes = el("rx-bytes");
    var out = el("rx-out");
    if (bits) {
      bits.innerHTML =
        '<p class="p-2 text-slate-600" data-empty>Waiting for symbols\u2026</p>';
    }
    if (bytes) {
      bytes.innerHTML =
        '<p class="p-2 text-slate-600" data-empty>Waiting for a full byte\u2026</p>';
    }
    if (out) out.textContent = "";
  }

  function handle(data) {
    var bits = el("rx-bits");
    var bytes = el("rx-bytes");
    var out = el("rx-out");

    switch (data.type) {
      case "symbol":
        if (bits) append(bits, bitPairRow(data.pair, data.slot));
        break;
      case "byte":
        if (bytes) append(bytes, byteRow(data.value, data.symbols, data.char));
        break;
      case "assembled":
        if (out) out.textContent = data.text || "";
        break;
      case "marker":
        if (bits) systemNote(bits, "\u2014 byte boundary \u2014");
        break;
      case "end":
        // End-of-message tone received: stop receiving and keep everything on
        // screen. Returning true tells the caller to close the stream for good
        // so the browser's EventSource doesn't auto-reconnect and replay this.
        setStatus(false);
        if (!ended && bits) systemNote(bits, "end of message", "warn");
        ended = true;
        return true;
      case "running":
      case "status":
        setStatus(!!data.running || data.type === "running");
        break;
      case "stopped":
        setStatus(false);
        break;
      case "error":
        if (bits) systemNote(bits, "receiver error \u2014 check the server log", "warn");
        setStatus(false);
        break;
    }
    return false;
  }

  function connect() {
    if (userStopped) return;
    if (source) source.close();
    source = new EventSource("/rx/events");
    source.onmessage = function (evt) {
      var data;
      try {
        data = JSON.parse(evt.data);
      } catch (e) {
        return;
      }
      retry = 0;
      if (handle(data)) {
        // End of message (or its replay): stop the native auto-reconnect for
        // good so the stream isn't reopened and the note isn't repeated.
        userStopped = true;
        if (source) {
          source.close();
          source = null;
        }
      }
    };
    source.onerror = function () {
      // The browser auto-reconnects EventSource on transient network errors;
      // if the server closed the stream we reconnect ourselves with backoff,
      // unless reception has ended or the user pressed Stop.
      if (source && source.readyState === EventSource.CLOSED && !userStopped) {
        source.close();
        retry = Math.min(retry + 1, 8);
        window.setTimeout(connect, retry * 500);
      }
    };
  }

  // The Stop button posts to /rx/stop via HTMX; remember that choice so our
  // reconnect logic doesn't immediately reopen the mic. Start re-establishes it.
  document.addEventListener("click", function (evt) {
    var btn = evt.target && evt.target.closest
      ? evt.target.closest("[hx-post='/rx/stop'], [hx-post=\"/rx/stop\"]")
      : null;
    if (btn) {
      userStopped = true;
      if (source) {
        source.close();
        source = null;
      }
    }
    var start = evt.target && evt.target.closest
      ? evt.target.closest("[hx-post='/rx/start'], [hx-post=\"/rx/start\"]")
      : null;
    if (start) {
      userStopped = false;
      ended = false;
      clearColumns();
      window.setTimeout(connect, 100);
    }
  });

  connect();
})();
