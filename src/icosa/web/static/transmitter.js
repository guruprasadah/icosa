// Transmitter: render live playback progress from the /tx/events SSE stream.
//
// The stream only yields events once a transmission is running, so we open an
// EventSource when the server raises the `tx-started` HX-Trigger, and close it
// when the stream reports `done`/`error`.
(function () {
  "use strict";

  var MAX_ITEMS = 400;
  var source = null;

  function el(id) {
    return document.getElementById(id);
  }

  function hint(text) {
    var h = el("tx-progress-hint");
    if (h) h.textContent = text;
  }

  function reset() {
    var log = el("tx-progress");
    if (!log) return;
    log.innerHTML =
      '<div id="tx-progress-placeholder" class="p-2 text-slate-600">Press Transmit to start\u2026</div>';
  }

  function append(node) {
    var log = el("tx-progress");
    if (!log) return;
    var placeholder = el("tx-progress-placeholder");
    if (placeholder) placeholder.remove();
    log.appendChild(node);
    while (log.children.length > MAX_ITEMS) log.removeChild(log.firstChild);
    log.scrollTop = log.scrollHeight;
  }

  function line(text, tone) {
    var row = document.createElement("div");
    row.className =
      "rounded-md px-2 py-0.5 " +
      (tone === "accent"
        ? "bg-sky-500/10 text-sky-300"
        : tone === "warn"
        ? "text-amber-400"
        : tone === "done"
        ? "text-emerald-300"
        : "text-slate-500");
    row.textContent = text;
    append(row);
  }

  function handle(data, close) {
    switch (data.type) {
      case "idle":
        hint("idle");
        close();
        break;
      case "started":
        hint("transmitting");
        line("start \u2014 " + data.bytes + " bytes", "accent");
        break;
      case "symbol":
        line("symbol " + data.index + "  slot " + data.slot + "  pair " + data.pair);
        break;
      case "marker":
        line("marker tone", "accent");
        break;
      case "done":
        hint("done");
        line("transmission complete", "done");
        close();
        break;
      case "error":
        hint("error");
        line("transmission error \u2014 check the server log", "warn");
        close();
        break;
    }
  }

  function connect() {
    if (!el("tx-progress")) return;
    if (source) source.close();
    reset();
    hint("starting\u2026");
    source = new EventSource("/tx/events");
    source.onmessage = function (evt) {
      var data;
      try {
        data = JSON.parse(evt.data);
      } catch (e) {
        return;
      }
      handle(data, function () {
        source.close();
        source = null;
      });
    };
    source.onerror = function () {
      source.close();
      source = null;
    };
  }

  // Fired by HTMX from the `HX-Trigger: tx-started` response header.
  document.body.addEventListener("tx-started", connect);
  document.body.addEventListener("tx-busy", function () {
    hint("already transmitting\u2026");
  });
})();
