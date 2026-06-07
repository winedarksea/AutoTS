/*
 * AutoTS client facade — the single language-neutral seam between the frontend
 * and the Python runtime.
 *
 * It exposes a tiny global `autotsClient` with three methods:
 *   initRuntime(wheelUrl, pyodideUrl) -> Promise   (resolves when ready)
 *   callTool(command, argsJson)       -> Promise<string>  (JSON result)
 *   setProgressHandler(fn)            -> void
 *
 * The Rust/Leptos app binds to this via wasm-bindgen, but the exact same API is
 * what a TypeScript + Plotly rewrite would import. Keep all worker/protocol
 * details on this side of the seam so the view layer stays swappable.
 */
(function () {
  let worker = null;
  let nextId = 1;
  const pending = new Map();
  let progressHandler = null;
  let readyResolve = null;
  let readyReject = null;
  let ready = null;

  function handleMessage(event) {
    const m = event.data;
    switch (m.type) {
      case 'ready':
        if (readyResolve) readyResolve();
        break;
      case 'init_error':
        if (readyReject) readyReject(new Error(m.error));
        break;
      case 'status':
        if (progressHandler) progressHandler(m.msg);
        break;
      case 'progress':
        if (progressHandler && pending.has(m.id)) progressHandler(m.text);
        break;
      case 'result': {
        const p = pending.get(m.id);
        if (p) { pending.delete(m.id); p.resolve(m.result); }
        break;
      }
      case 'error': {
        const p = pending.get(m.id);
        if (p) { pending.delete(m.id); p.reject(new Error(m.error)); }
        break;
      }
      default:
        break;
    }
  }

  // micropip needs the real PEP-427 wheel filename; the build publishes
  // /autots_wheel.json with the exact URL. Fall back to a conventional name.
  async function resolveWheelUrl(explicit) {
    if (explicit) return explicit;
    if (self.AUTOTS_WHEEL_URL) return self.AUTOTS_WHEEL_URL;
    try {
      const res = await fetch('/autots_wheel.json', { cache: 'no-store' });
      if (res.ok) {
        const m = await res.json();
        if (m && m.url) return m.url;
      }
    } catch (_) { /* ignore, use fallback */ }
    return '/autots-1.0.3-py3-none-any.whl';
  }

  function initRuntime(wheelUrl, pyodideUrl) {
    if (ready) return ready;
    ready = new Promise((resolve, reject) => {
      readyResolve = resolve;
      readyReject = reject;
    });
    const workerUrl = self.AUTOTS_WORKER_URL || '/pyodide_worker.js';
    worker = new Worker(workerUrl);
    worker.onmessage = handleMessage;
    worker.onerror = (e) => { if (readyReject) readyReject(new Error(String(e.message || e))); };
    resolveWheelUrl(wheelUrl).then((resolvedWheel) => {
      worker.postMessage({
        type: 'init',
        wheelUrl: resolvedWheel,
        pyodideUrl: pyodideUrl || self.AUTOTS_PYODIDE_URL ||
          'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js',
      });
    });
    return ready;
  }

  function callTool(command, argsJson) {
    return (ready || Promise.reject(new Error('runtime not initialized'))).then(
      () => new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'call', id, command, argsJson: argsJson || '{}' });
      })
    );
  }

  function setProgressHandler(fn) {
    progressHandler = fn;
  }

  self.autotsClient = { initRuntime, callTool, setProgressHandler };
})();
