/*
 * AutoTS client facade — the single language-neutral seam between the frontend
 * and the Python runtime.
 *
 * It exposes a tiny global `autotsClient` with three methods:
 *   initRuntime(wheelUrl, pyodideUrl) -> Promise   (resolves when ready)
 *   callTool(command, argsJson)       -> Promise<string>  (JSON result)
 *   cancelForecast()                  -> Promise           (fresh runtime ready)
 *   setProgressHandler(fn)            -> void
 *   setLifecycleHandler(fn)           -> void
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
  let lifecycleHandler = null;
  let readyResolve = null;
  let readyReject = null;
  let ready = null;
  let generation = 0;
  let callQueue = Promise.resolve();
  let runtimeConfig = null;

  function notifyLifecycle(state) {
    if (lifecycleHandler) lifecycleHandler(state);
  }

  function handleMessage(workerGeneration, event) {
    if (workerGeneration !== generation) return;
    const m = event.data;
    switch (m.type) {
      case 'ready':
        notifyLifecycle('ready');
        if (readyResolve) readyResolve();
        break;
      case 'init_error':
        notifyLifecycle('failed');
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

  function rejectPending(error) {
    for (const request of pending.values()) request.reject(error);
    pending.clear();
  }

  function startWorker() {
    const workerGeneration = ++generation;
    ready = new Promise((resolve, reject) => {
      readyResolve = resolve;
      readyReject = reject;
    });
    notifyLifecycle('restarting');
    const workerUrl = runtimeConfig.workerUrl;
    worker = new Worker(workerUrl, { name: 'AutoTS Forecast Worker' });
    worker.onmessage = (event) => handleMessage(workerGeneration, event);
    worker.onerror = (e) => {
      if (workerGeneration !== generation) return;
      notifyLifecycle('failed');
      if (readyReject) readyReject(new Error(String(e.message || e)));
    };
    worker.postMessage({
      type: 'init',
      wheelUrl: runtimeConfig.wheelUrl,
      pyodideUrl: runtimeConfig.pyodideUrl,
    });
    return ready;
  }

  // micropip needs an absolute wheel URL — relative URLs are ambiguous inside a
  // Pyodide web worker (the worker base URL differs from the page base URL).
  // autots_wheel.json (relative to the page) carries the exact filename; we
  // resolve it to an absolute URL here in the main-thread context.
  async function resolveWheelUrl(explicit) {
    if (explicit) return explicit;
    if (self.AUTOTS_WHEEL_URL) return self.AUTOTS_WHEEL_URL;
    const base = (typeof document !== 'undefined' && document.baseURI) ||
                 (typeof location !== 'undefined' && location.href) || '';
    try {
      const res = await fetch('autots_wheel.json', { cache: 'no-store' });
      if (res.ok) {
        const m = await res.json();
        if (m && m.url) return base ? new URL(m.url, base).href : m.url;
      }
    } catch (_) { /* ignore, use fallback */ }
    return base ? new URL('autots-1.0.4-py3-none-any.whl', base).href
                : 'autots-1.0.4-py3-none-any.whl';
  }

  function initRuntime(wheelUrl, pyodideUrl) {
    if (ready) return ready;
    return resolveWheelUrl(wheelUrl).then((resolvedWheel) => {
      runtimeConfig = {
        workerUrl: self.AUTOTS_WORKER_URL || 'pyodide_worker.js',
        wheelUrl: resolvedWheel,
        pyodideUrl: pyodideUrl || self.AUTOTS_PYODIDE_URL ||
          'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js',
      };
      return startWorker();
    });
  }

  function callTool(command, argsJson) {
    const execute = () => (ready || Promise.reject(new Error('runtime not initialized'))).then(
      () => new Promise((resolve, reject) => {
        const id = nextId++;
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'call', id, command, argsJson: argsJson || '{}' });
      })
    );
    const result = callQueue.then(execute, execute);
    callQueue = result.catch(() => {});
    return result;
  }

  function cancelForecast() {
    if (!runtimeConfig) return Promise.reject(new Error('runtime not initialized'));
    notifyLifecycle('cancelling');
    const cancellationError = new Error('Forecast cancelled');
    cancellationError.name = 'AutoTSCancelledError';
    rejectPending(cancellationError);
    if (worker) worker.terminate();
    worker = null;
    ready = null;
    callQueue = Promise.resolve();
    return startWorker();
  }

  function setProgressHandler(fn) {
    progressHandler = fn;
  }

  function setLifecycleHandler(fn) {
    lifecycleHandler = fn;
  }

  self.autotsClient = {
    initRuntime,
    callTool,
    cancelForecast,
    setProgressHandler,
    setLifecycleHandler,
    workerLabel: 'AutoTS Forecast Worker',
  };
})();
