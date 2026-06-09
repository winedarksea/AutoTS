/*
 * AutoTS client facade — the single language-neutral seam between the frontend
 * and the Python runtime.
 *
 * It exposes a tiny global `autotsClient` with three methods:
 *   initRuntime(wheelUrl, pyodideUrl) -> Promise   (resolves when ready)
 *   callTool(command, argsJson)       -> Promise<string>  (JSON result)
 *   cancelForecast()                  -> Promise           (fresh runtime ready)
 *   installApp()                      -> Promise<boolean>  (show saved prompt)
 *   setProgressHandler(fn)            -> void
 *   setLifecycleHandler(fn)           -> void
 *   setInstallHandler(fn)             -> void
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
  let installHandler = null;
  let deferredInstallPrompt = null;
  let offlineReady = false;
  let appInstalled = false;
  let serviceWorkerReady = null;
  let readyResolve = null;
  let readyReject = null;
  let ready = null;
  let generation = 0;
  let callQueue = Promise.resolve();
  let runtimeConfig = null;

  function notifyLifecycle(state) {
    if (lifecycleHandler) lifecycleHandler(state);
  }

  function notifyInstallState() {
    if (installHandler) {
      installHandler(JSON.stringify({
        offlineReady,
        installAvailable: Boolean(deferredInstallPrompt),
        installed: appInstalled,
      }));
    }
  }

  function waitForServiceWorkerControl() {
    if (navigator.serviceWorker.controller) return Promise.resolve();
    return new Promise((resolve) => {
      navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true });
    });
  }

  function waitForWorkerActivation(worker) {
    if (!worker || worker.state === 'activated') return Promise.resolve();
    return new Promise((resolve, reject) => {
      worker.addEventListener('statechange', () => {
        if (worker.state === 'activated') resolve();
        if (worker.state === 'redundant') {
          reject(new Error('Updated service worker became redundant'));
        }
      });
    });
  }

  function enableInstallManifest() {
    if (
      typeof document === 'undefined' ||
      !document.createElement ||
      document.querySelector('link[rel="manifest"]')
    ) return;
    const manifestLink = document.createElement('link');
    manifestLink.rel = 'manifest';
    manifestLink.href = 'manifest.webmanifest';
    document.head.appendChild(manifestLink);
  }

  function initializeOfflineSupport() {
    if (serviceWorkerReady) return serviceWorkerReady;
    if (
      !self.isSecureContext ||
      typeof navigator === 'undefined' ||
      !navigator.serviceWorker
    ) {
      serviceWorkerReady = Promise.resolve(false);
      return serviceWorkerReady;
    }
    serviceWorkerReady = navigator.serviceWorker.register('service_worker.js')
      .then(async (registration) => {
        const updatingWorker = registration.installing || registration.waiting;
        const previousController = navigator.serviceWorker.controller;
        let controllerChanged = null;
        if (updatingWorker && previousController) {
          controllerChanged = new Promise((resolve) => {
            navigator.serviceWorker.addEventListener(
              'controllerchange',
              resolve,
              { once: true }
            );
          });
        }
        await waitForWorkerActivation(updatingWorker);
        await navigator.serviceWorker.ready;
        if (
          controllerChanged &&
          navigator.serviceWorker.controller === previousController
        ) {
          await controllerChanged;
        } else {
          await waitForServiceWorkerControl();
        }
        return true;
      })
      .catch((error) => {
        console.warn('AutoTS offline support could not start:', error);
        return false;
      });
    return serviceWorkerReady;
  }

  if (self.addEventListener) {
    self.addEventListener('beforeinstallprompt', (event) => {
      event.preventDefault();
      deferredInstallPrompt = event;
      notifyInstallState();
    });
    self.addEventListener('appinstalled', () => {
      appInstalled = true;
      deferredInstallPrompt = null;
      notifyInstallState();
    });
  }

  function handleMessage(workerGeneration, event) {
    if (workerGeneration !== generation) return;
    const m = event.data;
    switch (m.type) {
      case 'ready':
        Promise.resolve().then(() => {
          offlineReady = Boolean(
            typeof navigator !== 'undefined' &&
            navigator.serviceWorker &&
            navigator.serviceWorker.controller
          );
          if (offlineReady) enableInstallManifest();
          notifyInstallState();
          notifyLifecycle(offlineReady ? 'offline-ready' : 'ready');
          if (readyResolve) readyResolve();
        });
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
      dependencyUrls: runtimeConfig.dependencyUrls,
      pyodideUrl: runtimeConfig.pyodideUrl,
    });
    return ready;
  }

  // micropip needs an absolute wheel URL — relative URLs are ambiguous inside a
  // Pyodide web worker (the worker base URL differs from the page base URL).
  // autots_wheel.json (relative to the page) carries the exact filename; we
  // resolve it to an absolute URL here in the main-thread context.
  async function resolveRuntimeAssets(explicitWheelUrl) {
    if (explicitWheelUrl) {
      return { wheelUrl: explicitWheelUrl, dependencyUrls: [] };
    }
    if (self.AUTOTS_WHEEL_URL) {
      return { wheelUrl: self.AUTOTS_WHEEL_URL, dependencyUrls: [] };
    }
    const base = (typeof document !== 'undefined' && document.baseURI) ||
                 (typeof location !== 'undefined' && location.href) || '';
    try {
      const res = await fetch('autots_wheel.json', { cache: 'no-store' });
      if (res.ok) {
        const m = await res.json();
        if (m && m.url) {
          const resolveUrl = (url) => base ? new URL(url, base).href : url;
          return {
            wheelUrl: resolveUrl(m.url),
            dependencyUrls: Array.isArray(m.dependencies)
              ? m.dependencies.map((dependency) => resolveUrl(dependency.url))
              : [],
          };
        }
      }
    } catch (_) { /* ignore, use fallback */ }
    return {
      wheelUrl: base ? new URL('autots-1.0.4-py3-none-any.whl', base).href
                     : 'autots-1.0.4-py3-none-any.whl',
      dependencyUrls: [],
    };
  }

  function initRuntime(wheelUrl, pyodideUrl) {
    if (ready) return ready;
    return initializeOfflineSupport().then(() => resolveRuntimeAssets(wheelUrl)).then((runtimeAssets) => {
      runtimeConfig = {
        workerUrl: self.AUTOTS_WORKER_URL || 'pyodide_worker.js',
        wheelUrl: runtimeAssets.wheelUrl,
        dependencyUrls: runtimeAssets.dependencyUrls,
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

  function setInstallHandler(fn) {
    installHandler = fn;
    notifyInstallState();
  }

  async function installApp() {
    if (!offlineReady || !deferredInstallPrompt) return false;
    const prompt = deferredInstallPrompt;
    deferredInstallPrompt = null;
    await prompt.prompt();
    const choice = await prompt.userChoice;
    notifyInstallState();
    return choice && choice.outcome === 'accepted';
  }

  self.autotsClient = {
    initRuntime,
    callTool,
    cancelForecast,
    installApp,
    setProgressHandler,
    setLifecycleHandler,
    setInstallHandler,
    workerLabel: 'AutoTS Forecast Worker',
  };
})();
