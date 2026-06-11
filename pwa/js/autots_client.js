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
  let offlinePreparation = null;
  let readyResolve = null;
  let readyReject = null;
  let ready = null;
  let generation = 0;
  let callQueue = Promise.resolve();
  let runtimeConfig = null;
  const RUNTIME_ASSET_CACHE_KEY = 'runtime-asset-v1';

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
      .then(() => navigator.serviceWorker.ready)
      .catch((error) => {
        console.warn('AutoTS offline support could not start:', error);
        return null;
      });
    return serviceWorkerReady;
  }

  function prepareOfflineSupport() {
    if (offlinePreparation) return offlinePreparation;
    offlinePreparation = initializeOfflineSupport().then((registration) => {
      if (!registration || !registration.active || typeof MessageChannel === 'undefined') {
        return false;
      }
      return new Promise((resolve) => {
        const channel = new MessageChannel();
        const timeout = setTimeout(() => resolve(false), 10 * 60 * 1000);
        channel.port1.onmessage = (event) => {
          clearTimeout(timeout);
          resolve(Boolean(event.data && event.data.ok));
        };
        registration.active.postMessage(
          { type: 'prepare-offline' },
          [channel.port2]
        );
      });
    }).catch((error) => {
      console.warn('AutoTS offline cache could not complete:', error);
      return false;
    });
    return offlinePreparation;
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
        notifyLifecycle('ready');
        if (readyResolve) readyResolve();
        prepareOfflineSupport().then((prepared) => {
          offlineReady = prepared;
          if (offlineReady) {
            enableInstallManifest();
            notifyLifecycle('offline-ready');
          }
          notifyInstallState();
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
    const addRuntimeAssetCacheKey = (url) => {
      if (!base) return url;
      const resolvedUrl = new URL(url, base);
      resolvedUrl.searchParams.set('autots-cache', RUNTIME_ASSET_CACHE_KEY);
      return resolvedUrl.href;
    };
    try {
      const res = await fetch(
        addRuntimeAssetCacheKey('autots_wheel.json'),
        { cache: 'no-store' }
      );
      if (res.ok) {
        const m = await res.json();
        if (m && m.url) {
          return {
            wheelUrl: addRuntimeAssetCacheKey(m.url),
            dependencyUrls: Array.isArray(m.dependencies)
              ? m.dependencies.map(
                (dependency) => addRuntimeAssetCacheKey(dependency.url)
              )
              : [],
          };
        }
      }
    } catch (_) { /* report one consistent runtime-asset error below */ }
    throw new Error(
      'Unable to load autots_wheel.json; the forecasting runtime assets are unavailable'
    );
  }

  function initRuntime(wheelUrl, pyodideUrl) {
    if (ready) return ready;
    initializeOfflineSupport();
    return resolveRuntimeAssets(wheelUrl).then((runtimeAssets) => {
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
