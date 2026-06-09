const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const handlers = {};
const deletedCaches = [];
const cachedAssets = [];
const cachedRequests = [];
const cache = {
  addAll: async (urls) => { cachedAssets.push(...urls); },
  match: async () => null,
  put: async (request) => { cachedRequests.push(request.url || request); },
};
const context = {
  self: {
    registration: { scope: 'https://example.test/app/' },
    location: { origin: 'https://example.test' },
    clients: { claim: async () => {} },
    skipWaiting: async () => {},
    addEventListener(type, handler) { handlers[type] = handler; },
  },
  caches: {
    open: async () => cache,
    keys: async () => ['unrelated', 'autots-pwa-old'],
    delete: async (name) => { deletedCaches.push(name); },
  },
  fetch: async (request) => {
    const url = typeof request === 'string' ? request : request.url;
    if (url.endsWith('offline_assets.json')) {
      return {
        ok: true,
        json: async () => ({
          assets: ['index.html', 'app.js', 'app_bg.wasm', 'autots.whl'],
        }),
      };
    }
    return {
      ok: true,
      type: 'basic',
      clone() { return this; },
    };
  },
  URL,
  Promise,
  Error,
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(require.resolve('../worker/service_worker.js'), 'utf8'),
  context
);

(async () => {
  let installPromise;
  handlers.install({ waitUntil: (promise) => { installPromise = promise; } });
  await installPromise;
  assert.deepEqual(cachedAssets, [
    'https://example.test/app/index.html',
    'https://example.test/app/app.js',
    'https://example.test/app/app_bg.wasm',
    'https://example.test/app/autots.whl',
  ]);

  let activatePromise;
  handlers.activate({ waitUntil: (promise) => { activatePromise = promise; } });
  await activatePromise;
  assert.deepEqual(deletedCaches, ['autots-pwa-old']);

  let fetchPromise;
  handlers.fetch({
    request: {
      method: 'GET',
      mode: 'cors',
      url: 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/numpy.whl',
    },
    respondWith: (promise) => { fetchPromise = promise; },
  });
  await fetchPromise;
  assert.deepEqual(cachedRequests, [
    'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/numpy.whl',
  ]);

  console.log('service worker tests passed');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
