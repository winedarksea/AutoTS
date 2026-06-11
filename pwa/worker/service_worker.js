const CACHE_PREFIX = 'autots-pwa-';
const CACHE_NAME = `${CACHE_PREFIX}__AUTOTS_CACHE_VERSION__`;
const PYODIDE_BASE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';
const PYODIDE_OFFLINE_ASSETS = [
  'pyodide.js',
  'pyodide.asm.js',
  'pyodide.asm.wasm',
  'python_stdlib.zip',
  'pyodide-lock.json',
  'joblib-1.4.0-py3-none-any.whl',
  'micropip-0.8.0-py3-none-any.whl',
  'numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'openblas-0.3.26.zip',
  'packaging-24.2-py3-none-any.whl',
  'pandas-2.2.3-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'patsy-0.5.6-py2.py3-none-any.whl',
  'python_dateutil-2.9.0.post0-py2.py3-none-any.whl',
  'pytz-2024.1-py2.py3-none-any.whl',
  'scikit_learn-1.6.1-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'scipy-1.14.1-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'six-1.16.0-py2.py3-none-any.whl',
  'statsmodels-0.14.4-cp312-cp312-pyodide_2024_0_wasm32.whl',
  'threadpoolctl-3.5.0-py3-none-any.whl',
];

function scopedUrl(path) {
  return new URL(path, self.registration.scope).href;
}

async function precachePublishedAssets() {
  const manifestResponse = await fetch(scopedUrl('offline_assets.json'), {
    cache: 'no-store',
  });
  if (!manifestResponse.ok) {
    throw new Error(`Offline asset manifest returned ${manifestResponse.status}`);
  }
  const manifest = await manifestResponse.json();
  const assetUrls = manifest.assets.map(scopedUrl);
  await cacheUrls(assetUrls);
}

async function cacheUrls(urls) {
  for (let index = 0; index < urls.length; index += 3) {
    await Promise.all(urls.slice(index, index + 3).map((url) => cacheFirst(url)));
  }
}

async function prepareOfflineCache() {
  await precachePublishedAssets();
  await cacheUrls(
    PYODIDE_OFFLINE_ASSETS.map((filename) => `${PYODIDE_BASE_URL}${filename}`)
  );
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok || response.type === 'opaque') {
    await cache.put(request, response.clone());
  }
  return response;
}

function isRuntimePackageRequest(request) {
  const requestUrl = new URL(request.url);
  // Only same-origin runtime assets need this network-first + content-type guard:
  // our SPA host can answer a missing .whl/manifest with an HTML fallback page,
  // which must never poison the cache. Pyodide's CDN wheels (jsdelivr) are never
  // HTML and stay on cacheFirst so the large scientific wheels serve from cache.
  if (requestUrl.origin !== self.location.origin) return false;
  return requestUrl.pathname.endsWith('.whl') ||
    requestUrl.pathname.endsWith('/autots_wheel.json');
}

function isValidRuntimePackageResponse(request, response) {
  if (!response.ok || response.type === 'opaque') return false;
  const requestUrl = new URL(request.url);
  const contentType = response.headers && response.headers.get
    ? response.headers.get('content-type') || ''
    : '';
  if (requestUrl.pathname.endsWith('.whl')) {
    return !contentType.toLowerCase().includes('text/html');
  }
  return contentType.toLowerCase().includes('application/json');
}

async function runtimePackageResponse(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    if (isValidRuntimePackageResponse(request, response)) {
      await cache.put(request, response.clone());
      return response;
    }
    return (await cache.match(request)) || response;
  } catch (_) {
    return cache.match(request);
  }
}

async function navigationResponse(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(scopedUrl('index.html'), response.clone());
    return response;
  } catch (_) {
    return cache.match(scopedUrl('index.html'));
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(
        names
          .filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const requestUrl = new URL(event.request.url);
  if (event.request.mode === 'navigate') {
    event.respondWith(navigationResponse(event.request));
    return;
  }
  if (isRuntimePackageRequest(event.request)) {
    event.respondWith(runtimePackageResponse(event.request));
    return;
  }
  if (
    requestUrl.origin === self.location.origin ||
    event.request.url.startsWith(PYODIDE_BASE_URL)
  ) {
    event.respondWith(cacheFirst(event.request));
  }
});

self.addEventListener('message', (event) => {
  if (!event.data || event.data.type !== 'prepare-offline') return;
  const responsePort = event.ports && event.ports[0];
  event.waitUntil(
    prepareOfflineCache()
      .then(() => {
        if (responsePort) responsePort.postMessage({ ok: true });
      })
      .catch((error) => {
        if (responsePort) {
          responsePort.postMessage({
            ok: false,
            error: String(error && error.message ? error.message : error),
          });
        }
      })
  );
});
