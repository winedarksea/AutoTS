const CACHE_PREFIX = 'autots-pwa-';
const CACHE_NAME = `${CACHE_PREFIX}__AUTOTS_CACHE_VERSION__`;
const PYODIDE_BASE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';

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
  const cache = await caches.open(CACHE_NAME);
  await cache.addAll(assetUrls);
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
  event.waitUntil(precachePublishedAssets().then(() => self.skipWaiting()));
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
  if (
    requestUrl.origin === self.location.origin ||
    event.request.url.startsWith(PYODIDE_BASE_URL)
  ) {
    event.respondWith(cacheFirst(event.request));
  }
});
