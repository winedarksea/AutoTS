/*
 * Durable browser storage for user-visible AutoTS artifacts.
 *
 * Python runtime objects are deliberately excluded: IndexedDB contains only
 * versioned JSON snapshots that can be reopened after a tab refresh.
 */
(function () {
  const DATABASE_NAME = 'autots-pwa';
  const DATABASE_VERSION = 1;
  const STORE_NAME = 'artifacts';
  const FORECAST_LIMIT = 25;
  const DATASET_LIMIT = 25;
  const BYTE_LIMIT = 200 * 1024 * 1024;
  let databasePromise = null;

  function openDatabase() {
    if (databasePromise) return databasePromise;
    databasePromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          const store = database.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('kind', 'kind', { unique: false });
          store.createIndex('last_accessed_at', 'last_accessed_at', { unique: false });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('IndexedDB open failed'));
    });
    return databasePromise;
  }

  function requestResult(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('IndexedDB request failed'));
    });
  }

  async function readAll() {
    const database = await openDatabase();
    const transaction = database.transaction(STORE_NAME, 'readonly');
    return requestResult(transaction.objectStore(STORE_NAME).getAll());
  }

  function utf8Size(value) {
    return new TextEncoder().encode(JSON.stringify(value)).byteLength;
  }

  function sortOldestFirst(artifacts) {
    return artifacts.slice().sort((left, right) =>
      String(left.last_accessed_at || left.created_at).localeCompare(
        String(right.last_accessed_at || right.created_at)
      )
    );
  }

  function chooseEvictions(artifacts, protectedIds) {
    const retained = new Map(artifacts.map((artifact) => [artifact.id, artifact]));
    const evicted = [];
    const remove = (artifact) => {
      if (!retained.has(artifact.id) || protectedIds.has(artifact.id)) return false;
      retained.delete(artifact.id);
      evicted.push(artifact.id);
      return true;
    };

    const trimKind = (kind, limit) => {
      const candidates = sortOldestFirst(
        Array.from(retained.values()).filter((artifact) => artifact.kind === kind)
      );
      while (candidates.filter((artifact) => retained.has(artifact.id)).length > limit) {
        const candidate = candidates.find((artifact) => retained.has(artifact.id) && !protectedIds.has(artifact.id));
        if (!candidate || !remove(candidate)) break;
      }
    };
    trimKind('forecast', FORECAST_LIMIT);

    const referencedDatasetIds = () => new Set(
      Array.from(retained.values())
        .filter((artifact) => artifact.kind === 'forecast')
        .map((artifact) => artifact.parent_id)
        .filter(Boolean)
    );
    const standaloneDatasets = () => Array.from(retained.values()).filter(
      (artifact) => artifact.kind === 'dataset' && !referencedDatasetIds().has(artifact.id)
    );
    while (standaloneDatasets().length > DATASET_LIMIT) {
      const candidate = sortOldestFirst(standaloneDatasets())
        .find((artifact) => !protectedIds.has(artifact.id));
      if (!candidate || !remove(candidate)) break;
    }

    const totalBytes = () => Array.from(retained.values())
      .reduce((sum, artifact) => sum + Number(artifact.payload_size || 0), 0);
    while (totalBytes() > BYTE_LIMIT) {
      const forecasts = sortOldestFirst(
        Array.from(retained.values()).filter((artifact) => artifact.kind === 'forecast')
      );
      let candidate = forecasts.find((artifact) => !protectedIds.has(artifact.id));
      if (!candidate) {
        candidate = sortOldestFirst(standaloneDatasets())
          .find((artifact) => !protectedIds.has(artifact.id));
      }
      if (!candidate || !remove(candidate)) break;
    }
    return evicted;
  }

  function collectDeletionIds(artifacts, id) {
    const target = artifacts.find((artifact) => artifact.id === id);
    if (!target) return [];
    const deletedIds = [id];
    if (target.kind === 'dataset') {
      artifacts
        .filter((artifact) => artifact.kind === 'forecast' && artifact.parent_id === id)
        .forEach((artifact) => deletedIds.push(artifact.id));
    }
    return deletedIds;
  }

  async function putArtifact(artifactJson, protectedIdsJson) {
    const artifact = JSON.parse(artifactJson);
    const protectedIds = new Set(JSON.parse(protectedIdsJson || '[]'));
    if (!artifact.id) {
      artifact.id = crypto.randomUUID
        ? crypto.randomUUID()
        : `artifact-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
    const now = new Date().toISOString();
    artifact.schema_version = 1;
    artifact.created_at = artifact.created_at || now;
    artifact.last_accessed_at = now;
    artifact.payload_size = utf8Size(artifact);
    if (artifact.payload_size > BYTE_LIMIT) {
      throw new Error('Artifact exceeds the 200 MiB persistence limit');
    }

    const database = await openDatabase();
    let transaction = database.transaction(STORE_NAME, 'readwrite');
    await requestResult(transaction.objectStore(STORE_NAME).put(artifact));

    const allArtifacts = await readAll();
    protectedIds.add(artifact.id);
    const evictedIds = chooseEvictions(allArtifacts, protectedIds);
    if (evictedIds.length) {
      transaction = database.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      await Promise.all(evictedIds.map((id) => requestResult(store.delete(id))));
    }
    return JSON.stringify({ artifact, evicted_ids: evictedIds });
  }

  async function listArtifacts() {
    const artifacts = await readAll();
    artifacts.sort((left, right) => String(right.created_at).localeCompare(String(left.created_at)));
    return JSON.stringify(artifacts);
  }

  async function getArtifact(id) {
    const database = await openDatabase();
    let transaction = database.transaction(STORE_NAME, 'readonly');
    const artifact = await requestResult(transaction.objectStore(STORE_NAME).get(id));
    if (!artifact) throw new Error(`Artifact ${id} was not found`);
    artifact.last_accessed_at = new Date().toISOString();
    transaction = database.transaction(STORE_NAME, 'readwrite');
    await requestResult(transaction.objectStore(STORE_NAME).put(artifact));
    return JSON.stringify(artifact);
  }

  async function deleteArtifact(id) {
    const artifacts = await readAll();
    const deletedIds = collectDeletionIds(artifacts, id);
    if (!deletedIds.length) return JSON.stringify({ deleted_ids: [] });
    const database = await openDatabase();
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    await Promise.all(deletedIds.map((artifactId) => requestResult(store.delete(artifactId))));
    return JSON.stringify({ deleted_ids: deletedIds });
  }

  async function storageSummary() {
    const artifacts = await readAll();
    return JSON.stringify({
      total_bytes: artifacts.reduce((sum, artifact) => sum + Number(artifact.payload_size || 0), 0),
      artifact_count: artifacts.length,
      forecast_limit: FORECAST_LIMIT,
      dataset_limit: DATASET_LIMIT,
      byte_limit: BYTE_LIMIT,
    });
  }

  async function requestPersistentStorage() {
    if (!navigator.storage || !navigator.storage.persist) return false;
    return navigator.storage.persist();
  }

  self.autotsArtifactStore = {
    putArtifact,
    listArtifacts,
    getArtifact,
    deleteArtifact,
    storageSummary,
    requestPersistentStorage,
  };
  self.autotsArtifactStoreTest = { chooseEvictions, collectDeletionIds };
})();
