const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  self: {},
  navigator: {},
  TextEncoder,
  console,
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(require.resolve('./artifact_store.js'), 'utf8'),
  context
);

const { chooseEvictions, collectDeletionIds } = context.self.autotsArtifactStoreTest;
const timestamp = (index) => `2026-01-${String(index + 1).padStart(2, '0')}T00:00:00`;

{
  const artifacts = Array.from({ length: 26 }, (_, index) => ({
    id: `forecast-${index}`,
    kind: 'forecast',
    parent_id: 'dataset-active',
    last_accessed_at: timestamp(index),
    payload_size: 10,
  }));
  const evicted = chooseEvictions(artifacts, new Set(['forecast-0']));
  assert.deepEqual(Array.from(evicted), ['forecast-1']);
}

{
  const artifacts = [
    { id: 'dataset-1', kind: 'dataset' },
    { id: 'forecast-1', kind: 'forecast', parent_id: 'dataset-1' },
    { id: 'forecast-2', kind: 'forecast', parent_id: 'dataset-1' },
    { id: 'dataset-2', kind: 'dataset' },
  ];
  assert.deepEqual(
    Array.from(collectDeletionIds(artifacts, 'dataset-1')),
    ['dataset-1', 'forecast-1', 'forecast-2']
  );
  assert.deepEqual(Array.from(collectDeletionIds(artifacts, 'forecast-1')), ['forecast-1']);
}

console.log('artifact_store tests passed');
