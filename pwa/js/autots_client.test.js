const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const workers = [];
class MockWorker {
  constructor(url, options) {
    this.url = url;
    this.options = options;
    this.messages = [];
    this.terminated = false;
    workers.push(this);
  }

  postMessage(message) {
    this.messages.push(message);
    if (message.type === 'init') {
      queueMicrotask(() => this.onmessage({ data: { type: 'ready' } }));
    }
  }

  terminate() {
    this.terminated = true;
  }

  emit(data) {
    this.onmessage({ data });
  }
}

const context = {
  self: { AUTOTS_WORKER_URL: 'worker.js' },
  Worker: MockWorker,
  URL,
  Promise,
  Error,
  queueMicrotask,
  console,
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(require.resolve('./autots_client.js'), 'utf8'),
  context
);

(async () => {
  const client = context.self.autotsClient;
  await client.initRuntime('wheel.whl', 'pyodide.js');
  assert.equal(workers[0].options.name, 'AutoTS Forecast Worker');

  const first = client.callTool('first', '{}');
  const second = client.callTool('second', '{}');
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(
    Array.from(workers[0].messages.filter((message) => message.type === 'call'), (message) => message.command),
    ['first']
  );
  const firstCall = workers[0].messages.find((message) => message.type === 'call');
  workers[0].emit({ type: 'result', id: firstCall.id, result: '{}' });
  await first;
  await new Promise((resolve) => setImmediate(resolve));
  const secondCall = workers[0].messages.find(
    (message) => message.type === 'call' && message.command === 'second'
  );
  workers[0].emit({ type: 'result', id: secondCall.id, result: '{}' });
  await second;

  const cancelled = client.callTool('search_all_night', '{}');
  const cancellationAssertion = assert.rejects(
    cancelled,
    (error) => error.name === 'AutoTSCancelledError'
  );
  await new Promise((resolve) => setImmediate(resolve));
  await client.cancelForecast();
  await cancellationAssertion;
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);

  console.log('autots_client tests passed');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
