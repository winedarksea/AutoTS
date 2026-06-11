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

  const installEvents = {};
  const installWorkers = [];
  class InstallWorker extends MockWorker {
    constructor(url, options) {
      super(url, options);
      installWorkers.push(this);
    }
  }
  const installPrompt = {
    promptCalls: 0,
    prompt() { this.promptCalls += 1; return Promise.resolve(); },
    userChoice: Promise.resolve({ outcome: 'accepted' }),
    preventDefault() {},
  };
  const installStates = [];
  const fetchedRuntimeAssetUrls = [];
  class MockMessageChannel {
    constructor() {
      this.port1 = { onmessage: null };
      this.port2 = {
        reply: (data) => queueMicrotask(() => this.port1.onmessage({ data })),
      };
    }
  }
  const activeServiceWorker = {
    postMessage(_message, ports) {
      ports[0].reply({ ok: true });
    },
  };
  const installContext = {
    self: {
      AUTOTS_WORKER_URL: 'worker.js',
      isSecureContext: true,
      addEventListener(type, handler) { installEvents[type] = handler; },
    },
    navigator: {
      serviceWorker: {
        controller: {},
        register: async () => ({ active: activeServiceWorker }),
        ready: Promise.resolve({ active: activeServiceWorker }),
        addEventListener() {},
      },
    },
    fetch: async (url) => {
      fetchedRuntimeAssetUrls.push(url);
      return {
        ok: true,
        json: async () => ({
          url: 'autots.whl',
          dependencies: [
            { name: 'et-xmlfile', url: 'et_xmlfile.whl' },
            { name: 'openpyxl', url: 'openpyxl.whl' },
          ],
        }),
      };
    },
    document: { baseURI: 'https://example.test/app/' },
    Worker: InstallWorker,
    URL,
    Promise,
    Error,
    JSON,
    MessageChannel: MockMessageChannel,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    console,
  };
  vm.createContext(installContext);
  vm.runInContext(
    fs.readFileSync(require.resolve('./autots_client.js'), 'utf8'),
    installContext
  );
  const installClient = installContext.self.autotsClient;
  installClient.setInstallHandler((state) => installStates.push(JSON.parse(state)));
  installEvents.beforeinstallprompt(installPrompt);
  assert.equal(await installClient.installApp(), false);
  await installClient.initRuntime('', '');
  assert.deepEqual(fetchedRuntimeAssetUrls, [
    'https://example.test/app/autots_wheel.json?autots-cache=runtime-asset-v1',
  ]);
  const initMessage = installWorkers[0].messages.find((message) => message.type === 'init');
  assert.equal(
    initMessage.wheelUrl,
    'https://example.test/app/autots.whl?autots-cache=runtime-asset-v1'
  );
  assert.deepEqual(
    Array.from(initMessage.dependencyUrls),
    [
      'https://example.test/app/et_xmlfile.whl?autots-cache=runtime-asset-v1',
      'https://example.test/app/openpyxl.whl?autots-cache=runtime-asset-v1',
    ]
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(installStates.at(-1).offlineReady, true);
  assert.equal(installStates.at(-1).installAvailable, true);
  assert.equal(await installClient.installApp(), true);
  assert.equal(installPrompt.promptCalls, 1);

  let resolveServiceWorkerRegistration;
  const delayedWorkers = [];
  class DelayedWorker extends MockWorker {
    constructor(url, options) {
      super(url, options);
      delayedWorkers.push(this);
    }
  }
  const delayedContext = {
    self: {
      AUTOTS_WORKER_URL: 'worker.js',
      isSecureContext: true,
      addEventListener() {},
    },
    navigator: {
      serviceWorker: {
        controller: null,
        register: () => new Promise((resolve) => {
          resolveServiceWorkerRegistration = resolve;
        }),
        ready: new Promise(() => {}),
      },
    },
    Worker: DelayedWorker,
    URL,
    Promise,
    Error,
    JSON,
    queueMicrotask,
    console,
  };
  vm.createContext(delayedContext);
  vm.runInContext(
    fs.readFileSync(require.resolve('./autots_client.js'), 'utf8'),
    delayedContext
  );
  await delayedContext.self.autotsClient.initRuntime('wheel.whl', 'pyodide.js');
  assert.equal(delayedWorkers.length, 1);
  assert.ok(resolveServiceWorkerRegistration, 'service worker registration should start');

  console.log('autots_client tests passed');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
