const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const messages = [];
const installCalls = [];
const micropip = {
  install: async (value) => { installCalls.push(value); },
};
const pyodide = {
  loadPackage: async () => {},
  pyimport: (name) => {
    assert.equal(name, 'micropip');
    return micropip;
  },
  runPythonAsync: async () => {},
  globals: {
    get: () => async () => '{}',
  },
};
const context = {
  self: {
    postMessage: (message) => messages.push(message),
  },
  importScripts: () => {},
  loadPyodide: async () => pyodide,
  postMessage: (message) => messages.push(message),
  String,
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(require.resolve('../worker/pyodide_worker.js'), 'utf8'),
  context
);

(async () => {
  await context.self.onmessage({
    data: {
      type: 'init',
      wheelUrl: 'autots.whl',
      dependencyUrls: ['et_xmlfile.whl', 'openpyxl.whl'],
      pyodideUrl: 'pyodide.js',
    },
  });
  assert.deepEqual(
    installCalls.map((value) => Array.isArray(value) ? Array.from(value) : value),
    [
      ['et_xmlfile.whl', 'openpyxl.whl'],
      'autots.whl',
    ]
  );
  assert.equal(messages.at(-1).type, 'ready');
  console.log('pyodide worker tests passed');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
