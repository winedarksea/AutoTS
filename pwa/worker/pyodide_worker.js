/*
 * AutoTS Pyodide Web Worker.
 *
 * Hosts a full Python runtime (Pyodide) off the UI thread so forecasting never
 * blocks the frontend. It is intentionally framework-agnostic: it speaks a tiny
 * JSON message protocol that ANY frontend (Leptos today, a TypeScript rewrite
 * tomorrow) can drive via autots_client.js.
 *
 * Protocol (postMessage):
 *   in:  {type:'init', wheelUrl, pyodideUrl}
 *        {type:'call', id, command, argsJson}
 *   out: {type:'status', msg}            // load progress
 *        {type:'ready'} | {type:'init_error', error}
 *        {type:'progress', id, text}     // per-call progress
 *        {type:'result', id, result}     // result is a JSON string
 *        {type:'error', id, error}
 */

let runCommand = null;

async function init(wheelUrl, pyodideUrl) {
  importScripts(pyodideUrl);
  postMessage({ type: 'status', msg: 'Starting Python runtime…' });
  self.pyodide = await loadPyodide();

  postMessage({ type: 'status', msg: 'Loading scientific packages…' });
  await self.pyodide.loadPackage([
    'micropip',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'statsmodels',
  ]);

  postMessage({ type: 'status', msg: 'Installing AutoTS…' });
  const micropip = self.pyodide.pyimport('micropip');
  await micropip.install(wheelUrl);

  postMessage({ type: 'status', msg: 'Importing AutoTS Python API…' });
  await self.pyodide.runPythonAsync(
    'from autots.mcp.pyodide_api import run_command_json'
  );
  runCommand = self.pyodide.globals.get('run_command_json');
  postMessage({ type: 'ready' });
}

self.onmessage = async (event) => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      await init(msg.wheelUrl, msg.pyodideUrl);
    } catch (err) {
      // Use err.message for Python exceptions (Pyodide converts them to JS Errors
      // with the Python traceback in .message); fall back to .stack or String(err).
      const detail = (err && err.message) ? err.message : (err && err.stack) ? err.stack : String(err);
      postMessage({ type: 'init_error', error: detail });
    }
    return;
  }

  if (msg.type === 'call') {
    const { id, command, argsJson } = msg;
    const progress = (text) => postMessage({ type: 'progress', id, text });
    try {
      const resultStr = await runCommand(command, argsJson || '{}', progress);
      postMessage({ type: 'result', id, result: resultStr });
    } catch (err) {
      postMessage({ type: 'error', id, error: String(err && err.message ? err.message : err) });
    }
  }
};
