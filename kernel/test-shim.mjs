// Checks the WASI shim in Node against the same module the browser runs.
//
//   node kernel/test-shim.mjs
//
// It cannot compare against PyTorch — that needs torch — but it does check that the shim loads
// the model, runs to completion and produces the JSON shape the model was trained on. The
// stronger parity test is kernel/parity.py.

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { run } from './wasi-mini.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const wasmBytes = await readFile(join(here, '..', 'docs', 'lm.wasm'));
const modelBytes = await readFile(join(here, '..', 'docs', 'model.lmk'));

const r = await run({ wasmBytes, modelBytes, prompt: 'Job postings from 185 sources', tokens: 40 });
const out = r.stdout.trim();

const checks = [
  ['exits cleanly', r.code === 0],
  ['produced output', out.length > 10],
  ['starts a JSON object', out.includes('{')],
  ['carries a schema key', /"api"|"tool"|"cat"/.test(out)],
];
let bad = 0;
for (const [name, ok] of checks) {
  console.log((ok ? '  ok   ' : '  FAIL ') + name);
  if (!ok) bad++;
}
if (bad) {
  console.error('\noutput was: ' + JSON.stringify(out.slice(0, 200)));
  process.exit(1);
}
console.log('\nshim runs the model: ' + JSON.stringify(out.slice(0, 80)));
