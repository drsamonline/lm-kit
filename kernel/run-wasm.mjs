// Runs lm.wasm under Node's WASI.  node run-wasm.mjs model.lmk "prompt" 40
import { WASI } from 'node:wasi';
import { readFile } from 'node:fs/promises';
import { argv } from 'node:process';
import { basename, dirname, resolve } from 'node:path';

const [model, prompt = 'Job postings', n = '40'] = argv.slice(2);
const wasi = new WASI({
  version: 'preview1',
  args: ['lm', basename(model), prompt, n],
  env: {},
  preopens: { '/': resolve(dirname(model)) },
});
const mod = await WebAssembly.compile(await readFile(new URL('./lm.wasm', import.meta.url)));
const inst = await WebAssembly.instantiate(mod, wasi.getImportObject());
wasi.start(inst);
