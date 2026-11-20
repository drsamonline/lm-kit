// wasi-mini.mjs — just enough WASI to run lm.wasm, in Node and in a browser.
//
// The kernel needs four things from the host: argv, a file to read, somewhere to write, and a
// way to exit. That is what is implemented here. Every other import the module declares is filled
// with a stub that returns ENOSYS, so a missing function fails loudly in the kernel's own error
// path instead of crashing the link with a LinkError nobody can read.
//
// Used by docs/demo.html — no build step, no dependency — and by kernel/t-shim.mjs, which checks
// the shim against Node's own WASI implementation.

const ENOENT = 44, EBADF = 8, ENOSYS = 52, EINVAL = 28;

export class WASIExit extends Error {
  constructor(code) { super("exit " + code); this.code = code; }
}

function concat(chunks) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const out = new Uint8Array(n);
  let p = 0;
  for (const c of chunks) { out.set(c, p); p += c.length; }
  return out;
}

export function createWASI({ argv = [], files = {}, onStdout = null }) {
  let mem = null;
  const u8 = () => new Uint8Array(mem.buffer);
  const dv = () => new DataView(mem.buffer);
  const encode = new TextEncoder();
  const decode = new TextDecoder();

  // fd table: 0 stdin, 1 stdout, 2 stderr, 3 the preopened directory, 4+ opened files
  const fds = new Map();
  fds.set(0, { kind: "stdin" });
  fds.set(1, { kind: "stdout", buf: [] });
  fds.set(2, { kind: "stderr", buf: [] });
  fds.set(3, { kind: "dir", name: "/" });

  function writeTo(fd, bytes) {
    const f = fds.get(fd);
    if (!f) return EBADF;
    if (f.kind === "stdout" || f.kind === "stderr") {
      f.buf.push(bytes);
      if (onStdout) onStdout(bytes, fd === 2);
      return 0;
    }
    return EBADF;
  }

  const impl = {
    args_sizes_get(argcPtr, bufSizePtr) {
      const d = dv();
      d.setUint32(argcPtr, argv.length, true);
      d.setUint32(bufSizePtr, argv.reduce((n, a) => n + encode.encode(a).length + 1, 0), true);
      return 0;
    },
    args_get(argvPtr, argvBufPtr) {
      const d = dv(), m = u8();
      let p = argvBufPtr;
      argv.forEach((a, i) => {
        d.setUint32(argvPtr + i * 4, p, true);
        const b = encode.encode(a);
        m.set(b, p);
        p += b.length;
        m[p++] = 0;
      });
      return 0;
    },
    environ_sizes_get(countPtr, sizePtr) {
      const d = dv();
      d.setUint32(countPtr, 0, true);
      d.setUint32(sizePtr, 0, true);
      return 0;
    },
    environ_get() { return 0; },

    fd_write(fd, iovsPtr, iovsLen, nwrittenPtr) {
      const d = dv(), m = u8();
      let total = 0;
      for (let i = 0; i < iovsLen; i++) {
        const ptr = d.getUint32(iovsPtr + i * 8, true);
        const len = d.getUint32(iovsPtr + i * 8 + 4, true);
        if (writeTo(fd, m.slice(ptr, ptr + len)) !== 0) return EBADF;
        total += len;
      }
      d.setUint32(nwrittenPtr, total, true);
      return 0;
    },

    fd_read(fd, iovsPtr, iovsLen, nreadPtr) {
      const f = fds.get(fd);
      if (!f || f.kind !== "file") return EBADF;
      const d = dv(), m = u8();
      let total = 0;
      for (let i = 0; i < iovsLen; i++) {
        const ptr = d.getUint32(iovsPtr + i * 8, true);
        const len = d.getUint32(iovsPtr + i * 8 + 4, true);
        const take = Math.min(len, f.data.length - f.pos);
        if (take > 0) {
          m.set(f.data.subarray(f.pos, f.pos + take), ptr);
          f.pos += take;
          total += take;
        }
        if (take < len) break;
      }
      d.setUint32(nreadPtr, total, true);
      return 0;
    },

    fd_seek(fd, offset, whence, newOffsetPtr) {
      const f = fds.get(fd);
      if (!f || f.kind !== "file") return EBADF;
      const base = whence === 0 ? 0 : whence === 1 ? f.pos : f.data.length;
      const next = base + Number(offset);
      if (next < 0) return EINVAL;
      f.pos = next;
      dv().setBigUint64(newOffsetPtr, BigInt(next), true);
      return 0;
    },

    fd_close(fd) { return fds.delete(fd) ? 0 : EBADF; },
    fd_fdstat_set_flags() { return 0; },
    fd_sync() { return 0; },

    fd_fdstat_get(fd, ptr) {
      const f = fds.get(fd);
      if (!f) return EBADF;
      const d = dv();
      d.setUint8(ptr, f.kind === "dir" ? 3 : 4);      // filetype: directory / regular file
      d.setUint16(ptr + 2, 0, true);
      d.setBigUint64(ptr + 8, 0xffffffffffffffffn, true);
      d.setBigUint64(ptr + 16, 0xffffffffffffffffn, true);
      return 0;
    },

    fd_prestat_get(fd, ptr) {
      const f = fds.get(fd);
      if (!f || f.kind !== "dir") return EBADF;
      const d = dv();
      d.setUint8(ptr, 0);                             // preopentype: dir
      d.setUint32(ptr + 4, f.name.length, true);
      return 0;
    },

    fd_prestat_dir_name(fd, ptr, len) {
      const f = fds.get(fd);
      if (!f || f.kind !== "dir") return EBADF;
      u8().set(encode.encode(f.name).subarray(0, len), ptr);
      return 0;
    },

    path_open(dirfd, dirflags, pathPtr, pathLen, oflags, rightsBase, rightsInheriting,
              fdflags, fdPtr) {
      const name = decode.decode(u8().subarray(pathPtr, pathPtr + pathLen));
      const key = Object.keys(files).find(k => k === name || k.endsWith("/" + name));
      if (!key) return ENOENT;
      const fd = 4 + fds.size;
      fds.set(fd, { kind: "file", data: files[key], pos: 0 });
      dv().setUint32(fdPtr, fd, true);
      return 0;
    },

    clock_time_get(id, precision, ptr) {
      const now = typeof performance !== "undefined" ? performance.now() : Date.now();
      dv().setBigUint64(ptr, BigInt(Math.round(now * 1e6)), true);
      return 0;
    },

    random_get(ptr, len) {
      const m = u8();
      for (let i = 0; i < len; i++) m[ptr + i] = (Math.random() * 256) | 0;
      return 0;
    },

    proc_exit(code) { throw new WASIExit(code); },
  };

  return {
    impl,
    setMemory(m) { mem = m; },
    stdoutText() { return decode.decode(concat(fds.get(1).buf)); },
    stderrText() { return decode.decode(concat(fds.get(2).buf)); },
  };
}

/** Build the import object the module actually asks for, stubbing anything unknown. */
export function buildImports(mod, wasi) {
  const out = {};
  for (const { module, name, kind } of WebAssembly.Module.imports(mod)) {
    if (kind !== "function") continue;
    out[module] = out[module] || {};
    const fn = module.startsWith("wasi") ? wasi.impl[name] : undefined;
    out[module][name] = fn || (() => ENOSYS);
  }
  return out;
}

export async function run({ wasmBytes, modelBytes, modelName = "model.lmk", prompt, tokens = 40 }) {
  const wasi = createWASI({
    argv: ["lm", modelName, prompt, String(tokens)],
    files: { [modelName]: new Uint8Array(modelBytes) },
  });
  const mod = await WebAssembly.compile(wasmBytes);
  const imports = buildImports(mod, wasi);
  const inst = await WebAssembly.instantiate(mod, imports);
  wasi.setMemory(inst.exports.memory);
  let code = 0;
  try {
    inst.exports._start();
  } catch (e) {
    if (e instanceof WASIExit) code = e.code;
    else throw e;
  }
  return { stdout: wasi.stdoutText(), stderr: wasi.stderrText(), code };
}
