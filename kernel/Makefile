CC      ?= cc
ZIG     ?= python3 -m ziglang cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

lm: lm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# 44 KB, no debug info. -Os matters: the default build carries 1.1 MB of it.
lm.wasm: lm.c
	$(ZIG) -target wasm32-wasi -Os -std=c11 -o $@ $< -lm

wasm: lm.wasm
	node run-wasm.mjs ../run-act/model-fp32.lmk "Job postings" 40

clean:
	rm -f lm lm.wasm

.PHONY: clean wasm
