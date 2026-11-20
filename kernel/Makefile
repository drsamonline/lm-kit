CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

lm: lm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f lm

.PHONY: clean
