# Builds doom_host: the PrBoom engine (engine/, from retro-go via oaustegard/m5)
# plus the Linux platform layer and planner (src/host_main.c).
CC ?= gcc
CFLAGS ?= -O2 -g
ENGINE_CFLAGS = -DHAVE_CONFIG_H -Iengine -w -fno-strict-aliasing
# d_server.c is PrBoom's standalone network server, a separate program
SRC = $(filter-out engine/d_server.c,$(wildcard engine/*.c)) src/host_main.c
OBJ = $(patsubst %.c,build/%.o,$(SRC))

doom_host: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) -lm

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(ENGINE_CFLAGS) -c $< -o $@

clean:
	rm -rf build doom_host
.PHONY: clean
