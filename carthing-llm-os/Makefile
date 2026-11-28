# make            build for QEMU virt     make run      boot it at EL1
# make run-el2    boot it at EL2 (the path U-Boot takes on hardware)
# make PLATFORM=carthing    build for the Car Thing (untested on hardware)
# make check-host  C forward pass vs PyTorch, natively    make run-model  Qwen3-0.6B in QEMU
PLATFORM ?= qemu
B        := build/$(PLATFORM)
CC       := clang --target=aarch64-none-elf
CFLAGS   := -mcpu=cortex-a53 -O2 -ffreestanding -fno-stack-protector -fno-pic -static -fno-math-errno \
            -Wall -Wextra -Wno-unused-parameter -include platform/$(PLATFORM).h -Isrc
SRCS     := src/boot.S src/main.c src/q4.c src/qwen3.c src/uart.c src/smp.c src/mmu.c src/lib.c
OBJS     := $(patsubst src/%,$(B)/%.o,$(SRCS))

MODEL    := work/qwen3-0.6b-q4_0.bin
QEMU     := qemu-system-aarch64 -M virt -cpu cortex-a53 -smp 4 -m 512M -nographic -kernel $(B)/kernel.bin
LOADER   = -device loader,file=$(1),addr=0x48200000,force-raw=on

all: $(B)/kernel.bin build/testvec.bin

$(B)/%.S.o: src/%.S platform/$(PLATFORM).h | $(B)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/mmu.c.o: src/mmu.c src/kernel.h platform/$(PLATFORM).h | $(B)
	$(CC) $(CFLAGS) -mstrict-align -c $< -o $@
$(B)/lib.c.o: src/lib.c src/kernel.h | $(B)
	$(CC) $(CFLAGS) -fno-builtin -c $< -o $@
$(B)/%.c.o: src/%.c src/kernel.h src/q4.h src/qwen3.h platform/$(PLATFORM).h | $(B)
	$(CC) $(CFLAGS) -c $< -o $@

$(B)/link.ld: src/link.ld platform/$(PLATFORM).h | $(B)
	clang -E -P -x c -include platform/$(PLATFORM).h $< -o $@
$(B)/kernel.elf: $(OBJS) $(B)/link.ld
	ld.lld -T $(B)/link.ld $(OBJS) -o $@
$(B)/kernel.bin: $(B)/kernel.elf
	llvm-objcopy -O binary $< $@
	@echo "$@: $$(stat -c %s $@) bytes"

build/testvec.bin: tools/gen_testvec.py
	python3 $< $@

work/Qwen3-0.6B/model.safetensors:
	python3 -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-0.6B', local_dir='work/Qwen3-0.6B', allow_patterns=['*.json', '*.safetensors', 'merges.txt'])"
$(MODEL): tools/convert.py work/Qwen3-0.6B/model.safetensors
	python3 $<

# The C forward pass built natively, for checking against PyTorch in seconds.
build/host/host_run: tools/host_run.c src/qwen3.c src/q4.c src/qwen3.h src/q4.h
	mkdir -p build/host
	clang -O2 -march=native -fno-math-errno -pthread -Wall -Wextra -Isrc tools/host_run.c src/qwen3.c src/q4.c -o $@
build/host/host_run_exact: tools/host_run.c src/qwen3.c src/q4.c src/qwen3.h src/q4.h
	mkdir -p build/host
	clang -O2 -march=native -fno-math-errno -pthread -Wall -Wextra -DQW_EXACT -Isrc tools/host_run.c src/qwen3.c src/q4.c -o $@
host: build/host/host_run build/host/host_run_exact
check-host: build/host/host_run build/host/host_run_exact $(MODEL)
	python3 tools/check_host.py
	build/host/host_run $(MODEL) gen --write-expect-dev

$(B):
	mkdir -p $@

run: all
	timeout 300 $(QEMU) $(call LOADER,build/testvec.bin)
run-el2: all
	timeout 300 $(QEMU) -M virt,virtualization=on $(call LOADER,build/testvec.bin)
run-model: $(B)/kernel.bin $(MODEL)
	timeout 1800 $(QEMU) $(call LOADER,$(MODEL))

clean:
	rm -r build
.PHONY: all run run-el2 run-model host check-host clean
