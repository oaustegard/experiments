# make            build for QEMU virt     make run      boot it at EL1
# make run-el2    boot it at EL2 (the path U-Boot takes on hardware)
# make PLATFORM=carthing    build for the Car Thing (untested on hardware)
PLATFORM ?= qemu
B        := build/$(PLATFORM)
CC       := clang --target=aarch64-none-elf
CFLAGS   := -mcpu=cortex-a53 -O2 -ffreestanding -fno-stack-protector -fno-pic -static -fno-math-errno \
            -Wall -Wextra -Wno-unused-parameter -include platform/$(PLATFORM).h -Isrc
SRCS     := src/boot.S src/main.c src/q4.c src/uart.c src/smp.c src/mmu.c src/lib.c
OBJS     := $(patsubst src/%,$(B)/%.o,$(SRCS))

QEMU     := timeout 300 qemu-system-aarch64 -M virt -cpu cortex-a53 -smp 4 -m 512M -nographic \
            -kernel $(B)/kernel.bin \
            -device loader,file=build/testvec.bin,addr=0x50000000,force-raw=on

all: $(B)/kernel.bin build/testvec.bin

$(B)/%.S.o: src/%.S platform/$(PLATFORM).h | $(B)
	$(CC) $(CFLAGS) -c $< -o $@
$(B)/mmu.c.o: src/mmu.c src/kernel.h platform/$(PLATFORM).h | $(B)
	$(CC) $(CFLAGS) -mstrict-align -c $< -o $@
$(B)/lib.c.o: src/lib.c src/kernel.h | $(B)
	$(CC) $(CFLAGS) -fno-builtin -c $< -o $@
$(B)/%.c.o: src/%.c src/kernel.h src/q4.h platform/$(PLATFORM).h | $(B)
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

$(B):
	mkdir -p $@

run: all
	$(QEMU)
run-el2: all
	$(QEMU) -M virt,virtualization=on

clean:
	rm -r build
.PHONY: all run run-el2 clean
