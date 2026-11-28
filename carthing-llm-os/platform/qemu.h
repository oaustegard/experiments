/* QEMU `virt` machine, cortex-a53 x4, 512 MB (the Car Thing's RAM size). */
#pragma once
#define PLAT_NAME      "qemu-virt"
#define UART_PL011     1
#define UART_BASE      0x09000000
#define RAM_BASE       0x40000000
#define RAM_SIZE       0x20000000   /* 512 MB */
#define LOAD_ADDR      0x40080000   /* RAM_BASE + TEXT_OFFSET */
#define TEXT_OFFSET    0x00080000
#define BLOB_ADDR      0x48200000     /* -device loader puts the blob here, past QEMU's DTB at 0x48000000-0x48100000 */
#define BW_ADDR        0x54000000     /* 64 MB scratch for the bandwidth test */
#define NCORES         4
#define PSCI_AUTO      1              /* hvc if booted at EL1, smc if booted at EL2 */
