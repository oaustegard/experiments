/* Spotify Car Thing (superbird): Amlogic S905D2, 4x Cortex-A53, 512 MB.
 * UNVERIFIED on hardware. Addresses are from the Amlogic G12A memory map and
 * the community superbird U-Boot; check them against a UART log before trusting. */
#pragma once
#define PLAT_NAME      "carthing"
#define UART_MESON     1
#define UART_BASE      0xff803000   /* UART_AO_A, already set up by U-Boot (115200 8N1) */
#define RAM_BASE       0x00000000
#define RAM_SIZE       0x20000000   /* 512 MB */
#define LOAD_ADDR      0x01080000   /* the usual Amlogic kernel address; booti uses TEXT_OFFSET */
#define TEXT_OFFSET    0x01080000
#define BLOB_ADDR      0x08000000   /* `ext4load`/`mmc read` the weights here before `booti` */
#define BW_ADDR        0x10000000     /* 64 MB scratch for the bandwidth test */
#define NCORES         4
#define PSCI_AUTO      0              /* BL31 owns PSCI: always smc */
