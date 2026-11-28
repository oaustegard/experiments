/* Identity map with 4 KB granule and a 32-bit VA (walk starts at level 1).
 * RAM's 1 GB slot is split into 2 MB blocks so only real RAM is cacheable;
 * everything else is Device-nGnRnE. Built with -mstrict-align: it runs with
 * the MMU off, when an unaligned access faults. */
#include "kernel.h"

#define DESC_TABLE   0x3UL
#define DESC_BLOCK   0x1UL
#define ATTR_IDX(i)  ((uint64_t)(i) << 2)
#define SH_INNER     (3UL << 8)
#define AF           (1UL << 10)
#define PXN          (1UL << 53)
#define UXN          (1UL << 54)
#define NORMAL       (DESC_BLOCK | ATTR_IDX(1) | SH_INNER | AF)
#define DEVICE       (DESC_BLOCK | ATTR_IDX(0) | AF | PXN | UXN)

#define GB           (1UL << 30)
#define MB2          (1UL << 21)
#define RAM_SLOT     ((uint64_t)RAM_BASE / GB)

_Static_assert(((uint64_t)RAM_BASE + RAM_SIZE - 1) / GB == (uint64_t)RAM_BASE / GB,
               "RAM must sit inside one 1 GB slot");
_Static_assert((uint64_t)RAM_BASE % MB2 == 0 && RAM_SIZE % MB2 == 0, "RAM must be 2 MB aligned");

static uint64_t l1[512] __attribute__((aligned(4096)));
static uint64_t l2_ram[512] __attribute__((aligned(4096)));

void mmu_build(void) {
    for (uint64_t s = 0; s < 4; s++)
        l1[s] = (s == RAM_SLOT) ? ((uint64_t)l2_ram | DESC_TABLE) : ((s * GB) | DEVICE);
    for (uint64_t i = 0; i < 512; i++) {
        uint64_t pa = RAM_SLOT * GB + i * MB2;
        int ram = pa >= (uint64_t)RAM_BASE && pa < (uint64_t)RAM_BASE + RAM_SIZE;
        l2_ram[i] = pa | (ram ? NORMAL : DEVICE);
    }
    /* U-Boot ran with caches on: drop any stale lines over the tables. */
    dsb(sy);
    for (uint64_t a = (uint64_t)l1; a < (uint64_t)l1 + sizeof l1; a += 64)
        __asm__ volatile("dc ivac, %0" :: "r"(a) : "memory");
    for (uint64_t a = (uint64_t)l2_ram; a < (uint64_t)l2_ram + sizeof l2_ram; a += 64)
        __asm__ volatile("dc ivac, %0" :: "r"(a) : "memory");
    dsb(sy);
}

/* Reads no variables, only link-time constants, so a secondary core can call
 * it with its caches off while core 0's copies of shared data sit in cache. */
void mmu_enable(void) {
    sysreg_write(mair_el1, 0xff00UL);            /* attr0 Device-nGnRnE, attr1 Normal WB RA/WA */
    sysreg_write(tcr_el1,
                 32UL                            /* T0SZ: 4 GB VA */
                 | (1UL << 8) | (1UL << 10)      /* table walks: inner/outer WB cacheable */
                 | (3UL << 12)                   /* inner shareable */
                 | (1UL << 23));                 /* EPD1: no TTBR1 walks; IPS=0 (4 GB PA) */
    sysreg_write(ttbr0_el1, (uint64_t)l1);
    isb();
    __asm__ volatile("tlbi vmalle1\n dsb nsh\n ic iallu\n dsb nsh" ::: "memory");
    isb();
    uint64_t sctlr = sysreg_read(sctlr_el1);
    sctlr |= (1UL << 0) | (1UL << 2) | (1UL << 12);   /* M, C, I */
    sctlr &= ~(1UL << 1);                             /* A: allow unaligned */
    sysreg_write(sctlr_el1, sctlr);
    isb();
}
