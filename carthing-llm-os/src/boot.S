/* Entry for core 0 and for secondaries started with PSCI CPU_ON.
 * The image starts with an arm64 Linux Image header, so U-Boot's `booti` and
 * QEMU's `-kernel` both take it: they clean and disable the caches, then jump
 * here at EL2 or EL1 with the MMU off. */

#define STACK_SHIFT 16                /* 64 KB per core */

    .section .text.boot, "ax"
    .globl _start
_start:
    b       primary_entry             /* code0 */
    .long   0                         /* code1 */
    .quad   TEXT_OFFSET               /* text_offset */
    .quad   _image_size               /* image_size, bss included */
    .quad   0x2                       /* flags: LE, 4 KB pages, 2 MB-aligned base */
    .quad   0, 0, 0
    .ascii  "ARM\x64"                 /* magic */
    .long   0

primary_entry:
    mrs     x19, CurrentEL
    ubfx    x19, x19, #2, #2          /* boot EL, kept for PSCI conduit choice */
    mrs     x0, mpidr_el1
    and     x0, x0, #0xff
    cbnz    x0, park                  /* only core 0 comes in this way */

    bl      el_setup
    ldr     x0, =core_stacks
    add     sp, x0, #(1 << STACK_SHIFT)

    ldr     x0, =__bss_start          /* zero bss: 16-byte aligned stores, legal with MMU off */
    ldr     x1, =__bss_end
1:  cmp     x0, x1
    b.hs    2f
    stp     xzr, xzr, [x0], #16
    b       1b
2:  ldr     x0, =boot_el
    str     x19, [x0]
    bl      kmain
    b       park

    .globl secondary_entry
secondary_entry:                      /* x0 = context_id = core index */
    mov     x20, x0
    bl      el_setup
    ldr     x1, =core_stacks
    add     x2, x20, #1
    lsl     x2, x2, #STACK_SHIFT
    add     sp, x1, x2
    bl      mmu_enable                /* tables already built by core 0 */
    mov     x0, x20
    bl      secondary_main
    .globl park
park:
    wfe
    b       park

/* Drop EL2 -> EL1 if needed and enable FP/SIMD. Clobbers x9, keeps x30. */
el_setup:
    mrs     x9, CurrentEL
    ubfx    x9, x9, #2, #2
    cmp     x9, #2
    b.ne    1f
    mov     x9, #(1 << 31)            /* HCR_EL2.RW: EL1 is AArch64 */
    msr     hcr_el2, x9
    mov     x9, #3                    /* EL1 may read the physical counter and timer */
    msr     cnthctl_el2, x9
    msr     cntvoff_el2, xzr
    mov     x9, #0x33ff               /* CPTR_EL2: no FP/SIMD trap */
    msr     cptr_el2, x9
    msr     hstr_el2, xzr
    ldr     x9, =0x30d00800           /* SCTLR_EL1 RES1 bits, MMU and caches off */
    msr     sctlr_el1, x9
    mov     x9, #0x3c5                /* EL1h, DAIF masked */
    msr     spsr_el2, x9
    adr     x9, 1f
    msr     elr_el2, x9
    eret
1:  mov     x9, #(3 << 20)            /* CPACR_EL1.FPEN: no FP/SIMD trap at EL1 */
    msr     cpacr_el1, x9
    isb
    ret

/* Exception vectors: print the syndrome and stop. */
    .macro vec idx
    .balign 0x80
    mov     x0, #\idx
    mrs     x1, esr_el1
    mrs     x2, elr_el1
    mrs     x3, far_el1
    b       exc_entry
    .endm

    .balign 0x800
    .globl vectors
vectors:
    vec 0;  vec 1;  vec 2;  vec 3
    vec 4;  vec 5;  vec 6;  vec 7
    vec 8;  vec 9;  vec 10; vec 11
    vec 12; vec 13; vec 14; vec 15
exc_entry:
    bl      exc_handler
    b       park

    .bss
    .balign 16
    .globl core_stacks
core_stacks:
    .space  (NCORES << STACK_SHIFT)
