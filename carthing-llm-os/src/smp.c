#include "kernel.h"

#define PSCI_VERSION     0x84000000UL
#define PSCI_CPU_ON      0xC4000003UL
#define PSCI_SYSTEM_OFF  0x84000008UL

uint64_t boot_el;
extern char secondary_entry[];

static int use_smc(void) {
#if PSCI_AUTO
    return boot_el >= 2;     /* QEMU virt: smc with virtualization=on, hvc otherwise */
#else
    return 1;
#endif
}

int64_t psci_call(uint64_t fn, uint64_t a1, uint64_t a2, uint64_t a3) {
    register uint64_t x0 __asm__("x0") = fn, x1 __asm__("x1") = a1,
                      x2 __asm__("x2") = a2, x3 __asm__("x3") = a3;
    if (use_smc())
        __asm__ volatile("smc #0" : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3) :: "memory");
    else
        __asm__ volatile("hvc #0" : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3) :: "memory");
    return (int64_t)x0;
}

uint32_t psci_version(void) { return (uint32_t)psci_call(PSCI_VERSION, 0, 0, 0); }

void psci_system_off(void) {
    psci_call(PSCI_SYSTEM_OFF, 0, 0, 0);
    for (;;) __asm__ volatile("wfe");
}

/* One job slot. Core 0 publishes fn/arg and bumps gen; workers run it and
 * count themselves done. Waits are wfe loops: an sev that lands between the
 * check and the wfe sets the event register, so no wakeup is lost. */
static struct {
    job_fn fn;
    void *arg;
    int ncores;
    uint32_t gen;
    uint32_t done;
} job;
static uint32_t online = 1;

void secondary_main(int core) {
    sysreg_write(vbar_el1, (uint64_t)vectors);
    isb();
    __atomic_fetch_add(&online, 1, __ATOMIC_ACQ_REL);
    __asm__ volatile("sev");
    uint32_t seen = 0;
    for (;;) {
        uint32_t g;
        while ((g = __atomic_load_n(&job.gen, __ATOMIC_ACQUIRE)) == seen)
            __asm__ volatile("wfe");
        seen = g;
        if (core < job.ncores) job.fn(core, job.ncores, job.arg);
        __atomic_fetch_add(&job.done, 1, __ATOMIC_ACQ_REL);
        __asm__ volatile("sev");
    }
}

int smp_init(void) {
    for (int i = 1; i < NCORES; i++) {
        int64_t r = psci_call(PSCI_CPU_ON, (uint64_t)i, (uint64_t)secondary_entry, (uint64_t)i);
        if (r != 0) kprintf("  CPU_ON core %d failed: %ld\n", i, r);
    }
    uint64_t t0 = ticks(), limit = tick_hz();          /* 1 s */
    while (__atomic_load_n(&online, __ATOMIC_ACQUIRE) < NCORES && ticks() - t0 < limit)
        __asm__ volatile("wfe");
    return (int)__atomic_load_n(&online, __ATOMIC_ACQUIRE);
}

int smp_cores(void) { return (int)__atomic_load_n(&online, __ATOMIC_ACQUIRE); }

void smp_run(job_fn fn, void *arg, int ncores) {
    int n = smp_cores();
    if (ncores > n) ncores = n;
    if (ncores <= 1) { fn(0, 1, arg); return; }
    job.fn = fn; job.arg = arg; job.ncores = ncores;
    __atomic_store_n(&job.done, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&job.gen, 1, __ATOMIC_RELEASE);
    __asm__ volatile("sev");
    fn(0, ncores, arg);
    while (__atomic_load_n(&job.done, __ATOMIC_ACQUIRE) < (uint32_t)(n - 1))
        __asm__ volatile("wfe");
}
