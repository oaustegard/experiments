#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define sysreg_read(r) ({ uint64_t _v; __asm__ volatile("mrs %0, " #r : "=r"(_v)); _v; })
#define sysreg_write(r, v) __asm__ volatile("msr " #r ", %0" :: "r"((uint64_t)(v)))
#define isb() __asm__ volatile("isb" ::: "memory")
#define dsb(o) __asm__ volatile("dsb " #o ::: "memory")

extern uint64_t boot_el;
extern char vectors[];

/* uart.c */
void uart_putc(char c);
void kprintf(const char *fmt, ...);
void kvprintf(const char *fmt, va_list ap);
void print_lock(void);
void print_unlock(void);

/* lib.c */
void *memset(void *d, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);
int memcmp(const void *a, const void *b, size_t n);

/* timer */
static inline uint64_t ticks(void) { isb(); return sysreg_read(cntvct_el0); }
static inline uint64_t tick_hz(void) { return sysreg_read(cntfrq_el0); }
static inline double seconds(uint64_t dt) { return (double)dt / (double)tick_hz(); }

/* mmu.c */
void mmu_build(void);
void mmu_enable(void);

/* psci.c / smp.c */
int64_t psci_call(uint64_t fn, uint64_t a1, uint64_t a2, uint64_t a3);
uint32_t psci_version(void);
void psci_system_off(void) __attribute__((noreturn));
int smp_init(void);
int smp_cores(void);
typedef void (*job_fn)(int core, int ncores, void *arg);
void smp_run(job_fn fn, void *arg, int ncores);
