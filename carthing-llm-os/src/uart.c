#include "kernel.h"

static inline void mmio_w32(uint64_t a, uint32_t v) { *(volatile uint32_t *)a = v; }
static inline uint32_t mmio_r32(uint64_t a) { return *(volatile uint32_t *)a; }

void uart_putc(char c) {
    if (c == '\n') uart_putc('\r');
#if defined(UART_PL011)
    while (mmio_r32(UART_BASE + 0x18) & (1u << 5)) ;   /* FR.TXFF */
    mmio_w32(UART_BASE + 0x00, (uint8_t)c);             /* DR */
#elif defined(UART_MESON)
    while (mmio_r32(UART_BASE + 0x0c) & (1u << 21)) ;  /* STATUS.TX_FULL */
    mmio_w32(UART_BASE + 0x00, (uint8_t)c);             /* WFIFO */
#endif
}

/* Spinlock usable only once the MMU is on (exclusives need cacheable memory). */
static volatile uint32_t plock;
static volatile int mmu_up;
void print_lock(void) {
    if (!mmu_up) { mmu_up = (sysreg_read(sctlr_el1) & 1); if (!mmu_up) return; }
    while (__atomic_exchange_n(&plock, 1, __ATOMIC_ACQUIRE)) ;
}
void print_unlock(void) {
    if (mmu_up) __atomic_store_n(&plock, 0, __ATOMIC_RELEASE);
}

static void put_u64(uint64_t v, unsigned base, int width, char pad) {
    char buf[24]; int i = 0;
    do { unsigned d = v % base; buf[i++] = d < 10 ? '0' + d : 'a' + d - 10; v /= base; } while (v);
    while (i < width) buf[i++] = pad, width--;
    while (i) uart_putc(buf[--i]);
}

static void put_f64(double v, int prec) {
    if (v != v) { uart_putc('n'); uart_putc('a'); uart_putc('n'); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    uint64_t scale = 1;
    for (int i = 0; i < prec; i++) scale *= 10;
    uint64_t x = (uint64_t)(v * (double)scale + 0.5);
    put_u64(x / scale, 10, 0, ' ');
    if (prec) { uart_putc('.'); put_u64(x % scale, 10, prec, '0'); }
}

static void put_e64(double v, int prec) {
    if (v < 0) { uart_putc('-'); v = -v; }
    int e = 0;
    if (v != 0) {
        while (v >= 10) { v /= 10; e++; }
        while (v < 1) { v *= 10; e--; }
    }
    put_f64(v, prec);
    uart_putc('e'); uart_putc(e < 0 ? '-' : '+');
    put_u64((uint64_t)(e < 0 ? -e : e), 10, 2, '0');
}

/* %d %u %x %s %c %p %f %e, with l/ll, a width ('-' to left-justify a string),
 * '0' padding and .N precision. */
void kvprintf(const char *f, va_list ap) {
    for (; *f; f++) {
        if (*f != '%') { uart_putc(*f); continue; }
        f++;
        char pad = ' '; int width = 0, prec = 6, lng = 0, left = 0;
        if (*f == '-') { left = 1; f++; }
        if (*f == '0') { pad = '0'; f++; }
        while (*f >= '0' && *f <= '9') width = width * 10 + (*f++ - '0');
        if (*f == '.') { f++; prec = 0; while (*f >= '0' && *f <= '9') prec = prec * 10 + (*f++ - '0'); }
        while (*f == 'l') { lng++; f++; }
        switch (*f) {
        case 'd': {
            int64_t v = lng ? va_arg(ap, int64_t) : va_arg(ap, int);
            if (v < 0) { uart_putc('-'); v = -v; if (width) width--; }
            put_u64((uint64_t)v, 10, width, pad); break;
        }
        case 'u': put_u64(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 10, width, pad); break;
        case 'x': put_u64(lng ? va_arg(ap, uint64_t) : va_arg(ap, unsigned), 16, width, pad); break;
        case 'p': uart_putc('0'); uart_putc('x'); put_u64((uint64_t)va_arg(ap, void *), 16, 0, ' '); break;
        case 's': {
            const char *s = va_arg(ap, const char *);
            int len = 0; while (s[len]) len++;
            if (!left) for (int i = len; i < width; i++) uart_putc(' ');
            while (*s) uart_putc(*s++);
            if (left) for (int i = len; i < width; i++) uart_putc(' ');
            break;
        }
        case 'c': uart_putc((char)va_arg(ap, int)); break;
        case 'f': put_f64(va_arg(ap, double), prec); break;
        case 'e': put_e64(va_arg(ap, double), prec); break;
        case '%': uart_putc('%'); break;
        default: uart_putc('?'); break;
        }
    }
}

void kprintf(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    print_lock(); kvprintf(fmt, ap); print_unlock();
    va_end(ap);
}

static void kprintf_raw(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); kvprintf(fmt, ap); va_end(ap);
}

/* Called from the vector table. No lock: the faulting core may hold it. */
void exc_handler(uint64_t idx, uint64_t esr, uint64_t elr, uint64_t far) {
    static const char *kind[] = { "sync", "irq", "fiq", "serror" };
    kprintf_raw("\n!! %s exception (vector %u) on core %u: ESR=%lx EC=%lx ELR=%lx FAR=%lx\n",
                kind[idx & 3], (unsigned)idx, (unsigned)(sysreg_read(mpidr_el1) & 0xff),
                esr, esr >> 26, elr, far);
}
