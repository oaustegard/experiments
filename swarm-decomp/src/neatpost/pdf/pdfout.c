#include <stdarg.h>
#include <stdio.h>
static void pdfout(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    pdf_pos += vfprintf(stdout, fmt, ap);
    va_end(ap);
}
