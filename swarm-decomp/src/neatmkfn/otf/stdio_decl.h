#pragma once
#include <stddef.h>

/*
 * stdio declarations for this TU. <stdio.h> itself must not be included: at -O2 its
 * inline putchar() turns every putchar() call into putc(), which the target does not do.
 */
typedef struct _IO_FILE FILE;
extern FILE *stdout, *stderr;
extern int printf(const char *fmt, ...);
extern int fprintf(FILE *f, const char *fmt, ...);
extern int sprintf(char *buf, const char *fmt, ...);
extern int snprintf(char *buf, size_t n, const char *fmt, ...);
extern int putchar(int c);
extern int fputc(int c, FILE *f);
