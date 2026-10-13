#include <stdarg.h>
#include <stdio.h>
#include "outf.h"

static void outf(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stdout, fmt, ap);
	va_end(ap);
}
