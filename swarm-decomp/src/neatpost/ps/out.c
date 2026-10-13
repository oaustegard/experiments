#include <stdarg.h>
#include <stdio.h>
#include "outf.h"
#include "ps_state.h"

void out(const char *fmt, ...)
{
	va_list ap;

	o_flush();
	va_start(ap, fmt);
	vfprintf(stdout, fmt, ap);
	va_end(ap);
}
