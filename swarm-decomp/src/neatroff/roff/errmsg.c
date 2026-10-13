#include <stdarg.h>
#include <stdio.h>
#include "fn_errmsg.h"

void errmsg(char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}
