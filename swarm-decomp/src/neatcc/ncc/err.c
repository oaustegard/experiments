#include "sys.h"
#include "fn_die.h"
#include "fn_tok_addr.h"
#include "fn_cpp_loc.h"

void err(char *fmt, ...)
{
	va_list ap;
	char buf[512];

	va_start(ap, fmt);
	vsprintf(buf, fmt, ap);
	va_end(ap);
	die("%s: %s", cpp_loc(tok_addr()), buf);
}
