#include <stdio.h>
#include "sbuf.h"
#include "fn_sbuf_buf.h"

void trfn_cdefs(void)
{
	fputs(sbuf_buf(sbuf_char), stdout);
}
