#include "fn_mem_buf.h"
#include "tok_read.h"

char *tok_get(void)
{
	if (!tok_set && tok_read())
		return "";
	tok_set = 0;
	return mem_buf(&tok);
}
