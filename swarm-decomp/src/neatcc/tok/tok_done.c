#include "fn_mem_done.h"
void tok_done(void)
{
	mem_done(&tok);
	mem_done(&tok_mem);
}
