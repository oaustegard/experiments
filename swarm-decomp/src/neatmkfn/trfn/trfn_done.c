#include "sbuf.h"
#include "fn_sbuf_free.h"
#include "fn_tab_free.h"

void trfn_done(void)
{
	sbuf_free(sbuf_char);
	tab_free(tab_alts);
	if (tab_agl)
		tab_free(tab_agl);
}
