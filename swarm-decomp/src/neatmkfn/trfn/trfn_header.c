#include <stdio.h>
#include "mkfn_swid.h"
#include "mkfn_noligs.h"
#include "fn_trfn_header.h"

void trfn_header(void)
{
	printf("spacewidth %d\n", mkfn_swid);
	if (!mkfn_noligs)
		printf("ligatures %s%s0\n", mkfn_ligs, mkfn_ligs2);
}
