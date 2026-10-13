#include "otf_chain.h"

static void
gctx_lookahead(struct otf_chain *c, int k)
{
	int i, j;

	if (c == NULL)
		return;
	for (i = c->seq + k; i < c->ni; i++)
		printf(" =@%d", c->in[i]);
	for (j = 0; j < c->nl; j++)
		printf(" =@%d", c->la[j]);
}
