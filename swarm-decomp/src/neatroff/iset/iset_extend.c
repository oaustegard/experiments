#include "iset.h"
#include "fn_mextend.h"

static void iset_extend(struct iset *s, int n)
{
	s->tab = mextend(s->tab, s->n, n, 8);
	s->rowcap = mextend(s->rowcap, s->n, n, 4);
	s->rowlen = mextend(s->rowlen, s->n, n, 4);
	s->n = n;
}
