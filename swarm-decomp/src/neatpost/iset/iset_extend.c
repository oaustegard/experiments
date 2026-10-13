#include "iset.h"
#include "fn_mextend.h"

static void iset_extend(struct iset *s, int n)
{
	s->a = mextend(s->a, s->n, n, 8);
	s->b = mextend(s->b, s->n, n, 4);
	s->c = mextend(s->c, s->n, n, 4);
	s->n = n;
}
