#include <string.h>
static struct array *readarrays(struct type *t)
{
	struct array *r = NULL;
	long v;
	long sz[16];
	int n = 0;
	int i;
	char *p;

	for (;;) {
		p = tok_see();
		if (strcmp("[", p) != 0)
			break;
		tok_get();
		v = 0;
		p = tok_see();
		if (*p == ']' && !p[1]) {
			tok_get();
		} else {
			readexpr();
			ts_de(1);
			nts--;
			if (o_popnum(&v))
				err("const expr expected\n");
			p = tok_get();
			if (p[0] != ']' || p[1])
				err("syntax error (expected <%s> but got <%s>)\n", "]", p);
		}
		sz[n] = v;
		n++;
	}
	if (n == 0)
		return NULL;
	i = n - 1;
	for (; i >= 0; i--) {
		t->arr = array_add(t, sz[i]);
		if (!r)
			r = (struct array *)arrays + t->arr;
		t->kind = 8;
		t->flags = 1;
		t->ptr = 0;
	}
	return r;
}
