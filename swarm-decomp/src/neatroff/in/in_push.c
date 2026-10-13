#include "vars.h"
#include "fn_in_push.h"

void
in_push(char *s, char **args)
{
	struct src *p, *prev;
	char **a = NULL;
	int n, i, l;

	n = strlen(s);
	p = xmalloc(sizeof *p);
	memset(p, 0, sizeof *p);
	prev = buf;
	buf = p;
	p->prev = prev;
	p->name = xmalloc(n + 1);
	buf->len = n;
	strcpy(buf->name, s);
	if (args != NULL) {
		a = xmalloc(32 * sizeof(char *));
		for (i = 0; i < 32; i++) {
			a[i] = NULL;
			if (args[i]) {
				l = strlen(args[i]) + 1;
				a[i] = xmalloc(l);
				memcpy(a[i], args[i], l);
			}
		}
	}
	buf->args = a;
}
