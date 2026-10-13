#include "cpp_macro_protos.h"

static void buf_macro(struct macro *m)
{
	struct mem mem;
	char name[128];
	char *p;
	unsigned char *q;
	unsigned char *r;
	char *w;
	char *s;
	char c;
	unsigned char nx;
	int quote = 0;
	int hash;
	int k;
	int nargs;
	int n;
	int len;

	mem_init(&mem);
	p = m->body;
	c = *p;
	if (!c)
		goto end;
top:
	if (quote) {
		if (c == quote)
			goto d8;
		q = (unsigned char *)p;
		if (c == '\\') {
			mem_putc(&mem, '\\');
			q++;
		}
		goto e1;
	}
qb:
	if (c == '"' || c == '\'') {
		quote = c;
		q = (unsigned char *)p;
		goto e1;
	}
d8:
	if (c == '#')
		goto hash;
dc:
	quote = 0;
	q = (unsigned char *)p;
e1:
	if (!*q)
		goto end;
	mem_putc(&mem, *q);
	p = (char *)q + 1;
	c = *p;
	if (c)
		goto top;
	goto end;
hash:
	nx = p[1];
	q = (unsigned char *)p + 1;
	hash = 1;
	if (nx == '#') {
		hash = 2;
		nx = p[2];
		q = (unsigned char *)p + 2;
	}
	if (!nx)
		goto dc;
	k = bufs_n;
	w = name;
	while (nx && w - name <= 126 && (isalnum(nx) || nx == '_')) {
		*w++ = nx;
		nx = *++q;
	}
	*w = 0;
	nargs = m->nargs;
	for (n = 0; n < nargs; n++)
		if (!strcmp(name, m->arg[n]))
			break;
	if (n == nargs)
		goto dc;
found:
	s = bufs[k].arg[n];
	if (hash == 1) {
		mem_putc(&mem, '"');
		for (r = (unsigned char *)s; *r; ) {
			if (*r == '"')
				mem_putc(&mem, '\\');
			mem_putc(&mem, *r++);
		}
		mem_putc(&mem, '"');
	} else {
		for (r = (unsigned char *)s; *r; )
			mem_putc(&mem, *r++);
	}
	p = (char *)q;
	c = *p;
	if (!c)
		goto end;
	goto qb;
end:
	len = mem_len(&mem);
	buf_new(1, mem_get(&mem), len);
	mem_done(&mem);
	bufs[bufs_n - 1].macro = m;
}
