#include "buf_macro.h"

static void macro_expand(char *name)
{
	int i;
	int j;
	int h;
	int n;
	int argc;
	char *p;
	struct macro *m;

	i = buf_arg_find(name);
	if (i >= 0) {
		m = bufs[i].macro;
		j = -1;
		for (n = 0; n < m->nargs; n++)
			if (!strcmp(name, m->arg[n])) {
				j = n;
				break;
			}
		buf_new(2, bufs[i].arg[j], strlen(bufs[i].arg[j]));
		bufs[bufs_n - 1].argi = i;
		return;
	}
	h = (unsigned char)name[0];
	for (i = mhead[h]; i > 0; i = mnext[i]) {
		m = &macros[i];
		if (!strcmp(name, m->name) && m->flags == 0)
			break;
	}
	if (i <= 0) {
		i = -1;
		m = &macros[i];
	}
	if (macros[i].isfunc == 0) {
		buf_macro(m);
		return;
	}
	jumpws();
	if (buf[cur] != '(')
		return;
	n = bufs_n;
	cur++;
	jumpws();
	argc = 0;
	if (cur < len) {
		p = bufs[n].arg[0];
		while (buf[cur] != ')') {
			argc++;
			readarg(p);
			jumpws();
			if (buf[cur] != ',')
				break;
			cur++;
			p += 1024;
			jumpws();
			if (cur >= len)
				break;
		}
	}
	for (j = argc; j < macros[i].nargs; j++)
		bufs[n].arg[j][0] = 0;
	cur++;
	buf_macro(m);
}
