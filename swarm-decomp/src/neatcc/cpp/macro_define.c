#include "cpp_macro_protos.h"

static void macro_define(void)
{
	char name[128];
	int i;
	int h;

	read_word(name);
	h = (unsigned char)name[0];
	i = mhead[h];
	while (i > 0 && strcmp(name, macros[i].name))
		i = mnext[i];
	if (i <= 0) {
		i = mcount;
		if (i > 0xfff)
			die("nomem: NDEFS reached!\n");
		mcount = i + 1;
		strcpy(macros[i].name, name);
		mnext[i] = mhead[h];
		mhead[h] = i;
	}
	macros[i].nargs = 0;
	macros[i].isfunc = 0;
	macros[i].flags = 0;
	if (buf[cur] == '(') {
		cur++;
		jumpws();
		if (cur < len && buf[cur] != ')') {
			do {
				int n = macros[i].nargs;
				macros[i].nargs = n + 1;
				readarg(macros[i].arg[n]);
				jumpws();
				if (buf[cur] != ',')
					break;
				cur++;
				jumpws();
			} while (cur < len && buf[cur] != ')');
		}
		cur++;
		macros[i].isfunc = 1;
	}
	read_tilleol(macros[i].body);
}
