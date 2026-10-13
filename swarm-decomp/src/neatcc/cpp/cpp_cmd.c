#include "cpp_macro_protos.h"
#include "cpp_state.h"

static int cpp_cmd(void)
{
	char name[128];
	char file[128];
	char path[1024];
	char c;
	char *e;
	long n;
	int i;
	int k;

	cur++;
	read_word(name);
	if (!strcmp(name, "define")) {
		macro_define();
		return 0;
	}
	if (!strcmp(name, "undef")) {
		read_word(path);
		for (k = mhead[(unsigned char)path[0]]; k > 0; k = mnext[k])
			if (!strcmp(path, macros[k].name) && macros[k].flags == 0) {
				macros[k].flags = 1;
				break;
			}
		return 0;
	}
	if (!strcmp(name, "ifdef") || !strcmp(name, "ifndef") || !strcmp(name, "if")) {
		c = name[2];
		if (c == 0) {
			if (!cpp_eval())
				jumpifs(0);
			return 0;
		}
		read_word(path);
		if (c == 'n') {
			for (k = mhead[(unsigned char)path[0]]; k > 0; k = mnext[k])
				if (!strcmp(path, macros[k].name) && macros[k].flags == 0) {
					jumpifs(0);
					return 0;
				}
			return 0;
		}
		for (k = mhead[(unsigned char)path[0]]; k > 0; k = mnext[k])
			if (!strcmp(path, macros[k].name) && macros[k].flags == 0)
				return 0;
		jumpifs(0);
		return 0;
	}
	if (!strcmp(name, "else") || !strcmp(name, "elif")) {
		jumpifs(1);
		return 0;
	}
	if (!strcmp(name, "endif"))
		return 0;
	if (!strcmp(name, "include")) {
		jumpws();
		e = strchr(buf + cur + 1, buf[cur] == '"' ? '"' : '>');
		n = e - (buf + cur + 1);
		memcpy(file, buf + cur + 1, n);
		cur += n + 2;
		file[n] = 0;
		for (i = nlocs - (*e == '>'); i >= 0; i--) {
			if (locs[i])
				sprintf(path, "%s/%s", locs[i], file);
			else
				strcpy(path, file);
			if (include_file(path) == 0)
				return 0;
		}
		err("cannot include <%s>\n", file);
		return 0;
	}
	err("unknown directive <%s>\n", name);
	return 1;
}
