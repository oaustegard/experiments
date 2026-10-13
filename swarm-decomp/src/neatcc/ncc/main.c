#include <fcntl.h>
#include "sys.h"
#include "fn_cpp_define.h"
#include "fn_cpp_init.h"
#include "fn_cpp_read.h"
#include "fn_cpp_path.h"
#include "fn_out_init.h"
#include "fn_o_write.h"
#include "fn_tok_see.h"
#include "fn_tok_get.h"
#include "fn_tok_done.h"
#include "fn_err.h"
#include "fn_die.h"
#include "basetype.h"
#include "readdefs.h"
#include "readname.h"
#include "globaldef.h"
#include "typedefdef.h"
#include "tok_grp.h"


int main(int argc, char **argv)
{
	int i, fd = 0;
	char oname[128] = {0};
	int ebp;
	unsigned char c;

	cpp_define("__STDC__", "");
	cpp_define("__linux__", "");
	cpp_define("__x86_64__", "");
	cpp_define("__neatcc__", "");
	cpp_define("const", "");
	cpp_define("register", "");
	cpp_define("volatile", "");
	cpp_define("inline", "");
	cpp_define("restrict", "");
	cpp_define("__inline__", "");
	cpp_define("__restrict__", "");
	cpp_define("__attribute__(x)", "");
	cpp_define("__builtin_va_list__", "long");

	for (i = 1; i < argc; i++) {
		if (argv[i][0] != '-')
			break;
		c = argv[i][1];
		if (c == 'I') {
			if (!argv[i][2]) {
				i++;
				cpp_path(argv[i]);
			} else {
				cpp_path(argv[i] + 2);
			}
			c = argv[i][1];
		}
		if (c == 'O') {
			if (!argv[i][2])
				ncc_opt = 2;
			else
				ncc_opt = strtol(argv[i] + 2, 0, 10);
			continue;
		}
		if (c == 'E') {
			fd = 1;
			continue;
		}
		if (c == 'D') {
			char *name = argv[i] + 2;
			char *val = strchr(name, '=');
			if (val) {
				*val = 0;
				val++;
			} else {
				val = "";
			}
			cpp_define(name, val);
			c = argv[i][1];
		}
		if (c == 'o') {
			char *arg = argv[i] + 2;
			if (!argv[i][2]) {
				i++;
				arg = argv[i];
			}
			strcpy(oname, arg);
			c = argv[i][1];
		}
		if (c == 'h') {
			printf("Usage: %s [options] source\n", argv[0]);
			__builtin_putchar(10);
			puts("Options:");
			puts("  -I dir     \tspecify a header directory");
			puts("  -o out     \tspecify output file name");
			puts("  -E         \tpreprocess only");
			puts("  -Dname=val \tdefine a macro");
			puts("  -On        \toptimize (-O0 to disable)");
			return 0;
		}
	}

	if (i == argc)
		die("neatcc: no file given\n");
	if (cpp_init(argv[i]))
		die("neatcc: cannot open <%s>\n", argv[i]);

	if (fd) {
		char *out;
		long outn;

		if (oname[0])
			fd = open(oname, 0x241, 0x180);
		while (!cpp_read(&out, &outn))
			write(fd, out, outn);
		if (oname[0])
			close(fd);
		return 0;
	}

	out_init(0);
	while (*tok_see()) {
		if (strcmp(tok_see(), "typedef") == 0) {
			char *tk;
			tok_get();
			readdefs(typedefdef, 0);
			tk = tok_get();
			if (*tk == ';' && !tk[1])
				continue;
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
			continue;
		}
		int storage = 0;
		struct type t;

		ebp = basetype(&t, &storage);
		if (ebp) {
			if (tok_grp() != 'a')
				goto skip;
			t.kind = 0x14;
			memset(t.rest, 0, sizeof t.rest);
			ebp = 1;
		}
		if (*tok_see() != ';' || tok_see()[1]) {
			for (;;) {
				struct decl d;

				memset(&d, 0, sizeof d);
				if (readname(d.name, &d, &t))
					break;
				if (ebp && tok_grp() == 'a')
					err("type missing!\n");
				globaldef(0, &d, storage);
				if (*tok_see() != ',' || tok_see()[1])
					break;
				tok_get();
			}
		}
	skip:
		if (*tok_see() == ';' && !tok_see()[1])
			tok_get();
	}

	if (!oname[0]) {
		char *p = argv[i];
		char *b = strrchr(p, '/');
		int len;
		if (b)
			p = b + 1;
		len = stpcpy(oname, p) - oname;
		oname[len - 1] = 'o';
	}
	free(locals);
	free(globals);
	free(label_name);
	free(label_ids);
	free(funcs);
	free(typedefs);
	free(structs);
	free(arrays);
	tok_done();
	fd = open(oname, 0x241, 0x180);
	o_write(fd);
	close(fd);
	return 0;
}
