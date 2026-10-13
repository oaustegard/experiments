#include "fn_err.h"
#include "fn_tok_addr.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "fn_tok_len.h"
#include "fn_tok_jump.h"
#include "fn_o_popnum.h"
#include "tok_grp.h"
#include "readexpr.h"
#include "ts_de.h"
#include "ncc_data.h"
#include "initsize.h"

static int initsize(void)
{
	char *tok = tok_addr();
	char *p;
	long x;
	int n = 0;
	int depth;

	if (tok_see()[0] == '=' && !tok_see()[1]) {
		tok_get();
		if (tok_grp() == '"') {
			tok_get();
			n = tok_len() - 1;
			tok_jump(tok);
			return n;
		}
		p = tok_get();
		if (p[0] != '{' || p[1])
			err("syntax error (expected <%s> but got <%s>)\n", "{", p);
		for (;;) {
			if (tok_see()[0] == '}' && !tok_see()[1]) {
				tok_get();
				tok_jump(tok);
				return n;
			}
			x = n;
			if (tok_see()[0] == '[' && !tok_see()[1]) {
				tok_get();
				readexpr();
				ts_de(1);
				nts--;
				o_popnum(&x);
				p = tok_get();
				if (p[0] != ']' || p[1])
					err("syntax error (expected <%s> but got <%s>)\n", "]", p);
				p = tok_get();
				if (p[0] != '=' || p[1])
					err("syntax error (expected <%s> but got <%s>)\n", "=", p);
			}
			if (n <= x)
				n = x + 1;
			for (;;) {
				if (tok_see()[0] == '}' && !tok_see()[1])
					break;
				if (tok_see()[0] == ',' && !tok_see()[1])
					break;
				p = tok_get();
				if (p[0] == '{' && !p[1]) {
					depth = 0;
					for (;;) {
						if (tok_see()[0] == '}' && !tok_see()[1]) {
							if (depth == 0)
								break;
							depth--;
						}
						p = tok_get();
						if (p[0] == '{' && !p[1])
							depth++;
					}
					p = tok_get();
					if (p[0] != '}' || p[1])
						err("syntax error (expected <%s> but got <%s>)\n", "}", p);
				}
			}
			if (tok_see()[0] == ',' && !tok_see()[1])
				tok_get();
		}
	}
	return 0;
}
