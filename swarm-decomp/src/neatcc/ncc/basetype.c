#include <stdlib.h>
#include <string.h>
#include "enumdef.h"
#include "tdef.h"
#include "type.h"
#include "ncc_data.h"
#include "ncc_data_enums.h"
#include "ncc_data_stmt.h"
#include "ncc_data_typedefs.h"
#include "fn_err.h"
#include "fn_o_popnum.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "tok_grp.h"
#include "ts_de.h"
#include "readexpr.h"
#include "readdefs.h"
#include "basetype.h"

static int basetype(struct type *t, int *storage)
{
	int kind = 4, sign = 1, nkw = 0;
	char name[0x80] = "";
	char ebuf[0x80];

	*storage = 0;
	t->flags = 0;
	t->ptr = 0;
	t->tag = 0;
	for (;;) {
		if (strcmp("static", tok_see()) == 0) {
			tok_get();
			*storage |= 1;
			nkw++;
			continue;
		}
		if (strcmp("extern", tok_see()) == 0) {
			tok_get();
			*storage |= 2;
			nkw++;
			continue;
		}
		if (strcmp("void", tok_see()) == 0) {
			tok_get();
			kind = 0;
			goto done;
		}
		if (strcmp("int", tok_see()) == 0) {
			tok_get();
			sign <<= 4;
			kind |= sign;
			goto done;
		}
		if (strcmp("char", tok_see()) == 0) {
			tok_get();
			sign <<= 4;
			kind = sign | 1;
			goto done;
		}
		if (strcmp("short", tok_see()) == 0) {
			tok_get();
			kind = 2;
			nkw++;
			continue;
		}
		if (strcmp("long", tok_see()) == 0) {
			tok_get();
			kind = 8;
			nkw++;
			continue;
		}
		if (strcmp("signed", tok_see()) == 0) {
			tok_get();
			sign = 1;
			nkw++;
			continue;
		}
		if (strcmp("unsigned", tok_see()) == 0) {
			tok_get();
			sign = 0;
			nkw++;
			continue;
		}
		if (strcmp("union", tok_see()) == 0 || strcmp("struct", tok_see()) == 0) {
			int is_union = strcmp("union", tok_get()) == 0;
			int n;

			if (tok_grp() == 'a')
				strcpy(name, tok_get());
			if (strcmp("{", tok_see()) != 0) {
				n = struct_find(name, is_union);
			} else {
				char *tok;

				n = struct_find(name, is_union);
				tok = tok_get();
				if (strcmp("{", tok) != 0)
					err("syntax error (expected <%s> but got <%s>)\n", "{", tok);
				while (strcmp("}", tok_see()) != 0) {
					readdefs(structdef, n);
					tok = tok_get();
					if (strcmp(";", tok) != 0)
						err("syntax error (expected <%s> but got <%s>)\n", ";", tok);
				}
				tok_get();
			}
			t->flags |= 2;
			t->arr = n;
			t->kind = 8;
			return 0;
		}
		if (strcmp("enum", tok_see()) == 0) {
			long v = 0;

			tok_get();
			if (tok_grp() == 'a')
				tok_get();
			if (strcmp("{", tok_see()) != 0) {
				kind = 0x14;
				goto done;
			}
			if (strcmp("{", tok_get()) != 0)
				err("syntax error (expected <%s> but got <%s>)\n", "{", "{");
			for (;;) {
				char *end;
				long cur;
				int n;

				if (strcmp("}", tok_see()) == 0) {
					tok_get();
					break;
				}
				end = stpcpy(ebuf, tok_get());
				if (strcmp("=", tok_see()) == 0) {
					tok_get();
					readexpr();
					ts_de(1);
					nts--;
					if (o_popnum(&v))
						err("const expr expected!\n");
				}
				cur = v;
				v = cur + 1;
				n = enums_n;
				if (n >= enums_sz) {
					int sz = enums_sz * 2;
					void *p;

					if (sz < 0x80)
						sz = 0x80;
					enums_sz = sz;
					p = malloc((long)sz * 0x84);
					memcpy(p, enums, (long)n * 0x84);
					memset((char *)p + (long)n * 0x84, 0, ((long)sz - n) * 0x84);
					free(enums);
					enums = p;
				}
				enums_n = n + 1;
				((struct enumdef *)enums + n)->val = 0;
				memcpy((struct enumdef *)enums + n, ebuf, end - ebuf + 1);
				((struct enumdef *)enums + n)->val = cur;
				if (strcmp(",", tok_see()) == 0)
					tok_get();
			}
			kind = 0x14;
			goto done;
		}
		if (tok_grp() == 'a') {
			char *nm = tok_see();
			int i;

			for (i = typedefs_n - 1; i >= 0; i--) {
				struct tdef *td = (struct tdef *)typedefs + i;

				if (strcmp(nm, td->name) == 0) {
					tok_get();
					td = (struct tdef *)typedefs + i;
					*t = td->t;
					return 0;
				}
			}
		}
		if (nkw == 0)
			return 1;
		kind |= sign << 4;
		goto done;
	}
done:
	t->kind = kind;
	return 0;
}
