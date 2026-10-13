#include <string.h>
#include "fn_o_mklocal.h"
#include "fn_o_jz.h"
#include "fn_o_jmp.h"
#include "fn_o_label.h"
#include "fn_o_num.h"
#include "fn_o_local.h"
#include "fn_o_tmpdrop.h"
#include "fn_o_uop.h"
#include "fn_o_popnum.h"
#include "fn_o_mark.h"
#include "fn_o_rmlocal.h"
#include "fn_o_back.h"
#include "o_assign.h"
#include "o_deref.h"
#include "o_tmpswap.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "fn_err.h"
#include "ts_de.h"
#include "ts_binop.h"
#include "readxor.h"
#include "readand.h"
#include "readcexpr.h"

static void readcexpr(void)
{
	long v;
	long tmp;
	long mk;
	int lf, le;
	int k, fl, pt;
	char *p;

	readxor();
	for (;;) {
		p = tok_see();
		if (*p != '|')
			break;
		if (p[1])
			break;
		tok_get();
		readxor();
		ts_binop(0x13);
	}
	readand();
	p = tok_see();
	if (strcmp("||", p) == 0) {
		tmp = o_mklocal(4);
		lf = label + 1;
		le = label + 2;
		label = le;
		ts_de(1);
		nts--;
		o_uop(0x102);
		o_jz(lf);
		for (;;) {
			p = tok_see();
			if (strcmp("||", p) != 0)
				break;
			tok_get();
			readxor();
			for (;;) {
				p = tok_see();
				if (*p != '|')
					break;
				if (p[1])
					break;
				tok_get();
				readxor();
				ts_binop(0x13);
			}
			readand();
			ts_de(1);
			nts--;
			o_uop(0x102);
			o_jz(lf);
		}
		o_num(0);
		o_local(tmp);
		o_tmpswap();
		o_assign(0x14);
		o_tmpdrop(1);
		o_jmp(le);
		o_label(lf);
		o_num(1);
		o_local(tmp);
		o_tmpswap();
		o_assign(0x14);
		o_tmpdrop(1);
		o_label(le);
		o_local(tmp);
		o_deref(0x14);
		o_rmlocal(tmp, 0x14);
		ts[nts].kind = 0x14;
		ts[nts].flags = 0;
		ts[nts].ptr = 0;
		ts[nts].tag = 0;
		nts++;
	}
	p = tok_see();
	if (p[0] == '?' && !p[1]) {
		tok_get();
		ncexpr++;
		ts_de(1);
		nts--;
		if (o_popnum(&v)) {
			lf = label + 1;
			le = label + 2;
			label = le;
			o_jz(lf);
			readcexpr();
			ts_de(1);
			nts--;
			k = ts[nts].kind;
			fl = ts[nts].flags;
			pt = ts[nts].ptr;
			if (k | fl | pt) {
				tmp = o_mklocal(8);
				o_local(tmp);
				o_tmpswap();
				o_assign(8);
				o_tmpdrop(1);
				o_jmp(le);
				p = tok_get();
				if (p[0] != ':' || p[1])
					err("syntax error (expected <%s> but got <%s>)\n", ":", p);
				o_label(lf);
				readcexpr();
				ts_de(1);
				o_local(tmp);
				o_tmpswap();
				o_assign(8);
				o_tmpdrop(1);
				o_label(le);
				o_local(tmp);
				o_deref(8);
				o_rmlocal(tmp, 8);
			} else {
				o_jmp(le);
				p = tok_get();
				if (p[0] != ':' || p[1])
					err("syntax error (expected <%s> but got <%s>)\n", ":", p);
				o_label(lf);
				readcexpr();
				ts_de(1);
				o_label(le);
			}
		} else {
			mk = 0;
			if (v == 0)
				mk = o_mark();
			readcexpr();
			ts_de(1);
			nts--;
			p = tok_get();
			if (p[0] != ':' || p[1])
				err("syntax error (expected <%s> but got <%s>)\n", ":", p);
			if (v == 0) {
				o_back(mk);
				o_tmpdrop(1);
			} else {
				mk = o_mark();
			}
			readcexpr();
			ts_de(1);
			if (v != 0) {
				o_back(mk);
				o_tmpdrop(1);
			}
		}
		ncexpr--;
	}
}
