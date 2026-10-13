#include "fn_o_tmpdrop.h"
#include "fn_o_label.h"
#include "fn_o_jmp.h"
#include "fn_o_jz.h"
#include "fn_o_ret.h"
#include "fn_o_uop.h"
#include "fn_o_mklocal.h"
#include "fn_o_rmlocal.h"
#include "fn_o_local.h"
#include "fn_tok_get.h"
#include "fn_tok_see.h"
#include "fn_err.h"
#include "readexpr.h"
#include "ts_de.h"
#include "readdefs.h"
#include "typedefdef.h"
#include "localdef.h"
#include "label_id.h"

static void readstmt(void)
{
	char *tk;
	int has, lbase, lfalse, lend, top, cont_save, brk_save;

	o_tmpdrop(-1);
	nts = 0;
	if (strcmp("{", tk = tok_see()) == 0) {
		int gn, ln, en, an, tn, sn, fn;
		char *s;

		tok_get();
		gn = globals_n;
		ln = locals_n;
		en = enums_n;
		an = arrays_n;
		tn = typedefs_n;
		sn = structs_n;
		fn = funcs_n;
		for (;;) {
			s = tok_see();
			if (strcmp("}", s) != 0) {
				readstmt();
				continue;
			}
			break;
		}
		tok_get();
		locals_n = ln;
		enums_n = en;
		arrays_n = an;
		typedefs_n = tn;
		structs_n = sn;
		funcs_n = fn;
		globals_n = gn;
		return;
	}
	if (readdefs(localdef, 0)) {
		if (strcmp("typedef", tok_see()) != 0)
			goto statement;
		tok_get();
		readdefs(typedefdef, 0);
	}
	tk = tok_get();
	if (strcmp(";", tk) == 0)
		return;
	err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
	return;

statement:
	if (strcmp("if", tok_see()) == 0) {
		tok_get();
		lbase = label;
		lfalse = lbase + 1;
		lend = lbase + 2;
		label = lend;
		tk = tok_get();
		if (tk[0] != '(')
			err("syntax error (expected <%s> but got <%s>)\n", "(", tk);
		for (;;) {
			o_tmpdrop(-1);
			nts = 0;
			readexpr();
			if (strcmp(",", (tk = tok_see())) != 0)
				break;
			tok_get();
		}
		tk = tok_get();
		if (tk[0] != ')')
			err("syntax error (expected <%s> but got <%s>)\n", ")", tk);
		ts_de(1);
		nts--;
		o_jz(lfalse);
		readstmt();
		if (strcmp("else", tok_see()) == 0) {
			tok_get();
			o_jmp(lend);
			o_label(lfalse);
			readstmt();
			o_label(lend);
			return;
		}
		o_label(lfalse);
		return;
	}
	if (strcmp("while", tok_see()) != 0)
		goto notwhile;
	{
		tok_get();
		cont_save = l_cont;
		brk_save = l_break;
		top = label + 2;
		l_break = label + 1;
		label = top;
		l_cont = top;
		o_label(top);
		tk = tok_get();
		if (tk[0] != '(')
			err("syntax error (expected <%s> but got <%s>)\n", "(", tk);
		for (;;) {
			o_tmpdrop(-1);
			nts = 0;
			readexpr();
			if (strcmp(",", (tk = tok_see())) != 0)
				break;
			tok_get();
		}
		tk = tok_get();
		if (tk[0] != ')')
			err("syntax error (expected <%s> but got <%s>)\n", ")", tk);
		ts_de(1);
		nts--;
		o_jz(l_break);
		readstmt();
		o_jmp(l_cont);
		o_label(l_break);
		l_cont = cont_save;
		l_break = brk_save;
		return;
	}
notwhile:
	tk = tok_see();
	if (tk[0] != 'd' || tk[1] != 'o' || tk[2])
		goto notdo;
	{
		tok_get();
		cont_save = l_cont;
		brk_save = l_break;
		top = label + 1;
		l_break = label + 2;
		label = label + 3;
		l_cont = label;
		o_label(top);
		readstmt();
		tk = tok_get();
		if (strcmp("while", tk) != 0)
			err("syntax error (expected <%s> but got <%s>)\n", "while", tk);
		tk = tok_get();
		if (tk[0] != '(')
			err("syntax error (expected <%s> but got <%s>)\n", "(", tk);
		o_label(l_cont);
		readexpr();
		ts_de(1);
		nts--;
		o_uop(0x102);
		o_jz(top);
		tk = tok_get();
		if (tk[0] != ')')
			err("syntax error (expected <%s> but got <%s>)\n", ")", tk);
		o_label(l_break);
		tk = tok_get();
		if (tk[0] != ';')
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		l_break = brk_save;
		l_cont = cont_save;
		return;
	}
notdo:
	if (strcmp("for", tok_see()) == 0) {
		tok_get();
		lbase = label;
		cont_save = l_cont;
		brk_save = l_break;
		l_cont = lbase + 3;
		label = lbase + 4;
		l_break = lbase + 4;
		tk = tok_get();
		if (tk[0] != '(')
			err("syntax error (expected <%s> but got <%s>)\n", "(", tk);
		tk = tok_see();
		if (strcmp(";", tk) != 0) {
			for (;;) {
				o_tmpdrop(-1);
				nts = 0;
				readexpr();
				if (strcmp(",", (tk = tok_see())) != 0)
					break;
				tok_get();
			}
		}
		tk = tok_get();
		if (strcmp(";", tk) != 0)
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		lfalse = lbase + 1;
		lend = lbase + 2;
		o_label(lfalse);
		tk = tok_see();
		if (strcmp(";", tk) != 0) {
			for (;;) {
				o_tmpdrop(-1);
				nts = 0;
				readexpr();
				if (strcmp(",", (tk = tok_see())) != 0)
					break;
				tok_get();
			}
			ts_de(1);
			nts--;
			o_jz(l_break);
		}
		tk = tok_get();
		if (strcmp(";", tk) != 0)
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		o_jmp(lend);
		o_label(l_cont);
		tk = tok_see();
		if (strcmp(")", tk) != 0) {
			for (;;) {
				o_tmpdrop(-1);
				nts = 0;
				readexpr();
				if (strcmp(",", (tk = tok_see())) != 0)
					break;
				tok_get();
			}
		}
		tk = tok_get();
		if (strcmp(")", tk) != 0)
			err("syntax error (expected <%s> but got <%s>)\n", ")", tk);
		o_jmp(lfalse);
		o_label(lend);
		readstmt();
		o_jmp(l_cont);
		o_label(l_break);
		l_break = brk_save;
		l_cont = cont_save;
		return;
	}
	if (strcmp("switch", tok_see()) != 0)
		goto notswitch;
	{
		long loc, sz;
		int dflt, ncase, body;
		long test;

		tok_get();
		brk_save = l_break;
		loc = o_mklocal(8);
		lbase = label;
		label = lbase + 3;
		l_break = label;
		tk = tok_get();
		if (tk[0] != '(')
			err("syntax error (expected <%s> but got <%s>)\n", "(", tk);
		readexpr();
		ts_de(1);
		nts--;
		sz = ts[nts].ptr ? 8 : ts[nts].kind;
		o_local(loc);
		o_tmpswap();
		o_assign(sz);
		o_tmpdrop(1);
		tk = tok_get();
		if (tk[0] != ')')
			err("syntax error (expected <%s> but got <%s>)\n", ")", tk);
		tk = tok_get();
		if (tk[0] != '{')
			err("syntax error (expected <%s> but got <%s>)\n", "{", tk);
		test = lbase + 1;
		ncase = 0;
		body = lbase + 2;
		dflt = 0;
		for (;;) {
			char *s = tok_see();
			if (strcmp("}", s) == 0)
				break;
			if (strcmp("case", tok_see()) == 0 || strcmp("default", tok_see()) == 0) {
				if (ncase)
					o_jmp(body);
				tk = tok_get();
				if (strcmp("case", tk) == 0) {
					o_label(test);
					caseexpr = 1;
					test = label + 1;
					label = test;
					readexpr();
					ts_de(1);
					nts--;
					caseexpr = 0;
					o_local(loc);
					o_deref(sz);
					o_bop(0x82);
					o_jz(test);
					o_tmpdrop(1);
				} else {
					if (!ncase)
						o_jmp(test);
					dflt = label + 1;
					label = dflt;
					o_label(dflt);
				}
				tk = tok_get();
				if (tk[0] != ':')
					err("syntax error (expected <%s> but got <%s>)\n", ":", tk);
				o_label(body);
				ncase++;
				body = label + 1;
				label = body;
			} else {
				readstmt();
			}
		}
		tok_get();
		o_rmlocal(loc, 8);
		o_jmp(l_break);
		o_label(test);
		if (dflt)
			o_jmp(dflt);
		o_label(l_break);
		l_break = brk_save;
		return;
	}
notswitch:
	if (strcmp("return", tok_see()) != 0)
		goto notreturn;
	{
		tok_get();
		has = tok_see()[0] != ';';
		if (has) {
			readexpr();
			ts_de(1);
			nts--;
		}
		tk = tok_get();
		if (tk[0] != ';')
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		o_ret(has);
		return;
	}
notreturn:
	if (strcmp("break", tok_see()) != 0)
		goto notbreak;
	{
		tok_get();
		tok_get();
		tk = tok_get();
		if (tk[0] != ';')
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		o_jmp(l_break);
		return;
	}
notbreak:
	if (strcmp("continue", tok_see()) != 0)
		goto notcontinue;
	{
		tok_get();
		tok_get();
		tk = tok_get();
		if (tk[0] != ';')
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		o_jmp(l_cont);
		return;
	}
notcontinue:
	if (strcmp("goto", tok_see()) != 0)
		goto notgoto;
	{
		tok_get();
		tk = tok_get();
		o_jmp(label_id(tk));
		tk = tok_get();
		if (tk[0] != ';')
			err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
		return;
	}
notgoto:
	o_tmpdrop(-1);
	nts = 0;
	readexpr();
	if ((tk = tok_see())[0] == ':' && !tk[1]) {
		tok_get();
		o_label(label_id(tok_previden));
		return;
	}
	ts_de(1);
	nts--;
	tk = tok_get();
	if (tk[0] != ';')
		err("syntax error (expected <%s> but got <%s>)\n", ";", tk);
}
