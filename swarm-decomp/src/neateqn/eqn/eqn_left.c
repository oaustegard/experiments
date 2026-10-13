#include "libc.h"
#include "box.h"
#include "fn_tok_jmp.h"
#include "fn_tok_get.h"
#include "fn_tok_poptext.h"
#include "fn_tok_type.h"
#include "fn_tok_chops.h"
#include "fn_tok_pop.h"
#include "fn_tok_eqnout.h"
#include "fn_ts_sub.h"
#include "fn_ts_sup.h"
#include "fn_nreg.h"
#include "fn_escarg.h"
#include "fn_box_alloc.h"
#include "fn_box_free.h"
#include "fn_box_merge.h"
#include "fn_box_puttext.h"
#include "fn_box_putf.h"
#include "fn_box_move.h"
#include "fn_box_size.h"
#include "fn_box_sqrt.h"
#include "fn_box_sub.h"
#include "fn_box_from.h"
#include "fn_box_wrap.h"
#include "fn_box_vcenter.h"
#include "fn_box_accent.h"
#include "fn_box_bar.h"
#include "fn_box_under.h"
#include "fn_box_empty.h"
#include "eqn_boxuntil.h"
#include "eqn_pile.h"
#include "eqn_commands.h"
#include "eqn_left.h"

static struct box *eqn_left(int sz, struct box *prev, int ps, char *text)
{
	unsigned short s16 = (unsigned short) sz;
	char ldl[32] = {0}, rdl[32] = {0}, txt[32] = {0};
	struct box *b, *m, *sb, *spb, *fromb, *supb, *nb;
	int h, v, cur, reg, t, n, c;
	const char *fnt, *s;
	char *tk;

	if (text != NULL)
		strcpy(txt, text);
	while (!eqn_commands())
		;
	b = box_alloc(ps, prev ? prev->flags : 0, s16);
	if (eqn_gaps(b, ps) == 0) {
		while (eqn_gaps(b, ps) == 0)
			;
		return b;
	}
	cur = ps;
	h = 0;
	v = 0;
	for (;;) {
		if (tok_jmp("fat") == 0)
			continue;
		if (tok_jmp("roman") == 0) {
			strcpy(txt, grfont);
			continue;
		}
		if (tok_jmp("italic") == 0) {
			strcpy(txt, gfont);
			continue;
		}
		if (tok_jmp("bold") == 0) {
			strcpy(txt, gbfont);
			continue;
		}
		if (tok_jmp("font") == 0) {
			strcpy(txt, tok_poptext(1));
			continue;
		}
		if (tok_jmp("size") == 0) {
			cur = box_size(b, tok_poptext(1));
			continue;
		}
		if (tok_jmp("fwd") == 0) {
			h += strtol(tok_poptext(1), NULL, 10);
			continue;
		}
		if (tok_jmp("back") == 0) {
			h -= strtol(tok_poptext(1), NULL, 10);
			continue;
		}
		if (tok_jmp("down") == 0) {
			v += strtol(tok_poptext(1), NULL, 10);
			continue;
		}
		if (tok_jmp("up") == 0) {
			v -= strtol(tok_poptext(1), NULL, 10);
			continue;
		}
		break;
	}
	if (tok_jmp("sqrt") == 0) {
		m = eqn_left(s16 | 1, NULL, cur, txt);
		printf(".ft %s\n", grfont);
		box_sqrt(b, m);
		box_free(m);
	} else if (tok_jmp("type") == 0) {
		t = 0x10;
		tk = tok_poptext(1);
		s = tk;
		if (strcmp("ord", s) != 0 && strcmp("ordinary", s) != 0)
			t = typenum(s);
		m = eqn_left(sz, prev, cur, txt);
		m->first = t;
		m->flags = t;
		box_merge(b, m, 1);
		box_free(m);
		b->flags = t;
	} else if (tok_jmp("pile") == 0 || tok_jmp("cpile") == 0) {
		eqn_pile(b, cur, txt, 'c');
	} else if (tok_jmp("lpile") == 0) {
		eqn_pile(b, cur, txt, 'l');
	} else if (tok_jmp("rpile") == 0) {
		eqn_pile(b, cur, txt, 'r');
	} else if (tok_jmp("matrix") == 0) {
		eqn_matrix(b, cur, txt);
	} else if (tok_jmp("vcenter") == 0) {
		m = eqn_left(sz, prev, cur, txt);
		box_vcenter(b, m);
		box_free(m);
	} else if (tok_jmp("{") == 0) {
		eqn_boxuntil(b, cur, txt, "}");
	} else if (tok_jmp("left") == 0) {
		nb = box_alloc(cur, 0, s16);
		snprintf(ldl, 32, "%s", tok_quotes(tok_poptext(0)));
		eqn_boxuntil(nb, cur, txt, "right");
		snprintf(rdl, 32, "%s", tok_quotes(tok_poptext(0)));
		printf(".ft %s\n", grfont);
		box_wrap(b, nb, ldl[0] ? ldl : NULL, rdl[0] ? rdl : NULL);
		box_free(nb);
	} else {
		if (tok_get() == NULL)
			goto L2d0;
		if (tok_type() == 3)
			goto L2d0;
		if ((h | v) != 0)
			box_move(b, h, v);
		box_putf(b, "\\s%s", escarg(nreg(cur)));
		t = tok_type();
		fnt = txt[0] != '\0' ? txt : ((t & ~2) == 0x11 ? gfont : grfont);
		tk = tok_get();
		c = tk ? tk[0] : 0;
		box_puttext(b, 0, "\\f%s%s", fnt, tk ? tk : "");
		(void) c;
	}

L2d0:
	for (;;) {
		if (tok_get() == NULL)
			break;
		if (tok_jmp("dyad") == 0) {
			printf(".ft %s\n", grfont);
			box_accent(b, "\\(ab");
			continue;
		}
		if (tok_jmp("bar") == 0) {
			printf(".ft %s\n", grfont);
			box_bar(b);
			continue;
		}
		if (tok_jmp("under") == 0) {
			printf(".ft %s\n", grfont);
			box_under(b);
			continue;
		}
		if (tok_jmp("vec") == 0) {
			printf(".ft %s\n", grfont);
			box_accent(b, "\\s[\\n(.s/2u]\\(->\\s0");
			continue;
		}
		if (tok_jmp("tilde") == 0) {
			printf(".ft %s\n", grfont);
			box_accent(b, "\\s[\\n(.s*3u/4u]\\(ap\\s0");
			continue;
		}
		if (tok_jmp("hat") == 0) {
			printf(".ft %s\n", grfont);
			box_accent(b, "\xcb\x86");
			continue;
		}
		if (tok_jmp("dot") == 0) {
			printf(".ft %s\n", grfont);
			box_accent(b, ".");
			continue;
		}
		if (tok_jmp("dotdot") != 0)
			break;
		printf(".ft %s\n", grfont);
		box_accent(b, "..");
	}

	reg = nregmk();
	fromb = NULL;
	supb = NULL;
	sb = NULL;
	spb = NULL;
	if (tok_jmp("sub") == 0) {
		sizesub(reg, ps, ts_sup(s16), s16);
		spb = eqn_left(ts_sup(s16) | 0x20000, NULL, reg, text);
		if (spb == NULL)
			goto L32b;
		if (tok_jmp("sup") == 0) {
			sizesub(reg, ps, ts_sub(s16), s16);
			sb = eqn_left(ts_sub(s16), NULL, reg, text);
		}
		goto L389;
	}
L32b:
	if ((sz & 0x20000) != 0)
		goto L6a8;
	if (tok_jmp("sup") != 0)
		goto L6a8;
	sizesub(reg, ps, ts_sub(s16), s16);
	sb = eqn_left(ts_sub(s16), NULL, reg, text);
	if (sb == NULL)
		goto L6a8;
	spb = NULL;
L389:
	box_sub(b, spb, sb);
L397:
	if (tok_jmp("from") == 0)
		goto L646;
	if ((sz & 0x40000) != 0)
		goto L5e2;
	if (tok_jmp("to") != 0)
		goto L5e2;
	sizesub(reg, ps, ts_sup(s16), s16);
	supb = eqn_left(ts_sup(s16), NULL, reg, text);
	if (supb == NULL)
		goto L5e2;
	goto L40f;
L6a8:
	spb = NULL;
	sb = NULL;
	goto L397;
L646:
	sizesub(reg, ps, ts_sub(s16), s16);
	fromb = eqn_left(ts_sub(s16) | 0x40000, NULL, reg, text);
	if (fromb == NULL)
		goto L3a9;
	if (tok_jmp("to") == 0) {
		sizesub(reg, ps, ts_sup(s16), s16);
		supb = eqn_left(ts_sup(s16), NULL, reg, text);
	}
	goto L40f;
L3a9:
	if ((sz & 0x40000) != 0)
		goto L5e2;
	if (tok_jmp("to") != 0)
		goto L5e2;
	goto L3c9;
L3c9:
	sizesub(reg, ps, ts_sup(s16), s16);
	supb = eqn_left(ts_sup(s16), NULL, reg, text);
	if (supb == NULL)
		goto L5e2;
L40f:
	nb = box_alloc(ps, 0, s16);
	box_from(nb, b, fromb, supb);
	box_free(b);
	nregrm(reg);
	if (spb != NULL)
		box_free(spb);
	if (sb != NULL)
		box_free(sb);
	if (fromb != NULL)
		box_free(fromb);
	if (supb != NULL)
		box_free(supb);
	return nb;
L5e2:
	nregrm(reg);
	if (spb != NULL)
		box_free(spb);
	if (sb != NULL)
		box_free(sb);
	return b;
}
