#include "ic_ent.h"
#include "ic_tab.h"
#include "ic_put.h"
#include "ic_num.h"
#include "ic_off.h"
#include "iv_tab.h"
#include "opt.h"
#include "fn_i_imm.h"
#include "fn_i_reg.h"

/* binary operator t: fold constants and strength-reduce against the two operands on iv */
void o_bop(long t)
{
	int a, b;
	struct ic_ent *e, *f;
	long x0, x, y, n, op, k, top, shift, lr, s1;
	long p1, p2, p3, p4, p5;
	unsigned long u;
	int i, s, r;
	long kk;
	static const int cond[] = {5, 4, 2, 3, 1, 0};

	b = iv[iv_n - 1];
	a = iv[iv_n - 2];
	iv_n -= 2;
	if (opt(1)) {
		if ((ic[a].op & 0x400 && ic[a].op & 0x700000)
		    && !(ic[b].op & 0x400 && ic[b].op & 0x700000)) {
			ic_put(ic[a].op, ic[a].a, ic[a].b, ic[a].c);
			iv_n--;
			a = iv[iv_n];
		}
	}
	ic_put(t, a, b, 0);
	if (!opt(1))
		return;
	io_num();
	n = iv_n - 1;
	top = iv[n];
	e = &ic[top];
	op = e->op;
	if (!(op & 0x40))
		goto stage;
	k = op & 0xffffff;
	if (k == 0x40 && !ic_num(ic, e->a, &x0)) {
		long tmp = e->a;

		e->a = e->b;
		e->b = tmp;
	}
	if (ic_num(ic, e->b, &x0))
		goto operand;
	u = x0;
	if (u & 1) {
		if (u >> 1)
			goto operand;
		shift = 0;
	} else {
		i = 0;
		do {
			s = i;
			i++;
			if (i == 64)
				goto zero;
		} while (!(u & (1u << i)));
		if (u >> (s + 2))
			goto operand;
		shift = i;
	}
have_shift:
	if (k == 0x40)
		goto mul_pow2;
	goto kind_chk;
zero:
	shift = 64;
	if (u)
		goto have_shift;
	if (k != 0x40)
		goto kind_chk;
	n = n < 0 ? 0 : n;
	iv_n = n;
	ic_put(0x100400, 0, 0, 0);
	goto after_put;
kind_chk:
	if (k == 0x41)
		goto div_pow2;
	if (k == 0x42)
		goto mod_pow2;
	goto operand;
mul_pow2:
	n = n < 0 ? 0 : n;
	iv_n = n;
	if (x0 == 1)
		goto replace_a;
	ic_put(0x100400, shift, 0, 0);
	iv_n--;
	ic_put(0x8000020, ic[top].a, iv[iv_n], 0);
	goto after_put;
div_pow2:
	if (op & 0x10000000)
		goto operand;
	n = n < 0 ? 0 : n;
	iv_n = n;
	if (x0 == 1)
		goto replace_a;
	ic_put(0x100400, shift, 0, 0);
	iv_n--;
	ic_put(0x8000021, ic[top].a, iv[iv_n], 0);
	goto after_put;
mod_pow2:
	if (op & 0x10000000)
		goto operand;
	n = n < 0 ? 0 : n;
	iv_n = n;
	if (x0 == 1) {
		ic_put(0x100400, 0, 0, 0);
		goto after_put;
	}
	ic_put(0x100400, 64 - shift, 0, 0);
	iv_n--;
	ic_put(0x8000020, ic[top].a, iv[iv_n], 0);
	iv_n--;
	s1 = iv[iv_n];
	ic_put(0x100400, 64 - shift, 0, 0);
	iv_n--;
	ic_put(0x8000021, s1, iv[iv_n], 0);
	goto after_put;
replace_a:
	iv[n] = e->a;
	iv_n = n + 1;
	goto stage;
after_put:
	n = iv_n - 1;
	top = iv[n];
	goto stage;
operand:
	top = iv[n];
stage:
	r = ic_off(ic, top, &x, &y);
	if (!r && x != top) {
		f = &ic[x];
		if ((f->op & 0x200400) == 0x200400) {
			n = iv_n - 1;
			iv_n = n < 0 ? 0 : n;
			ic_put(0x200400, f->a, y + f->b, 0);
			goto tail;
		}
		if ((f->op & 0x400400) == 0x400400) {
			n = iv_n - 1;
			iv_n = n < 0 ? 0 : n;
			ic_put(0x400400, f->a, y + f->b, 0);
			goto tail;
		}
	}
tail:
	e = &ic[ic_n - 1];
	op = e->op;
	if (op & 0x700000)
		return;
	k = op & 0xffffff;
	if (k > 0x40) {
		if (k == 0x82 || k == 0x83)
			goto swap_plain;
		if (k == 0x80 || k == 0x81)
			goto swap_mirror;
		if (k == 0x84)
			goto swap_mirror;
		if (k != 0x85)
			goto t278;
		goto swap_mirror;
	}
	if (!(op & 0xfffff0))
		goto t278;
	if (k == 0x10 || k == 0x12 || k == 0x13 || k == 0x14 || k == 0x30)
		goto swap_plain;
	goto t278;
swap_plain:
	if (!ic_num(ic, e->a, &x)) {
		long tmp = e->a;

		e->a = e->b;
		e->b = tmp;
	}
	goto t278;
swap_mirror:
	if (!ic_num(ic, e->a, &x)) {
		long oa = e->a, ob = e->b;

		e->b = (int)oa;
		e->a = ob;
		e->op = (int)((op & ~0xfL) | cond[op & 0xf]);
	}
	goto t278;
t278:
	if (op & 0x4000)
		goto t3a8;
t281:
	if (!(op & 0xf0))
		return;
	if (ic_num(ic, e->b, &x))
		return;
	kk = k - 0x10;
	if ((unsigned long)kk <= 1 || (op & 0x20)) {
		if (x == 0)
			goto t714;
	}
	lr = i_reg(e->op | 0x100000, &p1, &p2, &p3, &p4, &p5);
	if (lr)
		return;
	if (!i_imm(p3, x))
		return;
	e->op |= 0x100000;
	e->b = x;
	return;
t714:
	n = iv_n - 1;
	if (n < 0)
		n = 0;
	iv[n] = e->a;
	iv_n = n + 1;
	return;
t3a8:
	if (!ic_num(ic, e->a, &x)) {
		long oa = e->a, ob = e->b;

		e->b = (int)oa;
		e->a = ob;
		e->op = (int)((op & ~0xfL) | cond[op & 0xf]);
	}
	if (ic_num(ic, e->b, &x))
		goto t281;
	lr = i_reg(e->op | 0x100000, &p1, &p2, &p3, &p4, &p5);
	if (lr)
		goto t281;
	if (i_imm(p3, x)) {
		e->op |= 0x100000;
		e->b = x;
		return;
	}
	goto t281;
}
