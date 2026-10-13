#include "cpp_macro_protos.h"
#include "cpp_etok.h"

static long evalpre(void)
{
	long v = 1;
	int np;
	int i;

	if (enext == -1)
		enext = eval_tok();
top:
	if (enext == '!') {
		enext = -1;
		if (evalpre())
			v = 0;
		return v;
	}
	if (enext == -1)
		enext = eval_tok();
	if (enext == '-') {
		v = -v;
		enext = eval_tok();
		goto top;
	}
	if (enext == -1)
		enext = eval_tok();
	if (enext == '~') {
		enext = -1;
		return v * ~evalpre();
	}
	if (enext == -1)
		enext = eval_tok();
	if (enext == 0x101) {
		enext = -1;
		v *= strtol(etok, NULL, 10);
		return v;
	}
	if (enext == -1)
		enext = eval_tok();
	if (enext == 0x100) {
		enext = eval_tok();
		if (enext == '(') {
			np = 0;
			enext = eval_tok();
		} else {
			np = 1;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x100)
			enext = -1;
		for (i = mhead[(unsigned char)etok[0]]; i > 0; i = mnext[i])
			if (!strcmp(etok, macros[i].name) && !macros[i].flags)
				break;
		if (i <= 0)
			v = 0;
		if (np)
			return v;
		if (enext == -1)
			enext = eval_tok();
		if (enext == ')')
			enext = -1;
		return v;
	}
	if (enext == -1)
		enext = eval_tok();
	if (enext == '(') {
		long x;
		enext = -1;
		x = evalcexpr();
		if (enext == -1)
			enext = eval_tok();
		v *= x;
		if (enext == ')')
			enext = -1;
		return v;
	}
	v = -v;
	return v;
}
