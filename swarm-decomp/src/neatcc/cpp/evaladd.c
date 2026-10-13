#include "cpp_protos.h"

static long evaladd(void)
{
	long v = evalmul();
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == '+') {
			enext = -1;
			v += evalmul();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == '-') {
			enext = -1;
			v -= evalmul();
			continue;
		}
		return v;
	}
}
