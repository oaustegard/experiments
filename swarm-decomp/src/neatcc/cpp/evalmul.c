#include "cpp_protos.h"

static long evalmul(void)
{
	long v = evalpre();
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == '*') {
			enext = -1;
			v *= evalpre();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == '/') {
			enext = -1;
			v /= evalpre();
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == '%') {
			enext = -1;
			v %= evalpre();
			continue;
		}
		return v;
	}
}
