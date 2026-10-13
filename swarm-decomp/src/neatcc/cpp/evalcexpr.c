#include "cpp_protos.h"

static long evalcexpr(void)
{
	long v = evalor();
	if (enext == -1)
		enext = eval_tok();
	if (enext == '?') {
		enext = -1;
		if (!v)
			while (eval_tok() != ':')
				;
		return evalor();
	}
	return v;
}
