#include "cpp_protos.h"

static long evaleq(void)
{
	long v = evalcmp();
	for (;;) {
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x3d3d00) {
			enext = -1;
			v = evalcmp() == v;
			continue;
		}
		if (enext == -1)
			enext = eval_tok();
		if (enext == 0x213d00) {
			enext = -1;
			v = evalcmp() != v;
			continue;
		}
		return v;
	}
}
